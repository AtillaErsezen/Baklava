"""
LLM Factory — crafter agent. Fine-tunes several open models on your data in parallel, scores
them on held-out test rows, and exports the winner (text models also as GGUF for Ollama).

  uv run --extra llm llm_factory.py chats.jsonl --goal "answer support questions in our tone"
  uv run --extra llm llm_factory.py tickets.csv --goal "route tickets" --kind decision --no-llm
  uv run --extra llm llm_factory.py hf:org/dataset --goal "..." --models Qwen/Qwen2.5-0.5B-Instruct
  uv run llm_factory.py chats.jsonl --goal "..." --kaggle run.ipynb    # no training here: notebook for Kaggle GPUs

Data: a .jsonl/.json/.csv/.parquet file or hf:org/name[:config[:split]]. Kinds: text, decision,
vision, stt, tts (see llm_train.py). The agent LLM is configured as in agent.py; --no-llm
trains the default candidates for the kind.
"""
import argparse
import base64
import json
import os
import pprint
import re
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

import pandas as pd

import agent
import llm_train
from agent import _tool

MAX_ROUNDS = 2
MAX_CANDIDATES = 4
EMBED_LIMIT = 5 * 2 ** 20  # local data files up to this size travel inside the Kaggle notebook
HUB_TASK = {"text": "text-generation", "decision": "fill-mask", "vision": "image-text-to-text",
            "stt": "automatic-speech-recognition", "tts": "text-to-speech"}
ALIASES = {
    "prompt": ("prompt", "instruction", "question", "input", "query"),
    "response": ("response", "output", "answer", "completion", "target"),
    "image": ("image", "image_path", "img", "picture"),
    "audio": ("audio", "audio_path", "wav", "path", "file"),
    "text": ("text", "sentence", "transcript", "transcription", "content"),
}
NEEDS = {"text": ("prompt", "response"), "vision": ("image", "prompt", "response"), "stt": ("audio", "text"),
         "tts": ("audio", "text"), "decision": ()}

SYSTEM_PROMPT = f"""You are the crafter agent of LLM Factory. You get a dataset and a goal and must deliver the best
fine-tuned open model for it. A Python harness trains and scores every candidate; you choose what to train.

Model kinds
- text: chat or instruction LLM (LoRA). Data: prompt + response columns, or a messages column.
- decision: a "System One" model that returns typed decisions from one encoder pass and never generates text.
  Each question is a column of the data: choice (one of its options), noul (yes/no), score (number in a range).
  Use it when the goal is to classify, route, flag or rate. Base models are encoders such as ModernBERT.
- vision: vision-language model (LoRA). Data: image + prompt + response.
- stt: speech-to-text, Whisper-family models. Data: audio + text.
- tts: text-to-speech. Only SpeechT5 checkpoints (microsoft/speecht5_tts) fit the trainer; vary the
  learning rate, epochs or max_steps instead of the base model.

Workflow
1. get_data_profile. Pick the kind from the goal and the columns. Map the columns the kind needs.
2. Optionally search_models to find base models on Hugging Face that fit the hardware.
3. train_candidates with 2-{MAX_CANDIDATES} candidates that differ in a way worth testing: different base models, or one
   base model with different learning_rate / epochs / lora_r. They train in parallel and are scored on the same
   held-out test rows. Budget: {MAX_ROUNDS} rounds. A second round should refine what the first one showed.
4. write_report. The harness exports the candidate with the highest test score; you explain the result.

Rules
- Only use base_model ids returned by search_models or named by the user, and respect the parameter budget in
  the profile. A model that does not fit the hardware fails and wastes the round.
- Before every tool call, write 1-2 plain sentences explaining your reasoning.
- Never invent numbers; only use numbers returned by tools. The harness appends the score table to your report.
- Report markdown: the data and goal, what was trained and why, the winner and how to use it, caveats."""

_CAND = {"type": "object", "required": ["name", "base_model"], "properties": {
    "name": {"type": "string", "description": "Short unique name, e.g. qwen05-lr1e4."},
    "base_model": {"type": "string", "description": "Hugging Face model id."},
    "learning_rate": {"type": "number"}, "epochs": {"type": "integer", "minimum": 1, "maximum": 20},
    "batch_size": {"type": "integer", "minimum": 1}, "max_steps": {"type": "integer", "minimum": 1},
    "lora_r": {"type": "integer", "description": "LoRA rank for text and vision. Default 16."}}}
TOOLS = [
    _tool("get_data_profile", "Columns, sample rows, row count, the guessed kind and column mapping, inferred "
          "decision questions, and the hardware with its parameter budget."),
    _tool("search_models", "Open models on Hugging Face for a kind, most downloaded first, within the parameter "
          "budget. Returns id, parameters in millions and downloads.", {
              "kind": {"type": "string", "enum": list(llm_train.KINDS)},
              "query": {"type": "string", "description": "Optional search words, e.g. 'instruct', 'turkish'."}},
          ["kind"]),
    _tool("train_candidates", "Fine-tune the candidates in parallel and score each on the held-out test rows. "
          "Returns a leaderboard; higher score is better. kind, columns and questions are fixed by the first round.", {
              "rationale": {"type": "string", "description": "Why these candidates. Shown to the user."},
              "kind": {"type": "string", "enum": list(llm_train.KINDS)},
              "columns": {"type": "object", "description": "Field -> column name. Fields: prompt, response, messages, "
                          "image, audio, text; for decision 'inputs' is the list of input columns. Omit to use the "
                          "mapping guessed in the profile."},
              "questions": {"type": "object", "description": "decision only. Column -> {type: choice, options: [...]} "
                            "| {type: noul} | {type: score, min, max}. Omit to use the profile's."},
              "system": {"type": "string", "description": "text only: optional system prompt baked into training."},
              "candidates": {"type": "array", "minItems": 1, "maxItems": MAX_CANDIDATES, "items": _CAND}},
          ["rationale", "kind", "candidates"]),
    _tool("write_report", "Export the candidate with the highest test score and save the report. Call once, at the end.",
          {"markdown": {"type": "string"}}, ["markdown"]),
]


# ------------------------------------------------------------------ data profile (no ML libraries needed)

def _api(url):
    """GET a Hugging Face API URL as JSON. Sends HF_TOKEN when set (gated or private repos)."""
    headers = {"User-Agent": "llm-factory"}
    if os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.load(r)


def load_sample(source):
    """(sample DataFrame, total rows or None, media) for a local file or an hf: dataset.
    media maps a column to 'image', 'audio' or, for class-label columns, its list of class
    names. Hub datasets are sampled through the dataset viewer API (first 100 rows)."""
    if not source.startswith("hf:"):
        ext = source.rsplit(".", 1)[-1].lower()
        if ext in ("jsonl", "json"):
            df = pd.read_json(source, lines=ext == "jsonl")
        elif ext in ("csv", "parquet"):
            df = getattr(pd, f"read_{ext}")(source)
        else:
            raise SystemExit(f"Unsupported file type '.{ext}'. Use .jsonl, .json, .csv or .parquet, or hf:org/name.")
        return df, len(df), {}
    name, config, split = (source[3:].split(":") + [None, None])[:3]
    viewer = "https://datasets-server.huggingface.co"
    try:
        if not config:
            config = _api(f"{viewer}/splits?dataset={urllib.parse.quote(name)}")["splits"][0]["config"]
        q = urllib.parse.urlencode({"dataset": name, "config": config, "split": split or "train"})
        first = _api(f"{viewer}/first-rows?{q}")
        sizes = _api(f"{viewer}/size?{q}").get("size", {})
    except Exception as e:
        raise SystemExit(f"Could not read Hugging Face dataset '{name}': {e}") from None
    df = pd.DataFrame([r["row"] for r in first["rows"]])
    media = {}
    for f in first["features"]:
        kind = f["type"].get("_type")
        if kind in ("Image", "Audio"):
            media[f["name"]] = kind.lower()
        elif kind == "ClassLabel":  # class ids -> names, as the trainer does
            names = media[f["name"]] = f["type"]["names"]  # the sample may not show every class
            df[f["name"]] = df[f["name"]].map(lambda i, names=names: names[i] if isinstance(i, int) and i >= 0 else None)
    return df, (sizes.get("split") or sizes.get("config") or {}).get("num_rows"), media


def infer_setup(df, media=None):
    """Guess {'kind', 'columns', 'questions'} from a sample. kind is None when the data alone
    cannot tell (audio is stt or tts depending on the goal)."""
    media = media or {}
    lower = {c.lower(): c for c in df.columns}
    cols = {f: lower[a] for f, names in ALIASES.items() for a in reversed(names) if a in lower}

    is_text = lambda s: s.dtype == object or pd.api.types.is_string_dtype(s)  # pandas 3 strings are not object

    def looks_like(kind, exts):
        hits = [c for c, k in media.items() if k == kind]
        hits += [c for c in df.columns if is_text(df[c]) and
                 df[c].dropna().astype(str).str.lower().str.endswith(exts).mean() > 0.9]
        return hits[0] if hits else None

    image = looks_like("image", (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    audio = looks_like("audio", (".wav", ".mp3", ".flac", ".ogg", ".m4a"))
    if audio:
        return {"kind": None, "columns": {**cols, "audio": audio}, "questions": {}}
    if image:
        return {"kind": "vision", "columns": {**cols, "image": image}, "questions": {}}
    if "messages" in lower or ("prompt" in cols and "response" in cols):
        return {"kind": "text", "columns": cols, "questions": {}}

    questions, inputs = {}, []
    for c in df.columns:
        if isinstance(media.get(c), list):
            questions[c] = {"type": "choice", "options": media[c]}
            continue
        s = df[c].dropna()
        values = set(s.astype(str).str.strip().str.lower())
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if pd.api.types.is_bool_dtype(s) or (len(values) == 2 and values <= {"yes", "no", "true", "false", "0", "1"}):
            questions[c] = {"type": "noul"}
        elif numeric and s.nunique() > 10:
            questions[c] = {"type": "score", "min": float(s.min()), "max": float(s.max())}
        elif 1 < s.nunique() <= min(50, max(2, len(s) // 2)):
            questions[c] = {"type": "choice", "options": sorted(s.astype(str).unique())}
        elif is_text(s):
            inputs.append(c)
    kind = "decision" if questions and inputs else None
    return {"kind": kind, "columns": {**cols, "inputs": inputs}, "questions": questions}


def hardware(kaggle=False):
    """(description, parameter budget) for the machine that will train."""
    if kaggle:
        return "Kaggle: 2x NVIDIA T4 (16 GB each), about 30 GPU hours a week, 12 h per session", 3_000_000_000
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        gpus = [line.split(", ") for line in out.stdout.strip().splitlines()] if out.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        gpus = []
    if not gpus:
        return f"No GPU, {os.cpu_count()} CPU cores: only very small models, slowly", 600_000_000
    vram = min(int(g[1]) for g in gpus) / 1024
    return f"{len(gpus)}x {gpus[0][0]} ({vram:.0f} GB each)", int(vram * 200_000_000)  # ~3B on 16 GB with LoRA


def search_models(kind, query=None, max_params=None):
    """Most-downloaded open transformers models on the Hub for a kind, within max_params."""
    query = {"stt": "whisper", "tts": "speecht5"}.get(kind, query)  # the only families these trainers fit
    q = urllib.parse.urlencode({"pipeline_tag": HUB_TASK[kind], "library": "transformers", "sort": "downloads",
                                "direction": "-1", "limit": "60", **({"search": query} if query else {})})
    found = _api(f"https://huggingface.co/api/models?{q}&expand[]=safetensors&expand[]=downloads&expand[]=gated")
    out = []
    for m in found:
        params = (m.get("safetensors") or {}).get("total")
        # no safetensors size means a GGUF/ONNX-only repo: nothing the trainer can load or size
        if not params or (m.get("gated") and not os.getenv("HF_TOKEN")) or (max_params and params > max_params):
            continue
        out.append({"id": m["id"], "params_millions": round(params / 1e6), "downloads": m.get("downloads")})
    return out[:15]


# ------------------------------------------------------------------ Kaggle notebook export

def export_notebook(plan, path, goal=None):
    """Write a Kaggle notebook that runs this plan on Kaggle's free GPUs: installs what is
    missing, carries llm_train.py verbatim, trains the candidates in parallel (one per GPU),
    and leaves the winner as a zip in the notebook's output. Returns the path."""
    plan = dict(plan)
    data, kind = plan["data"], plan["kind"]
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    data_cell, data_note = None, f"Data: Hugging Face dataset `{data[3:]}`."
    if not data.startswith("hf:"):
        name = re.sub(r"[^\w.-]", "_", os.path.basename(data))
        if kind in ("text", "decision") and os.path.getsize(data) <= EMBED_LIMIT:
            with open(data, "rb") as f:
                payload = base64.b64encode(f.read()).decode()
            plan["data"] = f"/kaggle/working/{name}"
            data_cell = code(f'import base64\nwith open({plan["data"]!r}, "wb") as f:\n'
                             f'    f.write(base64.b64decode("{payload}"))')
            data_note = f"Data: `{name}` is embedded in this notebook, so keep the notebook private if the data is."
        else:
            plan["data"] = f"/kaggle/input/YOUR-DATASET/{name}"
            data_note = (f"Data: upload `{name}` (and any image or audio files it points to) as a Kaggle dataset, add "
                         "it to this notebook with *Add input*, then fix the `data` path in the PLAN cell.")
    with open(llm_train.__file__) as f:
        source = f.read()
    cells = [
        md(f"# LLM Factory: {kind} fine-tuning\n\n{f'Goal: {goal}' + chr(10) * 2 if goal else ''}"
           "Before running, open **Settings** in the right panel: set **Accelerator** to **GPU T4 x2** and turn "
           f"**Internet** on. Then **Run all**.\n\n{data_note}\n\nThe candidates in `PLAN` train in parallel, one per "
           "GPU. Each is scored on the same held-out test rows, and the best one is exported."),
        code("!pip install -q peft librosa soundfile"),
        code("%%writefile llm_train.py\n" + source),
        *([data_cell] if data_cell else []),
        code("PLAN = " + pprint.pformat(plan, sort_dicts=False, width=100)),
        code("import os\n\ntry:  # optional: a Kaggle secret named HF_TOKEN unlocks gated models (Add-ons > Secrets)\n"
             "    from kaggle_secrets import UserSecretsClient\n"
             '    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")\n'
             "except Exception:\n    pass\n\nimport llm_train\n\n"
             'out = llm_train.run_plan(PLAN, out_dir="/kaggle/working/run")'),
        md("## Download\n\nThe winner is the `*-model.zip` under `run/` in this notebook's **Output** (use *Save "
           "Version* first if you want it kept). Unzip it and read `README.txt` for how to use it. Text models "
           "include `model.gguf` and a `Modelfile`: `ollama create my-model -f Modelfile`, then `ollama run my-model`."),
    ]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"}}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    return path


# ------------------------------------------------------------------ the agent

class LLMFactoryRun(agent.FactoryRun):
    """One run: profile the data, let the agent (or the default portfolio) choose candidates,
    train them in parallel, export the best by test score. Reuses FactoryRun's tool loop,
    event stream and saving; only the prompt, the tools and the opening message differ."""

    system_prompt, tools = SYSTEM_PROMPT, TOOLS

    def __init__(self, data, goal=None, kind=None, models=None, no_llm=False, kaggle=None, parallel=None,
                 max_rows=None):
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        if not data.startswith("hf:"):
            if not os.path.isfile(data):
                raise SystemExit(f"Data file not found: {data}")
            data = os.path.abspath(data)  # workers and the notebook export need a stable path
        self.data, self.goal, self.kind, self.models = data, goal, kind, models or []
        self.kaggle, self.parallel, self.max_rows = kaggle, parallel, max_rows
        self.sample, n_rows, media = load_sample(data)
        self.setup = infer_setup(self.sample, media)
        described, self.max_params = hardware(bool(kaggle))
        short = lambda v: v if not isinstance(v, str) or len(v) <= 200 else v[:200] + "…"
        self.profile = {
            "source": data, "n_rows": n_rows, "columns": {c: str(t) for c, t in self.sample.dtypes.items()},
            "sample_rows": [{c: short(v) for c, v in r.items()}
                            for r in json.loads(self.sample.head(3).to_json(orient="records", default_handler=str))],
            "guessed": self.setup, "hardware": described,
            "parameter_budget_millions": round(self.max_params / 1e6),
            "trains_on": "Kaggle (notebook export)" if kaggle else "this machine"}
        self.out_dir = f"runs/{self.run_id}_llm"
        self.plan, self.rounds = None, 0
        self.results, self.events = [], []
        self.finalists, self.final, self.report = None, None, None
        self.tokens = {"prompt": 0, "completion": 0}
        self.client = None if no_llm else agent._llm()

    # ---- tools
    def tool_get_data_profile(self, _):
        """The profile computed in __init__. Takes no arguments."""
        return self.profile

    def tool_search_models(self, inp):
        """Hub search for base models of a kind, limited to this run's parameter budget."""
        if inp["kind"] not in llm_train.KINDS:
            return {"error": f"kind must be one of {list(llm_train.KINDS)}"}
        return {"models": search_models(inp["kind"], inp.get("query"), self.max_params),
                "parameter_budget_millions": round(self.max_params / 1e6)}

    def tool_train_candidates(self, inp):
        """Validate a round and train its candidates in parallel (or, with --kaggle, write
        the notebook that will). The first round fixes kind, columns and questions, so
        every candidate of the run is scored on the same test rows the same way."""
        if self.rounds >= MAX_ROUNDS:
            return {"error": f"Budget exhausted ({MAX_ROUNDS} rounds). Call write_report."}
        kind = inp["kind"]
        if kind not in llm_train.KINDS:
            return {"error": f"kind must be one of {list(llm_train.KINDS)}"}
        if self.plan and kind != self.plan["kind"]:
            return {"error": f"kind is fixed to '{self.plan['kind']}' by the first round."}
        base = self.plan or {"kind": kind, "data": self.data, "system": inp.get("system"), "max_rows": self.max_rows,
                             "columns": {**self.setup["columns"], **(inp.get("columns") or {})},
                             "questions": inp.get("questions") or self.setup["questions"]}
        known, cols = set(self.sample.columns), base["columns"]
        missing = [f for f in NEEDS[kind] if cols.get(f, f) not in known]
        if kind == "text" and cols.get("messages", "messages") in known:
            missing = []
        if kind == "decision":
            missing = [c for c in (cols.get("inputs") or ["text"]) + list(base["questions"]) if c not in known]
            if not base["questions"]:
                return {"error": "decision needs questions: {column: {type: choice|noul|score, ...}}."}
        if missing:
            return {"error": f"Columns not found for {kind}: {missing}. Columns: {sorted(known)}. Pass `columns`."}

        cands, taken = [], {r["name"] for r in self.results}
        for c in (inp.get("candidates") or [])[:MAX_CANDIDATES]:
            if not c.get("name") or not c.get("base_model"):
                return {"error": "Every candidate needs name and base_model."}
            name = re.sub(r"[^\w.-]+", "-", c["name"])  # it becomes a folder name
            name = name if name not in taken else f"{name}_r{self.rounds + 1}"
            taken.add(name)
            cands.append({**c, "name": name})
        if not cands:
            return {"error": "Give at least one candidate."}
        self.plan, self.rounds = base, self.rounds + 1
        plan = {**base, "candidates": cands}
        self.emit("round_start", {"round": self.rounds, "rationale": inp["rationale"], "kind": kind,
                                  "candidates": cands})
        if self.kaggle:
            path = export_notebook(plan, self.kaggle, self.goal)
            self.report = {"recommended": None, "spoken_summary": "", "markdown":
                           f"# Kaggle notebook exported\n\n`{path}` trains {len(cands)} {kind} candidates on Kaggle "
                           "GPUs and exports the best one. Import it on kaggle.com (Create > Notebook > File > Import)."}
            self.emit("report", self.report)
            return {"exported": path}

        t0 = time.time()
        results = llm_train.train_all(plan, self.out_dir, self.parallel, on_result=lambda r: self.emit(
            "candidate", {"round": self.rounds, "of": len(cands), **{k: v for k, v in r.items() if k != "model_dir"}}))
        self.results += [{**r, "round": self.rounds} for r in results]
        rows = [{"name": r["name"], "base_model": r["base_model"], "score": r["score"], "metrics": r["metrics"],
                 "train_seconds": r["train_seconds"]} if r["ok"] else
                {"name": r["name"], "base_model": r["base_model"], "error": r["error"][-600:]} for r in results]
        rows.sort(key=lambda r: -r.get("score", float("-inf")))
        self.emit("leaderboard", {"round": self.rounds, "rows": rows, "wall_seconds": round(time.time() - t0, 1)})
        return {"round": self.rounds, "rounds_left": MAX_ROUNDS - self.rounds, "leaderboard": rows,
                "note": "Higher score is better. The harness exports the highest test score."}

    def tool_write_report(self, inp):
        """Export the best candidate by test score (GGUF + Modelfile for text models, one
        zip to download), drop the losers' weights, and store the report with the
        harness-computed score table appended. Ends the run."""
        winner = llm_train.pick_winner(self.results)
        if not winner:
            return {"error": "No candidate has trained successfully yet. Call train_candidates."}
        self.emit("status", {"msg": f"Exporting '{winner['name']}'"})
        export = llm_train.export_winner(winner, self.plan, self.out_dir, [r for r in self.results if r is not winner])
        self.final = {**winner, **export, "model_path": export["zip"]}  # model_path: what save() prints as the model
        self.emit("final_model", self.final)
        table = ["## Candidates (scored by the harness on the same held-out test rows)", "",
                 "| candidate | base model | test score | metrics | train s |", "|---|---|---|---|---|"]
        for r in sorted(self.results, key=lambda r: -r.get("score", float("-inf"))):
            star = " ★" if r is winner else ""
            table.append(f"| {r['name']}{star} | {r['base_model']} | {r['score']} | {json.dumps(r['metrics'])} "
                         f"| {r['train_seconds']} |" if r["ok"] else f"| {r['name']} | {r['base_model']} | failed | | |")
        table += ["", f"Download: `{export['zip']}`" + (f". {export['note']}" if export["note"] else "")]
        self.report = {"recommended": winner["name"], "spoken_summary": "",
                       "markdown": inp["markdown"].rstrip() + "\n\n" + "\n".join(table) + "\n"}
        self.emit("report", self.report)
        return {"saved": True, "winner": winner["name"], "download": export["zip"], "ollama": export["ollama"]}

    # ---- loop
    def _start(self):
        """Emit run_start and return the opening user message for the LLM."""
        self.emit("run_start", {"dataset": self.data, "goal": self.goal, "model": agent.MODEL,
                                "rows": self.profile["n_rows"], "hardware": self.profile["hardware"]})
        hints = [f"Goal: {self.goal}" if self.goal else "No goal was given; infer it from the data."]
        if self.kind:
            hints.append(f"The user says the kind is {self.kind}.")
        if self.models:
            hints.append(f"The user wants these base models tried: {', '.join(self.models)}.")
        if self.kaggle:
            hints.append("Training will run on Kaggle: train_candidates writes the notebook and ends the run, so "
                         "choose the candidates carefully in one round.")
        return f"Dataset: {self.data}. " + " ".join(hints) + " Start by reading the data profile."

    def run(self):
        """With an LLM: the inherited tool loop. Without: train the user's models, or the
        default portfolio for the kind, then export the winner."""
        if self.client:
            return super().run()
        self._start()
        kind = self.kind or self.setup["kind"]
        if not kind:
            raise SystemExit("Could not tell the model kind from the data. Pass --kind "
                             f"({', '.join(llm_train.KINDS)}).")
        cands = [{"name": m.split("/")[-1].lower(), "base_model": m} for m in self.models] or llm_train.PORTFOLIO[kind]
        out = self.tool_train_candidates({"rationale": "No LLM: default candidates.", "kind": kind,
                                          "candidates": cands})
        if "error" not in out and not self.kaggle:
            out = self.tool_write_report({"markdown": "# Report (no LLM)"})
        if "error" in out:
            self.emit("status", {"msg": out["error"]})
        self.emit("run_end", {"tokens": self.tokens, "completed": self.report is not None})
        self.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune open models on your data and keep the best one.")
    ap.add_argument("data", help=".jsonl/.json/.csv/.parquet file, or hf:org/name[:config[:split]]")
    ap.add_argument("--goal", help="what the model is for; the agent picks the kind and candidates from it")
    ap.add_argument("--kind", choices=llm_train.KINDS, help="skip the guess: which kind of model to train")
    ap.add_argument("--models", help="comma-separated Hugging Face model ids to try")
    ap.add_argument("--no-llm", action="store_true", help="no agent: train --models or the default candidates")
    ap.add_argument("--kaggle", metavar="NOTEBOOK.ipynb", help="write a Kaggle notebook instead of training here")
    ap.add_argument("--parallel", type=int, help="candidates trained at once (default: one per GPU, else 2)")
    ap.add_argument("--max-rows", type=int, help="use at most this many rows")
    a = ap.parse_args()
    LLMFactoryRun(a.data, a.goal, a.kind, a.models.split(",") if a.models else None, a.no_llm, a.kaggle,
                  a.parallel, a.max_rows).run()
