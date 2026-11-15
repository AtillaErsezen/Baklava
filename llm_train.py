"""LLM Factory — training core. One self-contained file on purpose: it runs locally and is
embedded verbatim in the exported Kaggle notebook. Only the standard library is imported at
the top; torch and friends load inside the functions that need them.

  python llm_train.py candidate spec.json          # worker: train + score one candidate
  python llm_train.py decide MODEL_DIR "some text" # run an exported decision model

Model kinds:
  text      chat / instruction LLM, LoRA           score: token F1 of generated answers
  decision  "System One" model: typed decisions    score: agreement with the labels
            (choice, noul = yes/no, score) from one encoder pass, no text generation
  vision    vision-language model, LoRA            score: token F1 of generated answers
  stt       speech-to-text (Whisper family)        score: 1 - word error rate
  tts       text-to-speech (SpeechT5)              score: minus spectrogram loss
"""
import json
import os
import re
import subprocess
import sys
import time

SEED = 42
KINDS = ("text", "decision", "vision", "stt", "tts")
MAX_TEST = 200  # test examples per run; generation is the slow part of scoring

# Default candidates when no agent picks them. All open, all fit one free Kaggle T4.
PORTFOLIO = {
    "text": [{"name": "smollm2-360m", "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct"},
             {"name": "qwen2.5-0.5b", "base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
             {"name": "qwen2.5-1.5b", "base_model": "Qwen/Qwen2.5-1.5B-Instruct"}],
    "decision": [{"name": "modernbert-base", "base_model": "answerdotai/ModernBERT-base"},
                 {"name": "distilbert", "base_model": "distilbert/distilbert-base-uncased"},
                 {"name": "xlm-roberta", "base_model": "FacebookAI/xlm-roberta-base"}],
    "vision": [{"name": "smolvlm-256m", "base_model": "HuggingFaceTB/SmolVLM-256M-Instruct"},
               {"name": "smolvlm-500m", "base_model": "HuggingFaceTB/SmolVLM-500M-Instruct"}],
    "stt": [{"name": "whisper-tiny", "base_model": "openai/whisper-tiny"},
            {"name": "whisper-base", "base_model": "openai/whisper-base"},
            {"name": "whisper-small", "base_model": "openai/whisper-small"}],
    "tts": [{"name": "speecht5-lr1e-5", "base_model": "microsoft/speecht5_tts", "learning_rate": 1e-5},
            {"name": "speecht5-lr5e-5", "base_model": "microsoft/speecht5_tts", "learning_rate": 5e-5}],
}
DEFAULTS = {  # per kind: learning rate, epochs, batch size
    "text": (2e-4, 3, 4), "decision": (3e-5, 4, 16), "vision": (1e-4, 3, 2), "stt": (1e-5, 3, 8), "tts": (1e-5, 10, 8),
}


# ------------------------------------------------------------------ scoring (pure Python, tested offline)

def _words(s):
    return re.sub(r"[^\w\s]", " ", str(s).lower()).split()


def token_f1(pred, ref):
    """SQuAD-style word-overlap F1 between a generated answer and the reference."""
    p, r = _words(pred), _words(ref)
    if not p or not r:
        return float(p == r)
    pool, common = list(r), 0
    for w in p:
        if w in pool:
            pool.remove(w)
            common += 1
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def wer(preds, refs):
    """Word error rate over a list of transcripts: word edit distance / reference words."""
    errors = total = 0
    for pred, ref in zip(preds, refs):
        p, r = _words(pred), _words(ref)
        row = list(range(len(p) + 1))
        for i, rw in enumerate(r, 1):
            prev, row[0] = row[0], i
            for j, pw in enumerate(p, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != pw))
        errors, total = errors + row[-1], total + len(r)
    return errors / max(total, 1)


def decision_agreement(preds, rows, questions):
    """Mean agreement with the labels over all questions: accuracy for choice and noul,
    1 - (absolute error / range) for score. preds: one {question: value} per row."""
    per_q = {}
    for q, spec in questions.items():
        if spec["type"] == "score":
            span = (spec["max"] - spec["min"]) or 1
            vals = [1 - min(abs(p[q] - float(r[q])) / span, 1) for p, r in zip(preds, rows)]
        elif spec["type"] == "noul":
            vals = [float(p[q] == _truthy(r[q])) for p, r in zip(preds, rows)]
        else:
            vals = [float(str(p[q]) == str(r[q])) for p, r in zip(preds, rows)]
        per_q[q] = sum(vals) / max(len(vals), 1)
    return sum(per_q.values()) / max(len(per_q), 1), per_q


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


# ------------------------------------------------------------------ data

def load_rows(plan):
    """(train_rows, test_rows) as lists of dicts, normalised to the kind's field names.
    Source: a local .jsonl/.json/.csv/.parquet file, or 'hf:org/name[:config[:split]]' for
    a Hugging Face dataset. The split is seeded, so every candidate gets the same test rows."""
    from datasets import Audio, ClassLabel, Image, load_dataset

    src, cols, kind = plan["data"], plan.get("columns") or {}, plan["kind"]
    if src.startswith("hf:"):
        name, config, split = (src[3:].split(":") + [None, None])[:3]
        ds = load_dataset(name, config or None, split=split or "train")
    else:
        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}[src.rsplit(".", 1)[-1].lower()]
        ds = load_dataset(fmt, data_files=src, split="train")
    for c, feature in ds.features.items():  # decode media ourselves: avoids codec differences between versions
        if isinstance(feature, (Audio, Image)):
            ds = ds.cast_column(c, type(feature)(decode=False))
        elif isinstance(feature, ClassLabel):  # integer class ids -> their names, via a new column so the type changes
            names = feature.names
            ds = ds.map(lambda r, c=c, names=names: {c + "__name": names[r[c]] if r[c] is not None and r[c] >= 0 else None})
            ds = ds.remove_columns(c).rename_column(c + "__name", c)
    ds = ds.shuffle(seed=SEED)
    max_rows = plan.get("max_rows") or 20000  # rows are held in memory; raise max_rows in the plan for more
    if len(ds) > max_rows:
        print(f"Using {max_rows} of {len(ds)} rows (max_rows).", flush=True)
        ds = ds.select(range(max_rows))
    n_test = max(1, min(MAX_TEST, len(ds) // 10))
    rows = [_normalise(r, kind, cols, plan) for r in ds]
    rows = [r for r in rows if r is not None]
    return rows[n_test:], rows[:n_test]


def _normalise(row, kind, cols, plan):
    """One raw row -> the fields its kind trains on, or None if something required is missing."""
    get = lambda field: row.get(cols.get(field, field))
    if kind == "text":
        if row.get(cols.get("messages", "messages")):
            return {"messages": list(get("messages"))}
        if get("prompt") is None or get("response") is None:
            return None
        system = [{"role": "system", "content": plan["system"]}] if plan.get("system") else []
        return {"messages": system + [{"role": "user", "content": str(get("prompt"))},
                                      {"role": "assistant", "content": str(get("response"))}]}
    if kind == "decision":
        inputs = cols.get("inputs") or ["text"]
        text = str(row[inputs[0]]) if len(inputs) == 1 else "\n".join(f"{c}: {row[c]}" for c in inputs)
        for q, s in plan["questions"].items():  # skip rows with a missing answer or a choice outside the options
            if row.get(q) is None or (s["type"] == "choice" and str(row[q]) not in s["options"]):
                return None
        return {"text": text, **{q: row[q] for q in plan["questions"]}}
    needed = {"vision": ("image", "prompt", "response"), "stt": ("audio", "text"), "tts": ("audio", "text")}[kind]
    out = {f: get(f) for f in needed}
    return None if any(v is None for v in out.values()) else out


def _audio(x, rate=16000):
    """Mono float waveform at `rate` from a file path, an {'bytes'|'path'} dict or an {'array'} dict."""
    import io

    import librosa
    import numpy as np
    import soundfile as sf

    if isinstance(x, dict) and x.get("array") is not None:
        wave, sr = x["array"], x["sampling_rate"]
    elif isinstance(x, dict) and x.get("bytes"):
        wave, sr = sf.read(io.BytesIO(x["bytes"]), dtype="float32")
    else:
        wave, sr = librosa.load(x["path"] if isinstance(x, dict) else x, sr=None)
    wave = np.asarray(wave, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return librosa.resample(wave, orig_sr=sr, target_sr=rate) if sr != rate else wave


def _image(x):
    """RGB PIL image from a PIL image, a file path, a URL or an {'bytes'|'path'} dict."""
    import io
    import urllib.request

    from PIL import Image

    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict):
        x = io.BytesIO(x["bytes"]) if x.get("bytes") else x["path"]
    elif str(x).startswith(("http://", "https://")):
        x = io.BytesIO(urllib.request.urlopen(x, timeout=30).read())
    return Image.open(x).convert("RGB")


# ------------------------------------------------------------------ models: one builder per kind
# Each builder returns (model, collate, predict, save):
#   collate(rows) -> batch dict for model(**batch), which must return an object with .loss
#   predict(rows) -> one prediction per row, in the form the kind's score expects
#   save(out_dir) -> writes the final, loadable model folder

def _lora(model, spec, **kwargs):
    from peft import LoraConfig, get_peft_model

    r = spec.get("lora_r", 16)
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, **kwargs))


def _build_text(spec, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["base_model"])
    tok.pad_token = tok.pad_token or tok.eos_token
    model = _lora(AutoModelForCausalLM.from_pretrained(spec["base_model"]), spec,
                  task_type="CAUSAL_LM", target_modules="all-linear")
    max_len, templated = spec.get("max_length", 1024), bool(tok.chat_template)

    def render(messages, prompt_only=False):
        if templated:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=prompt_only)
        text = "".join(f"### {m['role'].title()}:\n{m['content']}\n\n" for m in messages)
        return text + "### Assistant:\n" if prompt_only else text + tok.eos_token

    def encode(texts, side):
        tok.padding_side = side  # right for training, left so generation continues from the prompt
        return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                   return_token_type_ids=False,
                   add_special_tokens=not templated)  # chat templates already contain the special tokens

    def collate(rows):
        enc = encode([render(r["messages"]) for r in rows], "right")
        return {**enc, "labels": enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)}

    def predict(rows):
        enc = encode([render(r["messages"][:-1], prompt_only=True) for r in rows], "left").to(device)
        out = model.generate(**enc, max_new_tokens=spec.get("max_new_tokens", 128), do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def save(out_dir):
        model.merge_and_unload().half().save_pretrained(out_dir)  # fp16 halves the download
        tok.save_pretrained(out_dir)

    return model, collate, predict, save


def _decision_net(backbone, questions):
    """Encoder + one small head per question, all answered in a single forward pass."""
    import types

    import torch
    import torch.nn.functional as F

    class DecisionNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            width = backbone.config.hidden_size
            self.heads = torch.nn.ModuleDict({
                q: torch.nn.Linear(width, len(s["options"]) if s["type"] == "choice" else 1)
                for q, s in questions.items()})

        def forward(self, input_ids, attention_mask, targets=None):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float()
            logits = {q: head(pooled) for q, head in self.heads.items()}
            loss = None
            if targets is not None:
                loss = 0
                for q, s in questions.items():
                    if s["type"] == "choice":
                        loss = loss + F.cross_entropy(logits[q], targets[q])
                    elif s["type"] == "noul":
                        loss = loss + F.binary_cross_entropy_with_logits(logits[q].squeeze(-1), targets[q])
                    else:  # score: regress its position inside [min, max]
                        loss = loss + F.mse_loss(torch.sigmoid(logits[q].squeeze(-1)), targets[q])
            return types.SimpleNamespace(loss=loss, logits=logits)

    return DecisionNet()


def _decide(model, tok, questions, texts, device, max_len=512):
    """Typed answers for a batch of texts: one {question: {'value', 'confidence'[, 'probabilities']}} each."""
    import torch

    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"]).logits
    out = [{} for _ in texts]
    for q, s in questions.items():
        if s["type"] == "choice":
            probs = torch.softmax(logits[q], -1).tolist()
            for o, p in zip(out, probs):
                best = max(range(len(p)), key=p.__getitem__)
                o[q] = {"value": s["options"][best], "confidence": round(p[best], 4),
                        "probabilities": {opt: round(v, 4) for opt, v in zip(s["options"], p)}}
        else:
            vals = torch.sigmoid(logits[q].squeeze(-1)).tolist()
            for o, v in zip(out, vals):
                if s["type"] == "noul":
                    o[q] = {"value": v >= 0.5, "confidence": round(max(v, 1 - v), 4)}
                else:
                    o[q] = {"value": round(s["min"] + v * (s["max"] - s["min"]), 4)}
    return out


def _build_decision(spec, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    questions = spec["questions"]
    tok = AutoTokenizer.from_pretrained(spec["base_model"])
    model = _decision_net(AutoModel.from_pretrained(spec["base_model"]), questions)
    max_len = spec.get("max_length", 512)

    def collate(rows):
        enc = tok([r["text"] for r in rows], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        targets = {}
        for q, s in questions.items():
            if s["type"] == "choice":
                targets[q] = torch.tensor([s["options"].index(str(r[q])) for r in rows])
            elif s["type"] == "noul":
                targets[q] = torch.tensor([float(_truthy(r[q])) for r in rows])
            else:
                span = (s["max"] - s["min"]) or 1
                targets[q] = torch.tensor([min(max((float(r[q]) - s["min"]) / span, 0.0), 1.0) for r in rows])
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "targets": targets}

    def predict(rows):
        answers = _decide(model, tok, questions, [r["text"] for r in rows], device, max_len)
        return [{q: a[q]["value"] for q in questions} for a in answers]

    def save(out_dir):
        model.backbone.save_pretrained(f"{out_dir}/backbone")
        tok.save_pretrained(f"{out_dir}/backbone")
        torch.save(model.heads.state_dict(), f"{out_dir}/heads.pt")
        with open(f"{out_dir}/questions.json", "w") as f:
            json.dump(questions, f, indent=2)
        with open(__file__) as src, open(f"{out_dir}/llm_train.py", "w") as dst:
            dst.write(src.read())  # the folder carries its own loader: python llm_train.py decide . "text"

    return model, collate, predict, save


def decide(model_dir, texts):
    """Load an exported decision model and answer its questions for each text."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    with open(f"{model_dir}/questions.json") as f:
        questions = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _decision_net(AutoModel.from_pretrained(f"{model_dir}/backbone"), questions)
    model.heads.load_state_dict(torch.load(f"{model_dir}/heads.pt", map_location="cpu"))
    return _decide(model.to(device).eval(), AutoTokenizer.from_pretrained(f"{model_dir}/backbone"), questions,
                   texts, device)


def _build_vision(spec, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(spec["base_model"])
    model = _lora(AutoModelForImageTextToText.from_pretrained(spec["base_model"]), spec,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    def encode(rows, with_answer, side):
        proc.tokenizer.padding_side = side
        texts = []
        for r in rows:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(r["prompt"])}]}]
            if with_answer:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": str(r["response"])}]})
            texts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=not with_answer))
        return proc(text=texts, images=[[_image(r["image"])] for r in rows], return_tensors="pt", padding=True)

    def collate(rows):
        enc = encode(rows, True, "right")
        return {**enc, "labels": enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)}

    def predict(rows):
        enc = encode(rows, False, "left").to(device)
        out = model.generate(**enc, max_new_tokens=spec.get("max_new_tokens", 128), do_sample=False)
        return proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def save(out_dir):
        model.merge_and_unload().half().save_pretrained(out_dir)
        proc.save_pretrained(out_dir)

    return model, collate, predict, save


def _build_stt(spec, device):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(spec["base_model"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(spec["base_model"])
    if spec.get("language"):  # Whisper: fix the language instead of auto-detecting it
        proc.tokenizer.set_prefix_tokens(language=spec["language"], task="transcribe")
        model.generation_config.language, model.generation_config.task = spec["language"], "transcribe"

    def features(rows):
        return proc.feature_extractor([_audio(r["audio"]) for r in rows], sampling_rate=16000,
                                      return_tensors="pt").input_features

    def collate(rows):
        lab = proc.tokenizer([str(r["text"]) for r in rows], return_tensors="pt", padding=True, truncation=True,
                             max_length=448)
        labels = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]  # the model prepends the start token itself
        return {"input_features": features(rows), "labels": labels}

    def predict(rows):
        out = model.generate(input_features=features(rows).to(device), max_new_tokens=225)
        return proc.batch_decode(out, skip_special_tokens=True)

    def save(out_dir):
        model.save_pretrained(out_dir)
        proc.save_pretrained(out_dir)

    return model, collate, predict, save


XVECTORS = "https://huggingface.co/api/datasets/Matthijs/cmu-arctic-xvectors/parquet/default/validation/0.parquet"


def _build_tts(spec, device):
    import io
    import urllib.request

    import pandas as pd
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

    proc = SpeechT5Processor.from_pretrained(spec["base_model"])
    model = SpeechT5ForTextToSpeech.from_pretrained(spec["base_model"])
    # shortcut: one fixed speaker embedding for every clip, so single-speaker data only;
    # compute an x-vector per clip (speechbrain spkrec-xvect-voxceleb) for multi-speaker data.
    table = pd.read_parquet(io.BytesIO(urllib.request.urlopen(XVECTORS, timeout=60).read()))
    speaker = torch.tensor(table["xvector"].iloc[7306], dtype=torch.float32)
    reduction = model.config.reduction_factor

    def collate(rows):
        items = [proc(text=str(r["text"]), audio_target=_audio(r["audio"]), sampling_rate=16000,
                      return_attention_mask=False) for r in rows]
        batch = proc.pad(input_ids=[{"input_ids": i["input_ids"]} for i in items],
                         labels=[{"input_values": i["labels"][0]} for i in items], return_tensors="pt")
        labels = batch["labels"].masked_fill(batch["decoder_attention_mask"].unsqueeze(-1).ne(1), -100)
        length = max(len(i["labels"][0]) for i in items)
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                "labels": labels[:, :length - length % reduction],  # target length must divide by the reduction factor
                "speaker_embeddings": speaker.repeat(len(rows), 1)}

    def save(out_dir):
        model.save_pretrained(out_dir)
        proc.save_pretrained(out_dir)
        torch.save(speaker, f"{out_dir}/speaker_embedding.pt")

    return model, collate, None, save  # no predict: tts is scored by its test spectrogram loss


BUILDERS = {"text": _build_text, "decision": _build_decision, "vision": _build_vision, "stt": _build_stt,
            "tts": _build_tts}


# ------------------------------------------------------------------ train + score one candidate

def _to(batch, device):
    return {k: ({q: t.to(device) for q, t in v.items()} if isinstance(v, dict) else v.to(device))
            for k, v in batch.items()}


def train_candidate(spec):
    """Fine-tune one candidate on the train rows, score it on the test rows, save the model to
    spec['out_dir']. Returns the result dict; raises on failure (the worker wrapper reports it)."""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    kind = spec["kind"]
    lr, epochs, batch_size = DEFAULTS[kind]
    lr, epochs = spec.get("learning_rate", lr), spec.get("epochs", epochs)
    batch_size, accum = spec.get("batch_size", batch_size), spec.get("grad_accum", 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_rows(spec)
    if not train:
        raise ValueError(f"No usable training rows: check the column mapping {spec.get('columns')}.")
    model, collate, predict, save = BUILDERS[kind](spec, device)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    steps = max(1, min(spec.get("max_steps") or 10 ** 9, epochs * len(loader) // accum))
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    amp = device.type == "cuda"  # weights stay fp32; autocast gives the speed without fp16 gradient issues
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    t0, step, done = time.time(), 0, False
    model.train()
    while not done:
        for i, batch in enumerate(loader):
            with torch.autocast(device.type, dtype=torch.float16 if amp else torch.bfloat16, enabled=amp):
                loss = model(**_to(batch, device)).loss
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                if step % 20 == 0 or step == steps:
                    print(f"[{spec['name']}] step {step}/{steps} loss {loss.item():.4f}", flush=True)
                if step >= steps:
                    done = True
                    break
    train_seconds = time.time() - t0

    model.eval()
    chunks = [test[i:i + batch_size] for i in range(0, len(test), batch_size)]
    with torch.no_grad():
        losses = [model(**_to(collate(c), device)).loss.item() for c in chunks]
        preds = [p for c in chunks for p in predict(c)] if predict else None
    metrics = {"test_loss": round(sum(losses) / len(losses), 4)}
    if kind in ("text", "vision"):
        refs = [r["messages"][-1]["content"] if kind == "text" else str(r["response"]) for r in test]
        f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
        metrics.update(token_f1=round(sum(f1) / len(f1), 4),
                       exact_match=round(sum(_words(p) == _words(r) for p, r in zip(preds, refs)) / len(refs), 4))
        score = metrics["token_f1"]
    elif kind == "decision":
        score, per_q = decision_agreement(preds, test, spec["questions"])
        metrics.update(agreement=round(score, 4), per_question={q: round(v, 4) for q, v in per_q.items()})
    elif kind == "stt":
        metrics["wer"] = round(wer(preds, [r["text"] for r in test]), 4)
        score = 1 - metrics["wer"]
    else:
        score = -metrics["test_loss"]

    os.makedirs(spec["out_dir"], exist_ok=True)
    save(spec["out_dir"])
    return {"name": spec["name"], "ok": True, "base_model": spec["base_model"], "score": round(score, 4),
            "metrics": metrics, "train_rows": len(train), "test_rows": len(test), "steps": step,
            "train_seconds": round(train_seconds, 1), "trainable_params": sum(p.numel() for p in trainable),
            "model_dir": spec["out_dir"], "device": str(device)}


# ------------------------------------------------------------------ orchestration: parallel candidates, winner, export

def gpu_count():
    """Number of NVIDIA GPUs, without importing torch."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return len(out.stdout.strip().splitlines()) if out.returncode == 0 else 0
    except (OSError, subprocess.TimeoutExpired):
        return 0


def train_all(plan, out_dir, parallel=None, on_result=None):
    """Train every candidate of the plan at the same time, one worker process each, at most
    `parallel` running together (default: one per GPU, or 2 without a GPU). Returns one result
    per candidate, failures included. on_result(result) is called as each one finishes."""
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    os.makedirs(out_dir, exist_ok=True)
    gpus = gpu_count()
    parallel = parallel or gpus or 2
    slots = Queue()
    for i in range(parallel):
        slots.put(i)

    def work(cand):
        spec = {**{k: v for k, v in plan.items() if k != "candidates"}, **cand,
                "out_dir": os.path.join(out_dir, cand["name"])}
        spec_path, result_path, log_path = (os.path.join(out_dir, f"{cand['name']}.{ext}")
                                            for ext in ("spec.json", "result.json", "log"))
        with open(spec_path, "w") as f:
            json.dump({**spec, "result_path": result_path}, f)
        slot = slots.get()
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(slot % gpus)  # each worker gets a GPU to itself
        try:
            with open(log_path, "w") as log:
                code = subprocess.run([sys.executable, os.path.abspath(__file__), "candidate", spec_path],
                                      stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        finally:
            slots.put(slot)
        if code == 0 and os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
        else:
            with open(log_path, errors="replace") as f:
                result = {"name": cand["name"], "ok": False, "base_model": cand.get("base_model"),
                          "error": f.read()[-1500:]}
        if on_result:
            on_result(result)
        return result

    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(work, plan["candidates"]))


def pick_winner(results):
    """The successful candidate with the highest test score, or None."""
    ok = [r for r in results if r.get("ok")]
    return max(ok, key=lambda r: r["score"]) if ok else None


LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"


def export_winner(winner, plan, out_dir, losers=()):
    """Turn the winner's folder into the download: for text models also a GGUF file and a
    Modelfile for Ollama, then one zip. The losers' model folders are removed to free disk
    (their logs and results stay). Returns {'model_dir', 'zip', 'ollama', 'note'}."""
    import shutil

    for r in losers:
        if r.get("model_dir"):
            shutil.rmtree(r["model_dir"], ignore_errors=True)

    model_dir, name = winner["model_dir"], re.sub(r"[^a-z0-9._-]+", "-", winner["name"].lower())
    out = {"model_dir": model_dir, "ollama": False, "note": None}
    usage = {
        "decision": 'python llm_train.py decide . "your text"   # typed answers with confidences, as JSON',
        "vision": "transformers: AutoProcessor + AutoModelForImageTextToText.from_pretrained(this folder)",
        "stt": 'transformers: pipeline("automatic-speech-recognition", model=this folder)',
        "tts": "transformers: SpeechT5ForTextToSpeech.from_pretrained(this folder), speaker_embedding.pt and the "
               "microsoft/speecht5_hifigan vocoder",
        "text": "transformers: AutoModelForCausalLM.from_pretrained(this folder)",
    }[plan["kind"]]
    if plan["kind"] == "text":
        try:  # the trained model stays usable even if the Ollama conversion fails
            cache = os.path.expanduser("~/.cache/llm-factory/llama.cpp")
            if not os.path.exists(cache):
                subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP, cache], check=True)
            subprocess.run([sys.executable, f"{cache}/convert_hf_to_gguf.py", model_dir, "--outfile",
                            f"{model_dir}/model.gguf", "--outtype", "q8_0"], check=True)
            with open(f"{model_dir}/Modelfile", "w") as f:
                f.write("FROM ./model.gguf\n" + (f"SYSTEM {json.dumps(plan['system'])}\n" if plan.get("system") else ""))
            out["ollama"] = True
            usage = f"ollama create {name} -f Modelfile && ollama run {name}\n  or " + usage
        except (OSError, subprocess.CalledProcessError) as e:
            out["note"] = f"Ollama export failed ({e}); the Hugging Face model folder is complete."
    else:
        out["note"] = f"Ollama runs text-generation models only; this {plan['kind']} model is a Hugging Face folder."
    with open(f"{model_dir}/README.txt", "w") as f:
        f.write(f"{winner['name']} (from {winner['base_model']}), {plan['kind']} model, test score {winner['score']}\n"
                f"Metrics: {json.dumps(winner['metrics'])}\n\nUse it:\n  {usage}\n")
    out["zip"] = shutil.make_archive(os.path.join(out_dir, f"{name}-model"), "zip", model_dir)
    return out


def run_plan(plan, out_dir="llm_runs", parallel=None):
    """The whole pipeline without an agent (this is what the Kaggle notebook calls): train all
    candidates in parallel, print the leaderboard, export the best one."""
    results = train_all(plan, out_dir, parallel, on_result=lambda r: print(
        f"finished {r['name']}: " + (f"score {r['score']} {r['metrics']}" if r["ok"] else "FAILED\n" + r["error"]),
        flush=True))
    winner = pick_winner(results)
    if not winner:
        raise SystemExit("Every candidate failed; see the *.log files in " + out_dir)
    print("\nLeaderboard (test score, higher is better)")
    for r in sorted((r for r in results if r["ok"]), key=lambda r: -r["score"]):
        print(f"  {r['score']:>8}  {r['name']}  ({r['base_model']}, {r['train_seconds']} s)")
    export = export_winner(winner, plan, out_dir, [r for r in results if r is not winner])
    print(f"\nWinner: {winner['name']}\nDownload: {export['zip']}" + (f"\nNote: {export['note']}" if export["note"] else ""))
    return {"results": results, "winner": winner, "export": export}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "candidate":  # worker for train_all
        with open(sys.argv[2]) as f:
            spec = json.load(f)
        result = train_candidate(spec)  # a crash exits non-zero; train_all reads the log tail
        with open(spec["result_path"], "w") as f:
            json.dump(result, f)
    elif sys.argv[1] == "decide":
        print(json.dumps(decide(sys.argv[2], sys.argv[3:]), indent=2))
