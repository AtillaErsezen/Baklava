"""
ML Factory agent (open-weight LLM on Nebius Token Factory).

  uv run --env-file .env agent.py data/churn.csv --target Churn
  uv run --env-file .env agent.py data/houses.csv --target SalePrice --task regression
  uv run agent.py data/churn.csv --target Churn --provider scripted   # no API key
"""
import argparse
import json
import os
import time
import uuid
import modal
import pandas as pd

from modal_train import APP_NAME, MODEL_MENU, check_name, check_params, load_local
from factory_tools import GPU_MODELS, HIGHER_IS_BETTER, PipelineTools, compact
from providers import NebiusClient, ScriptedClient, parse_arguments, to_openai_tools

MAX_STEPS = 20
MAX_ROUNDS = 3
MAX_CANDIDATES = 8
TOKEN_BUDGET = 40_000
PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "system.md")

TOOLS = [
    {
        "name": "diag_summary",
        "description": "Ranked findings (leakage, id columns, drift, imbalance, nonlinearity...) and meta-features "
                       "from 33 statistical checks on the dev split. Free, precomputed. Call first.",
        "input_schema": {"type": "object", "properties": {
            "response_format": {"type": "string", "enum": ["concise", "detailed"]}}},
    },
    {
        "name": "diag_run",
        "description": "Run one named check (see diag_summary detailed catalog), optionally on given columns.",
        "input_schema": {"type": "object", "properties": {"check": {"type": "string"},
                                                          "columns": {"type": "array", "items": {"type": "string"}}},
                         "required": ["check"]},
    },
    {
        "name": "run_search",
        "description": "Generate ~3000 candidate pipelines, rank them by a benchmark prior, and race the top 243 on "
                       "statistically sized growing subsamples on Modal with paired early dropping. Returns the "
                       "funnel, the sample size n* with its justification, and the top 3.",
        "input_schema": {"type": "object", "properties": {
            "rationale": {"type": "string", "description": "Why this setup. Shown to the user."},
            "task": {"type": "string", "enum": ["classification", "regression"]},
            "primary_metric": {"type": "string", "enum": list(HIGHER_IS_BETTER)},
            "cv": {"type": "string", "enum": ["kfold", "walk_forward", "purged"]},
            "time_column": {"type": "string"},
            "drop_columns": {"type": "array", "items": {"type": "string"}},
            "veto_families": {"type": "array", "items": {"type": "string"}},
            "purpose": {"type": "object", "properties": {"interpretable": {"type": "boolean"},
                                                         "max_predict_ms": {"type": "number"},
                                                         "class_weighting": {"type": "boolean"}}}},
            "required": ["rationale", "task", "primary_metric"]},
    },
    {
        "name": "confirm_and_test",
        "description": "Confirm the race top-k with 10 paired folds (corrected t, Bayesian ROPE, Holm), pick by the "
                       "1-SE rule, then score the locked hidden holdout once (DeLong/McNemar, calibration), with "
                       "Big-O, fit time, predict ms and a Pareto flag.",
        "input_schema": {"type": "object", "properties": {"k": {"type": "integer", "minimum": 2, "maximum": 5}}},
    },
    {
        "name": "data_search",
        "description": "Search public dataset sites (Tavily) for data that could enrich this dataset. Candidates only.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                          "purpose": {"type": "string", "enum": ["enrich", "more_rows"]}},
                         "required": ["query"]},
    },
    {
        "name": "data_try",
        "description": "Fetch one public csv/parquet, left-join its columns on a key, and measure base vs enriched "
                       "with the current best config (paired test). Only a verdict of improves justifies a claim.",
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}, "left_key": {"type": "string"},
                                                          "right_key": {"type": "string"},
                                                          "columns": {"type": "array", "items": {"type": "string"}}},
                         "required": ["url", "left_key", "right_key"]},
    },
    {
        "name": "inspect_column",
        "description": "Sample values, value counts and (for classification) target rate per value for one column. "
                       "Use to confirm leakage or decide whether to drop a column.",
        "input_schema": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]},
    },
    {
        "name": "run_experiments",
        "description": "Cross-validate candidates in parallel on Modal. Returns a leaderboard with validation "
                       "mean/std, train mean and overfit_gap for the primary metric, plus all metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string", "description": "Why these candidates. Shown to the user."},
                "task": {"type": "string", "enum": ["classification", "regression"]},
                "primary_metric": {"type": "string", "enum": list(HIGHER_IS_BETTER)},
                "cv": {"type": "string", "enum": ["kfold", "timeseries"]},
                "cv_folds": {"type": "integer", "minimum": 3, "maximum": 10},
                "time_column": {"type": "string", "description": "Sort by this before timeseries CV."},
                "drop_columns": {"type": "array", "items": {"type": "string"}},
                "candidates": {
                    "type": "array",
                    "maxItems": MAX_CANDIDATES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "model": {"type": "string", "enum": sorted({m for ms in MODEL_MENU.values() for m in ms})},
                            "params": {"type": "object", "description": "Constructor kwargs."},
                            "preprocessing": {
                                "type": "object",
                                "properties": {
                                    "numeric_impute": {"type": "string", "enum": ["median", "mean"]},
                                    "scale": {"type": "boolean"},
                                    "categorical": {"type": "string", "enum": ["onehot", "ordinal"]},
                                    "max_categories": {"type": "integer"},
                                },
                            },
                        },
                        "required": ["name", "model"],
                    },
                },
            },
            "required": ["rationale", "task", "primary_metric", "candidates"],
        },
    },
    {
        "name": "finalize_model",
        "description": "Train the chosen candidate on all data, save it, return model path and top features.",
        "input_schema": {"type": "object", "properties": {"candidate_name": {"type": "string"}},
                         "required": ["candidate_name"]},
    },
    {
        "name": "write_report",
        "description": "Save the final report. Call once, at the very end.",
        "input_schema": {
            "type": "object",
            "properties": {"markdown": {"type": "string"}, "spoken_summary": {"type": "string"}},
            "required": ["markdown", "spoken_summary"],
        },
    },
]


OPENAI_TOOLS = to_openai_tools(TOOLS)
with open(PROMPT_PATH, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().format(max_rounds=MAX_ROUNDS, token_budget=TOKEN_BUDGET,
                                     tool_catalog="\n".join(f"- {t['name']}: {t['description']}" for t in TOOLS))


# ------------------------------------------------------------------ data profiling (local, cheap)

def _r(v):
    """Round a possibly-NaN scalar to 4 decimals, or None if it's NaN."""
    return round(float(v), 4) if pd.notna(v) else None


def profile_data(df: pd.DataFrame, target: str) -> dict:
    """Build the JSON profile the agent reads instead of raw rows: shape, target
    distribution/task guess, and per-column type/missingness/cardinality with
    flags (id_like, constant, high_cardinality, mostly_missing, possible_leakage)."""
    n = len(df)
    y = df[target]
    y_numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    n_classes = int(y.nunique())

    if not y_numeric or (pd.api.types.is_integer_dtype(y) and n_classes <= 20):
        task = "classification"
        balance = y.value_counts(normalize=True).head(10)
        target_info = {"suggested_task": task, "n_classes": n_classes,
                       "class_balance": {str(k): round(float(v), 3) for k, v in balance.items()}}
        y_code = pd.Series(pd.factorize(y)[0], index=y.index).where(y.notna()) if n_classes == 2 else None
    else:
        task = "regression"
        target_info = {"suggested_task": task, "min": _r(y.min()), "max": _r(y.max()),
                       "mean": _r(y.mean()), "skew": _r(y.skew())}
        y_code = y
    target_info["missing_pct"] = round(float(y.isna().mean() * 100), 1)

    cols = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        nun = int(s.nunique(dropna=True))
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        else:
            kind = "categorical"
        info = {"name": c, "kind": kind, "missing_pct": round(float(s.isna().mean() * 100), 1), "n_unique": nun}
        flags = []
        if kind == "numeric":
            info.update(min=_r(s.min()), max=_r(s.max()), mean=_r(s.mean()), skew=_r(s.skew()))
            if y_code is not None:
                corr = s.corr(y_code)
                if pd.notna(corr):
                    info["corr_with_target"] = _r(corr)
                    if abs(corr) > 0.95:
                        flags.append("possible_leakage")
            if pd.api.types.is_integer_dtype(s) and nun == n and n > 50:
                flags.append("id_like")
        elif kind in ("categorical", "boolean"):
            info["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(3).items()}
            if nun > 50:
                flags.append("high_cardinality")
            if nun >= 0.95 * n and n > 50:
                flags.append("id_like")
        if nun <= 1:
            flags.append("constant")
        if info["missing_pct"] > 40:
            flags.append("mostly_missing")
        if flags:
            info["flags"] = flags
        cols.append(info)

    return {
        "n_rows": n,
        "n_features": len(cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "target": {"name": target, **target_info},
        "datetime_columns": [c["name"] for c in cols if c["kind"] == "datetime"],
        "columns": cols[:100],
        "note": "truncated to 100 columns" if len(cols) > 100 else None,
    }


# ------------------------------------------------------------------ the agent

def _maybe_supabase():
    """Return a Supabase client for live event streaming, or None if the env vars
    aren't set or the package/connection fails. Live UI is optional; jsonl replay
    always works, so callers should treat a None return as a no-op, not an error."""
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        print("Supabase disabled:", e)
        return None


class FactoryRun(PipelineTools):
    """One end-to-end run: load data, profile it, drive the agent loop through
    its tools (profile/inspect/experiment/finalize/report), and save the result.
    One instance = one dataset + target; not reused across runs."""

    def __init__(self, data_path: str, target: str, task_hint: str | None = None, provider: str = "nebius",
                 purpose: str | None = None):
        """Load the dataset, validate the target column, and profile it up front so
        get_data_profile is a free tool call (no recomputation during the agent loop)."""
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.df = load_local(data_path)
        if target not in self.df.columns:
            raise SystemExit(f"Target '{target}' not found. Columns: {list(self.df.columns)}")
        self.dataset_name = os.path.basename(data_path)
        self.target, self.task_hint = target, task_hint
        self.profile = profile_data(self.df, target)
        self.remote_path = None
        self.rounds = 0
        self.specs, self.results, self.events = {}, [], []
        self.final, self.report = None, None
        self.client = ScriptedClient(self.profile, MODEL_MENU) if provider == "scripted" else NebiusClient()
        self.fns = {n: modal.Function.from_name(APP_NAME, n) for n in (
            "train_candidate", "train_candidate_gpu", "fit_final", "fit_final_gpu", "predict_holdout", "predict_holdout_gpu")}
        self.train_fn, self.final_fn = self.fns["train_candidate"], self.fns["fit_final"]
        self.supabase = _maybe_supabase()
        self.setup_pipeline(task_hint or self.profile["target"]["suggested_task"], purpose)

    # ---- event stream: stdout + jsonl (replay mode for the demo) + optional Supabase (live UI)
    def emit(self, kind: str, payload: dict):
        """Record one timeline event (thought, tool_call, leaderboard, etc.) to the
        in-memory list that `save()` persists, print it, and best-effort push it to
        Supabase for the live UI. A Supabase failure is logged, never raised."""
        event = {"run_id": self.run_id, "ts": time.time(), "kind": kind,
                 "payload": json.loads(json.dumps(payload, default=str))}
        self.events.append(event)
        print(f"[{kind}] {json.dumps(payload, default=str, ensure_ascii=False)[:400]}")
        if self.supabase:
            try:
                self.supabase.table("events").insert(event).execute()
            except Exception as e:
                print("supabase insert failed:", e)

    # ---- tools
    # Each tool_* method mirrors one entry in TOOLS and is dispatched by name from
    # run(). All take the tool's `input` dict and return a JSON-serializable dict -
    # never raise on bad input, return {"error": ...} instead so the agent can react.

    def tool_get_data_profile(self, _):
        """Return the profile computed once in __init__. Ignores its input (the tool
        takes no arguments)."""
        return self.profile

    def tool_inspect_column(self, inp):
        """Drill into one column: dtype, sample values, value counts, and: for a
        classification target: the target rate per value, to help the agent confirm
        or rule out leakage/id-like columns flagged by the profile."""
        c = inp["column"]
        if c not in self.df.columns:
            return {"error": f"Unknown column '{c}'"}
        s = self.df[c]
        out = {"dtype": str(s.dtype), "sample": s.dropna().astype(str).head(10).tolist(),
               "value_counts": {str(k): int(v) for k, v in s.value_counts(dropna=False).head(15).items()}}
        if self.profile["target"]["suggested_task"] == "classification" and s.nunique() <= 50:
            ct = pd.crosstab(s.astype(str), self.df[self.target].astype(str), normalize="index").round(3).head(15)
            out["target_rate_by_value"] = {str(k): {str(kk): float(vv) for kk, vv in row.items()}
                                           for k, row in ct.iterrows()}
        return out

    def tool_run_experiments(self, inp):
        """Validate and enqueue a round of candidates, upload the dataset to Modal on
        first use, cross-validate them in parallel via `train_fn.map`, and return a
        leaderboard sorted by the primary metric (with overfit_gap and a
        suspiciously-high-score warning per row). Enforces MAX_ROUNDS/MAX_CANDIDATES
        and rejects models that don't belong to the given task."""
        if self.rounds >= MAX_ROUNDS:
            return {"error": f"Budget exhausted ({MAX_ROUNDS} rounds). Finalize the best candidate."}
        task, pm = inp["task"], inp["primary_metric"]
        if pm not in (("roc_auc", "f1_macro", "accuracy") if task == "classification" else ("rmse", "mae", "r2")):
            return {"error": f"Metric '{pm}' does not fit task '{task}'."}
        cands = inp["candidates"][:MAX_CANDIDATES]
        try:
            for c in cands:
                check_name(c["name"])
                check_params(c["model"], dict(c.get("params") or {}))
        except ValueError as e:
            return {"error": str(e)}
        bad = [c["name"] for c in cands if c["model"] not in MODEL_MENU[task]]
        if bad:
            return {"error": f"Invalid models for {task}: {bad}. Allowed: {MODEL_MENU[task]}"}

        if self.remote_path is None:
            self.remote_path = self._dev()
        self.rounds += 1

        base = {"dataset_path": self.remote_path, "target": self.target, "task": task,
                "cv": inp.get("cv", "kfold"), "cv_folds": min(max(int(inp.get("cv_folds", 5)), 3), 10),
                "time_column": inp.get("time_column"), "drop_columns": inp.get("drop_columns", [])}
        specs = []
        for c in cands:
            name = c["name"] if c["name"] not in self.specs else f"{c['name'][:56]}_r{self.rounds}"
            spec = {**base, "name": name, "model": c["model"],
                    "params": c.get("params") or {}, "preprocessing": c.get("preprocessing") or {}}
            self.specs[name] = spec
            specs.append(spec)
        self.emit("round_start", {"round": self.rounds, "rationale": inp["rationale"],
                                  "candidates": [{k: s[k] for k in ("name", "model", "params")} for s in specs]})

        t0 = time.time()
        hib = HIGHER_IS_BETTER[pm]
        rows = []
        for r in self.train_fn.map(specs):
            r["round"] = self.rounds
            self.results.append(r)
            if not r["ok"]:
                rows.append({"name": r["name"], "error": r["error"]})
                continue
            m = r["metrics"][pm]
            gap = (m["train_mean"] - m["mean"]) if hib else (m["mean"] - m["train_mean"])
            row = {"name": r["name"], pm: m["mean"], "std": m["std"], "train": m["train_mean"],
                   "overfit_gap": round(gap, 4), "fit_seconds": r["fit_seconds"],
                   "all_metrics": {k: v["mean"] for k, v in r["metrics"].items()}}
            if pm in ("roc_auc", "accuracy", "r2") and m["mean"] > 0.995:
                row["warning"] = "suspiciously_high_check_leakage"
            rows.append(row)

        ok = sorted([x for x in rows if pm in x], key=lambda x: x[pm], reverse=hib)
        leaderboard = ok + [x for x in rows if pm not in x]
        self.emit("leaderboard", {"round": self.rounds, "primary_metric": pm, "rows": leaderboard,
                                  "wall_seconds": round(time.time() - t0, 1)})
        return {"round": self.rounds, "rounds_left": MAX_ROUNDS - self.rounds,
                "primary_metric": pm, "leaderboard": leaderboard}

    def tool_finalize_model(self, inp):
        """Refit a previously-run candidate (by name) on the full dataset via Modal's
        `fit_final`, store the result as self.final, and return the saved model path
        and top features."""
        name = inp["candidate_name"]
        if name not in self.specs:
            return {"error": f"Unknown candidate. Known: {list(self.specs)}"}
        self.emit("status", {"msg": f"Training final model '{name}' on all data"})
        fn = self.fns["fit_final_gpu" if self.specs[name]["model"] in GPU_MODELS else "fit_final"]
        out = fn.remote(self.specs[name], self.run_id)
        self.final = {"name": name, "spec": self.specs[name], **out}
        self.emit("final_model", self.final)
        return {**out, "export": self.export_user_model(name)}

    def tool_write_report(self, inp):
        """Store the agent's final markdown report and short spoken summary. Calling
        this is also the loop's stop signal: `run()` breaks right after."""
        self.report = {"markdown": inp["markdown"], "spoken_summary": inp["spoken_summary"]}
        self.emit("report", self.report)
        return {"saved": True}

    # ---- loop
    def run(self):
        """Drive the tool-use loop in OpenAI chat format: stream assistant text as
        `thought` events, dispatch each tool call to the matching tool_* method, and
        send results back as role="tool" messages. Bad arguments go back to the model as
        a tool error (it self-corrects). Stops when the model stops calling tools, after
        write_report, or after MAX_STEPS, then always saves whatever was produced."""
        self.emit("run_start", {"dataset": self.dataset_name, "target": self.target,
                                "rows": self.profile["n_rows"], "features": self.profile["n_features"]})
        hint = f" The user says this is a {self.task_hint} task." if self.task_hint else ""
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Dataset '{self.dataset_name}', target column '{self.target}'.{hint} "
                                                "Build the best model you can. Start with diag_summary." + (f" Purpose: {self.purpose}" if self.purpose else "")}]
        schemas = {t["name"]: t["input_schema"] for t in TOOLS}

        for _ in range(MAX_STEPS):
            msg, finish = self.client.chat(compact(messages), OPENAI_TOOLS)
            calls = msg.tool_calls or []
            messages.append({"role": "assistant", "content": msg.content or "",
                             **({"tool_calls": [{"id": c.id, "type": "function",
                                                 "function": {"name": c.function.name,
                                                              "arguments": c.function.arguments}}
                                                for c in calls]} if calls else {})})
            if msg.content and msg.content.strip():
                self.emit("thought", {"text": msg.content})
            if not calls:
                break

            for call in calls:
                name = call.function.name
                args, err = (None, f"unknown tool '{name}'. tools: {list(schemas)}") if name not in schemas \
                    else parse_arguments(call.function.arguments, schemas[name])
                self.emit("tool_call", {"tool": name, "input": args if args is not None else call.function.arguments})
                if err:
                    out = {"error": err}
                else:
                    try:
                        out = getattr(self, f"tool_{name}")(args)
                    except Exception as e:
                        out = {"error": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(out, default=str, ensure_ascii=False, sort_keys=True)})
            if self.report:
                break

        self.emit("usage", self.client.ledger.totals())
        self.remember()
        self.save()

    def save(self):
        """Persist this run under runs/<run_id>_*: the full event timeline as jsonl
        (for the replay-mode fallback), profile/results/final as json, and the
        markdown report if one was written. Called once at the end of run()."""
        os.makedirs("runs", exist_ok=True)
        prefix = f"runs/{self.run_id}"
        with open(f"{prefix}_events.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in self.events)
        with open(f"{prefix}_results.json", "w", encoding="utf-8") as f:
            json.dump({"profile": self.profile, "results": self.results, "final": self.final}, f,
                      default=str, indent=2)
        if self.report:
            with open(f"{prefix}_report.md", "w", encoding="utf-8") as f:
                f.write(self.report["markdown"])
        print(f"\nSaved → {prefix}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--provider", choices=["nebius", "scripted"], default="nebius")
    ap.add_argument("--purpose", help="what the model is for, in plain words")
    ap.add_argument("--dry-run", action="store_true", help="alias for --provider scripted (no API key)")
    args = ap.parse_args()
    FactoryRun(args.data, args.target, args.task, "scripted" if args.dry_run else args.provider, args.purpose).run()
