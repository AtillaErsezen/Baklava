"""
ML Factory — Claude agent.

  python agent.py data/churn.csv --target Churn
  python agent.py data/houses.csv --target SalePrice --task regression
"""
import argparse
import json
import os
import time
import uuid
from types import SimpleNamespace
import queue
import re
import threading

import anthropic
import modal
import pandas as pd

from modal_train import APP_NAME, MODEL_MENU, upload_dataset
from tracking import TrackingLog, json_safe, load_tracked_dataset, supabase_client, verify_tracking_api

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_STEPS = 20
MAX_ROUNDS = 3
MAX_CANDIDATES = 8
HIGHER_IS_BETTER = {"roc_auc": True, "f1_macro": True, "accuracy": True, "r2": True, "rmse": False, "mae": False}

SYSTEM_PROMPT = f"""You are ML Factory, an autonomous ML engineer. You get a tabular dataset and a target column,
and you must deliver the best-performing *trustworthy* model within a fixed compute budget
({MAX_ROUNDS} experiment rounds, max {MAX_CANDIDATES} candidates per round).

Workflow
1. Call get_data_profile. Decide: task type; primary metric (binary → roc_auc, or f1_macro if the minority class
   matters and imbalance is heavy; multiclass → f1_macro; regression → rmse); CV scheme (timeseries if rows are
   ordered in time and the target is about the future, else kfold); columns to drop (id_like, constant,
   possible_leakage, free text). Use inspect_column when a flag is ambiguous.
2. Round 1, broad sweep: 4-6 diverse candidates with sensible defaults. Always include a linear baseline.
3. Round 2, refine: 3-6 variants of the top 1-2 families (depth, learning rate, n_estimators, regularization,
   class_weight / scale_pos_weight for imbalance, ordinal vs onehot encoding for tree models).
4. Round 3 is optional: only if round 2 changed the ranking or revealed a problem.
5. After each round: a large overfit_gap relative to std → prefer more regularization. A suspiciously_high
   warning → investigate leakage with inspect_column, drop the column, rerun.
6. finalize_model on the winner. If two candidates are within one std, pick the simpler/faster one and say why.
7. write_report, then stop.

Rules
- Models: {json.dumps(MODEL_MENU)}. params must be valid constructor kwargs for that library.
- Short unique candidate names, e.g. lgbm_d6_lr05.
- Before every tool call, write 1-3 plain-language sentences explaining your reasoning. This is streamed live
  to a non-expert audience.
- Never invent numbers; only report metrics returned by tools.
- Report: concise markdown with dataset summary, data issues and how you handled them, a table of what was tried,
  the winner with CV metric ± std, top features, caveats, next steps. spoken_summary: max 60 words, no numbers
  beyond the headline metric, written to be read aloud."""

TOOLS = [
    {
        "name": "get_data_profile",
        "description": "Schema, column types, missing values, cardinality, target distribution, and flags "
                       "(id_like, constant, high_cardinality, mostly_missing, possible_leakage).",
        "input_schema": {"type": "object", "properties": {}},
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


# ------------------------------------------------------------------ dry run (no API key)

class ScriptedClient:
    """Stand-in for anthropic.Anthropic(): replays a fixed tool sequence so the whole pipeline
    except Claude's reasoning (Modal, events, saving) can be tested without an API key."""

    def __init__(self, profile):
        self.profile, self.step, self.messages = profile, 0, self

    def create(self, messages, **_):
        """Mimic client.messages.create: return the next scripted tool call as a
        response-shaped object (stop_reason + content blocks). Script: profile ->
        one run_experiments round (every menu model, flagged columns dropped) ->
        finalize the leaderboard's top row -> write_report -> end_turn. Later steps
        read the previous tool result from `messages`. Other kwargs (model, tools...)
        are ignored."""
        self.step += 1
        last = json.loads(messages[-1]["content"][0]["content"]) if self.step > 1 else None
        task = self.profile["target"]["suggested_task"]
        if self.step == 1:
            name, inp = "get_data_profile", {}
        elif self.step == 2:
            bad = {"id_like", "possible_leakage", "constant"}
            name, inp = "run_experiments", {
                "rationale": "Dry run: every menu model with defaults, flagged columns dropped.",
                "task": task, "primary_metric": "roc_auc" if task == "classification" else "rmse",
                "drop_columns": [c["name"] for c in self.profile["columns"] if bad & set(c.get("flags", []))],
                "candidates": [{"name": m, "model": m} for m in MODEL_MENU[task]]}
        elif self.step == 3:
            name, inp = "finalize_model", {"candidate_name": last["leaderboard"][0]["name"]}
        elif self.step == 4:
            name, inp = "write_report", {"markdown": f"# Dry run\n\n```json\n{json.dumps(last, indent=2)}\n```\n",
                                         "spoken_summary": "Dry run complete."}
        else:
            return SimpleNamespace(stop_reason="end_turn", content=[])
        return SimpleNamespace(stop_reason="tool_use", content=[
            SimpleNamespace(type="text", text=f"(dry run) calling {name}"),
            SimpleNamespace(type="tool_use", id=f"dry{self.step}", name=name, input=inp)])


# ------------------------------------------------------------------ the agent

class FactoryRun:
    """One end-to-end run: load data, profile it, drive the Claude agent loop through
    its tools (profile/inspect/experiment/finalize/report), and save the result.
    One instance = one dataset + target; not reused across runs."""

    def __init__(self, data_path: str | None, target: str, task_hint: str | None = None,
                 dry_run: bool = False, dataset_id: str | None = None, offline: bool = False,
                 output_dir: str = "runs"):
        """Load the dataset, validate the target column, and profile it up front so
        get_data_profile is a free tool call (no recomputation during the agent loop)."""
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.supabase = supabase_client(offline)
        verify_tracking_api(self.supabase)
        self.df, self.dataset = load_tracked_dataset(data_path, dataset_id, self.supabase)
        if target not in self.df.columns:
            raise SystemExit(f"Target '{target}' not found. Columns: {list(self.df.columns)}")
        self.dataset_name = self.dataset["name"]
        self.target, self.task_hint = target, task_hint
        self.profile = profile_data(self.df, target)
        self.remote_path = None
        self.prepared_sha256 = None
        self.rounds = 0
        self.specs, self.results, self.candidate_ids = {}, [], {}
        self.final, self.report = None, None
        self.client = ScriptedClient(self.profile) if dry_run else anthropic.Anthropic()
        self.track_fn = modal.Function.from_name(APP_NAME, "track_training")
        self.output_dir = output_dir
        self.tracker = TrackingLog(self.run_id, self.dataset, target, task_hint, self.supabase, output_dir)
        self.events = self.tracker.events

    # ---- event stream: stdout + jsonl (replay mode for the demo) + optional Supabase (live UI)
    def emit(self, kind: str, payload: dict):
        """Append a durable replay event, then attempt atomic live delivery."""
        self.tracker.emit(kind, payload)

    def _stream_trainings(self, jobs, primary_metric=None):
        """Consume worker packets as they arrive, with all persistence on this thread."""
        packets = queue.Queue()

        def receive(training_id, spec, stage):
            try:
                for packet in self.track_fn.remote_gen(spec, self.run_id, stage):
                    packets.put((training_id, packet))
            except Exception as exc:
                packets.put((training_id, {"kind": "stream_error", "error": f"Worker stream failed ({type(exc).__name__}); completion unconfirmed."}))
            finally:
                packets.put((training_id, {"kind": "done"}))

        for training_id, spec, stage in jobs:
            threading.Thread(target=receive, args=(training_id, spec, stage), daemon=True).start()
        remaining = {job[0] for job in jobs}
        results = []
        while remaining:
            training_id, packet = packets.get()
            row = self.tracker.trainings[training_id]
            kind = packet["kind"]
            if kind == "done":
                remaining.discard(training_id)
            if row["status"] in ("succeeded", "failed"):
                continue
            if kind == "started":
                self.tracker.training_started(training_id, packet["started_at"])
            elif kind in ("finished", "stream_error", "done"):
                result = packet.get("result") or {"ok": False, "error": packet.get("error", "Worker stream ended without a result.")}
                result = {**result, "name": row["candidate_name"], "training_id": training_id, "round": row["round"]}
                result["ok"] = self.tracker.training_finished(training_id, result, primary_metric)
                if not result["ok"]:
                    result["error"] = row["error"]
                results.append(result)
        return results

    # ---- tools
    # Each tool_* method mirrors one entry in TOOLS and is dispatched by name from
    # run(). All take the tool's `input` dict and return a JSON-serializable dict —
    # never raise on bad input, return {"error": ...} instead so the agent can react.

    def tool_get_data_profile(self, _):
        """Return the profile computed once in __init__. Ignores its input (the tool
        takes no arguments)."""
        return self.profile

    def tool_inspect_column(self, inp):
        """Drill into one column: dtype, sample values, value counts, and — for a
        classification target — the target rate per value, to help the agent confirm
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
        first use, cross-validate them through parallel worker streams, and return a
        leaderboard sorted by the primary metric (with overfit_gap and a
        suspiciously-high-score warning per row). Enforces MAX_ROUNDS/MAX_CANDIDATES
        and rejects models that don't belong to the given task."""
        if self.rounds >= MAX_ROUNDS:
            return {"error": f"Budget exhausted ({MAX_ROUNDS} rounds). Finalize the best candidate."}
        task, pm = inp["task"], inp["primary_metric"]
        if task not in MODEL_MENU:
            return {"error": "Task must be classification or regression."}
        if pm not in (("roc_auc", "f1_macro", "accuracy") if task == "classification" else ("rmse", "mae", "r2")):
            return {"error": f"Metric '{pm}' does not fit task '{task}'."}
        cands = inp["candidates"][:MAX_CANDIDATES]
        if not cands or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["name"]) for c in cands):
            return {"error": "Provide candidates with names containing only letters, numbers, underscores, and hyphens."}
        bad = [c["name"] for c in cands if c["model"] not in MODEL_MENU[task]]
        if bad:
            return {"error": f"Invalid models for {task}: {bad}. Allowed: {MODEL_MENU[task]}"}

        self.tracker.start(self.profile)
        self.tracker.set_task(task)
        if self.remote_path is None:
            self.emit("status", {"msg": "Uploading dataset to Modal"})
            prepared = upload_dataset(self.df, with_metadata=True)
            self.remote_path, self.prepared_sha256 = prepared["path"], prepared["sha256"]
        self.rounds += 1

        base = {"dataset_path": self.remote_path, "target": self.target, "task": task,
                "dataset_id": self.dataset["id"], "source_sha256": self.dataset["source_sha256"],
                "prepared_sha256": self.prepared_sha256,
                "cv": inp.get("cv", "kfold"), "cv_folds": inp.get("cv_folds", 5),
                "time_column": inp.get("time_column"), "drop_columns": inp.get("drop_columns", [])}
        specs = []
        for c in cands:
            name = c["name"]
            suffix = 1
            while name in self.specs:
                name = f"{c['name']}_r{self.rounds}_{suffix}"
                suffix += 1
            spec = {**base, "name": name, "model": c["model"],
                    "params": c.get("params") or {}, "preprocessing": c.get("preprocessing") or {}}
            self.specs[name] = spec
            specs.append(spec)
        self.emit("round_start", {"round": self.rounds, "rationale": inp["rationale"],
                                  "candidates": [{k: s[k] for k in ("name", "model", "params")} for s in specs]})

        t0 = time.time()
        hib = HIGHER_IS_BETTER[pm]
        rows = []
        jobs = [(self.tracker.queue_training(spec, self.rounds), spec, "candidate_cv") for spec in specs]
        for r in self._stream_trainings(jobs, pm):
            r["round"] = self.rounds
            self.results.append(r)
            if not r["ok"]:
                rows.append({"name": r["name"], "error": r["error"]})
                continue
            self.candidate_ids[r["name"]] = r["training_id"]
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
        if name not in self.candidate_ids:
            return {"error": f"Select a successful CV candidate. Available: {list(self.candidate_ids)}"}
        self.emit("status", {"msg": f"Training final model '{name}' on all data"})
        parent_id = self.candidate_ids[name]
        training_id = self.tracker.queue_training(self.specs[name], self.tracker.trainings[parent_id]["round"], "final_fit", parent_id)
        out = self._stream_trainings([(training_id, self.specs[name], "final_fit")])[0]
        if out["ok"]:
            self.tracker.select(parent_id)
        self.final = {"name": name, "spec": self.specs[name], **out}
        self.emit("final_model", self.final)
        return out

    def tool_write_report(self, inp):
        """Store the agent's final markdown report and short spoken summary. Calling
        this is also the loop's stop signal — `run()` breaks right after."""
        self.report = {"markdown": inp["markdown"], "spoken_summary": inp["spoken_summary"]}
        self.tracker.report(self.report)
        return {"saved": True}

    # ---- loop
    def _run_loop(self):
        """Drive the Claude tool-use loop: send the system prompt + task, stream any
        text as `thought` events, dispatch each tool_use block to the matching
        tool_* method, and feed results back as the next user turn. Stops when Claude
        stops calling tools, when write_report has been called, or after MAX_STEPS —
        then always saves whatever was produced."""
        hint = f" The user says this is a {self.task_hint} task." if self.task_hint else ""
        messages = [{"role": "user", "content":
                     f"Dataset '{self.dataset_name}', target column '{self.target}'.{hint} "
                     "Build the best model you can. Start by reading the data profile."}]

        for _ in range(MAX_STEPS):
            resp = self.client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                                               tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            calls = []
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self.emit("thought", {"text": block.text})
                elif block.type == "tool_use":
                    calls.append(block)
            if resp.stop_reason != "tool_use" or not calls:
                break

            tool_results = []
            for call in calls:
                self.emit("tool_call", {"tool": call.name, "input": call.input})
                try:
                    out = getattr(self, f"tool_{call.name}")(call.input)
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
                tool_results.append({"type": "tool_result", "tool_use_id": call.id,
                                     "content": json.dumps(out, default=str, ensure_ascii=False),
                                     "is_error": "error" in out})
            messages.append({"role": "user", "content": tool_results})
            if self.report:
                break

    def run(self):
        self.tracker.start(self.profile)
        status, error = "incomplete", "Agent stopped without a successful final fit and report."
        original_error = None
        try:
            self._run_loop()
            if self.final and self.final.get("ok") and self.report and self.report["markdown"].strip():
                status, error = "completed", None
        except BaseException as exc:
            original_error = exc
            status, error = "failed", f"Run interrupted ({type(exc).__name__})."
            raise
        finally:
            cleanup_error = None
            try:
                self.tracker.end(status, error)
            except Exception as exc:
                cleanup_error = exc
            try:
                self.save()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error:
                print(f"Run cleanup failed ({type(cleanup_error).__name__}); inspect the partial local log.")
                if original_error is None:
                    raise RuntimeError("Could not finish saving the run.") from cleanup_error

    def save(self):
        """Save results/report even after interruption; events are already journaled."""
        os.makedirs(self.output_dir, exist_ok=True)
        prefix = f"{self.output_dir}/{self.run_id}"
        with open(f"{prefix}_results.json", "w", encoding="utf-8") as f:
            json.dump(json_safe({"dataset": self.dataset, "run": self.tracker.run,
                      "trainings": list(self.tracker.trainings.values()), "profile": self.profile,
                      "results": self.results, "final": self.final,
                      "pending_events": len(self.tracker.pending)}), f, allow_nan=False, indent=2)
        if self.report:
            with open(f"{prefix}_report.md", "w", encoding="utf-8") as f:
                f.write(self.report["markdown"])
        print(f"\nSaved → {prefix}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--dry-run", action="store_true", help="scripted agent, no Anthropic API key needed")
    ap.add_argument("--dataset-id", help="Registered Supabase dataset UUID; downloads and verifies its exact file")
    ap.add_argument("--offline", action="store_true", help="Keep tracking locally; requires a local data path")
    args = ap.parse_args()
    FactoryRun(args.data, args.target, args.task, args.dry_run, args.dataset_id, args.offline).run()
