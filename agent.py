"""
ML Factory — agent. Open-weight LLM on Nebius (OpenAI-compatible API) + Modal compute.

  uv run --env-file .env agent.py data/churn.csv --target Churn
  uv run --env-file .env agent.py data/houses.csv --target SalePrice --task regression
  uv run agent.py data/churn.csv --target Churn --dry-run      # scripted LLM, no API key
"""
import argparse
import json
import os
import time
import uuid
from types import SimpleNamespace

import modal
import numpy as np
import pandas as pd

import stats
from modal_train import APP_NAME, MODEL_MENU, SEED, load_local, upload_dataset

MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MAX_STEPS = 20
MAX_ROUNDS = 3
MAX_CANDIDATES = 8
HIDDEN_FRAC = 0.2
N_FINALISTS = 3
CONFIRM_FOLDS, CONFIRM_REPEATS = 2, 5  # 5x2 repeated CV for the significance test
ALPHA = 0.05
HIGHER_IS_BETTER = {"roc_auc": True, "f1_macro": True, "accuracy": True, "r2": True, "rmse": False, "mae": False}
TASK_METRICS = {"classification": ("roc_auc", "f1_macro", "accuracy"), "regression": ("rmse", "mae", "r2")}

BIG_O = {
    "logreg": "fit O(iters·n·d), predict O(d)",
    "ridge": "fit O(n·d² + d³), predict O(d)",
    "random_forest": "fit O(T·n·log n·√d), predict O(T·depth)",
    "lightgbm": "fit O(T·n·d) histogram, predict O(T·depth)",
    "xgboost": "fit O(T·n·d) histogram, predict O(T·depth)",
    "mlp": "fit O(epochs·n·W), predict O(W), W = weights",
}

# Round-1 portfolio: per family a default, a shallow/regularised, and a deep/slow config.
_FAMILY_CONFIGS = {
    "random_forest": [("rf_default", {}), ("rf_shallow", {"max_depth": 6, "min_samples_leaf": 5}),
                      ("rf_deep", {"n_estimators": 500, "max_features": 0.5})],
    "lightgbm": [("lgbm_default", {}),
                 ("lgbm_reg", {"num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5.0, "n_estimators": 200}),
                 ("lgbm_slow", {"learning_rate": 0.03, "n_estimators": 600, "num_leaves": 63, "subsample": 0.8,
                                "subsample_freq": 1, "colsample_bytree": 0.8})],
    "xgboost": [("xgb_default", {}),
                ("xgb_reg", {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 5.0, "n_estimators": 200}),
                ("xgb_slow", {"learning_rate": 0.03, "n_estimators": 600, "max_depth": 8, "subsample": 0.8,
                              "colsample_bytree": 0.8})],
    "mlp": [("mlp_default", {}), ("mlp_small", {"hidden_layer_sizes": [32], "alpha": 0.01}),
            ("mlp_big", {"hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0005})],
}
PORTFOLIO = {
    "classification": [{"name": n, "model": "logreg", "params": p} for n, p in
                       [("logreg_default", {}), ("logreg_c01", {"C": 0.1}), ("logreg_balanced", {"class_weight": "balanced"})]],
    "regression": [{"name": n, "model": "ridge", "params": p} for n, p in
                   [("ridge_default", {}), ("ridge_a10", {"alpha": 10.0}), ("ridge_a01", {"alpha": 0.1})]],
}
for _task in PORTFOLIO:
    PORTFOLIO[_task] += [{"name": n, "model": m, "params": p} for m, cfgs in _FAMILY_CONFIGS.items() for n, p in cfgs]

SYSTEM_PROMPT = f"""You are ML Factory, an autonomous ML engineer. You get a tabular dataset and a target column and
must deliver the best *trustworthy* model. A Python harness runs the training and all statistics; you make the
judgement calls and explain them. {int(HIDDEN_FRAC * 100)}% of the rows are held out as a hidden test set that
you never see until the end.

Workflow
1. get_data_profile. Decide: task; primary metric (binary → roc_auc, or f1_macro if the minority class matters and
   imbalance is heavy; multiclass → f1_macro; regression → rmse); CV scheme (timeseries if the profile reports a
   time order and the target is about the future, else kfold); columns to drop (id_like, constant, possible_leakage,
   free text). Use inspect_column when a flag is ambiguous.
2. run_experiments WITHOUT candidates: runs a fixed portfolio of {len(PORTFOLIO["classification"])} configs across all families.
3. Optionally 1-2 more run_experiments rounds WITH up to {MAX_CANDIDATES} candidates: variants of the top 1-2 families
   (depth, learning rate, n_estimators, regularization, class_weight / scale_pos_weight, ordinal vs onehot for
   trees). Skip if the top configs are already within one std of each other. Budget: {MAX_ROUNDS} rounds in total.
   A large overfit_gap → more regularization. A suspiciously_high warning → inspect_column, drop the column, and rerun
   run_experiments without candidates.
4. confirm_finalists: the harness takes the best config of the top {N_FINALISTS} model families, re-runs {CONFIRM_REPEATS}x{CONFIRM_FOLDS} CV, tests them
   against the best (Nadeau–Bengio, Holm), and scores them once on the hidden set.
5. write_report: recommend one model from the tie group. Within the tie group prefer the simpler / faster one;
   do not pick by hidden score. Then stop.

Rules
- Models: {json.dumps(MODEL_MENU)}. params must be valid constructor kwargs for that library.
- Short unique candidate names, e.g. lgbm_d6_lr05.
- Before every tool call, write 1-2 plain sentences explaining your reasoning; they are streamed live to a
  non-expert audience.
- Never invent numbers; only use numbers returned by tools. The harness appends the finalist table to your report.
- Report markdown: dataset summary, data issues and how you handled them, what was tried, the recommendation and why,
  caveats, next steps. spoken_summary: max 60 words, to be read aloud, only the headline metric as a number."""


def _tool(name, description, properties=None, required=()):
    """One tool definition in OpenAI function-calling format."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required)}}}


TOOLS = [
    _tool("get_data_profile", "Schema, column types, missing values, cardinality, target distribution, flags "
          "(id_like, constant, high_cardinality, mostly_missing, possible_leakage), split and time-order info."),
    _tool("inspect_column", "Sample values, value counts and (for classification) target rate per value for one "
          "column. Use to confirm leakage or decide whether to drop a column.",
          {"column": {"type": "string"}}, ["column"]),
    _tool("run_experiments", "Cross-validate candidates in parallel on Modal. Without candidates it runs the fixed "
          "portfolio. Returns a leaderboard: validation mean/std, train mean, overfit_gap, all metrics.", {
              "rationale": {"type": "string", "description": "Why this round. Shown to the user."},
              "task": {"type": "string", "enum": ["classification", "regression"]},
              "primary_metric": {"type": "string", "enum": list(HIGHER_IS_BETTER)},
              "cv": {"type": "string", "enum": ["kfold", "timeseries"]},
              "cv_folds": {"type": "integer", "minimum": 3, "maximum": 10},
              "time_column": {"type": "string", "description": "Sort by this before timeseries CV."},
              "drop_columns": {"type": "array", "items": {"type": "string"}},
              "candidates": {
                  "type": "array", "maxItems": MAX_CANDIDATES,
                  "description": "Omit to run the fixed portfolio.",
                  "items": {"type": "object", "required": ["name", "model"], "properties": {
                      "name": {"type": "string"},
                      "model": {"type": "string", "enum": sorted({m for ms in MODEL_MENU.values() for m in ms})},
                      "params": {"type": "object", "description": "Constructor kwargs."},
                      "preprocessing": {"type": "object", "properties": {
                          "numeric_impute": {"type": "string", "enum": ["median", "mean"]},
                          "scale": {"type": "boolean"},
                          "categorical": {"type": "string", "enum": ["onehot", "ordinal"]},
                          "max_categories": {"type": "integer"}}}}}}},
          ["rationale", "task", "primary_metric"]),
    _tool("confirm_finalists", f"Best config of the top {N_FINALISTS} model families in the latest setup: "
          f"{CONFIRM_REPEATS}x{CONFIRM_FOLDS} repeated CV, "
          "Nadeau–Bengio test vs the best with Holm correction (tie group), then one scoring on the hidden set "
          "with bootstrap CIs. Can only run once."),
    _tool("write_report", "Fit the recommended model on all data, save it and the report. Call once, at the end.", {
        "recommended": {"type": "string", "description": "A candidate name from the tie group."},
        "markdown": {"type": "string"}, "spoken_summary": {"type": "string"}},
        ["recommended", "markdown", "spoken_summary"]),
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


# ------------------------------------------------------------------ data: split + profile (local, cheap)

def _r(v):
    """Round a possibly-NaN scalar to 4 decimals, or None if it's NaN."""
    return round(float(v), 4) if pd.notna(v) else None


def guess_task(y: pd.Series) -> str:
    """classification for non-numeric targets or integers with ≤ 20 distinct values, else regression."""
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if not numeric or (pd.api.types.is_integer_dtype(y) and y.nunique() <= 20):
        return "classification"
    return "regression"


def sorted_time_column(df: pd.DataFrame, target: str):
    """The first datetime column (not the target) whose values are already in increasing
    order, or None. Sorted rows suggest time-ordered data: time split + timeseries CV."""
    for c in df.columns:
        s = df[c].dropna()
        if c != target and pd.api.types.is_datetime64_any_dtype(s) and s.nunique() > 1 and s.is_monotonic_increasing:
            return c
    return None


def split_holdout(df: pd.DataFrame, target: str, task: str, time_col=None):
    """Split off the hidden test set before anything else. Rows with a missing target are
    dropped. Time-ordered data: the last HIDDEN_FRAC of rows by time_col. Otherwise a
    seeded random split, stratified on the target for classification (when every
    class has ≥ 2 rows). Returns (dev, hidden), both with a fresh index."""
    from sklearn.model_selection import train_test_split

    df = df.dropna(subset=[target]).reset_index(drop=True)
    if time_col:
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
        cut = int(round(len(df) * (1 - HIDDEN_FRAC)))
        return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    y = df[target].astype(str)
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    dev, hidden = train_test_split(df, test_size=HIDDEN_FRAC, random_state=SEED, stratify=strat)
    return dev.reset_index(drop=True), hidden.reset_index(drop=True)


def cross_split_duplicates(dev: pd.DataFrame, hidden: pd.DataFrame, target: str) -> int:
    """Hidden rows whose features exactly match a dev row. They inflate the hidden score."""
    h = lambda d: pd.util.hash_pandas_object(d.drop(columns=[target]), index=False)
    return int(h(hidden).isin(set(h(dev))).sum())


def profile_data(df: pd.DataFrame, target: str) -> dict:
    """Build the JSON profile the agent reads instead of raw rows: shape, target
    distribution/task guess, and per-column type/missingness/cardinality with
    flags (id_like, constant, high_cardinality, mostly_missing, possible_leakage).
    Leakage: numeric |corr| with a binary/regression target > 0.95, or a categorical
    column (≤ 50 levels) whose levels predict the class with > 98% purity, clearly
    above the majority-class rate."""
    n = len(df)
    y = df[target]
    task = guess_task(y)
    n_classes = int(y.nunique())

    if task == "classification":
        balance = y.value_counts(normalize=True)
        majority = float(balance.iloc[0]) if len(balance) else 1.0
        target_info = {"suggested_task": task, "n_classes": n_classes,
                       "class_balance": {str(k): round(float(v), 3) for k, v in balance.head(10).items()}}
        y_code = pd.Series(pd.factorize(y)[0], index=y.index).where(y.notna()) if n_classes == 2 else None
    else:
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
            if task == "classification" and 1 < nun <= 50:
                ct = pd.crosstab(s, y)
                purity = float(ct.max(axis=1).sum() / max(ct.to_numpy().sum(), 1))
                info["target_purity"] = round(purity, 4)
                if purity > 0.98 and purity - majority > 0.05:
                    flags.append("possible_leakage")
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
    """Stand-in for the OpenAI client: replays a fixed tool sequence so the whole pipeline
    except the LLM's judgement (Modal, statistics, events, saving) can be tested without
    an API key. Same call shape as client.chat.completions.create."""

    def __init__(self, profile):
        self.profile, self.step = profile, 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, messages, **_):
        """Return the next scripted tool call as an OpenAI-shaped response. Script: profile
        -> portfolio round (flagged columns dropped) -> confirm_finalists -> write_report
        recommending the first tie-group member -> no more calls. Later steps read the
        previous tool result from `messages`. Other kwargs (model, tools...) are ignored."""
        self.step += 1
        last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else None
        task = self.profile["target"]["suggested_task"]
        if self.step == 1:
            name, inp = "get_data_profile", {}
        elif self.step == 2:
            bad = {"id_like", "possible_leakage", "constant"}
            name, inp = "run_experiments", {
                "rationale": "Dry run: fixed portfolio, flagged columns dropped.",
                "task": task, "primary_metric": "roc_auc" if task == "classification" else "rmse",
                "drop_columns": [c["name"] for c in self.profile["columns"] if bad & set(c.get("flags", []))]}
        elif self.step == 3:
            name, inp = "confirm_finalists", {}
        elif self.step == 4:
            name, inp = "write_report", {"recommended": last["tie_group"][0], "markdown": "# Dry run report",
                                         "spoken_summary": "Dry run complete."}
        else:
            name = None
        calls = [SimpleNamespace(id=f"dry{self.step}", type="function",
                                 function=SimpleNamespace(name=name, arguments=json.dumps(inp)))] if name else None
        msg = SimpleNamespace(content=f"(dry run) calling {name}" if name else "(dry run) done", tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


# ------------------------------------------------------------------ the agent

def _nebius():
    """OpenAI client pointed at Nebius. Exits with a hint if NEBIUS_API_KEY is missing."""
    from openai import OpenAI

    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise SystemExit("NEBIUS_API_KEY is not set. Use `uv run --env-file .env ...` or --dry-run.")
    return OpenAI(base_url=BASE_URL, api_key=key)


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


def _setup_key(spec):
    """Specs with the same key were evaluated on the same columns and CV scheme, so their scores compare."""
    return spec["task"], spec["cv"], spec.get("time_column"), tuple(sorted(spec.get("drop_columns") or []))


class FactoryRun:
    """One end-to-end run: load data, split off the hidden set, profile the dev set, drive
    the LLM tool loop (profile/inspect/experiment/confirm/report), and save the result.
    One instance = one dataset + target; not reused across runs."""

    def __init__(self, data_path: str, target: str, task_hint: str | None = None, dry_run: bool = False):
        """Load the dataset, validate the target, split dev/hidden, and profile the dev set
        up front so get_data_profile is a free tool call. The LLM only ever sees dev data."""
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.df = load_local(data_path)
        if target not in self.df.columns:
            raise SystemExit(f"Target '{target}' not found. Columns: {list(self.df.columns)}")
        self.dataset_name = os.path.basename(data_path)
        self.target, self.task_hint = target, task_hint
        self.time_col = sorted_time_column(self.df, target)
        self.dev, self.hidden = split_holdout(self.df, target, task_hint or guess_task(self.df[target]), self.time_col)
        self.profile = profile_data(self.dev, target)
        self.profile["split"] = {
            "dev_rows": len(self.dev), "hidden_rows": len(self.hidden),
            "method": f"last {int(HIDDEN_FRAC * 100)}% by {self.time_col}" if self.time_col else "random, stratified",
            "hidden_rows_duplicated_in_dev": cross_split_duplicates(self.dev, self.hidden, target)}
        if self.time_col:
            self.profile["time_order"] = {"column": self.time_col, "suggest": "cv=timeseries with this time_column"}
        self.paths = None
        self.rounds, self.pm, self.setup = 0, None, None
        self.specs, self.results, self.events = {}, [], []
        self.finalists, self.final, self.report = None, None, None
        self.tokens = {"prompt": 0, "completion": 0}
        self.client = ScriptedClient(self.profile) if dry_run else _nebius()
        self.train_fn = modal.Function.from_name(APP_NAME, "train_candidate")
        self.holdout_fn = modal.Function.from_name(APP_NAME, "predict_holdout")
        self.final_fn = modal.Function.from_name(APP_NAME, "fit_final")
        self.supabase = _maybe_supabase()

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

    def _upload(self):
        """Upload dev, hidden and full data to the Modal volume once; returns their paths."""
        if self.paths is None:
            self.emit("status", {"msg": "Uploading dataset to Modal"})
            self.paths = {"dev": upload_dataset(self.dev), "hidden": upload_dataset(self.hidden),
                          "full": upload_dataset(pd.concat([self.dev, self.hidden], ignore_index=True))}
        return self.paths

    # ---- tools
    # Each tool_* method mirrors one entry in TOOLS and is dispatched by name from
    # run(). All take the tool's `input` dict and return a JSON-serializable dict —
    # never raise on bad input, return {"error": ...} instead so the agent can react.

    def tool_get_data_profile(self, _):
        """Return the dev-set profile computed once in __init__. Takes no arguments."""
        return self.profile

    def tool_inspect_column(self, inp):
        """Drill into one dev-set column: dtype, sample values, value counts, and — for a
        classification target — the target rate per value, to help the agent confirm
        or rule out leakage/id-like columns flagged by the profile."""
        c = inp["column"]
        if c not in self.dev.columns:
            return {"error": f"Unknown column '{c}'. Columns: {list(self.dev.columns)}"}
        s = self.dev[c]
        out = {"dtype": str(s.dtype), "sample": s.dropna().astype(str).head(10).tolist(),
               "value_counts": {str(k): int(v) for k, v in s.value_counts(dropna=False).head(15).items()}}
        if self.profile["target"]["suggested_task"] == "classification" and s.nunique() <= 50:
            ct = pd.crosstab(s.astype(str), self.dev[self.target].astype(str), normalize="index").round(3).head(15)
            out["target_rate_by_value"] = {str(k): {str(kk): float(vv) for kk, vv in row.items()}
                                           for k, row in ct.iterrows()}
        return out

    def tool_run_experiments(self, inp):
        """Validate a round (the fixed portfolio when no candidates are given), cross-validate
        it in parallel via `train_fn.map`, and return a leaderboard sorted by the primary
        metric with overfit_gap and a suspiciously-high warning per row. Enforces
        MAX_ROUNDS/MAX_CANDIDATES and rejects models that don't belong to the task."""
        if self.rounds >= MAX_ROUNDS:
            return {"error": f"Budget exhausted ({MAX_ROUNDS} rounds). Call confirm_finalists."}
        task, pm = inp["task"], inp["primary_metric"]
        if task not in TASK_METRICS:
            return {"error": f"task must be one of {list(TASK_METRICS)}"}
        if pm not in TASK_METRICS[task]:
            return {"error": f"Metric '{pm}' does not fit task '{task}'. Allowed: {TASK_METRICS[task]}"}
        cands = (inp.get("candidates") or [])[:MAX_CANDIDATES] or PORTFOLIO[task]
        bad = [c.get("name") for c in cands if c.get("model") not in MODEL_MENU[task]]
        if bad:
            return {"error": f"Invalid models for {task}: {bad}. Allowed: {MODEL_MENU[task]}"}
        cv = inp.get("cv", "kfold")
        time_col = inp.get("time_column") or (self.time_col if cv == "timeseries" else None)
        drops = [c for c in inp.get("drop_columns") or [] if c != self.target]
        unknown = [c for c in drops + ([time_col] if time_col else []) if c not in self.dev.columns]
        if unknown:
            return {"error": f"Unknown columns {unknown}. Columns: {list(self.dev.columns)}"}

        paths = self._upload()
        self.rounds += 1
        base = {"dataset_path": paths["dev"], "target": self.target, "task": task, "cv": cv,
                "cv_folds": inp.get("cv_folds", 5), "time_column": time_col, "drop_columns": drops}
        self.pm, self.setup = pm, _setup_key(base)
        specs = []
        for c in cands:
            name = c["name"] if c["name"] not in self.specs else f"{c['name']}_r{self.rounds}"
            spec = {**base, "name": name, "model": c["model"],
                    "params": c.get("params") or {}, "preprocessing": c.get("preprocessing") or {}}
            self.specs[name] = spec
            specs.append(spec)
        self.emit("round_start", {"round": self.rounds, "rationale": inp["rationale"],
                                  "portfolio": not inp.get("candidates"),
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

    def tool_confirm_finalists(self, _):
        """Harness-only phase, runs once. Takes the best config of each of the top
        N_FINALISTS model families by CV score among specs of the latest setup, re-runs them with 5x2 repeated CV, tests each against the
        best with the Nadeau–Bengio corrected t-test + Holm (adjusted p ≥ ALPHA → tie
        with the best), then fits each on dev and scores the hidden set once with paired
        bootstrap CIs. The LLM does not choose the finalists."""
        if self.finalists is not None:
            return {"error": "Finalists already confirmed; the hidden set is scored only once.",
                    "finalists": self.finalists, "tie_group": [f["name"] for f in self.finalists if f["tie_with_best"]]}
        pm = self.pm
        if pm is None:
            return {"error": "Run run_experiments first."}
        hib = HIGHER_IS_BETTER[pm]
        pool = [r for r in self.results if r["ok"] and _setup_key(self.specs[r["name"]]) == self.setup]
        if not pool:
            return {"error": "No successful candidates in the latest setup."}
        pool.sort(key=lambda r: r["metrics"][pm]["mean"], reverse=hib)
        best_per_family = {}
        for r in pool:
            best_per_family.setdefault(self.specs[r["name"]]["model"], r["name"])
        names = list(best_per_family.values())[:N_FINALISTS]
        timeseries = self.setup[1] == "timeseries"
        self.emit("status", {"msg": f"Confirming finalists {names}"})

        # 1) repeated CV on identical folds -> paired significance test vs the best
        cspecs = [self.specs[n] if timeseries else {**self.specs[n], "cv_folds": CONFIRM_FOLDS,
                                                    "cv_repeats": CONFIRM_REPEATS} for n in names]
        conf = {r["name"]: r for r in self.train_fn.map(cspecs)}
        ok = [n for n in names if conf[n]["ok"]]
        if not ok:
            return {"error": "Confirmation CV failed for all finalists.",
                    "details": {n: conf[n].get("error") for n in names}}
        sign = 1 if hib else -1
        folds = {n: sign * np.asarray(conf[n]["metrics"][pm]["folds"]) for n in ok}
        best = max(ok, key=lambda n: folds[n].mean())
        k = cspecs[0]["cv_folds"]
        ratio = 2 / (k + 1) if timeseries else 1 / (k - 1)  # n_test / n_train per fold
        others = [n for n in ok if n != best]
        raw = [stats.nb_corrected_t(folds[best], folds[n], ratio)[1] for n in others]
        adj = dict(zip(others, stats.holm(raw))) if others else {}
        rows = []
        for n in ok:
            m = conf[n]["metrics"][pm]
            p = None if n == best else round(float(adj[n]), 4)
            rows.append({"name": n, "model": self.specs[n]["model"], "params": self.specs[n]["params"],
                         "cv_mean": m["mean"], "cv_std": m["std"], "n_folds": len(m["folds"]),
                         "p_vs_best_holm": p, "tie_with_best": n == best or p >= ALPHA,
                         "big_o": BIG_O[self.specs[n]["model"]]})
        self.emit("confirm", {"primary_metric": pm, "best": best, "rows": rows})

        # 2) hidden set, scored once
        hold = {h["name"]: h for h in self.holdout_fn.map([self.specs[n] for n in ok],
                                                          [self.paths["hidden"]] * len(ok))}
        scored = [n for n in ok if hold[n]["ok"]]
        boot = {}
        if best in scored:
            y = hold[best]["y"]
            boot = stats.paired_bootstrap(y, {n: hold[n]["pred"] for n in scored}, pm, best)
        for row in rows:
            h, b = hold[row["name"]], boot.get(row["name"])
            if b:
                row.update(hidden=b["score"], hidden_ci=b["ci"], hidden_diff_vs_best=b["diff_vs_best"],
                           hidden_diff_ci=b["diff_ci"], fit_seconds=h["fit_seconds"],
                           predict_ms_per_1k=h["predict_ms_per_1k"])
            else:
                row["hidden_error"] = h.get("error", "not scored: the best finalist failed on the hidden set")
        self.finalists = rows
        tie = [r["name"] for r in rows if r["tie_with_best"]]
        self.emit("hidden_test", {"primary_metric": pm, "hidden_rows": len(self.hidden), "rows": rows})
        return {"primary_metric": pm, "best_by_cv": best, "tie_group": tie, "finalists": rows,
                "note": "Recommend one model from tie_group; prefer the simpler/faster one, not the best hidden score."}

    def tool_write_report(self, inp):
        """Check the recommendation is in the tie group, fit it on all data (dev + hidden)
        via Modal's `fit_final`, and store the report with the harness-computed finalist
        table appended. Calling this is also the loop's stop signal."""
        if self.finalists is None:
            return {"error": "Call confirm_finalists first."}
        rec = inp["recommended"]
        tie = [r["name"] for r in self.finalists if r["tie_with_best"]]
        if rec not in tie:
            return {"error": f"recommended must be one of the tie group: {tie}"}
        self.emit("status", {"msg": f"Training final model '{rec}' on all data"})
        out = self.final_fn.remote({**self.specs[rec], "dataset_path": self.paths["full"]}, self.run_id)
        self.final = {"name": rec, "spec": self.specs[rec], **out}
        self.emit("final_model", self.final)
        markdown = inp["markdown"].rstrip() + "\n\n" + self._finalist_table(rec)
        self.report = {"recommended": rec, "markdown": markdown, "spoken_summary": inp["spoken_summary"]}
        self.emit("report", self.report)
        return {"saved": True, "model_path": out.get("model_path")}

    def _finalist_table(self, rec):
        """Markdown table of the finalists; every number comes from confirm_finalists."""
        pm = self.pm
        f = lambda v: "–" if v is None else f"{v:.4f}"
        ci = lambda c: f"[{c[0]:.4f}, {c[1]:.4f}]" if c else "–"
        lines = [f"## Finalists (computed by the harness, not the LLM)", "",
                 f"CV: {self.finalists[0]['n_folds']} folds on {len(self.dev)} dev rows. "
                 f"Hidden: {len(self.hidden)} rows, scored once. Δ = hidden {pm} minus the CV-best model's"
                 f"{' (lower is better)' if not HIGHER_IS_BETTER[pm] else ''}.", "",
                 f"| model | CV {pm} (mean ± std) | p vs best (Holm) | tie | hidden {pm} [95% CI] | Δ vs best [95% CI] "
                 "| fit s | predict ms/1k | Big-O |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in self.finalists:
            star = " ★" if r["name"] == rec else ""
            lines.append(f"| {r['name']}{star} | {r['cv_mean']:.4f} ± {r['cv_std']:.4f} | {f(r['p_vs_best_holm'])} "
                         f"| {'yes' if r['tie_with_best'] else 'no'} | {f(r.get('hidden'))} {ci(r.get('hidden_ci'))} "
                         f"| {f(r.get('hidden_diff_vs_best'))} {ci(r.get('hidden_diff_ci'))} "
                         f"| {r.get('fit_seconds', '–')} | {r.get('predict_ms_per_1k', '–')} | {r['big_o']} |")
        return "\n".join(lines) + "\n"

    def _dispatch(self, name, arguments):
        """Parse and check one tool call, run it, and return its result. Bad JSON, unknown
        tools and missing required fields come back as errors the LLM can fix."""
        if name not in REQUIRED:
            return {"error": f"Unknown tool '{name}'. Tools: {list(REQUIRED)}"}
        try:
            inp = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return {"error": f"Arguments are not valid JSON: {e}"}
        missing = [k for k in REQUIRED[name] if k not in inp]
        if missing:
            return {"error": f"Missing required fields {missing} for {name}."}
        self.emit("tool_call", {"tool": name, "input": inp})
        try:
            return getattr(self, f"tool_{name}")(inp)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- loop
    def run(self):
        """Drive the tool-use loop: send the system prompt + task, stream any text as
        `thought` events, dispatch each tool call, and feed results back. Stops when
        write_report succeeds or after MAX_STEPS; if the model stops calling tools
        early it is nudged twice. Always saves whatever was produced."""
        self.emit("run_start", {"dataset": self.dataset_name, "target": self.target, "model": MODEL,
                                "rows": len(self.df), "features": self.profile["n_features"],
                                "split": self.profile["split"]})
        hint = f" The user says this is a {self.task_hint} task." if self.task_hint else ""
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Dataset '{self.dataset_name}', target column '{self.target}'.{hint} "
                                                "Build the best model you can. Start by reading the data profile."}]
        nudges = 0
        for _ in range(MAX_STEPS):
            resp = self.client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS,
                                                       temperature=0, max_tokens=2048)
            if resp.usage:
                self.tokens["prompt"] += resp.usage.prompt_tokens
                self.tokens["completion"] += resp.usage.completion_tokens
            msg = resp.choices[0].message
            if msg.content and msg.content.strip():
                self.emit("thought", {"text": msg.content.strip()})
            calls = msg.tool_calls or []
            assistant = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                    "name": c.function.name, "arguments": c.function.arguments}} for c in calls]
            messages.append(assistant)
            if not calls:
                if nudges >= 2:
                    break
                nudges += 1
                messages.append({"role": "user", "content": "Continue with the next tool call. "
                                                            "The run ends only when write_report succeeds."})
                continue
            for c in calls:
                out = self._dispatch(c.function.name, c.function.arguments)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(out, default=str, ensure_ascii=False)})
            if self.report:
                break
        self.emit("run_end", {"tokens": self.tokens, "completed": self.report is not None})
        self.save()

    def save(self):
        """Persist this run under runs/<run_id>_*: the full event timeline as jsonl
        (for the replay-mode fallback), profile/results/finalists/final/tokens as json,
        and the markdown report if one was written. Called once at the end of run()."""
        os.makedirs("runs", exist_ok=True)
        prefix = f"runs/{self.run_id}"
        with open(f"{prefix}_events.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in self.events)
        with open(f"{prefix}_results.json", "w", encoding="utf-8") as f:
            json.dump({"profile": self.profile, "results": self.results, "finalists": self.finalists,
                       "final": self.final, "tokens": self.tokens}, f, default=str, indent=2)
        if self.report:
            with open(f"{prefix}_report.md", "w", encoding="utf-8") as f:
                f.write(self.report["markdown"])
        print(f"\nSaved → {prefix}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--dry-run", action="store_true", help="scripted LLM, no API key needed")
    args = ap.parse_args()
    FactoryRun(args.data, args.target, args.task, args.dry_run).run()
