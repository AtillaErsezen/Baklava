"""Offline checks of the harness tools: Modal stubbed with deterministic fold scores.
Run: uv run python test_factory_tools.py"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import agent
import factory_tools

# Offline by construction: the harness calls factory_tools.upload_dataset, so stub that name (stubbing
# agent.upload_dataset did nothing and let these tests upload to the real Modal volume whenever a token existed).
factory_tools.upload_dataset = lambda df: "/datasets/fake.parquet"

# true mean per family: the fake backend makes lightgbm best, logreg a close second
TRUE = {"ridge": 0.84, "lightgbm": 0.86, "logreg": 0.855, "catboost": 0.84, "xgboost": 0.83, "random_forest": 0.82,
        "mlp": 0.78, "tabicl": 0.80}


class FakeFn:
    def __init__(self, kind):
        self.kind, self.calls = kind, 0

    def _one(self, s):
        self.calls += 1
        rng = np.random.default_rng(abs(hash(s["name"])) % 2**32)
        n = s.get("train_rows") or 2100
        k = s.get("cv_folds", 5) * s.get("repeats", 1)
        folds = (TRUE[s["model"]] + rng.normal(0, 0.02 * np.sqrt(500 / n), k)).round(5).tolist()
        m = {"mean": float(np.mean(folds)), "std": float(np.std(folds)), "train_mean": 0.9, "folds": folds}
        return {"name": s["name"], "ok": True, "fit_seconds": 0.01 * n / 500, "n_rows": n,
                "metrics": {"roc_auc": m, "accuracy": m, "f1_macro": m, "mae": m, "rmse": m, "r2": m}}

    def map(self, specs):
        return [self._one(s) for s in specs]

    def remote(self, spec, *args):
        self.paths = getattr(self, "paths", []) + [args[0] if args else None]
        if self.kind in ("predict_holdout", "predict_holdout_gpu"):
            y = HOLD["y"]
            rng = np.random.default_rng(len(spec["name"]))
            proba = np.clip(0.5 * y + rng.uniform(0, 0.5, len(y)) * (1.2 - TRUE[spec["model"]]), 0, 1)
            return {"name": spec["name"], "y_true": y.tolist(), "pred": (proba > 0.5).astype(int).tolist(),
                    "proba": proba.tolist(), "classes": ["No", "Yes"]}
        return {"model_path": f"/models/x/{spec['name']}.joblib", "n_rows": 10, "top_features": []}


HOLD = {}


def make_run(tmp_path="runs/test_memory.jsonl"):
    fns = {}
    agent.modal.Function.from_name = lambda app, name: fns.setdefault(name, FakeFn(name))
    run = agent.FactoryRun("data/churn.csv", "Churn", provider="scripted", purpose="find churners, explainable is a plus")
    run.memory_path = tmp_path
    y = run.hidden_df["Churn"].map({"No": 0, "Yes": 1}).to_numpy()
    HOLD["y"] = y
    return run, fns


def test_splits_lock_hidden_rows():
    run, _ = make_run()
    n = len(run.df)
    assert len(run.dev_df) + len(run.val_df) + len(run.hidden_df) == n
    assert set(run.dev_df.index).isdisjoint(run.hidden_df.index)


def test_diag_summary_is_compact_and_flags_leak():
    run, _ = make_run()
    out = run.tool_diag_summary({})
    checks = [f["check"] for f in out["findings"]]
    assert "leakage" in checks and len(out["findings"]) <= 15
    assert "refund_issued" in str(out["findings"])
    assert len(str(out)) < 6000  # token budget: compact


def test_run_search_races_hundreds_of_configs():
    run, fns = make_run()
    out = run.tool_run_search({"rationale": "test", "task": "classification", "primary_metric": "roc_auc",
                               "drop_columns": ["customer_id", "refund_issued"]})
    assert out["space_size"] >= 2000 and out["raced"] >= 100
    assert out["top"][0]["name"] in run.specs and len(out["top"]) == 3
    assert out["fits"] < out["raced"] * len(out["schedule"])  # racing saved fits
    assert "n_star" in out and "justification" in out


def test_confirm_and_test_ranks_with_statistics():
    run, _ = make_run()
    run.tool_run_search({"rationale": "t", "task": "classification", "primary_metric": "roc_auc",
                         "drop_columns": ["customer_id", "refund_issued"]})
    out = run.tool_confirm_and_test({})
    rows = out["table"]
    assert len(rows) == 3 and {"p_vs_best", "ci", "hidden", "tie_group", "big_o", "pareto"} <= set(rows[0])
    assert out["recommendation"]["pick"] in [r["name"] for r in rows]


def test_scripted_run_end_to_end_exports_user_model():
    run, _ = make_run()
    run.run()
    kinds = [e["kind"] for e in run.events]
    assert "rung" in kinds and "confirm" in kinds and kinds[-1] == "usage"
    assert run.final and run.export and all(k in run.export for k in ("script", "params", "card"))



def test_model_that_stops_early_is_nudged_to_continue():
    run, _ = make_run()
    real = run.client.chat
    state = {"stalled": False}

    def stall_once(messages, tools, **kw):  # like Qwen: narrates a plan, calls no tool
        if run.search is None and not state["stalled"] and any(m["role"] == "tool" for m in messages):
            state["stalled"] = True
            return SimpleNamespace(content="I will now run the search.", tool_calls=None), "stop"
        return real(messages, tools, **kw)

    run.client.chat = stall_once
    run.run()
    assert state["stalled"] and run.report, "run ended at the stall instead of finishing"



def test_sorted_datetime_column_gives_a_time_split(tmp="runs/_time.csv"):
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 600
    pd.DataFrame({"when": pd.date_range("2024-01-01", periods=n, freq="D"),
                  "x": rng.normal(size=n), "y": rng.integers(0, 2, n)}).to_csv(tmp, index=False)
    agent.modal.Function.from_name = lambda app, name: FakeFn(name)
    run = agent.FactoryRun(tmp, "y", provider="scripted")
    assert run.time_column == "when"
    assert run.hidden_df["when"].min() > run.dev_df["when"].max()  # hidden = the latest rows
    run.tool_run_search({"rationale": "t", "task": "classification", "primary_metric": "roc_auc"})
    assert run.search["base"]["cv"] == "walk_forward" and run.search["base"]["time_column"] == "when"


def test_hidden_rows_repeating_dev_rows_are_flagged(tmp="runs/_dup.csv"):
    import pandas as pd
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"x": rng.normal(size=150), "z": rng.integers(0, 9, 150), "y": rng.integers(0, 2, 150)})
    pd.concat([base] * 4, ignore_index=True).to_csv(tmp, index=False)  # every record appears 4 times
    agent.modal.Function.from_name = lambda app, name: FakeFn(name)
    run = agent.FactoryRun(tmp, "y", provider="scripted")
    dup = [f for f in run.diag["findings"] if f["check"] == "cross_split_duplicates"]
    assert dup and dup[0]["severity"] >= 2


def test_data_try_more_rows_adds_rows_to_training_only():
    import factory_tools as ft

    run, fns = make_run()
    uploads, seen = [], []
    saved = ft.xd.find_file_links, ft.xd.safe_fetch, ft.upload_dataset
    ft.xd.find_file_links = lambda url: ["https://openml.org/data/more.csv"]
    ft.xd.safe_fetch = lambda url: run.val_df.rename(columns={"Churn": "churned"})  # public rows, same schema
    ft.upload_dataset = lambda df: uploads.append(df) or f"/datasets/up{len(uploads)}.parquet"
    try:
        run.tool_run_search({"rationale": "t", "task": "classification", "primary_metric": "roc_auc",
                             "drop_columns": ["customer_id", "refund_issued"]})
        real_map = fns["train_candidate"].map
        fns["train_candidate"].map = lambda specs: seen.extend(specs) or real_map(specs)
        out = run.tool_data_try({"url": "https://openml.org/d/1", "mode": "more_rows", "column_map": {"churned": "Churn"}})
        bad = run.tool_data_try({"url": "https://openml.org/d/1", "mode": "bogus"})
    finally:
        ft.xd.find_file_links, ft.xd.safe_fetch, ft.upload_dataset = saved
    assert out["verdict"] in ("improves", "no_gain", "worse") and out["mode"] == "more_rows"
    assert {"source", "delta", "ci", "p_one_sided", "coverage"} <= set(out) and out["coverage"] == 1.0
    assert out["extra_rows"] == len(uploads[-1]) > 0 and list(uploads[-1].columns) == list(run.dev_df.columns)
    base_s, aug_s = seen[-2:]
    assert "extra_train_path" not in base_s and aug_s["extra_train_path"] == f"/datasets/up{len(uploads)}.parquet"
    assert aug_s["dataset_path"] == base_s["dataset_path"] and aug_s["repeats"] == base_s["repeats"] == ft.CONFIRM_REPEATS
    assert "error" in bad and set(ft.DATA_TRY_SCHEMA_ADDITIONS) == {"mode", "column_map"}



def test_predictions_use_the_requested_split_not_hidden():
    """Ensemble weights must be fit on search_val; predicting hidden there would leak the final test."""
    run, fns = make_run()
    run.tool_run_search({"rationale": "t", "task": "classification", "primary_metric": "roc_auc",
                         "drop_columns": ["customer_id", "refund_issued"]})
    run.hidden_path = "/datasets/hidden.parquet"
    spec = run.specs[run.search["top"][0]["name"]]
    run._predict([spec], "/datasets/search_val.parquet")
    assert fns["predict_holdout"].paths[-1] == "/datasets/search_val.parquet"



def test_goal_maps_to_a_use_case():
    run, _ = make_run()  # purpose: "find churners, explainable is a plus"
    assert run.use_case and run.use_case["id"] == "customer_churn"
    assert run.tool_diag_summary({})["use_case"]["id"] == "customer_churn"


def test_forecasting_goal_builds_a_leak_free_supervised_table():
    import os
    if not os.path.exists("evals/data/drifting_sales.csv"):
        import subprocess
        subprocess.run(["uv", "run", "evals/make_golden.py"], check=True)
    agent.modal.Function.from_name = lambda app, name: FakeFn(name)
    run = agent.FactoryRun("evals/data/drifting_sales.csv", "units", provider="scripted",
                           purpose="forecast daily units sold for the next day")
    assert run.forecast and run.task == "regression" and "units_lag1" in run.df.columns
    assert run.time_column == "date" and run.hidden_df["date"].min() > run.dev_df["date"].max()
    assert any(f["check"] == "naive_baseline" for f in run.diag["findings"])
    # nothing about the hidden period may shape features or the naive bar the agent sees
    assert pd.Timestamp(run.forecast["fit_until"]) < run.hidden_df["date"].min()
    out = run.tool_run_search({"rationale": "t", "task": "regression", "primary_metric": "mae"})
    assert run.search["base"]["cv"] == "walk_forward" and run.search["base"]["time_column"] == "date"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
