"""Offline checks of the harness tools: Modal stubbed with deterministic fold scores.
Run: uv run python test_factory_tools.py"""
from types import SimpleNamespace

import numpy as np

import agent

# true mean per family: the fake backend makes lightgbm best, logreg a close second
TRUE = {"lightgbm": 0.86, "logreg": 0.855, "catboost": 0.84, "xgboost": 0.83, "random_forest": 0.82,
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
                "metrics": {"roc_auc": m, "accuracy": m, "f1_macro": m}}

    def map(self, specs):
        return [self._one(s) for s in specs]

    def remote(self, spec, *args):
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
    agent.upload_dataset = lambda df: "/datasets/fake.parquet"
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


def test_hidden_rows_repeating_dev_rows_are_flagged(tmp="runs/_dup.csv"):
    import pandas as pd
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"x": rng.normal(size=150), "z": rng.integers(0, 9, 150), "y": rng.integers(0, 2, 150)})
    pd.concat([base] * 4, ignore_index=True).to_csv(tmp, index=False)  # every record appears 4 times
    agent.modal.Function.from_name = lambda app, name: FakeFn(name)
    run = agent.FactoryRun(tmp, "y", provider="scripted")
    dup = [f for f in run.diag["findings"] if f["check"] == "cross_split_duplicates"]
    assert dup and dup[0]["severity"] >= 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
