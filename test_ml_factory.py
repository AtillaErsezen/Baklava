"""Offline checks, no network: `uv run test_ml_factory.py`. Needs data/ (`uv run make_demo_data.py`).
The last check runs the full dry-run agent loop with Modal replaced by the same training code, in-process."""
import glob
import os

import numpy as np
import pandas as pd

import agent
import modal_train as mt
import stats

churn = mt.load_local("data/churn.csv")


def test_split():
    dev, hidden = agent.split_holdout(churn, "Churn", "classification")
    assert len(hidden) == 600 and len(dev) == 2400
    assert not set(dev.customer_id) & set(hidden.customer_id)
    assert abs((dev.Churn == "Yes").mean() - (hidden.Churn == "Yes").mean()) < 0.01

    t = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=100), "x": range(100), "y": [0, 1] * 50})
    col = agent.sorted_time_column(t, "y")
    dev, hidden = agent.split_holdout(t, "y", "classification", col)
    assert col == "when" and dev.when.max() < hidden.when.min() and len(hidden) == 20


def test_profile_flags():
    dev, _ = agent.split_holdout(churn, "Churn", "classification")
    flags = {c["name"]: c.get("flags", []) for c in agent.profile_data(dev, "Churn")["columns"]}
    assert "possible_leakage" in flags["refund_issued"] and "id_like" in flags["customer_id"]
    assert not flags["contract"] and not flags["monthly_charges"]

    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b"], 1000)
    status = np.where(rng.random(1000) < 0.995, np.where(y == "a", "closed", "open"), "open")
    df = pd.DataFrame({"status": status, "noise": rng.choice(["p", "q", "r"], 1000), "y": y})
    flags = {c["name"]: c.get("flags", []) for c in agent.profile_data(df, "y")["columns"]}
    assert "possible_leakage" in flags["status"] and not flags["noise"]

    assert agent.cross_split_duplicates(churn, churn.head(5), "Churn") == 5


def test_stats():
    t, p = stats.nb_corrected_t([1, 2, 3, 4], [0, 0, 0, 0], 1.0)  # mean 2.5, var 5/3 → t = √3
    assert abs(t - 3 ** 0.5) < 1e-9 and 0.08 < p < 0.10
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    rng = np.random.default_rng(1)
    y = rng.normal(0, 5, 2000)
    out = stats.paired_bootstrap(y, {"a": y + rng.normal(0, 1, 2000), "b": y + rng.normal(0, 2, 2000)}, "rmse", "a",
                                 n_boot=500)
    lo, hi = out["b"]["diff_ci"]
    assert 0 < lo < 1 < hi, out["b"]  # true rmse diff ≈ 1, clearly above 0
    assert out["a"]["diff_ci"] == [0.0, 0.0]


def test_modal_core_local():
    dev, hidden = agent.split_holdout(churn, "Churn", "classification")
    spec = {"name": "lr", "model": "logreg", "task": "classification", "target": "Churn", "cv": "kfold",
            "cv_folds": 2, "cv_repeats": 5, "drop_columns": ["customer_id", "refund_issued"]}
    r = mt._train(dev, spec)
    assert r["ok"], r
    assert len(r["metrics"]["roc_auc"]["folds"]) == 10 and 0.6 < r["metrics"]["roc_auc"]["mean"] < 0.95
    h = mt._predict_holdout(dev, hidden, spec)
    assert h["ok"] and len(h["y"]) == len(h["pred"]) == 600, h
    assert stats.score("roc_auc", h["y"], h["pred"]) > 0.6
    assert not mt._train(dev, {**spec, "model": "nope"})["ok"]


class _Local:
    """Replaces a modal.Function: .map / .remote call a local function."""

    def __init__(self, f):
        self.f = f

    def map(self, *iterables):
        return map(self.f, *iterables)

    def remote(self, *args):
        return self.f(*args)


def test_dry_run_loop():
    store = {}

    def upload(df):
        store[f"/datasets/{len(store)}"] = df
        return f"/datasets/{len(store) - 1}"

    agent.upload_dataset = upload
    run = agent.FactoryRun("data/churn.csv", "Churn", dry_run=True)
    run.train_fn = _Local(lambda spec: mt._train(store[spec["dataset_path"]], spec))
    run.holdout_fn = _Local(lambda spec, hp: mt._predict_holdout(store[spec["dataset_path"]], store[hp], spec))
    run.final_fn = _Local(lambda spec, rid: {"model_path": f"/models/{rid}/{spec['name']}.joblib", "top_features": []})
    try:
        run.run()
        kinds = [e["kind"] for e in run.events]
        assert {"leaderboard", "confirm", "hidden_test", "final_model", "report"} <= set(kinds), kinds
        assert set(run.specs["logreg_default"]["drop_columns"]) == {"customer_id", "refund_issued"}
        ok = [r for r in run.results if r["ok"]]
        assert len(ok) >= 6, [r.get("error") for r in run.results]  # sklearn families; lightgbm/xgboost may be absent
        tie = [f["name"] for f in run.finalists if f["tie_with_best"]]
        assert run.report["recommended"] in tie and "hidden" in run.finalists[0]
        assert "| model |" in run.report["markdown"]
        print(run.report["markdown"])
    finally:
        for f in glob.glob(f"runs/{run.run_id}_*"):
            os.remove(f)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
