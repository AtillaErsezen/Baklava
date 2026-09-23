"""Offline tests of results_store with a fake Supabase client (no network)."""
import math

import numpy as np

from results_store import ResultsStore, candidate_row


class FakeQuery:
    def __init__(self, log, table):
        self.log, self.call = log, {"table": table}

    def _op(self, op, payload=None, **kw):
        self.call.update(op=op, payload=payload, **kw)
        return self

    def insert(self, rows):
        return self._op("insert", rows)

    def upsert(self, row, on_conflict=None):
        return self._op("upsert", row, on_conflict=on_conflict)

    def update(self, row):
        return self._op("update", row)

    def eq(self, col, val):
        self.call["eq"] = (col, val)
        return self

    def execute(self):
        self.log.append(self.call)
        return self


class FakeClient:
    def __init__(self):
        self.log = []

    def table(self, name):
        return FakeQuery(self.log, name)


class BrokenClient:
    def table(self, name):
        raise ConnectionError("supabase down")


RESULT = {"name": "lgbm_p00", "ok": True, "fit_seconds": 1.5,
          "metrics": {"roc_auc": {"mean": 0.83, "std": 0.01, "train_mean": 0.9, "folds": [0.82, 0.84]},
                      "accuracy": {"mean": 0.8, "std": 0.02, "train_mean": 0.85, "folds": [0.79, 0.81]}}}
CONFIG = {"name": "lgbm_p00", "model": "lightgbm", "family": "lightgbm", "params": {"num_leaves": 31}}


def test_candidate_row_maps_primary_metric():
    row = candidate_row("r1", "race", CONFIG, RESULT, "roc_auc", rung=2, train_rows=500)
    assert (row["cv_mean"], row["cv_std"], row["train_mean"]) == (0.83, 0.01, 0.9)
    assert row["family"] == "lightgbm" and row["stage"] == "race" and row["rung"] == 2 and row["ok"] is True
    assert row["metrics"]["accuracy"]["folds"] == [0.79, 0.81] and row["preprocessing"] == {}
    failed = candidate_row("r1", "race", {"model": "xgboost"}, {"name": "x", "ok": False, "error": "boom"}, "roc_auc")
    assert failed["ok"] is False and failed["error"] == "boom" and failed["cv_mean"] is None and failed["family"] == "xgboost"


def test_writes_go_to_the_right_tables():
    fake = FakeClient()
    store = ResultsStore(fake, "r1")
    store.start_run(dataset_name="churn.csv", n_rows=3000)
    store.add_candidates([candidate_row("r1", "race", CONFIG, RESULT, "roc_auc")])
    store.add_candidates([])  # empty batch: no request
    store.finish_run(status="completed", recommended="lgbm_p00")
    ops = [(c["table"], c["op"]) for c in fake.log]
    assert ops == [("runs", "upsert"), ("candidates", "insert"), ("runs", "update")]
    start, _, finish = fake.log
    assert start["payload"]["status"] == "running" and start["on_conflict"] == "run_id"
    assert finish["eq"] == ("run_id", "r1") and finish["payload"]["recommended"] == "lgbm_p00"
    assert finish["payload"]["finished_at"].endswith("+00:00")


def test_payloads_are_json_safe():
    fake = FakeClient()
    ResultsStore(fake, "r1").add_candidates([{"run_id": "r1", "p_vs_best": math.nan, "cv_mean": np.float64(0.5),
                                              "rung": np.int64(3), "tie_with_best": np.bool_(True),
                                              "metrics": {"x": {"folds": [np.float32(0.25), math.inf]}}}])
    row = fake.log[0]["payload"][0]
    assert row["p_vs_best"] is None and row["cv_mean"] == 0.5
    assert type(row["rung"]) is int and row["tie_with_best"] is True
    assert row["metrics"]["x"]["folds"] == [0.25, None]


def test_no_client_and_broken_client_never_raise():
    for client in (None, BrokenClient()):
        store = ResultsStore(client, "r1")
        store.start_run(n_rows=1)
        store.add_candidates([{"run_id": "r1"}])
        store.finish_run(status="completed")
