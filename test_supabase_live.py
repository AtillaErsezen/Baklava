"""Live Supabase checks: ping every table, then a full CRUD cycle on runs + candidates.

  uv run --env-file .env pytest -q test_supabase_live.py

Skipped unless SUPABASE_URL and SUPABASE_KEY are set. Needs docs/supabase.sql applied and the SECRET key
(RLS only lets the backend write). Test rows use a `test-` run_id and are deleted even when a check fails.
"""
import os
import uuid

import pytest

from results_store import ResultsStore, candidate_row

URL, KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
pytestmark = pytest.mark.skipif(not (URL and KEY), reason="SUPABASE_URL / SUPABASE_KEY not set")
TABLES = ("events", "experience", "runs", "candidates")


@pytest.fixture(scope="module")
def client():
    from supabase import create_client

    return create_client(URL, KEY)


@pytest.fixture
def run_id(client):
    rid = f"test-{uuid.uuid4().hex[:12]}"
    yield rid
    client.table("runs").delete().eq("run_id", rid).execute()  # cascades to candidates


def _rows(client, table, rid):
    return client.table(table).select("*").eq("run_id", rid).execute().data


@pytest.mark.parametrize("table", TABLES)
def test_ping_table(client, table):
    res = client.table(table).select("*").limit(1).execute()
    assert isinstance(res.data, list), f"{table}: unexpected response {res!r}"


def test_crud_run_and_candidates(client, run_id):
    store = ResultsStore(client, run_id)

    # create
    store.start_run(dataset_name="pytest.csv", n_rows=10, n_features=2, target="y", task="classification",
                    split={"dev": 7, "search_val": 1, "hidden": 2}, profile={"meta": {"n": 10}})
    result = {"name": "lr_1", "ok": True, "fit_seconds": 0.1,
              "metrics": {"roc_auc": {"mean": 0.8, "std": 0.05, "train_mean": 0.9, "folds": [0.75, 0.85]}}}
    store.add_candidates([
        candidate_row(run_id, "race", {"name": "lr_1", "model": "logreg", "params": {"C": 1.0}}, result, "roc_auc",
                      rung=1, train_rows=7),
        candidate_row(run_id, "confirm", {"name": "lr_1", "model": "logreg"}, result, "roc_auc", train_rows=7,
                      p_vs_best=float("nan"), tie_with_best=True, hidden_score=0.78),
    ])

    # read
    run = _rows(client, "runs", run_id)
    assert len(run) == 1 and run[0]["status"] == "running" and run[0]["split"]["hidden"] == 2
    cands = sorted(_rows(client, "candidates", run_id), key=lambda r: r["stage"])
    assert [c["stage"] for c in cands] == ["confirm", "race"]
    assert cands[1]["params"] == {"C": 1.0} and cands[1]["metrics"]["roc_auc"]["folds"] == [0.75, 0.85]
    assert cands[0]["p_vs_best"] is None and cands[0]["tie_with_best"] is True and cands[0]["hidden_score"] == 0.78

    # update
    store.finish_run(status="completed", primary_metric="roc_auc", recommended="lr_1", report_md="# ok",
                     usage={"calls": 3})
    run = _rows(client, "runs", run_id)[0]
    assert run["status"] == "completed" and run["recommended"] == "lr_1" and run["finished_at"] is not None
    assert run["usage"] == {"calls": 3}

    # delete (cascade)
    client.table("runs").delete().eq("run_id", run_id).execute()
    assert _rows(client, "runs", run_id) == [] and _rows(client, "candidates", run_id) == []


def test_constraints_reject_bad_rows(client, run_id):
    from postgrest.exceptions import APIError

    with pytest.raises(APIError):  # status outside the check constraint
        client.table("runs").insert({"run_id": run_id, "status": "bogus"}).execute()
    with pytest.raises(APIError):  # candidate for a run that does not exist (foreign key)
        client.table("candidates").insert({"run_id": run_id, "stage": "race", "name": "x", "ok": True}).execute()
