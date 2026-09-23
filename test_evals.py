"""Offline tests for the eval harness: scoring on synthetic events, catalog validity, idempotent fetch, dry run.
Run: uv run python -m pytest -q test_evals.py"""
import os
from urllib.parse import urlsplit

import pandas as pd
import pytest

import agent
import diagnostics as dg
import external_data as xd
from evals import fetch_public, make_golden, run_golden
from evals.public_catalog import PUBLIC, PUBLIC_DIR
from factory_tools import TASK_METRICS
from report import REMOVED

CV_ENUM = next(t for t in agent.TOOLS if t["name"] == "run_search")["input_schema"]["properties"]["cv"]["enum"]
EXPECT = {"finding": "leakage", "must_drop": ["refund"], "cv": ("walk_forward", "purged")}


def ev(kind: str, payload: dict) -> dict:
    return {"run_id": "t", "ts": 0.0, "kind": kind, "payload": payload}


def events(findings=("leakage",), search=None, table=None, final="m1", pm="roc_auc", markdown="AUC 0.8.", usage=True):
    """A complete run's events; each argument bends one part of it."""
    table = table if table is not None else [{"name": "m1", "tie_group": 0, "hidden": 0.80},
                                             {"name": "m2", "tie_group": 1, "hidden": 0.78}]
    out = [ev("run_start", {"dataset": "d.csv"}),
           ev("diagnostics", {"findings": [{"check": c, "severity": 2, "finding": "x"} for c in findings]}),
           ev("tool_call", {"tool": "run_search", "input": search if search is not None else
                            {"cv": "walk_forward", "drop_columns": ["refund", "id"]}}),
           ev("confirm", {"primary_metric": pm, "table": table, "recommendation": {"pick": "m1"}})]
    if final:
        out.append(ev("final_model", {"name": final}))
    if markdown is not None:
        out.append(ev("report", {"markdown": markdown, "spoken_summary": "Done."}))
    if usage:
        out.append(ev("usage", {"calls": 3, "input_tokens": 100, "output_tokens": 20, "usd": 0.01, "modal_usd": 0.02}))
    return out


# ---- scoring

def test_score_all_checks_pass_and_reads_usage():
    s = run_golden.score(events(), EXPECT)
    assert s["checks"] == {c: True for c in run_golden.CHECKS} and s["passed"]
    assert s["tokens"] == 120 and s["usd"] == 0.01 and s["modal_usd"] == 0.02


def test_finding_fails_when_expected_check_missing_or_no_diagnostics():
    assert run_golden.score(events(findings=("imbalance",)), EXPECT)["checks"]["finding"] is False
    no_diag = [e for e in events() if e["kind"] != "diagnostics"]
    assert run_golden.score(no_diag, EXPECT)["checks"]["finding"] is False


def test_finding_none_is_not_applicable_and_does_not_fail():
    s = run_golden.score(events(findings=()), {**EXPECT, "finding": None})
    assert s["checks"]["finding"] is None and s["passed"]


def test_drops_fail_when_a_must_drop_column_is_kept_or_search_never_ran():
    kept = events(search={"cv": "walk_forward", "drop_columns": ["id"]})
    assert run_golden.score(kept, EXPECT)["checks"]["drops"] is False
    no_search = [e for e in events() if e["kind"] != "tool_call"]
    assert run_golden.score(no_search, EXPECT)["checks"]["drops"] is False
    assert run_golden.score(kept, {**EXPECT, "must_drop": []})["checks"]["drops"] is None


def test_drops_and_cv_use_the_last_valid_run_search_call():
    evs = events()
    evs.insert(3, ev("tool_call", {"tool": "run_search", "input": {"cv": "kfold", "drop_columns": []}}))
    evs.insert(4, ev("tool_call", {"tool": "run_search", "input": "{not json"}))  # unparsed args are ignored
    assert run_golden.score(evs, EXPECT)["checks"]["cv"] is False
    evs.insert(5, ev("tool_call", {"tool": "run_search", "input": {"cv": "purged", "drop_columns": ["refund"]}}))
    assert run_golden.score(evs, EXPECT)["checks"] == {c: True for c in run_golden.CHECKS}


def test_cv_fails_on_wrong_scheme_and_defaults_to_kfold():
    assert run_golden.score(events(search={"cv": "kfold", "drop_columns": ["refund"]}), EXPECT)["checks"]["cv"] is False
    no_cv = events(search={"drop_columns": ["refund"]})
    assert run_golden.score(no_cv, {**EXPECT, "cv": "kfold"})["checks"]["cv"] is True
    assert run_golden.score(no_cv, EXPECT)["checks"]["cv"] is False


def test_pick_passes_on_best_hidden_outside_tie_group_zero():
    table = [{"name": "m1", "tie_group": 0, "hidden": 0.80}, {"name": "m2", "tie_group": 1, "hidden": 0.85}]
    assert run_golden.score(events(table=table, final="m2"), EXPECT)["checks"]["pick"] is True
    assert run_golden.score(events(final="m2"), EXPECT)["checks"]["pick"] is False  # tie 1, worse hidden


def test_pick_uses_lower_is_better_for_error_metrics():
    table = [{"name": "m1", "tie_group": 0, "hidden": 12.0}, {"name": "m2", "tie_group": 1, "hidden": 10.0},
             {"name": "m3", "tie_group": 2, "hidden": 11.0}]
    assert run_golden.score(events(table=table, final="m2", pm="rmse"), EXPECT)["checks"]["pick"] is True
    assert run_golden.score(events(table=table, final="m3", pm="rmse"), EXPECT)["checks"]["pick"] is False


def test_pick_fails_without_confirm_or_for_a_model_outside_the_table():
    assert run_golden.score(events(final="stray"), EXPECT)["checks"]["pick"] is False
    assert run_golden.score([e for e in events() if e["kind"] != "confirm"], EXPECT)["checks"]["pick"] is False
    assert run_golden.score(events(final=None), EXPECT)["checks"]["pick"] is True  # falls back to recommendation


def test_claims_fail_on_removed_number_marker_or_missing_report():
    assert run_golden.score(events(markdown=f"AUC {REMOVED} on hidden."), EXPECT)["checks"]["claims"] is False
    assert run_golden.score(events(markdown=None), EXPECT)["checks"]["claims"] is False


def test_empty_run_fails_and_costs_nothing():
    s = run_golden.score([], EXPECT)
    assert not s["passed"] and s["tokens"] == 0 and s["usd"] == 0.0


# ---- catalogs

def _check_expect(name: str, task: str, expect: dict) -> None:
    assert task in TASK_METRICS, name
    assert all(cv in CV_ENUM for cv in run_golden._allowed(expect["cv"])), name
    assert expect["finding"] is None or expect["finding"] in dg.DIAGNOSTICS, name
    assert isinstance(expect["must_drop"], list) and all(isinstance(c, str) for c in expect["must_drop"]), name


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_public_entry_is_valid(name):
    e = PUBLIC[name]
    _check_expect(name, e["task"], e["expect"])
    assert e["metric"] in TASK_METRICS[e["task"]]
    url = urlsplit(e["url"])
    assert url.scheme == "https" and not url.query
    assert xd._on_domain(url.hostname, tuple(xd.SEARCH_DOMAINS)) and not xd._on_domain(url.hostname, xd.AUTH_ONLY_DOMAINS)
    assert url.path.lower().endswith(xd.DATA_EXTS)
    text = " ".join(str(e[k]) for k in ("purpose", "source", "license"))
    assert len(e["purpose"]) > 30 and chr(0x2014) not in text and chr(0x2013) not in text
    path = os.path.join(PUBLIC_DIR, f"{name}.csv")
    if os.path.exists(path):  # fetched: target and every must_drop column really exist
        cols = set(pd.read_csv(path, nrows=0).columns)
        assert {e["target"], *e["expect"]["must_drop"]} <= cols


def test_golden_expectations_are_valid():
    for name, e in make_golden.EXPECT.items():
        _check_expect(name, e["task"], e)
    assert set(run_golden.datasets("all")) == set(make_golden.EXPECT) | set(PUBLIC)


# ---- fetch

def test_fetch_skips_existing_file(tmp_path):
    (tmp_path / "x.csv").write_text("a,y\n1,0\n", encoding="utf-8")

    def boom(url, **kw):
        raise AssertionError("must not download an existing file")

    path, status = fetch_public.fetch("x", {"url": "https://openml.org/x.csv", "target": "y"}, str(tmp_path), boom)
    assert status == "cached" and (tmp_path / "x.csv").read_text(encoding="utf-8") == "a,y\n1,0\n" and path == str(tmp_path / "x.csv")


def test_fetch_writes_new_file_and_rejects_missing_target(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
    path, status = fetch_public.fetch("n", {"url": "u", "target": "y"}, str(tmp_path), lambda url, **kw: df)
    assert status == "fetched" and pd.read_csv(path).equals(df)
    with pytest.raises(ValueError, match="target"):
        fetch_public.fetch("m", {"url": "u", "target": "nope"}, str(tmp_path), lambda url, **kw: df)
    assert not os.path.exists(tmp_path / "m.csv")


# ---- dry harness end to end

def test_dry_run_scores_repeats_and_writes_outputs(tmp_path):
    csv = tmp_path / "tiny.csv"
    make_golden.tiny_diabetes().to_csv(csv, index=False)
    entry = {"path": str(csv), "target": "outcome", "task": "classification", "purpose": None,
             "expect": {k: make_golden.EXPECT["tiny_diabetes"][k] for k in ("finding", "must_drop", "cv")}}
    real_from_name = agent.modal.Function.from_name
    results = run_golden.evaluate({"tiny": entry, "gone": {**entry, "path": str(tmp_path / "gone.csv")}}, 2, dry=True)
    assert agent.modal.Function.from_name == real_from_name  # fakes are unpatched afterwards
    tiny, gone = results
    assert tiny["k"] == 2 and tiny["pass_k"] and all(r["error"] is None for r in tiny["runs"])
    assert not gone["pass_k"] and "missing" in gone["runs"][0]["error"]
    md, js = run_golden.write(results, {"ts": "t", "sets": "test", "repeat": 2, "provider": "dry"}, str(tmp_path))
    with open(md, encoding="utf-8") as f:
        text = f.read()
    assert "| tiny | n/a | n/a | 2/2 | 2/2 | 2/2 | PASS |" in text and "pass^k: 1/2 datasets" in text
    assert os.path.exists(js)
