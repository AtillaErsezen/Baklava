"""Tests for report.py: deterministic report, claim check, prose call with fallback. No network."""
import json
from types import SimpleNamespace

from report import REMOVED, REPORT_PROSE_SCHEMA, build_report, claim_check, write_prose

DASHES = (chr(0x2014), chr(0x2013))


def live_state() -> dict:
    """Harness state shaped like the live churn run in runs/ (docs/EVENTS.md contract)."""
    lgbm = {"subsample": 0.775784726514, "subsample_freq": 1, "extra_trees": True, "colsample_bytree": 0.7732354787904,
            "learning_rate": 0.0090151778542, "min_child_samples": 4, "num_leaves": 2, "n_estimators": 3000}
    xgb = {"learning_rate": 0.0068171645251, "max_depth": 6, "max_leaves": 10, "n_estimators": 3000}
    lg2 = {"learning_rate": 0.01632, "num_leaves": 9, "min_child_samples": 30, "n_estimators": 1145}
    return {
        "dataset": {"name": "churn.csv", "rows": 3000, "features": 9, "target": "Churn", "task": "classification",
                    "purpose": "Catch customers about to churn early"},
        "split": {"dev": 2099, "search_val": 301, "hidden_locked": 600, "method": "random, stratified"},
        "findings": [
            {"check": "leakage", "severity": 3, "finding": "Probable leakage: refund_issued predict the target alone; drop them."},
            {"check": "id_like", "severity": 2, "finding": "Drop id-like columns ['customer_id']."},
            {"check": "imbalance", "severity": 1, "finding": "Imbalanced 3.9:1; use class_weight and roc_auc or f1_macro."}],
        "dropped_columns": ["refund_issued", "customer_id"],
        "search": {"space_size": 3000, "raced": 243, "n_star": 2099,
                   "justification": "n* = 2099: the learning curve flattens past 1800 rows.",
                   "schedule": [{"rung": 0, "n_rows": 500, "keep": 81}, {"rung": 1, "n_rows": 1500, "keep": 27},
                                {"rung": 2, "n_rows": 2099, "keep": 3}],
                   "rungs": [{"rung": 0, "survivors": 81}, {"rung": 1, "survivors": 27}, {"rung": 2, "survivors": 6}],
                   "fits": 351, "stopped": "done"},
        "confirm": {"primary_metric": "f1_macro", "table": [
            {"name": "lgbm_p02", "family": "lightgbm", "cv_mean": 0.6772, "ci": [0.6451, 0.7093], "p_vs_best": None,
             "tie_group": 0, "hidden": 0.6567, "ece": 0.025048, "dev_to_hidden_gap": -0.0205, "fit_s": 9.9,
             "predict_ms": 4.564, "big_o": {"train": "O(T n p) histogram splits", "predict": "O(T log L)"},
             "params": lgbm, "pareto": True},
            {"name": "xgb_p00", "family": "xgboost", "cv_mean": 0.6701, "ci": [0.6427, 0.6974], "p_vs_best": 0.5618,
             "tie_group": 0, "hidden": 0.6657, "ece": 0.029892, "dev_to_hidden_gap": -0.0044, "fit_s": 46.6,
             "predict_ms": 7.046, "big_o": {"train": "O(T n p) histogram splits", "predict": "O(T depth)"},
             "params": xgb, "pareto": False},
            {"name": "lgbm_s0283", "family": "lightgbm", "cv_mean": 0.6677, "ci": [0.6313, 0.7041], "p_vs_best": 0.33358,
             "tie_group": 0, "hidden": 0.6783, "ece": 0.150453, "dev_to_hidden_gap": 0.0106, "fit_s": 12.7,
             "predict_ms": 9.361, "big_o": {"train": "O(T n p) histogram splits", "predict": "O(T log L)"},
             "params": lg2, "pareto": False}],
            "recommendation": {"pick": "lgbm_p02", "best": "lgbm_p02", "within_1se": ["lgbm_p02", "xgb_p00"]},
            "tie_groups": [["lgbm_p02", "xgb_p00", "lgbm_s0283"]]},
        "final": {"name": "lgbm_p02", "top_features": [{"feature": "num__tenure_months", "importance": 735.0},
                                                       {"feature": "cat__contract", "importance": 498.0}]},
        "export": {"script": "runs/r1_export/train_lgbm_p02.py", "params": "runs/r1_export/params.json",
                   "card": "runs/r1_export/MODEL_CARD.md"},
        "usage": {"input_tokens": 30199, "output_tokens": 1531, "usd": 0.006958},
    }


# ---- claim_check
def test_claim_check_removes_wrong_feature_count():
    text, removed = claim_check("The data has 10 features.", live_state())
    assert removed == ["10"] and "10" not in text and REMOVED in text


def test_claim_check_keeps_matching_decimal():
    text, removed = claim_check("CV f1_macro is 0.6772.", {"confirm": {"cv_mean": 0.67720}})
    assert removed == [] and "0.6772" in text


def test_claim_check_keeps_percent_of_fraction():
    text, removed = claim_check("Accuracy reached 96.8% on 3,000 rows.", {"acc": 0.968, "rows": 3000})
    assert removed == [] and "96.8%" in text and "3,000" in text


def test_claim_check_strict_small_counts_whitelists_1_and_2():
    _, removed = claim_check("The top 3 beat 1 or 2 others and 7 baselines.", {"k": 3})
    assert removed == ["7"]


def test_claim_check_ignores_digits_inside_names():
    _, removed = claim_check("lgbm_p02 wins on f1_macro.", {})
    assert removed == []


# ---- build_report
SECTIONS = ["## Dataset card", "## Issues found and fixes", "## Search funnel", "## Top models",
            "## Recommendation", "## Sample size", "## External data", "## Caveats", "## How to retrain", "## Cost"]
PROSE = {"why": "The purpose asks to catch churners early; lgbm_p02 has the top f1_macro of 0.6772 on 9 features.",
         "caveats": ["The data has 10 features.", "Hidden rows number 600."],
         "spoken_summary": "The churn model scores 0.6567 f1 on the hidden holdout."}


def test_report_renders_all_sections_in_order():
    md = build_report(live_state(), PROSE)["markdown"]
    pos = [md.index(s) for s in SECTIONS]
    assert pos == sorted(pos)


def test_report_numbers_come_from_state():
    md = build_report(live_state(), PROSE)["markdown"]
    for s in ("Rows: 3000", "Features: 9", "3000 pipelines -> 243 raced -> 81 -> 27 -> 6 -> top 3",
              "0.6772 [0.6451, 0.7093]", "0.5618", "0.6657", "46.6", "7.046", "n* = 2099",
              "train_lgbm_p02.py data.csv --target Churn", "30199", "1531", "0.006958"):
        assert s in md, s
    assert "Features: 10" not in md


def test_report_tie_note_and_auto_caveats():
    md = build_report(live_state(), PROSE)["markdown"]
    assert "statistically tied with xgb_p00, lgbm_s0283; chosen by the 1-SE rule" in md
    assert "lgbm_s0283" in md.split("## Caveats")[1] and "0.1505" in md  # ece above 0.1
    assert "leakage" not in md.split("## Caveats")[1].split("## How")[0]  # refund_issued was dropped


def test_report_flags_unhandled_severity3_and_big_gap():
    st = live_state()
    st["dropped_columns"] = []
    st["confirm"]["table"][1]["dev_to_hidden_gap"] = -0.09
    cav = build_report(st, PROSE)["markdown"].split("## Caveats")[1].split("## How")[0]
    assert "leakage" in cav and "xgb_p00" in cav


def test_report_claim_checks_prose():
    out = build_report(live_state(), PROSE)
    md = out["markdown"]
    assert "0.6772 on 9 features" in md and "10 features" not in md and REMOVED in md
    assert "Claim check" in md and "0.6567" in out["spoken_summary"]


def test_report_escapes_cells_and_has_no_dashes():
    st = live_state()
    st["confirm"]["table"][0]["name"] = "bad|name\nx"
    st["findings"][0]["finding"] = "pipe | and\nnewline " + chr(0x2014) + " dash"
    prose = {**PROSE, "why": "Fast " + chr(0x2013) + " and simple."}
    out = build_report(st, prose)
    assert "bad\\|name x" in out["markdown"]
    assert not any(d in out["markdown"] + out["spoken_summary"] for d in DASHES)


def test_report_renders_partial_state():
    md = build_report({"dataset": {"name": "d.csv", "rows": 10}}, {"why": "", "caveats": [], "spoken_summary": ""})["markdown"]
    assert "Rows: 10" in md and "## Top models" not in md and "Not tested." in md


# ---- write_prose
class FakeLedger:
    def __init__(self):
        self.usages = []

    def add(self, usage):
        self.usages.append(usage)


def fake_client(content: str | None = None, error: Exception | None = None):
    """Mimics NebiusClient: .model, .ledger, .client.chat.completions.create(**kw)."""
    calls = []

    def create(**kw):
        calls.append(kw)
        if error:
            raise error
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=5))

    comp = SimpleNamespace(create=create)
    return SimpleNamespace(model="m", ledger=FakeLedger(), client=SimpleNamespace(chat=SimpleNamespace(completions=comp))), calls


def test_schema_is_strict():
    assert set(REPORT_PROSE_SCHEMA["required"]) == {"why", "caveats", "spoken_summary"}
    assert REPORT_PROSE_SCHEMA["additionalProperties"] is False
    assert REPORT_PROSE_SCHEMA["properties"]["caveats"]["maxItems"] == 5


def test_write_prose_one_constrained_call_without_tools():
    good = {"why": "Best f1_macro.", "caveats": ["Small data."], "spoken_summary": "Done."}
    client, calls = fake_client(json.dumps(good))
    assert write_prose(client, live_state()) == good
    kw = calls[0]
    assert "tools" not in kw and kw["temperature"] == 0 and kw["max_tokens"] == 700
    assert kw["response_format"]["json_schema"]["schema"] is REPORT_PROSE_SCHEMA
    assert kw["response_format"]["json_schema"]["strict"] is True
    assert len(client.ledger.usages) == 1


def test_write_prose_falls_back_on_error_bad_json_and_scripted():
    st = live_state()
    raising, _ = fake_client(error=RuntimeError("503"))
    broken, _ = fake_client("not json")
    scripted = SimpleNamespace(ledger=FakeLedger())  # ScriptedClient has no .client
    for c in (raising, broken, scripted):
        p = write_prose(c, st)
        assert set(p) == {"why", "caveats", "spoken_summary"} and "lgbm_p02" in p["why"]


def test_fallback_prose_passes_claim_check():
    p = write_prose(SimpleNamespace(), live_state())
    for text in [p["why"], p["spoken_summary"], *p["caveats"]]:
        assert claim_check(text, live_state())[1] == [], text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
