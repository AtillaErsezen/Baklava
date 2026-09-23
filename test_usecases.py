"""Tests for usecases.py: catalog shape, goal matching, target suggestion, leakage suspects. No network."""
import csv
import re
from pathlib import Path

from purpose import TASK_METRICS
from search_space import FAMILIES
from usecases import TASK_KIND, TIMES, USE_CASES, leakage_suspects, match, suggest_target

ROOT = Path(__file__).parent
CHURN_COLS = next(csv.reader(open(ROOT / "data" / "churn.csv")))
HOUSE_COLS = next(csv.reader(open(ROOT / "data" / "houses.csv")))
FIELDS = {"id", "name", "segments", "questions", "task", "target_hint", "unit", "time", "primary_metric",
          "leakage_patterns", "families_prior", "needed_columns"}
DASHES = (chr(0x2014), chr(0x2013))


def uc(uid: str) -> dict:
    return next(u for u in USE_CASES if u["id"] == uid)


def top1(goal: str, columns: list[str] | None = None, task_hint: str | None = None) -> str:
    return match(goal, columns or [], task_hint)[0]["id"]


# ---- catalog
def test_catalog_size_and_unique_ids():
    ids = [u["id"] for u in USE_CASES]
    assert 40 <= len(ids) <= 60 and len(ids) == len(set(ids))


def test_every_entry_has_valid_metric_for_its_task():
    for u in USE_CASES:
        assert FIELDS <= set(u), u["id"]
        assert u["primary_metric"] in TASK_METRICS[TASK_KIND[u["task"]]], u["id"]
        assert u["time"] in TIMES, u["id"]


def test_entries_are_well_formed():
    for u in USE_CASES:
        assert 4 <= len(u["questions"]) <= 8 and u["segments"] and u["target_hint"] and u["needed_columns"], u["id"]
        for p in u["target_hint"] + u["leakage_patterns"] + u["needed_columns"]:
            re.compile(p)
        fams = FAMILIES[TASK_KIND[u["task"]]]
        assert set(u["families_prior"]) <= fams and all(0 < w <= 3 for w in u["families_prior"].values()), u["id"]


def test_no_long_dashes_in_catalog_or_doc():
    text = repr(USE_CASES) + (ROOT / "docs" / "USE_CASES.md").read_text() + (ROOT / "usecases.py").read_text()
    assert not any(d in text for d in DASHES)


# ---- match
def test_example_goals_map_to_expected_use_case():
    assert top1("which customers churn in the next 90 days") == "customer_churn"
    assert top1("predict next month's price per product") == "price_forecast"
    assert top1("forecast weekly demand per store") == "demand_forecast"
    assert top1("flag fraud") == "transaction_fraud"


def test_paraphrases_outside_the_catalog_still_match():
    assert top1("who is going to cancel their subscription soon") == "customer_churn"
    assert top1("how many units will each shop sell next week") == "demand_forecast"
    assert top1("spot fraudulent card payments") == "transaction_fraud"
    assert top1("what will our product prices be in the coming months") == "price_forecast"
    assert top1("which employees might resign") == "employee_attrition"


def test_example_goals_hold_with_dataset_columns():
    assert top1("which customers churn in the next 90 days", CHURN_COLS, "classification") == "customer_churn"
    assert top1("estimate the sale price of each house", HOUSE_COLS, "regression") == "house_price"


def test_nonsense_goal_scores_low():
    for cols in ([], ["a", "b", "c"]):
        ms = match("purple elephants juggle quietly under the moon", cols, top_k=5)
        assert ms and all(m["score"] < 0.2 for m in ms)


def test_scores_bounded_sorted_deterministic_with_reasons():
    a = match("which customers churn in the next 90 days", CHURN_COLS, top_k=5)
    b = match("which customers churn in the next 90 days", CHURN_COLS, top_k=5)
    assert [m["id"] for m in a] == [m["id"] for m in b] and [m["score"] for m in a] == [m["score"] for m in b]
    assert len(a) == 5 and all(0 <= m["score"] <= 1 and m["reasons"] for m in a)
    assert [m["score"] for m in a] == sorted((m["score"] for m in a), reverse=True)


def test_columns_raise_score_and_are_cited():
    bare = match("which customers churn in the next 90 days", [])[0]
    with_cols = match("which customers churn in the next 90 days", CHURN_COLS)[0]
    assert with_cols["score"] > bare["score"] and any("Churn" in r for r in with_cols["reasons"])


def test_task_hint_mismatch_lowers_score():
    good = match("which customers churn", [], "classification")[0]["score"]
    bad = next(m for m in match("which customers churn", [], "regression", top_k=60) if m["id"] == "customer_churn")
    assert bad["score"] < good and any("task" in r for r in bad["reasons"])


def test_bad_task_hint_raises():
    try:
        match("churn", [], "clustering")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_empty_goal_falls_back_to_columns():
    assert match("", CHURN_COLS)[0]["id"] in {"customer_churn", "saas_renewal_churn"}


# ---- suggest_target / leakage_suspects
def test_suggest_target_finds_churn_in_demo():
    assert suggest_target(uc("customer_churn"), CHURN_COLS) == "Churn"
    assert suggest_target(uc("house_price"), HOUSE_COLS) == "SalePrice"


def test_suggest_target_skips_leaks_and_lag_features():
    assert suggest_target(uc("customer_churn"), ["churn_reason", "tenure"]) is None
    cols = ["date", "product_id", "price_lag_1", "price_rolling_mean_4", "price"]
    assert suggest_target(uc("price_forecast"), cols) == "price"


def test_leakage_suspects_flags_refund_for_churn():
    s = leakage_suspects(uc("customer_churn"), CHURN_COLS)
    assert "refund_issued" in s and "Churn" not in s and "tenure_months" not in s


def test_leakage_suspects_reads_camel_case():
    assert leakage_suspects(uc("booking_cancellation"), ["LeadTime", "ReservationStatusDate"]) == ["ReservationStatusDate"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
