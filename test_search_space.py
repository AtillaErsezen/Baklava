"""Checks for search_space.py. Run: uv run python test_search_space.py (or pytest)."""
import json

from search_space import (COMPLEXITY, FAMILIES, PORTFOLIO_2025, build_space, family_prior, rank_space,
                          sobol_configs)

SMALL = {"n_rows": 2000, "n_features": 20, "n_over_p": 100}
LARGE = {"n_rows": 500_000, "n_features": 60, "n_over_p": 8333, "nonlinearity_gap": 0.08}
GBDT = {"lightgbm", "xgboost", "catboost"}


def _key(c):
    return json.dumps([c["model"], c["params"], c["preprocessing"]], sort_keys=True)


def test_build_space_thousands_unique_and_valid():
    for task in ("classification", "regression"):
        space = build_space(task, 2000, 20, SMALL, {}, budget=3000)
        assert len(space) >= 2000, len(space)
        assert len({_key(c) for c in space}) == len(space)
        assert len({c["name"] for c in space}) == len(space)
        for c in space:
            assert c["model"] in FAMILIES[task] and c["family"] == c["model"]
            assert isinstance(c["prior"], float) and c["prior"] > 0
            json.dumps(c)
        assert [c["prior"] for c in space] == sorted((c["prior"] for c in space), reverse=True)


def test_interpretable_drops_black_boxes():
    space = build_space("classification", 2000, 20, SMALL, {"interpretable": True}, budget=500)
    assert space and {c["model"] for c in space} <= {"logreg", "random_forest"}
    assert all(c["params"].get("max_depth") is not None and c["params"]["max_depth"] <= 6
               for c in space if c["model"] == "random_forest")


def test_small_n_favours_tabicl():
    w = family_prior("classification", SMALL)
    assert max(w, key=w.get) == "tabicl"
    assert abs(sum(w.values()) - 1) < 1e-9 and min(w.values()) >= 0.05 - 1e-12


def test_large_n_favours_gbdt():
    for task in ("classification", "regression"):
        w = family_prior(task, LARGE)
        assert max(w, key=w.get) in GBDT


def test_sobol_deterministic():
    a, b = sobol_configs("lightgbm", "classification", 64, 7), sobol_configs("lightgbm", "classification", 64, 7)
    assert a == b and len(a) == 64
    assert a != sobol_configs("lightgbm", "classification", 64, 8)
    assert all(0.01 <= c["params"]["learning_rate"] <= 0.3 for c in a)


def test_portfolio_present():
    assert {c["model"] for c in PORTFOLIO_2025} == {"lightgbm", "xgboost", "catboost", "tabicl"}
    names = {c["name"] for c in build_space("regression", 5000, 30, {}, {}, budget=3000)}
    assert {c["name"] for c in PORTFOLIO_2025} <= names


def test_purpose_filters_and_class_weight():
    space = build_space("classification", 50_000, 20, {}, {"max_predict_ms": 5, "class_weighting": True}, budget=800)
    assert not {c["model"] for c in space} & {"mlp", "tabicl"}
    assert all(c["params"].get("class_weight") == "balanced" for c in space if c["model"] == "logreg")


def test_rank_space_sorts_and_complexity_covers_families():
    assert [c["prior"] for c in rank_space([{"prior": 0.1}, {"prior": 0.3}])] == [0.3, 0.1]
    assert set(COMPLEXITY) == FAMILIES["classification"] | FAMILIES["regression"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
