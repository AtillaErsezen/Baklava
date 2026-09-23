"""Tests for ensemble. Run: uv run python test_ensemble.py (or pytest)."""
import numpy as np

import ensemble as en


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _binary(n=2000, seed=0):
    """Latent signal, binary label, two models with independent errors, one strong model, one pure-noise model."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (z + rng.normal(scale=0.5, size=n) > 0).astype(int)
    preds = {"a": _sig(z + rng.normal(size=n)), "b": _sig(z + rng.normal(size=n)),
             "strong": _sig(3 * z), "noise": rng.uniform(size=n)}
    return y, preds


def _regression(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return y, {"a": y + rng.normal(size=n), "b": y + rng.normal(scale=1.2, size=n)}


def test_weights_sum_to_one_and_non_negative():
    y, p = _binary()
    r = en.caruana(p, y, "roc_auc", True)
    assert abs(sum(r["weights"].values()) - 1) < 1e-9, r
    assert all(w >= 0 for w in r["weights"].values())
    assert set(r["members"]) == {k for k, w in r["weights"].items() if w > 0}


def test_independent_errors_beat_each_single():
    y, p = _binary()
    two = {k: p[k] for k in ("a", "b")}
    r = en.caruana(two, y, "roc_auc", True)
    singles = {k: en.score("roc_auc", y, v) for k, v in two.items()}
    assert all(r["val_score"] > s for s in singles.values()), (r, singles)
    assert set(r["members"]) == {"a", "b"}


def test_noise_model_gets_zero_weight():
    y, p = _binary()
    r = en.caruana({"strong": p["strong"], "noise": p["noise"]}, y, "roc_auc", True)
    assert r["weights"]["noise"] == 0 and r["members"] == ["strong"], r


def test_lower_is_better_rmse():
    y, p = _regression()
    r = en.caruana(p, y, "rmse", False)
    assert r["val_score"] < min(en.score("rmse", y, v) for v in p.values()), r
    assert r["val_score"] == min(r["history"])
    assert abs(en.score("rmse", y, en.blend(p, r["weights"])) - r["val_score"]) < 1e-9


def test_compare_keep_false_when_ensemble_equals_best_single():
    for metric, hib, (y, p) in (("roc_auc", True, _binary(seed=5)), ("rmse", False, _regression(seed=6))):
        c = en.compare_on_hidden(p, y, {"a": 1.0, "b": 0.0}, "a", metric, hib)
        assert c["delta"] == 0 and c["p"] >= 0.05 and c["keep"] is False, (metric, c)


def test_compare_keep_true_on_real_gain():
    for metric, hib, (y, p) in (("roc_auc", True, _binary(seed=7)), ("rmse", False, _regression(seed=8))):
        c = en.compare_on_hidden(p, y, {"a": 0.5, "b": 0.5}, "a", metric, hib)
        assert c["keep"] is True and c["p"] < 0.05, (metric, c)


def test_deterministic_and_name_tie_break():
    y, p = _binary()
    assert en.caruana(p, y, "accuracy", True) == en.caruana(p, y, "accuracy", True)
    r = en.caruana({"b": p["a"], "a": p["a"].copy()}, y, "roc_auc", True)
    assert r["members"] == ["a"] and r["weights"] == {"a": 1.0, "b": 0.0}, r


def test_threshold_metrics_run():
    y, p = _binary()
    for m in ("accuracy", "f1_macro"):
        r = en.caruana(p, y, m, True)
        assert 0 <= r["val_score"] <= 1 and r["members"], (m, r)


def test_2d_input_raises():
    y, p = _binary()
    bad = {**p, "multi": np.column_stack([1 - p["a"], p["a"]])}
    for call in (lambda: en.caruana(bad, y, "roc_auc", True), lambda: en.blend(bad, {"multi": 1.0})):
        try:
            call()
        except ValueError as e:
            assert "multiclass" in str(e)
        else:
            raise AssertionError("2-D input did not raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
