"""Tests for stats_tests. Run: uv run python test_stats_tests.py (or pytest)."""
import math

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

import stats_tests as st


def test_nadeau_bengio_hand():
    a = [0.80, 0.82, 0.81, 0.79, 0.83]
    b = [0.78, 0.80, 0.80, 0.78, 0.80]
    d = [x - y for x, y in zip(a, b)]
    j = len(d)
    m = sum(d) / j
    s2 = sum((x - m) ** 2 for x in d) / (j - 1)
    t = m / math.sqrt((1 / j + 200 / 800) * s2)
    p = 2 * stats.t.sf(abs(t), j - 1)
    r = st.nadeau_bengio(a, b, n_train=800, n_test=200)
    assert abs(r["t"] - t) < 1e-5, (r, t)
    assert abs(r["p"] - p) < 1e-5
    assert abs(r["p_one_sided"] - stats.t.sf(t, j - 1)) < 1e-5
    assert r["ci_low"] < m < r["ci_high"]
    assert r["df"] == 4


def test_nadeau_bengio_identical():
    a = [0.8, 0.81, 0.79]
    assert st.nadeau_bengio(a, a, 800, 200)["p"] == 1.0


def test_holm_matches_statsmodels():
    pv = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.2}
    rej, adj, _, _ = multipletests(list(pv.values()), alpha=0.05, method="holm")
    r = st.holm(pv)
    for (k, _), rj, pa in zip(pv.items(), rej, adj):
        assert abs(r[k]["p_adj"] - pa) < 1e-6 and r[k]["reject"] == bool(rj), (k, r[k], pa)


def _mc_case(n, flip_a, flip_b, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    pa, pb = y.copy(), y.copy()
    pa[rng.random(n) < flip_a] ^= 1
    pb[rng.random(n) < flip_b] ^= 1
    return y, pa, pb


def test_mcnemar_matches_statsmodels():
    for n, fa, fb, exact in [(60, 0.05, 0.2, True), (2000, 0.1, 0.2, False)]:
        y, pa, pb = _mc_case(n, fa, fb, 0)
        r = st.mcnemar(y, pa, pb)
        ca, cb = pa == y, pb == y
        tab = [[np.sum(ca & cb), np.sum(ca & ~cb)], [np.sum(~ca & cb), np.sum(~ca & ~cb)]]
        assert (r["b"] + r["c"] < 25) == exact, r
        ref = sm_mcnemar(tab, exact=exact, correction=True)
        assert abs(r["p"] - ref.pvalue) < 1e-5, (r, ref.pvalue)


def test_delong():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    sa = y + rng.normal(0, 1.0, 500)
    sb = np.round(y + rng.normal(0, 1.5, 500), 1)  # ties exercise midranks
    r = st.delong(y, sa, sb)
    assert abs(r["auc_a"] - roc_auc_score(y, sa)) < 1e-6
    assert abs(r["auc_b"] - roc_auc_score(y, sb)) < 1e-6
    assert st.delong(y, sa, sa)["p"] == 1.0
    good = y + rng.normal(0, 0.3, 500)
    bad = rng.normal(0, 1, 500)
    assert st.delong(y, good, bad)["p"] < 0.01


def test_paired_bootstrap():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    acc = lambda yt, yp: float(np.mean(yt == yp))  # noqa: E731
    p = np.where(rng.random(300) < 0.7, y, 1 - y)
    r = st.paired_bootstrap(acc, y, p, p)
    assert r["ci_low"] <= 0 <= r["ci_high"] and r["p"] > 0.9, r
    good = np.where(rng.random(300) < 0.95, y, 1 - y)
    assert st.paired_bootstrap(acc, y, good, p)["p"] < 0.05


def test_calibration():
    rng = np.random.default_rng(3)
    prob = rng.random(20000)
    y = (rng.random(20000) < prob).astype(int)
    r = st.calibration(y, prob)
    assert r["ece"] < 0.03, r["ece"]
    assert abs(r["brier"] - brier_score_loss(y, prob)) < 1e-6
    assert sum(b["n"] for b in r["bins"]) == 20000


def test_kendall_tau():
    assert st.kendall_tau(["a", "b", "c", "x"], ["a", "b", "c", "y"]) == 1.0
    assert st.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_mann_kendall():
    r = st.mann_kendall(np.arange(20))
    assert r["trend"] == "increasing" and r["p"] < 0.01, r
    assert st.mann_kendall(np.random.default_rng(7).normal(size=30))["trend"] == "none"
    assert st.mann_kendall(-np.arange(20))["trend"] == "decreasing"


RESULTS = [
    {"name": "gbm", "family": "gbm", "mean": 0.850, "std": 0.03, "n_folds": 5},
    {"name": "logreg", "family": "linear", "mean": 0.840, "std": 0.02, "n_folds": 5},
    {"name": "rf", "family": "forest", "mean": 0.845, "std": 0.02, "n_folds": 5},
    {"name": "knn", "family": "knn", "mean": 0.700, "std": 0.02, "n_folds": 5},
]


def test_one_se_pick():
    r = st.one_se_pick(RESULTS, complexity_order=["linear", "forest", "gbm"])
    assert r["pick"] == "logreg" and r["best"] == "gbm", r
    assert set(r["within_1se"]) == {"gbm", "logreg", "rf"}


def test_tie_groups():
    pv = {"logreg": 0.4, "rf": 0.6, "knn": 0.0001}
    g = st.tie_groups(RESULTS, pv)
    assert g[0] == ["gbm", "rf", "logreg"] and g[1] == ["knn"], g


def test_bayes_correlated():
    rng = np.random.default_rng(4)
    a = 0.8 + rng.normal(0, 0.01, 10)
    b = a - 0.02 + rng.normal(0, 0.005, 10)
    r = st.bayes_correlated(a, b, rope=0.01, runs=2)
    assert abs(r["p_a_better"] + r["p_equiv"] + r["p_b_better"] - 1) < 1e-3, r
    assert r["p_a_better"] > 0.5



def test_test_train_ratio_matches_the_cv_scheme():
    assert abs(st.test_train_ratio("kfold", 5) - 0.25) < 1e-12          # 1 test fold vs 4 train folds
    assert abs(st.test_train_ratio("purged", 5) - 0.25) < 1e-12
    walk = sum(1 / i for i in range(1, 6)) / 5                           # fold i trains on i blocks, tests on 1
    assert abs(st.test_train_ratio("walk_forward", 5) - walk) < 1e-12
    assert st.test_train_ratio("walk_forward", 5) > st.test_train_ratio("kfold", 5)  # more conservative


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
