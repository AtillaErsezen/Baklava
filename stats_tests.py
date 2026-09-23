"""Statistical model-comparison toolkit. Every function returns small dicts of floats rounded to 6 decimals."""
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

Z95 = stats.norm.ppf(0.975)


def _r(x: float) -> float:
    return round(float(x), 6)


def test_train_ratio(cv: str, k: int) -> float:
    """n_test / n_train for the Nadeau-Bengio variance term, per CV scheme. kfold and purged: 1 / (k - 1).
    walk_forward / timeseries (TimeSeriesSplit): fold i trains on i blocks and tests on 1, so the average ratio is
    mean(1 / i). The correction was derived for exchangeable resamples; for time-ordered folds it is an
    approximation, and this ratio makes it more conservative, not less."""
    if cv in ("walk_forward", "timeseries"):
        return sum(1 / i for i in range(1, k + 1)) / k
    return 1 / (k - 1)


def nadeau_bengio(a: Sequence[float], b: Sequence[float], n_train: int, n_test: int, alpha: float = 0.05) -> dict:
    """Corrected resampled t-test (Nadeau and Bengio 2003) on paired per-fold scores; positive diff means a better."""
    d = np.asarray(a, float) - np.asarray(b, float)
    j = len(d)
    m, s2 = d.mean(), d.var(ddof=1)
    if s2 == 0:
        if m == 0:
            return {"mean_diff": 0.0, "t": 0.0, "df": j - 1, "p": 1.0, "p_one_sided": 0.5, "ci_low": 0.0, "ci_high": 0.0,
                    "note": "zero variance, identical scores"}
        return {"mean_diff": _r(m), "t": None, "df": j - 1, "p": 0.0, "p_one_sided": 0.0 if m > 0 else 1.0,
                "ci_low": _r(m), "ci_high": _r(m), "note": "zero variance, constant nonzero difference"}
    se = np.sqrt((1 / j + n_test / n_train) * s2)
    t = m / se
    q = stats.t.ppf(1 - alpha / 2, j - 1)
    return {"mean_diff": _r(m), "t": _r(t), "df": j - 1, "p": _r(2 * stats.t.sf(abs(t), j - 1)),
            "p_one_sided": _r(stats.t.sf(t, j - 1)), "ci_low": _r(m - q * se), "ci_high": _r(m + q * se)}


def bayes_correlated(a: Sequence[float], b: Sequence[float], rope: float, runs: int = 1) -> dict:
    """Bayesian correlated t-test via baycomp. baycomp uses diff = b - a and returns (p_left, p_rope, p_right),
    so p_left = P(a better), p_right = P(b better)."""
    try:
        import baycomp

        res = baycomp.two_on_single(np.asarray(a, float), np.asarray(b, float), rope=rope, runs=runs)
        left, eq, right = res if len(res) == 3 else (res[0], 0.0, res[1])
        return {"p_a_better": _r(left), "p_equiv": _r(eq), "p_b_better": _r(right)}
    except Exception as e:  # zero variance, bad shapes, etc.
        return {"error": f"{type(e).__name__}: {e}"}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment: p_adj_(i) = max_{j<=i} min(1, (m - j) * p_(j))."""
    names = sorted(pvals, key=pvals.get)
    m, run, out = len(names), 0.0, {}
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = {"p_adj": _r(run), "reject": bool(run <= alpha)}
    return {k: out[k] for k in pvals}


def mcnemar(y, pred_a, pred_b) -> dict:
    """McNemar test on paired correctness. b = a right and b wrong, c = a wrong and b right.
    Exact binomial p when b + c < 25, else chi2 with continuity correction."""
    y, pa, pb = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = pa == y, pb == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0, "exact": True}
    chi2 = (abs(b - c) - 1) ** 2 / n
    exact = n < 25
    p = stats.binomtest(min(b, c), n, 0.5).pvalue if exact else stats.chi2.sf(chi2, 1)
    return {"b": b, "c": c, "chi2": _r(chi2), "p": _r(p), "exact": exact}


def delong(y, score_a, score_b) -> dict:
    """Paired DeLong AUC test, fast O(n log n) version (Sun and Xu 2014) via midranks. y must be binary 0/1."""
    y = np.asarray(y).astype(bool)
    s = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    pos, neg = s[:, y], s[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        return {"error": "need both classes"}
    tx = stats.rankdata(pos, axis=1)  # rankdata 'average' is the midrank
    ty = stats.rankdata(neg, axis=1)
    tz = stats.rankdata(np.hstack([pos, neg]), axis=1)
    auc = (tz[:, :m].sum(1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = auc[0] - auc[1]
    se = float(np.sqrt(var)) if var > 1e-15 else 0.0
    z = diff / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z)) if se > 0 else 1.0
    ci = lambda k: [_r(max(0.0, auc[k] - Z95 * np.sqrt(cov[k, k]))), _r(min(1.0, auc[k] + Z95 * np.sqrt(cov[k, k])))]  # noqa: E731
    return {"auc_a": _r(auc[0]), "auc_b": _r(auc[1]), "diff": _r(diff), "se": _r(se), "z": _r(z), "p": _r(p),
            "ci_a": ci(0), "ci_b": ci(1)}


def paired_bootstrap(metric: Callable, y, pred_a, pred_b, n_boot: int = 2000, seed: int = 42,
                     higher_is_better: bool = True, chunk: int = 500) -> dict:
    """Paired bootstrap over rows. diff > 0 means a better (sign flipped when lower is better)."""
    y, pa, pb = map(np.asarray, (y, pred_a, pred_b))
    sign = 1.0 if higher_is_better else -1.0
    obs = sign * (metric(y, pa) - metric(y, pb))
    rng, n, diffs = np.random.default_rng(seed), len(y), []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, (min(chunk, n_boot - start), n))
        diffs.extend(sign * (metric(y[i], pa[i]) - metric(y[i], pb[i])) for i in idx)
    diffs = np.asarray(diffs)
    wrong = np.mean(diffs <= 0) if obs >= 0 else np.mean(diffs >= 0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": _r(obs), "ci_low": _r(lo), "ci_high": _r(hi), "p": _r(min(1.0, 2 * wrong))}


def calibration(y, prob, n_bins: int = 10) -> dict:
    """Brier score, ECE over equal-width bins, and non-empty bins for a reliability plot."""
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    idx = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for k in range(n_bins):
        mask = idx == k
        cnt = int(mask.sum())
        if cnt:
            conf, acc = prob[mask].mean(), y[mask].mean()
            ece += cnt / len(y) * abs(acc - conf)
            bins.append({"conf": _r(conf), "acc": _r(acc), "n": cnt})
    return {"brier": _r(np.mean((prob - y) ** 2)), "ece": _r(ece), "bins": bins}


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    """Kendall tau between two rankings restricted to their common items."""
    common = [x for x in rank_a if x in set(rank_b)]
    if len(common) < 2:
        return float("nan")
    return _r(stats.kendalltau([rank_a.index(x) for x in common], [rank_b.index(x) for x in common]).statistic)


def mann_kendall(series, alpha: float = 0.05) -> dict:
    """Mann-Kendall monotonic trend test with tie-corrected variance and continuity correction."""
    x = np.asarray(series, float)
    n = len(x)
    s = float(np.sign(x[None, :] - x[:, None])[np.triu_indices(n, 1)].sum())
    _, t = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(t * (t - 1) * (2 * t + 5))) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    trend = "none" if p >= alpha else ("increasing" if z > 0 else "decreasing")
    return {"s": s, "var": _r(var), "z": _r(z), "p": _r(p), "trend": trend}


def one_se_pick(results: list[dict], higher_is_better: bool = True, complexity_order: list[str] = ()) -> dict:
    """One-standard-error rule: among models within 1 SE of the best, pick the simplest family (ties by mean)."""
    sgn = 1 if higher_is_better else -1
    best = max(results, key=lambda r: sgn * r["mean"])
    se = best["std"] / np.sqrt(best["n_folds"])
    thr = best["mean"] - sgn * se
    within = [r for r in results if sgn * (r["mean"] - thr) >= 0]
    rank = {f: i for i, f in enumerate(complexity_order)}
    pick = min(within, key=lambda r: (rank.get(r["family"], len(rank)), -sgn * r["mean"]))
    return {"pick": pick["name"], "best": best["name"], "threshold": _r(thr), "within_1se": [r["name"] for r in within]}


def tie_groups(results: list[dict], pvals_vs_best: dict, alpha: float = 0.05, higher_is_better: bool = True) -> list[list[str]]:
    """Group 0 = best plus models not significantly different after Holm; the rest greedily grouped by mean."""
    sgn = 1 if higher_is_better else -1
    ordered = sorted(results, key=lambda r: -sgn * r["mean"])
    adj = holm(pvals_vs_best, alpha)
    best = ordered[0]["name"]
    groups = [[r["name"] for r in ordered if r["name"] == best or not adj.get(r["name"], {"reject": True})["reject"]]]
    # ponytail: no pairwise tests among the rest, so a later group = leader plus models within leader's 1 SE.
    rest = [r for r in ordered if r["name"] not in groups[0]]
    while rest:
        lead = rest[0]
        se = lead["std"] / np.sqrt(lead["n_folds"])
        grp = [r for r in rest if sgn * (lead["mean"] - r["mean"]) <= se]
        groups.append([r["name"] for r in grp])
        rest = [r for r in rest if r not in grp]
    return groups
