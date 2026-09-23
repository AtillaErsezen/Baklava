"""Statistics for comparing finalists: CV significance (Nadeau–Bengio + Holm) and a
paired bootstrap on the hidden set. Runs locally; numpy, scipy, sklearn.metrics."""
import warnings

import numpy as np
from scipy import stats
from sklearn import metrics as M


def nb_corrected_t(a, b, test_train_ratio):
    """Nadeau–Bengio corrected resampled t-test, one-sided, H1: a is better than b.
    a, b: per-fold scores on the SAME folds, oriented so higher is better.
    test_train_ratio: n_test / n_train of one fold. The correction inflates the variance
    because CV training sets overlap, which the plain paired t-test ignores.
    Returns (t, p) with df = J - 1 for J folds."""
    d = np.asarray(a, float) - np.asarray(b, float)
    j, mean, var = len(d), d.mean(), d.var(ddof=1)
    if var == 0:
        return (np.inf, 0.0) if mean > 0 else (-np.inf, 1.0) if mean < 0 else (0.0, 0.5)
    t = mean / np.sqrt((1 / j + test_train_ratio) * var)
    return float(t), float(stats.t.sf(t, j - 1))


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def score(metric, y, pred):
    """One metric from hidden-set labels and predictions. For classification, pred is the
    positive-class probability (binary, 1-D) or one probability row per sample
    (multiclass, 2-D); labels come from a 0.5 threshold or argmax."""
    y, pred = np.asarray(y), np.asarray(pred, float)
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - pred) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y - pred)))
    if metric == "r2":
        return float(M.r2_score(y, pred))
    k = pred.shape[1] if pred.ndim == 2 else 2
    if metric == "roc_auc":
        if pred.ndim == 2:
            return float(M.roc_auc_score(y, pred, multi_class="ovr", labels=np.arange(k)))
        return float(M.roc_auc_score(y, pred))
    labels = pred.argmax(1) if pred.ndim == 2 else (pred >= 0.5).astype(int)
    if metric == "accuracy":
        return float(M.accuracy_score(y, labels))
    if metric == "f1_macro":
        return float(M.f1_score(y, labels, average="macro", labels=np.arange(k), zero_division=0))
    raise ValueError(f"unknown metric {metric}")


def _safe_score(metric, y, pred):
    """score(), or NaN when a resample is degenerate (e.g. only one class drawn)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return score(metric, y, pred)
    except ValueError:
        return np.nan


def paired_bootstrap(y, preds, metric, best, n_boot=2000, seed=0):
    """Resample hidden rows with replacement; every model is scored on the same resamples,
    so differences are paired. preds: {name: pred}; best: the name to compare against.
    Returns {name: {"score", "ci", "diff_vs_best", "diff_ci"}}: 95% percentile intervals,
    diff = score(name) - score(best) in raw metric units (for rmse/mae, negative = better)."""
    y = np.asarray(y)
    preds = {k: np.asarray(v, float) for k, v in preds.items()}
    idx = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))
    boot = {k: np.array([_safe_score(metric, y[i], p[i]) for i in idx]) for k, p in preds.items()}
    full = {k: score(metric, y, p) for k, p in preds.items()}
    ci = lambda a: [round(float(v), 5) for v in np.nanpercentile(a, [2.5, 97.5])]
    return {k: {"score": round(full[k], 5), "ci": ci(boot[k]),
                "diff_vs_best": round(full[k] - full[best], 5), "diff_ci": ci(boot[k] - boot[best])}
            for k in preds}
