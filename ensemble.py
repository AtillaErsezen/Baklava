"""Caruana et al. (2004) greedy ensemble selection with replacement.

Fit with caruana() on search_val predictions only, then call compare_on_hidden() once on the hidden split.
Binary tasks pass positive-class probabilities, regression tasks pass predictions; multiclass is out of scope.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

import stats_tests as st

PATIENCE = 3  # additions without improvement before stopping
TOL = 1e-12   # score changes below this are float noise, not improvement
ALPHA = 0.05
METRICS = {
    "roc_auc": roc_auc_score,
    "accuracy": lambda y, p: accuracy_score(y, (p >= 0.5).astype(int)),
    "f1_macro": lambda y, p: f1_score(y, (p >= 0.5).astype(int), average="macro"),
    "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
    "mae": mean_absolute_error,
    "r2": r2_score,
}


def _arr(x, name: str) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1:
        raise ValueError(f"{name}: got a {a.ndim}-D array; multiclass probabilities are out of scope, pass the "
                         "positive-class column (binary) or 1-D predictions (regression)")
    return a


def score(metric: str, y, pred) -> float:
    """Metric on 1-D predictions; accuracy and f1_macro threshold probabilities at 0.5."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {sorted(METRICS)}")
    return float(METRICS[metric](np.asarray(y), _arr(pred, "pred")))


def caruana(preds: dict[str, np.ndarray], y: np.ndarray, metric: str, higher_is_better: bool,
            max_iter: int = 25, init_best: int = 1) -> dict:
    """Start from the init_best best singles, add with replacement the model that most improves the blend,
    stop after PATIENCE stale additions. Returns the best bag seen; history[0] is the initial bag."""
    if not preds:
        raise ValueError("preds is empty")
    if init_best < 1:
        raise ValueError("init_best must be >= 1")
    y = np.asarray(y)
    p = {k: _arr(v, k) for k, v in preds.items()}
    if any(len(v) != len(y) for v in p.values()):
        raise ValueError("every prediction array must have len(y) rows")
    names, sign = sorted(p), 1.0 if higher_is_better else -1.0
    rank = lambda s, n: (-sign * round(s, 12), n)  # noqa: E731  best score first, ties by name
    singles = {n: score(metric, y, p[n]) for n in names}
    counts = dict.fromkeys(names, 0)
    for n in sorted(names, key=lambda n: rank(singles[n], n))[:init_best]:
        counts[n] = 1
    total, k = sum(p[n] for n in names if counts[n]), sum(counts.values())
    best = score(metric, y, total / k)
    history, best_counts, stale = [best], dict(counts), 0
    for _ in range(max_iter):
        cand = {n: score(metric, y, (total + p[n]) / (k + 1)) for n in names}
        pick = min(names, key=lambda n: rank(cand[n], n))
        counts[pick], total, k = counts[pick] + 1, total + p[pick], k + 1
        history.append(cand[pick])
        if sign * (cand[pick] - best) > TOL:
            best, best_counts, stale = cand[pick], dict(counts), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    kb = sum(best_counts.values())
    return {"weights": {n: c / kb for n, c in best_counts.items()}, "val_score": best, "history": history,
            "members": [n for n in names if best_counts[n]]}


def blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average over the models with positive weight."""
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    used = {n: w for n, w in sorted(weights.items()) if w > 0}
    if not used:
        raise ValueError("weights has no positive entry")
    if missing := sorted(set(used) - set(preds)):
        raise ValueError(f"no predictions for {missing}")
    return sum(w * _arr(preds[n], n) for n, w in used.items()) / sum(used.values())


def compare_on_hidden(hidden_preds: dict[str, np.ndarray], y_hidden, weights: dict[str, float], best_single: str,
                      metric: str, higher_is_better: bool) -> dict:
    """One-shot hidden test of the blend vs the best single; delta = ensemble minus single in metric units.
    p: DeLong for roc_auc, paired bootstrap otherwise. keep only if delta improves and p < ALPHA."""
    if best_single not in hidden_preds:
        raise ValueError(f"no hidden predictions for best_single {best_single!r}")
    y = np.asarray(y_hidden)
    ens, single = blend(hidden_preds, weights), _arr(hidden_preds[best_single], best_single)
    es, ss = score(metric, y, ens), score(metric, y, single)
    if metric == "roc_auc":
        p = st.delong(y, ens, single).get("p", 1.0)
    else:
        p = st.paired_bootstrap(lambda a, b: score(metric, a, b), y, ens, single,
                                higher_is_better=higher_is_better)["p"]
    delta, sign = es - ss, 1.0 if higher_is_better else -1.0
    return {"ensemble_score": es, "best_single_score": ss, "delta": delta, "p": float(p),
            "keep": bool(sign * delta > TOL and p < ALPHA)}
