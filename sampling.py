"""Leak-free splits and statistically justified training subsamples."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from statsmodels.stats.multitest import multipletests

N_BINS = 10
TIME_BUCKETS = 4
RARE_SHARE = 0.01
ADV_AUC_MAX = 0.6
ADV_MAX_REST = 20_000


def _target_labels(y: pd.Series, task: str) -> np.ndarray:
    """Class labels, or quantile bin ids for regression."""
    if task == "regression":
        return pd.qcut(y.rank(method="first"), N_BINS, labels=False, duplicates="drop").fillna(-1).to_numpy()
    return y.astype(str).to_numpy()


def _strata(df: pd.DataFrame, target: str, task: str, time_column: str | None = None,
            extra: list[str] | None = None) -> np.ndarray:
    """Integer stratum id per row: target x extra strata x time bucket."""
    parts = [pd.Series(_target_labels(df[target], task)).astype(str)]
    parts += [df[c].astype(str).reset_index(drop=True) for c in extra or []]
    if time_column:
        t = df[time_column].rank(method="first").reset_index(drop=True)
        parts.append(pd.qcut(t, TIME_BUCKETS, labels=False, duplicates="drop").astype(str))
    key = parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]
    return pd.factorize(key)[0]


def _split(idx: np.ndarray, frac: float, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split of idx. Classes too small to stratify (under 2 rows) join the largest class so the
    rest stay stratified; plain random only if that still fails."""
    counts = pd.Series(labels).value_counts()
    merged = np.where(pd.Series(labels).map(counts).to_numpy() < 2, counts.index[0], labels)
    try:
        return train_test_split(idx, test_size=frac, random_state=seed, stratify=merged)
    except ValueError:
        return train_test_split(idx, test_size=frac, random_state=seed)


def sorted_time_column(df: pd.DataFrame, target: str) -> str | None:
    """First datetime column (not the target) already in increasing order: time-ordered data,
    so the hidden split must be the latest rows. Ported from Atilla's main (cf22121)."""
    for c in df.columns:
        s = df[c]
        if c != target and pd.api.types.is_datetime64_any_dtype(s) and s.nunique() > 1 and s.is_monotonic_increasing:
            return c
    return None


def cross_split_duplicates(dev: pd.DataFrame, hidden: pd.DataFrame, target: str) -> int:
    """Hidden rows whose features exactly repeat a dev row (memorization leak). Ported from cf22121."""
    cols = [c for c in dev.columns if c != target]
    seen = set(pd.util.hash_pandas_object(dev[cols], index=False))
    return int(sum(h in seen for h in pd.util.hash_pandas_object(hidden[cols], index=False)))


def split_three(df: pd.DataFrame, target: str, task: str, time_column: str | None = None, seed: int = 42,
                fractions: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> dict[str, np.ndarray]:
    """Disjoint positional index arrays dev / search_val / hidden covering every row."""
    n = len(df)
    f_dev, f_sv, f_hid = (f / sum(fractions) for f in fractions)
    if time_column:
        order = np.argsort(df[time_column].to_numpy(), kind="stable")
        n_hid, n_sv = round(n * f_hid), round(n * f_sv)
        return {"dev": order[: n - n_hid - n_sv], "search_val": order[n - n_hid - n_sv: n - n_hid],
                "hidden": order[n - n_hid:]}
    labels = _target_labels(df[target], task)
    rest, hidden = _split(np.arange(n), f_hid, labels, seed)
    dev, sv = _split(rest, f_sv / (f_dev + f_sv), labels[rest], seed + 1)
    return {"dev": np.sort(dev), "search_val": np.sort(sv), "hidden": np.sort(hidden)}


def hoeffding_n(eps: float, delta: float, k: int = 1) -> int:
    """Rows so all k bounded [0,1] metrics are within eps with prob 1 - delta (union bound)."""
    return math.ceil(math.log(2 * k / delta) / (2 * eps**2))


def precision_n(sigma: float, eps: float, alpha: float = 0.05) -> int:
    """Rows for a normal CI half width eps given per row std sigma."""
    return math.ceil((stats.norm.ppf(1 - alpha / 2) * sigma / eps) ** 2)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of AUC (Hanley and McNeil 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
    return math.sqrt(max(var, 0.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class PowerLaw:
    a: float
    b: float
    c: float
    ok: bool


def _pl(n, a, b, c):
    return a + b * np.power(n, -c)


def fit_power_law(ns, errs) -> PowerLaw:
    """Fit err(n) = a + b n^-c; on failure returns (min(errs), 0, 0, ok=False)."""
    ns, errs = np.asarray(ns, float), np.asarray(errs, float)
    fallback = PowerLaw(float(np.min(errs)), 0.0, 0.0, False)
    if ns.size < 3:
        return fallback
    a0 = max(float(errs.min()) * 0.5, 0.0)
    b0 = max(float(errs.max()) - a0, 1e-6) * float(ns.min()) ** 0.5
    try:
        (a, b, c), _ = curve_fit(_pl, ns, errs, p0=(a0, b0, 0.5),
                                 bounds=([0, 0, 1e-6], [np.inf, np.inf, 3 - 1e-6]), maxfev=20_000)
    except (RuntimeError, ValueError):
        return fallback
    return PowerLaw(float(a), float(b), float(c), True) if np.isfinite([a, b, c]).all() else fallback


def knee_n(a: float, b: float, c: float, eps: float, n_max: int) -> int:
    """Smallest grid n where doubling data gains less than eps error; n_max if never."""
    if n_max <= 100:
        return int(n_max)
    for n in np.unique(np.geomspace(100, n_max, 60).astype(int)):
        if _pl(n, a, b, c) - _pl(2 * n, a, b, c) < eps:
            return int(n)
    return int(n_max)


def _allocate(counts: np.ndarray, n: int) -> np.ndarray:
    """Proportional allocation, largest remainder, at least 1 per stratum when n allows."""
    quota = n * counts / counts.sum()
    alloc = np.minimum(np.floor(quota).astype(int), counts)
    if n >= len(counts):
        alloc = np.maximum(alloc, 1)
    diff = n - alloc.sum()
    while diff != 0:
        if diff > 0:
            cand = [i for i in np.argsort(-(quota - alloc)) if alloc[i] < counts[i]]
        else:
            cand = [i for i in np.argsort(-(alloc - quota)) if alloc[i] > 1]
        for i in cand[: abs(diff)]:
            alloc[i] += int(np.sign(diff))
        diff = n - alloc.sum()
    return alloc


def stratified_sample(df: pd.DataFrame, target: str, task: str, n: int, seed: int = 42,
                      time_column: str | None = None, extra_strata: list[str] | None = None) -> np.ndarray:
    """Exactly n unique positional indices, stratified without replacement."""
    if n >= len(df):
        return np.arange(len(df))
    ids = _strata(df, target, task, time_column, extra_strata)
    counts = np.bincount(ids)
    rng = np.random.default_rng(seed)
    alloc = _allocate(counts, n)
    groups = [np.flatnonzero(ids == s) for s in range(len(counts))]
    return np.sort(np.concatenate([rng.choice(g, k, replace=False) for g, k in zip(groups, alloc) if k]))


def _cat_pvalue(a: pd.Series, b: pd.Series) -> float | None:
    """Chi-square p value on level counts, levels under RARE_SHARE merged."""
    a, b = a.astype(str), b.astype(str)
    share = pd.concat([a, b]).value_counts(normalize=True)
    rare = set(share[share < RARE_SHARE].index)
    table = pd.concat([a.where(~a.isin(rare), "__rare__").value_counts(),
                       b.where(~b.isin(rare), "__rare__").value_counts()], axis=1).fillna(0)
    return None if len(table) < 2 else float(stats.chi2_contingency(table.T.to_numpy())[1])


def _adversarial_auc(sample: pd.DataFrame, rest: pd.DataFrame, seed: int = 0) -> float:
    """3 fold AUC of a small classifier telling sample rows from the rest."""
    if len(rest) > ADV_MAX_REST:  # ponytail: subsampled rest, raise cap if AUC looks noisy
        rest = rest.sample(ADV_MAX_REST, random_state=seed)
    X = pd.concat([sample, rest], ignore_index=True)
    X = X.apply(lambda s: s if pd.api.types.is_numeric_dtype(s) else pd.Series(pd.factorize(s)[0]).replace(-1, np.nan))
    y = np.r_[np.ones(len(sample)), np.zeros(len(rest))]
    if min(len(sample), len(rest)) < 3:
        return 0.5
    clf = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X.astype(float), y, cv=cv, scoring="roc_auc")))


def representativeness(sample: pd.DataFrame, full: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Per column KS / chi-square with BH FDR, plus adversarial AUC of sample vs other rows."""
    rest = full.loc[~full.index.isin(sample.index)]
    ref = rest if len(rest) else full
    cols, pvals = [], []
    for c in [c for c in full.columns if c in sample.columns]:
        a, b = sample[c].dropna(), ref[c].dropna()
        if a.empty or b.empty:
            continue
        p = (float(stats.ks_2samp(a, b).pvalue) if pd.api.types.is_numeric_dtype(full[c])
             and a.nunique() > 2 else _cat_pvalue(a, b))
        if p is not None:
            cols.append(c)
            pvals.append(p)
    rejected = [c for c, r in zip(cols, multipletests(pvals, alpha, "fdr_bh")[0]) if r] if pvals else []
    auc = _adversarial_auc(sample[full.columns], rest[full.columns]) if len(rest) else 0.5
    return {"ok": not rejected and auc < ADV_AUC_MAX, "rejected_columns": rejected, "adversarial_auc": auc}


def choose_n(n_dev: int, k_configs: int, eps: float = 0.02, delta: float = 0.05,
             pilot_sigma: float | None = None, curve: tuple | None = None) -> dict:
    """Training rows n_star with a one sentence justification."""
    n_hoef = hoeffding_n(eps, delta, k_configs)
    n_prec = precision_n(pilot_sigma, eps) if pilot_sigma is not None else None
    n_eval = max(n_hoef, n_prec or 0)
    n_knee = knee_n(*curve[:3], eps, n_dev) if curve is not None else None
    n_star = min(n_dev, max(n_eval, n_knee or 0))
    why = (f"Use n_star = {n_star} of {n_dev} dev rows because Hoeffding with eps = {eps}, delta = {delta} "
           f"and {k_configs} configs suggests about {n_hoef} rows as a rough guide "
           "(CV estimates are correlated, so this is approximate)")
    if n_prec is not None:
        why += f", a CI half width of {eps} at sigma = {pilot_sigma} needs {n_prec}"
    if n_knee is not None:
        why += f", and the fitted learning curve gains under {eps} per doubling from n = {n_knee}"
    why += ", capped at the dev size." if n_star == n_dev and n_dev < max(n_eval, n_knee or 0) else "."
    return {"n_star": n_star, "n_eval": n_eval, "n_knee": n_knee, "justification": why}


def draw_verified(df: pd.DataFrame, target: str, task: str, n: int, seed: int = 42,
                  time_column: str | None = None, max_tries: int = 5) -> dict:
    """Stratified draw, redrawn with seed + i until representativeness passes."""
    for i in range(max_tries):
        idx = stratified_sample(df, target, task, n, seed + i, time_column)
        report = representativeness(df.iloc[idx], df)
        if report["ok"]:
            break
    return {"indices": idx, "tries": i + 1, "report": report}
