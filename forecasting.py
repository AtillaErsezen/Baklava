"""Forecasting: turn a raw time series (one or many entities) into a leak-free supervised table.

Each output row is a forecast origin t. The label is the target `horizon` steps later in the
same series; every feature uses only values observed at or before t. leakage_audit verifies
that by rebuilding sampled rows from history truncated at their own timestamp.
"""
from __future__ import annotations

import itertools
import math
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.stattools import acf, adfuller

import sampling

DUP_TOL = 0.01        # (entity, time) duplicate share still called "nearly unique"
MAX_ENTITY_COLS = 3
MAX_CANDIDATES = 12
FREQ_SAMPLE = 50      # entities checked with pd.infer_freq
REG_TOL = 0.1         # a step within 10% of the median step counts as regular
ADF_MAX = 5000        # ponytail: ADF on the latest 5000 points only, raise if long series need it
AUDIT_ROWS = 30
ROLL_STATS = ("mean", "std", "min", "max")
# (min step days, max step days, candidate seasonal periods)
SEASON_CANDIDATES = [(0.5 / 24, 1.5 / 24, [24, 168]), (0.5, 1.5, [7, 365]), (5, 9, [52]),
                     (25, 35, [12]), (80, 100, [4])]


# ---------- structure ----------

def _as_time(s: pd.Series) -> pd.Series:
    """s as datetimes, parsed from strings if needed (unparseable values become NaT)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", format="mixed")


def _find_time_col(df: pd.DataFrame, target: str) -> str | None:
    """Sorted datetime column first (sampling), else any datetime or date-like string column."""
    found = sampling.sorted_time_column(df, target)
    if found:
        return found
    for c in df.columns:
        s = df[c]
        if c == target or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            if s.nunique() > 1:
                return c
            continue
        sample = s.dropna().astype(str).head(200)
        if len(sample) and sample.str.len().min() >= 6 and _as_time(sample).notna().mean() > 0.9:
            return c
    return None


def _entity_cols(df: pd.DataFrame, target: str, time_col: str) -> list[str]:
    """Fewest id-like columns that make (entity, time) nearly unique; [] for a single series."""
    if df.duplicated(subset=[time_col]).mean() <= DUP_TOL:
        return []
    n, cands = len(df), []
    for c in df.columns:
        s = df[c]
        if c in (target, time_col) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s) \
                or pd.api.types.is_datetime64_any_dtype(s):
            continue
        k = s.nunique()
        if 2 <= k <= n / 2:
            cands.append((pd.api.types.is_integer_dtype(s), k, c))  # text ids first, then low cardinality
    cands = [c for *_, c in sorted(cands)][:MAX_CANDIDATES]
    # ponytail: brute force over up to C(12, 3) combinations, fine below a few million rows
    for size in range(1, MAX_ENTITY_COLS + 1):
        scored = [(df.duplicated(subset=[*combo, time_col]).mean(), i, list(combo))
                  for i, combo in enumerate(itertools.combinations(cands, size))]
        ok = [s for s in scored if s[0] <= DUP_TOL]
        if ok:
            return min(ok)[2]
    return []


def _gid(df: pd.DataFrame, ents: list[str]) -> pd.Series:
    """Integer series id per row (all zeros without entity columns)."""
    if not ents:
        return pd.Series(0, index=df.index)
    return df.groupby(ents, sort=False, dropna=False).ngroup()


def _steps(df: pd.DataFrame, time_col: str, ents: list[str]) -> pd.DataFrame:
    """Rows (gid, t) sorted by series and time, plus the positive step to the previous row."""
    s = pd.DataFrame({"g": _gid(df, ents).to_numpy(), "t": df[time_col].to_numpy()}).dropna()
    s = s.sort_values(["g", "t"], kind="stable")
    s["d"] = s.groupby("g")["t"].diff()
    return s


def _frequency(steps: pd.DataFrame) -> dict:
    """Modal pd.infer_freq over sampled series, else the median step; regularity score."""
    d = steps["d"].dropna()
    d = d[d > pd.Timedelta(0)]
    med = d.median() if len(d) else pd.NaT
    freqs = []
    for _, t in itertools.islice(steps.groupby("g")["t"], FREQ_SAMPLE):  # ponytail: first 50 series only
        u = pd.DatetimeIndex(t.unique())
        if len(u) >= 3:
            freqs.append(pd.infer_freq(u))
    freqs = [f for f in freqs if f]
    source = "infer_freq" if freqs else "median_step"
    freq = Counter(freqs).most_common(1)[0][0] if freqs else (to_offset(med).freqstr if pd.notna(med) else None)
    reg = float(((d - med).abs() <= REG_TOL * med).mean()) if len(d) else 0.0
    days = med / pd.Timedelta(days=1) if pd.notna(med) else None
    return {"freq": freq, "freq_source": source, "median_step": str(med), "step_days": days, "regularity": reg}


def _candidates(step_days: float | None) -> list[int]:
    return next((p for lo, hi, p in SEASON_CANDIDATES if step_days is not None and lo <= step_days <= hi), [])


def _seasonality(agg: pd.Series, step_days: float | None) -> dict:
    """ACF of the differenced aggregate at candidate periods; period = strongest significant local peak."""
    x = agg.diff().dropna().to_numpy()
    fits = [p for p in _candidates(step_days) if len(x) > 2 * p + 1]
    if not fits or np.std(x) == 0:
        return {"acf": {}, "top": [], "period": None}
    r = acf(x, nlags=max(fits) + 1, fft=True)
    scores = {p: round(float(r[p]), 3) for p in fits}
    thr = 2 / math.sqrt(len(x))
    top = sorted((p for p in fits if r[p] > thr and r[p] >= r[p - 1] and r[p] >= r[p + 1]), key=lambda p: -r[p])
    return {"acf": scores, "threshold": round(thr, 3), "top": top, "period": top[0] if top else None}


def _adf_pvalue(agg: pd.Series) -> float | None:
    x = agg.dropna().to_numpy()[-ADF_MAX:]
    if len(x) < 20 or np.std(x) == 0:
        return None
    try:
        return float(adfuller(x, autolag="AIC", result_object=False)[1])
    except (ValueError, np.linalg.LinAlgError):
        return None


def detect_structure(df: pd.DataFrame, target: str, time_col: str | None = None) -> dict:
    """Time column, entity columns, frequency, regularity, series lengths, seasonality, stationarity."""
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in columns")
    time_col = time_col or _find_time_col(df, target)
    if time_col is None:
        return {"time_col": None, "entity_cols": [], "note": "no parseable datetime column"}
    data = df.assign(**{time_col: _as_time(df[time_col])})
    data = data[data[time_col].notna()]
    ents = _entity_cols(data, target, time_col)
    steps = _steps(data, time_col, ents)
    freq = _frequency(steps)
    sizes = steps.groupby("g").size()
    y = pd.to_numeric(data[target], errors="coerce")
    agg = y.groupby(data[time_col]).mean().sort_index()
    p = _adf_pvalue(agg)
    return {"time_col": time_col, "entity_cols": ents, **freq, "n_entities": int(len(sizes)),
            "series_length": {"min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max())},
            "seasonality": _seasonality(agg, freq["step_days"]), "adf_pvalue": p,
            "stationary": None if p is None else p < 0.05}


# ---------- supervised table ----------

def _past_mean(data: pd.DataFrame, col: str, y: pd.Series, time_col: str) -> pd.Series:
    """Mean target over rows sharing data[col] with a strictly earlier timestamp."""
    keys = [data[col], data[time_col]]
    agg = y.groupby(keys, dropna=False).agg(["sum", "count"])  # sorted by (col, time)
    prev = agg.groupby(level=0, dropna=False).cumsum().groupby(level=0, dropna=False).shift(1)
    enc = prev["sum"] / prev["count"]
    return pd.Series(enc.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy(), index=data.index)


def _calendar(t: pd.Series, p: dict) -> dict:
    """Calendar parts of the origin time, plus steps since the first timestamp."""
    doy = t.dt.dayofyear
    cal = {"cal_dow": t.dt.dayofweek, "cal_month": t.dt.month, "cal_week": t.dt.isocalendar().week.astype(int),
           "cal_quarter": t.dt.quarter, "cal_is_month_end": t.dt.is_month_end.astype(int),
           "cal_doy_sin": np.sin(2 * np.pi * doy / 365.25), "cal_doy_cos": np.cos(2 * np.pi * doy / 365.25)}
    if p["sub_daily"]:
        cal["cal_hour"] = t.dt.hour
    cal["time_index"] = (t - pd.Timestamp(p["t0"])).dt.total_seconds() / p["step_seconds"]
    return cal


def _features(data: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Point-in-time features for every row of data (sorted by entity, time); no label."""
    target, time_col, ents, h = p["target"], p["time_col"], p["entity_cols"], p["horizon"]
    gid = _gid(data, ents)
    y = data[target].astype(float)
    grp = y.groupby(gid)
    base = {c: data[c] for c in [time_col, *ents, *p["known"]]}
    out = {}
    for k in p["lags"]:
        out[f"{target}_lag{k}"] = grp.shift(k - h)  # value k steps before the label time
    for w in p["windows"]:
        roll = grp.rolling(w, min_periods=1)  # window ends at the origin t
        for stat in ROLL_STATS:
            out[f"{target}_roll{w}_{stat}"] = getattr(roll, stat)().droplevel(0)
    out[f"{target}_expanding_mean"] = grp.expanding().mean().droplevel(0)
    for c in ents:
        out[f"{c}_target_enc"] = _past_mean(data, c, y, time_col)
    for c in p["lag_cols"]:
        x = data[c].astype(float).groupby(gid)
        for k in p["lags"]:
            if k > h:
                out[f"{c}_lag{k}"] = x.shift(k - h)
    out.update(_calendar(data[time_col], p))
    clash = sorted(set(base) & set(out))
    if clash:
        raise ValueError(f"input columns {clash} clash with generated feature names; rename them")
    return pd.DataFrame({**base, **out}, index=data.index)


def _prepare(df: pd.DataFrame, target: str, time_col: str, ents: list[str]) -> pd.DataFrame:
    """Parsed time, numeric target, rows without a time dropped, sorted by entity then time."""
    missing = [c for c in [target, time_col, *ents] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in data: {missing}")
    data = df.assign(**{time_col: _as_time(df[time_col])})
    data = data[data[time_col].notna()]
    try:
        data[target] = pd.to_numeric(data[target])
    except (TypeError, ValueError) as e:
        raise ValueError(f"target {target!r} must be numeric to forecast it") from e
    return data.sort_values([*ents, time_col], kind="stable").reset_index(drop=True)


def _choose_lags(h: int, season: int | None, lags, max_hist: int) -> list[int]:
    """Lags >= h; defaults h..h+2 plus season multiples. h and the seasonal naive lag are always kept."""
    if lags is not None:
        lags = [int(k) for k in lags]
        bad = [k for k in lags if k < h]
        if bad:
            raise ValueError(f"lags {bad} are below horizon {h}: lag k is the value k steps before the label, "
                             f"so k < horizon would read the future")
        chosen = set(lags)
    else:
        chosen = {h, h + 1, h + 2} | ({season, 2 * season} if season else set())
    if season:
        chosen.add(math.ceil(h / season) * season)
    user = set(lags or [])
    return sorted(k for k in chosen | {h} if k - h <= max_hist or k in user)


def _choose_windows(season: int | None, windows, max_hist: int) -> list[int]:
    if windows is not None:
        if any(int(w) < 1 for w in windows):
            raise ValueError("windows must be positive")
        return sorted({int(w) for w in windows})
    return sorted(w for w in {3, season or 3} if w - 1 <= max_hist)


def _season(data: pd.DataFrame, target: str, time_col: str, step_days: float | None, n_min: int):
    """Detected seasonal period, else the frequency's first candidate that fits, else None."""
    agg = data[target].groupby(data[time_col]).mean().sort_index()
    found = _seasonality(agg, step_days)["period"]
    if found:
        return found, "acf"
    fallback = next((p for p in _candidates(step_days) if 2 * p <= n_min), None)
    return fallback, ("frequency default, not detected" if fallback else None)


def make_supervised(df: pd.DataFrame, target: str, time_col: str, entity_cols: list[str] | None,
                    horizon: int = 1, lags: list[int] | None = None, windows: list[int] | None = None,
                    unknown_future: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Leak-free supervised frame (label named `target`, sorted by time) and its manifest."""
    ents, unknown = list(entity_cols or []), list(unknown_future or [])
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    data = _prepare(df, target, time_col, ents)
    if unknown and not set(unknown) <= set(data.columns):
        raise KeyError(f"unknown_future columns not in data: {sorted(set(unknown) - set(data.columns))}")
    n_dup = int(data.duplicated([*ents, time_col]).sum())
    if n_dup:
        raise ValueError(f"{n_dup} rows repeat an (entity, time) pair; add entity columns or aggregate first")
    steps = _steps(data, time_col, ents)
    freq = _frequency(steps)
    n_min = int(steps.groupby("g").size().min())
    season, source = _season(data, target, time_col, freq["step_days"], n_min)
    lags = _choose_lags(horizon, season, lags, n_min // 2)
    windows = _choose_windows(season, windows, n_min // 2)
    others = [c for c in data.columns if c not in (target, time_col, *ents)]
    p = {"target": target, "time_col": time_col, "entity_cols": ents, "horizon": int(horizon), "lags": lags,
         "windows": windows, "known": [c for c in others if c not in unknown], "unknown_future": unknown,
         "lag_cols": [c for c in others if pd.api.types.is_numeric_dtype(data[c])],
         "t0": data[time_col].min().isoformat(), "sub_daily": bool((freq["step_days"] or 1) < 1),
         "step_seconds": float((freq["step_days"] or 1) * 86400)}
    feats = _features(data, p)
    gid = _gid(data, ents)
    label = data[target].astype(float).groupby(gid).shift(-horizon)
    short = data.groupby(gid).cumcount() < max(max(lags) - horizon, max(windows) - 1, 0)
    no_label = label.isna() & ~short
    frame = feats.assign(**{target: label})[~short & ~no_label]
    if frame.empty:
        raise ValueError("no rows left: series are shorter than the lags and windows need")
    frame = frame.sort_values([time_col, *ents], kind="stable").reset_index(drop=True)
    n_ent = int(gid.nunique())
    manifest = {"label": target, "time_col": time_col, "entity_cols": ents, "horizon": int(horizon),
                "lags": lags, "windows": windows, "season": season, "season_source": source,
                "features": [c for c in frame.columns if c not in (target, time_col)],
                "dropped_rows": {"missing_time": len(df) - len(data), "insufficient_history": int(short.sum()),
                                 "no_label": int(no_label.sum()), "total": len(df) - len(frame)},
                "baseline_columns": _baseline_columns(target, horizon, season, lags),
                "cv_gap_rows": (horizon - 1) * n_ent, "n_entities": n_ent,
                "assumptions": _assumptions(p, freq["regularity"], season, source, n_ent), "params": p}
    return frame, manifest


def _baseline_columns(target: str, h: int, season: int | None, lags: list[int]) -> dict:
    sn = math.ceil(h / season) * season if season else None
    return {"last_value": f"{target}_lag{h}", "seasonal_naive": f"{target}_lag{sn}" if sn in lags else None}


def _assumptions(p: dict, regularity: float, season: int | None, source: str | None, n_ent: int) -> list[str]:
    t, h = p["target"], p["horizon"]
    out = [f"Row time {p['time_col']} is the forecast origin t; the label {t} is the value {h} step(s) later "
           f"in the same series.",
           f"{t}_lag<k> is the target k steps before the label time (k >= {h}); rolling, expanding and "
           f"target encoding stats use values at or before t only.",
           "Calendar features and time_index describe the forecast origin t."]
    if p["known"]:
        out.append(f"Used at their value at t, assumed observed when the forecast is made: {p['known']}. "
                   "Move any column that arrives later into unknown_future.")
    if p["unknown_future"]:
        out.append(f"unknown_future columns {p['unknown_future']} are used only through lags (t-1 and earlier); "
                   "non-numeric ones are dropped.")
    gaps = " Series have gaps, so a lag can span more time than its number says." if regularity < 0.95 else ""
    out.append(f"Lags count rows within each series, not calendar periods (regularity {regularity:.2f}).{gaps}")
    out.append(f"Seasonal period: {season} ({source})." if season else "No seasonal period used.")
    if h > 1:
        out.append(f"CV must not shuffle; use walk_forward or purged with gap >= {(h - 1) * n_ent} rows, and drop "
                   f"the last {h - 1} step(s) of dev before the hidden split, so no training label falls after "
                   "a test origin.")
    else:
        out.append("CV must not shuffle; walk_forward or purged on rows sorted by time.")
    return out


# ---------- audit + baselines ----------

def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-7, atol=1e-9))
    except (TypeError, ValueError):
        return a == b


def _label_copies(frame: pd.DataFrame, target: str, cols: list[str]) -> list[str]:
    """Numeric columns equal to the (non-constant) label on every row."""
    lab = frame[target].astype(float)
    out = []
    for c in cols:
        s = frame[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        m = s.notna() & lab.notna()
        if m.sum() and lab[m].nunique() > 1 and np.allclose(s[m].astype(float), lab[m]):
            out.append(f"{c} equals the future label {target} on every row: it is the answer, not a feature.")
    return out


def _rebuild(frame: pd.DataFrame, data: pd.DataFrame, p: dict, rows: np.ndarray, cols: list[str]) -> Counter:
    """Per column, how many sampled rows change when rebuilt from history cut at their own time."""
    time_col, ents = p["time_col"], p["entity_cols"]
    bad = Counter()
    for i in rows:
        r = frame.iloc[i]
        hist = data[data[time_col] <= r[time_col]]
        if ents:  # rows of any series sharing an entity value (target encodings need them)
            hist = hist[np.logical_or.reduce([hist[c] == r[c] for c in ents])]
        f = _features(hist, p)
        m = f[time_col] == r[time_col]
        for c in ents:
            m &= f[c] == r[c]
        if m.sum() != 1:
            bad["<row not rebuildable from raw>"] += 1
            continue
        g = f[m].iloc[0]
        for c in cols:
            if not _same(r[c], g[c]):
                bad[c] += 1
    return bad


def leakage_audit(frame: pd.DataFrame, target: str, time_col: str, raw: pd.DataFrame | None = None,
                  manifest: dict | None = None, n_rows: int = AUDIT_ROWS, seed: int = 0) -> list[str]:
    """Problems found; [] means every checked feature is computable from data at or before its row time."""
    cols = [c for c in frame.columns if c not in (target, time_col)]
    issues = _label_copies(frame, target, cols)
    if raw is None or manifest is None:
        return issues + ["Point-in-time check skipped: pass raw= and manifest= from make_supervised."]
    p = manifest["params"]
    issues += [f"{c} was not built by make_supervised; its point-in-time availability is unverified."
               for c in cols if c not in manifest["features"]]
    data = _prepare(raw, p["target"], p["time_col"], p["entity_cols"])
    rows = np.sort(np.random.default_rng(seed).choice(len(frame), min(n_rows, len(frame)), replace=False))
    check = [c for c in manifest["features"] if c in frame.columns and c not in p["entity_cols"]]
    for c, n in _rebuild(frame, data, p, rows, check).items():
        issues.append(f"{c} changes on {n} of {len(rows)} sampled rows when data after the row time is removed: "
                      "it uses future values.")
    return issues


def naive_baselines(frame: pd.DataFrame, label_col: str, manifest: dict | None = None,
                    season: int | None = None, test_frac: float = 0.2) -> dict:
    """Last-value and seasonal naive MAE/RMSE on the latest test_frac of rows (same rows as split_three hidden)."""
    if manifest:
        time_col, cols = manifest["time_col"], dict(manifest["baseline_columns"])
    else:  # row order taken as time order; lag columns named <label>_lag<k>
        ks = sorted(int(m.group(1)) for c in frame.columns
                    if (m := re.fullmatch(rf"{re.escape(label_col)}_lag(\d+)", str(c))))
        if not ks:
            raise ValueError(f"no {label_col}_lag<k> columns; pass the manifest from make_supervised")
        time_col, cols = None, _baseline_columns(label_col, ks[0], season, ks)
    n = len(frame)
    order = np.argsort(frame[time_col].to_numpy(), kind="stable") if time_col else np.arange(n)
    n_test = round(n * test_frac)
    test = frame.iloc[order[n - n_test:]]
    out = {"test_rows": n_test, "split": f"latest {test_frac:.0%} of rows by time"}
    scores = {}
    for name, col in cols.items():
        if col is None or col not in test.columns:
            out[name] = None
            continue
        m = test[[label_col, col]].astype(float).dropna()
        err = m[label_col] - m[col]
        out[name] = {"column": col, "n": len(m), "mae": float(err.abs().mean()),
                     "rmse": float(np.sqrt((err ** 2).mean()))}
        scores[name] = out[name]["mae"]
    out["best"] = min(scores, key=scores.get) if scores else None
    return out
