import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import forecasting as fc
import sampling

HERE = os.path.dirname(os.path.abspath(__file__))
DRIFTING = os.path.join(HERE, "evals", "data", "drifting_sales.csv")


def _stores(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """3 stores, daily, weekly seasonality, stacked by store (so date is not globally sorted)."""
    rng = np.random.default_rng(seed)
    day = pd.date_range("2023-01-01", periods=n_days, freq="D")
    parts = []
    for store, level in [("A", 100.0), ("B", 150.0), ("C", 200.0)]:
        season = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        price = rng.normal(5, 1, n_days).round(2)
        sales = level + season - 2 * price + rng.normal(0, 2, n_days)
        parts.append(pd.DataFrame({"date": day, "store": store, "price": price, "sales": sales.round(2)}))
    return pd.concat(parts, ignore_index=True)


def _supervised(horizon: int = 1, **kw):
    df = _stores()
    frame, manifest = fc.make_supervised(df, "sales", "date", ["store"], horizon=horizon, **kw)
    return df, frame, manifest


def test_detect_structure_panel():
    df = _stores().sample(frac=1, random_state=1).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # unsorted, string dates
    s = fc.detect_structure(df, "sales")
    assert s["time_col"] == "date", s
    assert s["entity_cols"] == ["store"], s
    assert s["freq"] == "D" and s["n_entities"] == 3, s
    assert s["regularity"] > 0.99, s
    assert s["seasonality"]["period"] == 7, s
    assert s["series_length"]["min"] == 400, s
    assert 0 <= s["adf_pvalue"] <= 1, s


def test_make_supervised_has_no_leakage():
    df, frame, manifest = _supervised()
    assert fc.leakage_audit(frame, "sales", "date", raw=df, manifest=manifest) == []
    assert {"sales_lag1", "sales_lag7", "sales_roll7_mean", "store_target_enc", "cal_dow", "time_index"} <= set(frame)
    assert manifest["dropped_rows"]["total"] == len(df) - len(frame) > 0
    assert frame["sales"].notna().all() and frame["sales_lag7"].notna().all()


def test_leakage_audit_catches_future_features():
    df, frame, manifest = _supervised()
    bad = frame.copy()
    centred = (bad.groupby("store")["sales_lag1"]
               .transform(lambda s: s.rolling(7, center=True, min_periods=1).mean()))
    bad["sales_roll7_mean"] = centred.to_numpy()  # window reaches past t
    bad["cheat"] = bad["sales"]  # copy of the label
    issues = fc.leakage_audit(bad, "sales", "date", raw=df, manifest=manifest)
    text = " ".join(issues)
    assert "sales_roll7_mean" in text and "cheat" in text, issues


def test_lag_below_horizon_is_rejected():
    try:
        _supervised(horizon=7, lags=[3, 7])
    except ValueError as e:
        assert "future" in str(e)
    else:
        raise AssertionError("lag 3 with horizon 7 must raise")


def test_horizon_7_label_is_target_7_days_later():
    df, frame, manifest = _supervised(horizon=7)
    later = df.assign(date=df["date"] - pd.Timedelta(days=7)).rename(columns={"sales": "future"})
    m = frame[["store", "date", "sales"]].merge(later[["store", "date", "future"]], on=["store", "date"])
    assert len(m) == len(frame) and np.allclose(m["sales"], m["future"])
    now = frame[["store", "date", "sales_lag7"]].merge(df[["store", "date", "sales"]], on=["store", "date"])
    assert np.allclose(now["sales_lag7"], now["sales"])  # lag 7 of the label is the value at the origin
    assert manifest["horizon"] == 7 and min(manifest["lags"]) == 7
    assert fc.leakage_audit(frame, "sales", "date", raw=df, manifest=manifest) == []


def test_features_unchanged_when_future_rows_deleted():
    df, full, manifest = _supervised()
    cut = df["date"].min() + pd.Timedelta(days=200)
    part, _ = fc.make_supervised(df[df["date"] <= cut], "sales", "date", ["store"], horizon=1,
                                 lags=manifest["lags"], windows=manifest["windows"])
    key = ["store", "date"]
    m = part.merge(full, on=key, suffixes=("_p", "_f"))
    assert len(m) == len(part) > 100
    rolling = [c for c in part if "_roll" in c]
    assert rolling
    for c in [c for c in part if c not in key and c != "sales"]:
        assert np.allclose(m[f"{c}_p"].astype(float), m[f"{c}_f"].astype(float), equal_nan=True), c


def test_unknown_future_columns_are_only_lagged():
    _, known, _ = _supervised()
    _, unknown, manifest = _supervised(unknown_future=["price"])
    assert "price" in known and "price" not in unknown
    assert "price_lag2" in unknown and "price_lag2" in manifest["features"]
    assert any("price" in a for a in manifest["assumptions"])


def test_naive_baselines_finite():
    _, frame, manifest = _supervised()
    b = fc.naive_baselines(frame, "sales", manifest)
    for name in ("last_value", "seasonal_naive"):
        assert math.isfinite(b[name]["mae"]) and math.isfinite(b[name]["rmse"]), b
        assert b[name]["rmse"] >= b[name]["mae"] > 0
    assert b["best"] == "seasonal_naive", b  # weekly season: last week beats yesterday
    assert b["test_rows"] == round(len(frame) * 0.2)


def test_monthly_series_named_month():
    rng = np.random.default_rng(1)
    t = pd.date_range("2015-01-01", periods=96, freq="MS")
    raw = pd.DataFrame({"month": t.strftime("%Y-%m-%d"),
                        "price": 50 + 5 * np.sin(2 * np.pi * np.arange(96) / 12) + rng.normal(0, 0.5, 96)})
    s = fc.detect_structure(raw, "price")
    assert s["freq"] == "MS" and s["seasonality"]["period"] == 12, s
    frame, manifest = fc.make_supervised(raw, "price", "month", [])
    assert pd.api.types.is_datetime64_any_dtype(frame["month"]) and "price_lag12" in frame
    assert fc.leakage_audit(frame, "price", "month", raw=raw, manifest=manifest) == []


def test_drifting_sales_end_to_end():
    if not os.path.exists(DRIFTING):
        subprocess.run([sys.executable, os.path.join(HERE, "evals", "make_golden.py")], check=True)
    raw = pd.read_csv(DRIFTING, parse_dates=["date"])
    s = fc.detect_structure(raw, "units")
    assert s["time_col"] == "date" and s["entity_cols"] == [] and s["freq"] == "D", s
    assert s["seasonality"]["period"] == 7, s
    frame, manifest = fc.make_supervised(raw, "units", s["time_col"], s["entity_cols"])
    assert len(frame) > 1000 and "units_lag7" in frame
    assert fc.leakage_audit(frame, "units", "date", raw=raw, manifest=manifest) == []
    assert sampling.sorted_time_column(frame, "units") == "date"  # factory picks up the time split
    b = fc.naive_baselines(frame, "units", manifest)
    assert all(math.isfinite(b[k][m]) for k in ("last_value", "seasonal_naive") for m in ("mae", "rmse")), b



def test_fit_until_keeps_the_season_choice_out_of_the_future():
    """Season / lag choice must not see the hidden period: a weekly cycle that only exists late is ignored."""
    n = 1000
    t = pd.date_range("2020-01-01", periods=n, freq="D")
    y = np.random.default_rng(3).normal(0, 1, n)
    y[700:] += 10 * np.sin(2 * np.pi * np.arange(300) / 7)
    df = pd.DataFrame({"date": t, "y": y})
    _, full = fc.make_supervised(df, "y", "date", [], horizon=1)
    _, cut = fc.make_supervised(df, "y", "date", [], horizon=1, fit_until=t[699])
    # full series: the late weekly cycle is detected by ACF; dev only: nothing to detect (7 is just the daily default)
    assert full["season_source"] == "acf" and cut["season_source"] != "acf"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
