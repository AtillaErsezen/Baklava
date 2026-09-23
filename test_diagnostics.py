"""Tests for diagnostics.py. Run: uv run python test_diagnostics.py (or pytest)."""
import time
import numpy as np
import pandas as pd

import diagnostics as d

RNG = np.random.default_rng(0)


def _cls_frame(n: int = 2000) -> pd.DataFrame:
    x = RNG.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * RNG.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": y})


def test_leakage_planted():
    df = _cls_frame()
    flip = RNG.random(len(df)) < 0.01
    df["leak"] = np.where(flip, 1 - df["y"], df["y"])
    r = d.run_check("leakage", d.make_context(df, "y", "classification"))
    assert r["severity"] == 3, r
    assert "leak" in r["finding"], r


def test_nonlinearity_gap():
    n = 2000
    x = RNG.normal(size=(n, 2))
    lin = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": 2 * x[:, 0] - x[:, 1] + 0.1 * RNG.normal(size=n)})
    u = RNG.uniform(-3, 3, n)
    sin = pd.DataFrame({"x": u, "z": RNG.normal(size=n), "y": np.sin(3 * u) + 0.1 * RNG.normal(size=n)})
    g_lin = d.run_check("landmarkers", d.make_context(lin, "y", "regression"))["value"]["nonlinearity_gap"]
    g_sin = d.run_check("landmarkers", d.make_context(sin, "y", "regression"))["value"]["nonlinearity_gap"]
    assert g_lin < 0.1, g_lin
    assert g_sin > g_lin + 0.1, (g_lin, g_sin)


def test_imbalance():
    df = _cls_frame(2000)
    df["y"] = (RNG.random(len(df)) < 0.05).astype(int)
    r = d.run_check("imbalance", d.make_context(df, "y", "classification"))
    assert r["severity"] >= 2, r


def test_adversarial_drift():
    n = 2000
    t = np.linspace(0, 5, n)
    df = pd.DataFrame({"trend": t + RNG.normal(size=n) * 0.3, "noise": RNG.normal(size=n),
                       "y": RNG.integers(0, 2, n)})
    r = d.run_check("adversarial_drift", d.make_context(df, "y", "classification"))
    assert r["severity"] >= 2, r
    assert "trend" in r["finding"], r
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    r = d.run_check("adversarial_drift", d.make_context(shuffled, "y", "classification"))
    assert r["severity"] == 0, r


def test_id_like():
    df = _cls_frame()
    df["row_id"] = np.arange(len(df))
    r = d.run_check("id_like", d.make_context(df, "y", "classification"))
    assert r["severity"] >= 1 and "row_id" in r["value"], r


def test_errors_never_raise():
    @d.diagnostic("_boom", "quality", "always raises")
    def _boom(ctx, columns=None):
        raise RuntimeError("kaboom")
    try:
        ctx = d.make_context(_cls_frame(200), "y", "classification")
        r = d.run_check("_boom", ctx)
        assert r["severity"] == 0 and "kaboom" in r["error"], r
        r = d.run_check("nope", ctx)
        assert "error" in r and "leakage" in r["error"], r
    finally:
        d.DIAGNOSTICS.pop("_boom", None)


def test_run_all_churn():
    df = pd.read_csv("data/churn.csv")
    t0 = time.time()
    out = d.run_all(d.make_context(df, "Churn", "classification"))
    print(f"  run_all churn {time.time() - t0:.1f}s, {len(out['checks'])} checks")
    assert len(out["findings"]) <= 15
    assert out["checks"]["leakage"]["severity"] == 3 and "refund_issued" in out["checks"]["leakage"]["finding"]
    assert "customer_id" in out["checks"]["id_like"]["value"]
    assert all("error" not in r for r in out["checks"].values()), {k: r for k, r in out["checks"].items() if "error" in r}
    for k in ("n_rows", "n_features", "missing_rate", "imbalance_ratio", "nonlinearity_gap", "drift_auc"):
        assert k in out["meta"], k
    assert len(d.catalog()) >= 30



def test_high_cardinality_alone_is_not_drift():
    """~3 rows per level: halves separate by chance; the permutation null must absorb it."""
    rng = np.random.default_rng(3)
    n = 2800
    df = pd.DataFrame({"provider": "P" + pd.Series(rng.integers(0, 900, n)).astype(str),
                       "amount": rng.lognormal(7, 1, n), "y": rng.integers(0, 2, n)})
    ctx = d.make_context(df, "y", "classification")
    assert d.run_check("adversarial_drift", ctx)["severity"] == 0


def test_missing_placeholders_named():
    """Adult income style: '?' (and blanks, 'NA', 'null', '-') stand for missing in text columns."""
    df = _cls_frame(1000)
    df["work"] = RNG.choice(["gov", "private", " ?"], len(df), p=[0.45, 0.45, 0.1])
    df["country"] = RNG.choice(["us", "mx", "-", "null"], len(df), p=[0.8, 0.1, 0.05, 0.05])
    df["clean"] = RNG.choice(["a", "b"], len(df))
    r = d.run_check("missing_placeholders", d.make_context(df, "y", "classification"))
    assert r["severity"] >= 2 and set(r["value"]) == {"work", "country"}, r
    assert "work" in r["finding"] and "country" in r["finding"], r


def test_n_over_p_ignores_ids_and_caps_onehot():
    """Telco/Titanic: a unique customer id or a 600-level ticket column is not 600 one-hot features."""
    df = _cls_frame(700)
    df["customer_id"] = [f"C{i:05d}" for i in range(len(df))]
    df["ticket"] = RNG.integers(0, 600, len(df)).astype(str)
    r = d.run_check("n_over_p", d.make_context(df, "y", "classification"))
    assert r["severity"] == 0, r


def test_month_names_are_not_dates():
    """Bank marketing: 'may', 'jun' parse as year-1 dates in pandas but are a categorical."""
    df = _cls_frame(500)
    df["month"] = RNG.choice(["jan", "may", "jun", "oct"], len(df))
    ctx = d.make_context(df, "y", "classification")
    assert d.run_check("datetime_like", ctx)["severity"] == 0
    assert d.run_check("time_order", ctx)["severity"] == 0


def test_leakage_sum_of_two_columns():
    """Bike sharing: casual + registered = cnt; the two parts together are the target."""
    n = 2000
    casual = RNG.poisson(30, n) * RNG.integers(0, 3, n)
    registered = RNG.poisson(150, n) + 40 * RNG.integers(0, 4, n)
    df = pd.DataFrame({"temp": RNG.normal(size=n), "hour": RNG.integers(0, 24, n), "casual": casual,
                       "registered": registered, "cnt": casual + registered})
    r = d.run_check("leakage", d.make_context(df, "cnt", "regression"))
    assert r["severity"] == 3, r
    assert "casual" in r["finding"] and "registered" in r["finding"], r
    clean = df.assign(cnt=df["casual"] + df["registered"] + RNG.normal(0, 40, n))
    r = d.run_check("leakage", d.make_context(clean, "cnt", "regression"))
    assert "casual" not in r["finding"], r


def test_rare_classes_flagged():
    df = _cls_frame(1000)
    df["y"] = df["y"].astype(str)
    df.loc[:2, "y"] = "rare"
    r = d.run_check("rare_classes", d.make_context(df, "y", "classification"))
    assert r["severity"] >= 2 and r["value"] == {"rare": 3}, r
    assert d.run_check("rare_classes", d.make_context(_cls_frame(500), "y", "classification"))["severity"] == 0



def test_linear_leak_check_needs_enough_rows():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"a": rng.normal(size=6), "b": rng.normal(size=6), "y": rng.normal(size=6)})
    assert d._linear_leaks(d.make_context(df, "y", "regression")) == {}  # 6 rows: exact fits prove nothing


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
