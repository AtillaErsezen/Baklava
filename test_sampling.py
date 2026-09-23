import math

import numpy as np
import pandas as pd

import sampling as s


def _clf_df(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.exponential(size=n),
        "cat": rng.choice(list("abcd"), size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "t": np.arange(n),
        "y": (rng.random(n) < 0.1).astype(int),
    })


def test_split_three():
    df = _clf_df()
    sp = s.split_three(df, "y", "classification")
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == len(df) and len(np.unique(allidx)) == len(df)
    base = df["y"].mean()
    for k, idx in sp.items():
        assert abs(df["y"].iloc[idx].mean() - base) < 0.02, k
    tm = s.split_three(df.sample(frac=1, random_state=1).reset_index(drop=True), "y", "classification", time_column="t")
    assert len(np.unique(np.concatenate(list(tm.values())))) == len(df)
    assert tm["hidden"].size == 1000 and tm["search_val"].size == 500


def test_split_three_time_hidden_latest():
    df = _clf_df().sample(frac=1, random_state=3).reset_index(drop=True)
    sp = s.split_three(df, "y", "classification", time_column="t")
    t = df["t"].to_numpy()
    assert t[sp["hidden"]].min() > t[sp["search_val"]].max() > t[sp["dev"]].max()


def test_split_three_regression():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=2000), "y": rng.lognormal(size=2000)})
    sp = s.split_three(df, "y", "regression")
    assert sum(v.size for v in sp.values()) == 2000
    assert len(np.unique(np.concatenate(list(sp.values())))) == 2000



def test_split_three_singleton_class_keeps_other_strata():
    """One row of a third class makes stratify raise; the rest must stay stratified, not go fully random."""
    rng = np.random.default_rng(5)
    y = np.array(["no"] * 980 + ["yes"] * 19 + ["odd"], dtype=object)
    df = pd.DataFrame({"x": rng.normal(size=1000), "y": y})
    sp = s.split_three(df, "y", "classification")
    assert sorted(np.concatenate(list(sp.values())).tolist()) == list(range(1000))
    for k, want in (("hidden", 4), ("search_val", 2), ("dev", 13)):
        got = int((df["y"].iloc[sp[k]] == "yes").sum())
        assert abs(got - want) <= 1, (k, got)

def test_hoeffding_n():
    # ceil(ln(2*1000/0.05) / (2*0.02^2)) = ceil(10.5966 / 0.0008) = ceil(13245.7)
    assert s.hoeffding_n(0.02, 0.05, 1000) == math.ceil(math.log(40000) / 0.0008) == 13_246
    assert s.hoeffding_n(0.05, 0.05) == 738


def test_precision_n():
    assert s.precision_n(0.1, 0.01) == 385


def test_hanley_mcneil_se():
    a, n1, n0 = 0.8, 100, 100
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    exp = math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    got = s.hanley_mcneil_se(a, n1, n0)
    assert abs(got - exp) < 1e-12 and abs(got - 0.0314) < 1e-3


def test_fit_power_law():
    ns = np.array([100, 300, 1000, 3000, 10000])
    errs = 0.1 + 2 * ns ** -0.5
    f = s.fit_power_law(ns, errs)
    assert f.ok
    for got, want in ((f.a, 0.1), (f.b, 2), (f.c, 0.5)):
        assert abs(got - want) / want < 0.05, (got, want)
    bad = s.fit_power_law([100, 200], [0.3, 0.2])
    assert not bad.ok and bad.a == 0.2 and bad.b == 0


def test_knee_n_monotone():
    k1, k2 = s.knee_n(0.1, 2, 0.5, 0.001, 1_000_000), s.knee_n(0.1, 2, 0.5, 0.01, 1_000_000)
    assert k2 < k1 <= 1_000_000
    assert s.knee_n(0.1, 2, 0.5, 1e-12, 50_000) == 50_000


def test_stratified_sample():
    df = _clf_df(10_000)
    for n in (500, 1234):
        idx = s.stratified_sample(df, "y", "classification", n, extra_strata=["cat"], time_column="t")
        assert idx.size == n and np.unique(idx).size == n
        assert abs(df["y"].iloc[idx].mean() - df["y"].mean()) < 0.01
    assert s.stratified_sample(df, "y", "classification", 10**6).size == len(df)
    r = s.stratified_sample(df, "x1", "regression", 777)
    assert r.size == 777 and np.unique(r).size == 777


def test_representativeness():
    df = _clf_df(6000)
    good = s.representativeness(df.sample(1000, random_state=0), df)
    assert good["ok"], good
    bad = s.representativeness(df.sort_values("x1").head(1000), df)
    assert not bad["ok"] and "x1" in bad["rejected_columns"] and bad["adversarial_auc"] > 0.9


def test_choose_n():
    out = s.choose_n(50_000, 20, pilot_sigma=0.3, curve=(0.1, 2, 0.5))
    assert str(out["n_star"]) in out["justification"]
    assert out["n_eval"] >= s.hoeffding_n(0.02, 0.05, 20)
    assert out["n_star"] == min(50_000, max(out["n_eval"], out["n_knee"]))
    assert s.choose_n(500, 20)["n_star"] == 500
    assert "\u2014" not in out["justification"] and "\u2013" not in out["justification"]


def test_choose_n_hoeffding_is_a_rough_guide():
    why = s.choose_n(2099, 243)["justification"]
    assert s.hoeffding_n(0.02, 0.05, 243) == 11478
    assert why.startswith("Use n_star = 2099 of 2099 dev rows because Hoeffding with eps = 0.02, delta = 0.05 "
                          "and 243 configs suggests about 11478 rows as a rough guide (CV estimates are correlated, "
                          "so this is approximate)"), why
    assert "configs needs" not in why and "capped at the dev size" in why


def test_draw_verified():
    df = _clf_df(6000)
    out = s.draw_verified(df, "y", "classification", 1500)
    assert out["indices"].size == 1500 and out["report"]["ok"] and 1 <= out["tries"] <= 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
