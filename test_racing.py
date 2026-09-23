import numpy as np

import racing as r

N_MIN, N_MAX = 1000, 81000


def _fake(means: dict[str, float], seed: int, sigma: float = 0.01, fail: set[str] = frozenset()):
    rng = np.random.default_rng(seed)

    def evaluate(configs: list[dict], n_rows: int) -> list[dict]:
        sd = sigma * np.sqrt(1000 / n_rows)
        return [{"name": c["name"], "ok": c["name"] not in fail,
                 "folds": list(means[c["name"]] + rng.normal(0, sd, 5)), "fit_seconds": n_rows / 1e4}
                for c in configs]
    return evaluate


def _pool(n: int = 200):
    means = {f"c{i}": float(m) for i, m in enumerate(np.linspace(0.6, 0.9, n))}
    return [{"name": k} for k in means], means


def test_rung_schedule():
    s = r.rung_schedule(200, N_MIN, N_MAX)
    assert [x["n_rows"] for x in s] == [1000, 3000, 9000, 27000, 81000]
    assert s[-1]["n_rows"] == N_MAX and s[-1]["keep"] == 3 and s[0]["keep"] == 67
    assert r.rung_schedule(10, 1000, 5000)[-1]["n_rows"] == 5000
    assert r.schedule_cost(s, 200) == 200 * 1000 + 67 * 3000 + 23 * 9000 + 8 * 27000 + 3 * 81000


def test_true_best_in_top3():
    configs, means = _pool()
    s = r.rung_schedule(200, N_MIN, N_MAX)
    hits = 0
    for seed in range(10):
        out = r.race(configs, _fake(means, seed), s)
        hits += "c199" in [t["name"] for t in out["top"]]
        assert out["fits"] < 200 * len(s) / 3, out["fits"]
        assert len(out["top"]) == 3 and out["stopped"] in ("done", "tau")
    assert hits >= 9, hits


def test_protected_never_stat_dropped():
    means = {"lead": 0.9, **{f"p{i}": 0.5 for i in range(5)}, **{f"x{i}": 0.5 for i in range(10)}}
    configs = [{"name": k} for k in means]
    out = r.race(configs, _fake(means, 0, sigma=0.001), r.rung_schedule(16, N_MIN, N_MAX), protect=6)
    # 15 ties at 0.5 are all clearly worse; only 5 of them can hold ranks 1..5, so exactly 10 stat drops at rung 0
    stat = [k for k, (_, why) in out["dropped"].items() if why == "stat"]
    assert len(stat) == 10, out["dropped"]
    assert out["rungs"][0]["survivors"] == 6


def test_clearly_worse_dropped_by_stat_at_first_rung():
    means = {**{f"g{i}": 0.9 - 0.001 * i for i in range(8)}, "bad": 0.5}
    configs = [{"name": k} for k in means]
    out = r.race(configs, _fake(means, 1, sigma=0.002), r.rung_schedule(9, N_MIN, N_MAX), protect=2)
    assert out["dropped"]["bad"] == (0, "stat")


def test_budget_stops():
    configs, means = _pool()
    out = r.race(configs, _fake(means, 0), r.rung_schedule(200, N_MIN, N_MAX), budget_fits=250)
    assert out["stopped"] == "budget" and out["fits"] <= 250 and len(out["top"]) == 3


def test_failed_configs_removed():
    configs, means = _pool(30)
    seen = []
    out = r.race(configs, _fake(means, 0, fail={"c29", "c3"}), r.rung_schedule(30, N_MIN, N_MAX), on_rung=seen.append)
    assert out["dropped"]["c29"] == (0, "failed") and "c29" not in [t["name"] for t in out["top"]]
    assert seen and set(seen[0]) >= {"rung", "n_rows", "evaluated", "survivors", "dropped_stat", "dropped_rank",
                                     "leader", "leader_mean", "tau"}


def test_lower_is_better():
    configs, means = _pool(30)
    out = r.race(configs, _fake({k: -v for k, v in means.items()}, 0), r.rung_schedule(30, N_MIN, N_MAX),
                 higher_is_better=False)
    assert out["top"][0]["name"] in {"c29", "c28", "c27"}


def test_fit_time_exponent():
    ns = [1000, 3000, 9000, 27000]
    assert abs(r.fit_time_exponent([(n, n ** 1.5) for n in ns]) - 1.5) < 1e-9


def test_corrected_t():
    p = r._corrected_t_one_sided([0.9, 0.91, 0.89, 0.9, 0.92], [0.5, 0.51, 0.49, 0.5, 0.52], 800, 200)
    assert p == 0.0 or p < 1e-6
    assert r._corrected_t_one_sided([0.5, 0.52, 0.49, 0.5, 0.51], [0.9, 0.91, 0.89, 0.9, 0.92], 800, 200) > 0.99



def test_race_accepts_the_cv_test_train_ratio():
    """A larger ratio (walk-forward) inflates the variance correction, so borderline configs survive."""
    import inspect
    assert "test_train_ratio" in inspect.signature(r.race).parameters
    a = [0.80, 0.82, 0.81, 0.79, 0.83]
    b = [0.78, 0.80, 0.80, 0.78, 0.80]
    t1 = r._corrected_t_one_sided
    assert t1(a, b, 1.0, 0.4567) > t1(a, b, 4.0, 1.0)   # p grows with the ratio


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
