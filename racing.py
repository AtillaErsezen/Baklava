"""Statistically guarded successive halving: schedules rungs and decides who survives. Training lives elsewhere."""
import math
from typing import Callable

import numpy as np
from scipy import stats


def rung_schedule(n_configs: int, n_min: int, n_max: int, eta: int = 3, top_k: int = 3) -> list[dict]:
    """Rows grow by eta from n_min to exactly n_max; keep ~ ceil(survivors/eta), final rung keeps top_k."""
    rows, n = [], n_min
    while n < n_max:
        rows.append(int(n))
        n *= eta
    rows.append(int(n_max))
    out, surv = [], n_configs
    for i, nr in enumerate(rows):
        keep = top_k if i == len(rows) - 1 else max(top_k, math.ceil(surv / eta))
        out.append({"rung": i, "n_rows": nr, "keep": keep})
        surv = keep
    return out


def schedule_cost(schedule: list[dict], n_configs: int) -> int:
    """Planned rows trained: sum of configs at each rung times its n_rows."""
    cost, surv = 0, n_configs
    for s in schedule:
        cost += surv * s["n_rows"]
        surv = s["keep"]
    return cost


def _corrected_t_one_sided(a, b, n_train: int, n_test: int) -> float:
    """Nadeau-Bengio corrected paired t; one-sided p that a > b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    j = len(d)
    var = (1 / j + n_test / n_train) * d.var(ddof=1) if j > 1 else 0.0
    if var <= 0:
        return 0.0 if d.mean() > 0 else 1.0
    return float(stats.t.sf(d.mean() / math.sqrt(var), df=j - 1))


def fit_time_exponent(history: list[tuple[int, float]]) -> float:
    """Log-log slope of fit seconds vs rows (measured Big-O exponent)."""
    pts = np.array([(n, s) for n, s in history if n > 0 and s > 0], float)
    return float(np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1)[0])


def _optimistic(hist: list[tuple[int, float]], n_max: int) -> float:
    """LCCV-style: linear in log n extrapolation of signed mean, slope clipped >= 0."""
    x, y = np.log([h[0] for h in hist]), np.array([h[1] for h in hist])
    slope = max(0.0, float(np.polyfit(x, y, 1)[0]))
    return float(y[-1] + slope * (math.log(n_max) - x[-1]))


def race(configs: list[dict], evaluate: Callable[[list[dict], int], list[dict]], schedule: list[dict],
         higher_is_better: bool = True, alpha: float = 0.05, protect: int | None = None, n_folds: int = 5,
         budget_fits: int | None = None, tau_stop: float = 0.8, on_rung: Callable[[dict], None] | None = None) -> dict:
    """Successive halving with Bonferroni corrected-t drops, LCCV drops, Kendall-tau early stop and a fit budget."""
    sign = 1.0 if higher_is_better else -1.0
    if protect is None:
        eta = round(schedule[1]["n_rows"] / schedule[0]["n_rows"]) if len(schedule) > 1 else 3
        protect = 2 * eta
    top_k, n_max = schedule[-1]["keep"], schedule[-1]["n_rows"]
    by_name = {c["name"]: c for c in configs}
    survivors = [c["name"] for c in configs]
    hist: dict[str, list[tuple[int, float]]] = {}
    last: dict[str, dict] = {}
    dropped: dict[str, tuple[int, str]] = {}
    rungs, fits, stopped, prev_rank, i = [], 0, "done", None, 0
    while i < len(schedule) and survivors:
        sched = schedule[i]
        n_rows = sched["n_rows"]
        if budget_fits is not None and fits + len(survivors) > budget_fits:
            if last:
                stopped = "budget"
                break
            survivors = survivors[:budget_fits - fits]
        results = evaluate([by_name[n] for n in survivors], n_rows)
        fits += len(results)
        cur = {}
        for res in results:
            if not res.get("ok") or not res.get("folds"):
                dropped[res["name"]] = (i, "failed")
                continue
            f = np.asarray(res["folds"], float)
            cur[res["name"]] = {"name": res["name"], "mean": float(f.mean()), "std": float(f.std(ddof=1)) if len(f) > 1 else 0.0,
                                "folds": list(map(float, f)), "n_rows": n_rows}
            hist.setdefault(res["name"], []).append((n_rows, float(f.mean())))
        order = sorted(cur, key=lambda n: -sign * cur[n]["mean"])
        if not order:
            survivors, last = [], {}
            break
        lead = cur[order[0]]
        lead_folds = sign * np.asarray(lead["folds"])
        n_test = n_rows / n_folds
        m = len(order)
        keep, n_stat, n_rank = [], 0, 0
        for rank, name in enumerate(order):
            c = cur[name]
            if rank < protect:
                keep.append(name)
                continue
            cfolds = sign * np.asarray(c["folds"])
            p = _corrected_t_one_sided(lead_folds, cfolds, n_rows - n_test, n_test) if len(cfolds) == len(lead_folds) else 1.0
            signed_hist = [(n, sign * v) for n, v in hist[name]]
            lccv = len(signed_hist) >= 2 and _optimistic(signed_hist, n_max) < sign * lead["mean"] - lead["std"]
            if p < alpha / m or lccv:
                dropped[name], n_stat = (i, "stat"), n_stat + 1
            elif rank >= sched["keep"]:
                dropped[name], n_rank = (i, "rank"), n_rank + 1
            else:
                keep.append(name)
        rank_now = {n: k for k, n in enumerate(order)}
        common = [n for n in order if prev_rank and n in prev_rank]
        tau = None
        if len(common) >= 2:
            t = stats.kendalltau([rank_now[n] for n in common], [prev_rank[n] for n in common]).statistic
            tau = None if np.isnan(t) else float(t)
        summary = {"rung": i, "n_rows": n_rows, "evaluated": len(results), "survivors": len(keep),
                   "dropped_stat": n_stat, "dropped_rank": n_rank, "leader": lead["name"],
                   "leader_mean": lead["mean"], "tau": tau}
        rungs.append(summary)
        if on_rung:
            on_rung(summary)
        survivors, last, prev_rank = keep, {n: cur[n] for n in keep}, rank_now
        final = i == len(schedule) - 1
        if not final and tau is not None and tau >= tau_stop and len(keep) <= 2 * top_k:
            stopped, i = "tau", len(schedule) - 1
        else:
            i += 1
    top = sorted(last.values(), key=lambda c: -sign * c["mean"])[:top_k]
    for c in top:
        c["history"] = hist[c["name"]]
    return {"top": top, "rungs": rungs, "fits": fits, "stopped": stopped, "dropped": dropped}
