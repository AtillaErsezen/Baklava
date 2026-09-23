"""
Harness tools for FactoryRun: the math decides, the LLM judges.

  diag_summary / diag_run   33 statistical checks on the dev split (hidden rows never touched)
  run_search                thousands of pipelines -> prior-ranked -> racing on Modal with sized subsamples
  confirm_and_test          10-fold paired tests on the top-k, then ONE hidden-holdout test, Pareto ranking
  data_search / data_try    Tavily search, then a measured enrich trial (never a claimed gain)
Mixed into agent.FactoryRun; relies on self.df, self.target, self.emit, self.fns, self.specs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

import diagnostics as dg
import ensemble as ens
import external_data as xd
import forecasting as fc
import memory
import racing
import sampling
import search_space as ss
import stats_tests as st
import usecases as uc
from export import export_bundle
from modal_train import GPU_ENABLED, check_name, check_params, upload_dataset
from results_store import candidate_row

HIGHER_IS_BETTER = {"roc_auc": True, "f1_macro": True, "accuracy": True, "r2": True, "rmse": False, "mae": False}
TASK_METRICS = {"classification": ("roc_auc", "f1_macro", "accuracy"), "regression": ("rmse", "mae", "r2")}
SPACE_BUDGET = 3000  # pipelines generated per run
RACE_CONFIGS = 243   # 3^5 raced after zero-cost prior ranking
N_MIN = 500          # rows at the first racing rung
MAX_FITS = 1500      # Modal CV jobs per search
TOP_K = 3
CONFIRM_REPEATS = 2  # 5-fold x 2 repeats = 10 paired folds
ROPE = 0.005         # practical-equivalence band for the Bayesian test
GPU_MODELS = {"tabicl"}
DEV_FRACTION = 0.7  # split_three: dev 70 / search_val 10 / hidden 20; time data keeps that order
USE_CASE_MIN_SCORE = 0.3  # TF-IDF + column match; below this the goal is too vague to steer the search
COMPACT_KEEP = 2     # tool results kept verbatim in context; older ones are truncated
# Modal list prices (modal.com/pricing, 2026-09): per core-second, per GiB-second, per L4-second.
MODAL_CORE_S, MODAL_GIB_S, MODAL_L4_S = 0.0000131, 0.00000222, 0.000222
CPU_FN_COST_S = 4 * MODAL_CORE_S + 8 * MODAL_GIB_S       # train_candidate: cpu=4, 8 GiB
GPU_FN_COST_S = MODAL_L4_S + 4 * MODAL_CORE_S + 16 * MODAL_GIB_S
# Properties for agent.py's data_try input_schema (left_key/right_key are then only required for enrich).
DATA_TRY_SCHEMA_ADDITIONS = {
    "mode": {"type": "string", "enum": ["enrich", "more_rows"],
             "description": "enrich (default): left-join new columns on a key. more_rows: add public rows with the "
                            "same columns (target included) to the training folds only; validation stays on your rows."},
    "column_map": {"type": "object", "additionalProperties": {"type": "string"},
                   "description": "more_rows only: rename external columns to this dataset's names, {external: ours}."},
}


def _r(v, d=4):
    return None if v is None else round(float(v), d)


def _hidden_score(metric: str, y, pred, proba) -> float | None:
    """Metric on the hidden holdout from per-row predictions."""
    y, pred = np.asarray(y), np.asarray(pred)
    if metric == "roc_auc":
        return roc_auc_score(y, proba) if proba is not None and len(set(y.tolist())) == 2 else None
    fn = {"accuracy": accuracy_score, "f1_macro": lambda a, b: f1_score(a, b, average="macro"), "r2": r2_score,
          "mae": mean_absolute_error, "rmse": lambda a, b: float(np.sqrt(mean_squared_error(a, b)))}[metric]
    return float(fn(y, pred))


def _pareto(rows: list[dict], sign: float) -> set[str]:
    """Names not dominated on (higher score, lower fit time)."""
    keep = set()
    for r in rows:
        dominated = any(sign * o["cv_mean"] >= sign * r["cv_mean"] and o["fit_s"] <= r["fit_s"]
                        and (o["cv_mean"] != r["cv_mean"] or o["fit_s"] != r["fit_s"]) for o in rows)
        if not dominated:
            keep.add(r["name"])
    return keep


def _safe_config(c: dict) -> bool:
    """Drop configs (e.g. from stored memory) with unsafe names or non-allowlisted params."""
    try:
        check_name(c["name"])
        check_params(c["model"], dict(c.get("params") or {}))
        return True
    except (ValueError, KeyError):
        return False


class PipelineTools:
    """Tool methods added to FactoryRun. Each takes the tool input dict and returns a JSON-able dict."""

    def setup_pipeline(self, task: str, purpose: str | None) -> None:
        """Understand the goal, reshape forecasting data, lock the hidden split, then diagnose dev only."""
        self.task, self.purpose, self.forecast = task, purpose, None
        self.use_case = self._match_use_case(purpose)
        if self.use_case and self.use_case.get("time") == "forecast":
            self._to_forecast_table()
        self.time_column = sampling.sorted_time_column(self.df, self.target)
        idx = sampling.split_three(self.df, self.target, task, time_column=self.time_column)
        self.dev_df, self.val_df, self.hidden_df = (self.df.iloc[idx[k]] for k in ("dev", "search_val", "hidden"))
        self.ctx = dg.make_context(self.dev_df.reset_index(drop=True), self.target, task, time_column=self.time_column)
        self.diag = dg.run_all(self.ctx)
        self._use_case_findings()
        dups = sampling.cross_split_duplicates(self.dev_df, self.hidden_df, self.target)
        if dups:
            self.diag["findings"].insert(0, {"check": "cross_split_duplicates", "severity": 2,
                                             "finding": f"{dups} hidden rows repeat a dev row exactly; hidden scores "
                                                        "may be optimistic. Deduplicate before trusting them."})
        self.dev_path = self.hidden_path = self.val_path = self.full_data_path = None
        self.search = self.confirmed = self.export = self.purpose_spec = None
        self.memory_path = None
        self.modal_seconds = {"cpu": 0.0, "gpu": 0.0}
        self.external = []

    def _match_use_case(self, purpose: str | None) -> dict | None:
        """Token-free goal understanding: best catalog use case if confident enough."""
        if not purpose:
            return None
        hits = uc.match(purpose, [str(c) for c in self.df.columns], task_hint=self.task)
        return hits[0] if hits and hits[0].get("score", 0) >= USE_CASE_MIN_SCORE else None

    def _to_forecast_table(self) -> None:
        """Forecasting goal: lag / rolling / calendar features with no look-ahead, audited before use.
        Everything chosen from data (entity columns, frequency, season, lags, windows, the naive bar) looks only
        at rows up to the end of the future dev split, so the hidden period never shapes what the agent sees."""
        time_col = fc.detect_structure(self.df, self.target).get("time_col")
        if not time_col:
            return
        times = np.sort(pd.to_datetime(self.df[time_col]).to_numpy())
        fit_until = pd.Timestamp(times[max(int(len(times) * DEV_FRACTION) - 1, 0)])
        raw_dev = self.df[pd.to_datetime(self.df[time_col]) <= fit_until]
        s = fc.detect_structure(raw_dev, self.target, time_col)
        try:
            frame, man = fc.make_supervised(self.df, self.target, time_col, s["entity_cols"], horizon=1,
                                            fit_until=fit_until)
            issues = fc.leakage_audit(frame, self.target, time_col, raw=self.df, manifest=man)
        except (ValueError, KeyError) as e:
            self.forecast = {"skipped": f"{type(e).__name__}: {e}"[:200]}
            return
        if issues:
            self.forecast = {"skipped": "leakage audit failed", "issues": issues[:5]}
            return
        frame_times = pd.to_datetime(frame[time_col])
        naive = fc.naive_baselines(frame[frame_times <= fit_until], self.target, man)  # tail of dev, never hidden
        self.forecast = {"structure": s, "manifest": man, "naive": naive, "fit_until": fit_until.isoformat()}
        self.df, self.task = frame.reset_index(drop=True), "regression"
        self.df[time_col] = pd.to_datetime(self.df[time_col])

    def _use_case_findings(self) -> None:
        """Surface use-case leakage suspects and the naive forecast bar as diagnostics findings."""
        if self.use_case:
            sus = [c for c in uc.leakage_suspects(self.use_case, [str(c) for c in self.dev_df.columns]) if c != self.target]
            if sus:
                self.diag["findings"].insert(0, {"check": "use_case_leakage", "severity": 2, "finding":
                    f"For {self.use_case['name']}, columns like {sus[:4]} are usually known only after the outcome; "
                    "confirm with inspect_column before training on them."})
        naive = (self.forecast or {}).get("naive")
        if naive and naive.get("best"):
            b = naive[naive["best"]]
            self.diag["findings"].append({"check": "naive_baseline", "severity": 1, "finding":
                f"{naive['best'].replace('_', ' ')} baseline: MAE {b['mae']:.4g}, RMSE {b['rmse']:.4g} on the latest 20%. "
                "A model only helps if it beats this."})

    # ---- plumbing
    def _dev(self) -> str:
        if self.dev_path is None:
            self.emit("status", {"msg": "Uploading dev split to Modal (hidden rows stay local)"})
            self.dev_path = upload_dataset(self.dev_df)
        return self.dev_path

    def _map(self, specs: list[dict]) -> list[dict]:
        """CV specs in parallel; foundation models go to the GPU function."""
        cpu = [s for s in specs if s["model"] not in GPU_MODELS]
        gpu = [s for s in specs if s["model"] in GPU_MODELS]
        out = list(self.fns["train_candidate"].map(cpu)) if cpu else []
        out_gpu = list(self.fns["train_candidate_gpu"].map(gpu)) if gpu else []
        self.modal_seconds["cpu"] += sum(r.get("fit_seconds") or 0 for r in out)
        self.modal_seconds["gpu"] += sum(r.get("fit_seconds") or 0 for r in out_gpu)
        return out + out_gpu

    def modal_usd(self) -> float:
        """Approximate Modal spend from measured container seconds (list prices)."""
        return round(self.modal_seconds["cpu"] * CPU_FN_COST_S + self.modal_seconds["gpu"] * GPU_FN_COST_S, 4)

    def _upload_once(self, attr: str, df) -> str:
        if getattr(self, attr) is None:
            setattr(self, attr, upload_dataset(df))
        return getattr(self, attr)

    def _spec(self, base: dict, c: dict, n_rows: int | None) -> dict:
        spec = {**base, "name": c["name"], "model": c["model"], "params": c.get("params") or {},
                "preprocessing": c.get("preprocessing") or {}}
        if n_rows and n_rows < len(self.dev_df):
            spec["train_rows"] = int(n_rows)
        return spec

    # ---- diagnostics
    def tool_diag_summary(self, inp):
        """Ranked findings + meta-features from the 33-check battery (precomputed, free)."""
        out = {"findings": self.diag["findings"], "meta": {k: _r(v, 3) for k, v in self.diag["meta"].items()},
               "use_case": {k: self.use_case.get(k) for k in ("id", "name", "task", "time", "primary_metric", "score")}
               if self.use_case else None,
               "forecast": {"horizon": self.forecast["manifest"]["horizon"], "lags": self.forecast["manifest"]["lags"],
                            "cv": "walk_forward", "time_column": self.forecast["manifest"]["time_col"]}
               if (self.forecast or {}).get("manifest") else None,
               "rows": {"dev": len(self.dev_df), "search_val": len(self.val_df), "hidden_locked": len(self.hidden_df)},
               "split": f"time: latest 20% by {self.time_column}" if self.time_column else "random, stratified"}
        if self.purpose_spec:
            out["purpose_spec"] = self.purpose_spec
        self.emit("diagnostics", {"findings": out["findings"], "rows": out["rows"]})
        if inp.get("response_format") == "detailed":
            out["catalog"] = dg.catalog()
        return out

    def tool_diag_run(self, inp):
        """Re-run or drill into one check, optionally on specific columns."""
        return dg.run_check(inp["check"], self.ctx, inp.get("columns"))

    # ---- search
    def tool_run_search(self, inp):
        """Generate ~3000 pipelines, rank by the benchmark prior, race the top 243 on growing subsamples."""
        task, pm = inp["task"], inp["primary_metric"]
        if pm not in TASK_METRICS[task]:
            return {"error": f"metric '{pm}' does not fit task '{task}'; allowed: {list(TASK_METRICS[task])}"}
        veto = set(inp.get("veto_families") or []) | (set() if GPU_ENABLED else GPU_MODELS)
        meta = self.diag["meta"]
        n_dev, n_feat = len(self.dev_df), self.df.shape[1] - 1 - len(inp.get("drop_columns") or [])
        space = ss.build_space(task, n_dev, n_feat, meta, inp.get("purpose") or {}, budget=SPACE_BUDGET)
        warm = memory.warm_start_configs(meta, task, path=self.memory_path)
        pool, seen = [], set()
        for c in warm + [c for c in space if c.get("family", c.get("model")) not in veto]:
            if c.get("name") not in seen and c.get("model") in ss.FAMILIES[task] and _safe_config(c):
                seen.add(c["name"])
                pool.append(c)
        weights = (self.use_case or {}).get("families_prior") or {}
        if weights:  # the matched use case's typical winners get more of the race
            pool = sorted(pool, key=lambda c: -(c.get("prior") or 0) * weights.get(c.get("family", c.get("model")), 1.0))
        configs = pool[:RACE_CONFIGS]
        sizing = sampling.choose_n(n_dev, len(configs))
        n_star = max(min(N_MIN, n_dev), sizing["n_star"])
        schedule = racing.rung_schedule(len(configs), min(N_MIN, n_star), n_star, eta=3, top_k=TOP_K)
        man = (self.forecast or {}).get("manifest")
        time_col = inp.get("time_column") or (man["time_col"] if man else self.time_column)
        cv = inp.get("cv") or ("walk_forward" if time_col else "kfold")  # time-ordered data: never shuffle folds
        base = {"dataset_path": self._dev(), "target": self.target, "task": task, "cv": cv, "cv_folds": 5,
                "time_column": time_col,
                "drop_columns": inp.get("drop_columns") or []}
        if man and cv in ("walk_forward", "purged"):
            base["gap"] = int(man.get("cv_gap_rows") or 0)
        by_name = {c["name"]: c for c in configs}
        rung_of = {r["n_rows"]: r["rung"] for r in schedule}

        def evaluate(cfgs, n_rows):
            res = self._map([self._spec(base, c, n_rows) for c in cfgs])
            self.store.add_candidates([candidate_row(self.run_id, "race", by_name.get(r.get("name"), {}), r, pm,
                                                     rung=rung_of.get(n_rows), train_rows=n_rows) for r in res])
            return [{"name": r["name"], "ok": True, "folds": r["metrics"][pm]["folds"], "fit_seconds": r["fit_seconds"]}
                    if r.get("ok") else {"name": r.get("name"), "ok": False} for r in res]

        self.emit("search_plan", {"space": len(space), "raced": len(configs), "warm_start": len(warm),
                                  "n_star": n_star, "schedule": schedule, "rationale": inp.get("rationale"),
                                  "cv": base["cv"], "time_column": base["time_column"]})
        out = racing.race(configs, evaluate, schedule, higher_is_better=HIGHER_IS_BETTER[pm],
                          test_train_ratio=st.test_train_ratio(base["cv"], base["cv_folds"]),
                          budget_fits=MAX_FITS, on_rung=lambda s: self.emit("rung", s))
        for t in out["top"]:
            self.specs[t["name"]] = self._spec(base, by_name[t["name"]], n_star)
        self.search = {"pm": pm, "task": task, "base": base, "n_star": n_star, "top": out["top"], "by_name": by_name}
        top = [{"name": t["name"], "family": by_name[t["name"]].get("family", by_name[t["name"]]["model"]), "mean": _r(t["mean"]), "std": _r(t["std"]),
                "params": by_name[t["name"]].get("params")} for t in out["top"]]
        self.emit("leaderboard", {"round": "race", "primary_metric": pm, "rows": top})
        self._last_search = {"space_size": len(space), "raced": len(configs), "warm_start": len(warm),
                             "n_star": n_star, "justification": sizing["justification"], "schedule": schedule,
                             "fits": out["fits"], "stopped": out["stopped"], "rungs": out["rungs"], "top": top}
        return self._last_search

    # ---- confirmation + hidden test
    def tool_confirm_and_test(self, inp):
        """10 paired folds for the top-k, corrected t + Bayesian tests vs best, 1-SE pick, one hidden test."""
        if not self.search:
            return {"error": "run_search first; confirm_and_test needs its top-k."}
        pm, task, n = self.search["pm"], self.search["task"], self.search["n_star"]
        sign, names = (1.0 if HIGHER_IS_BETTER[pm] else -1.0), [t["name"] for t in self.search["top"][:inp.get("k", TOP_K)]]
        specs = [{**self.specs[k], "repeats": CONFIRM_REPEATS, "measure_latency": True} for k in names]
        res = {r["name"]: r for r in self._map(specs) if r.get("ok")}
        if not res:
            return {"error": "every confirmation fit failed; check Modal logs."}
        rows = []
        for k, r in res.items():
            f = r["metrics"][pm]["folds"]
            fam = self.search["by_name"][k].get("family", self.search["by_name"][k]["model"])
            rows.append({"name": k, "family": fam, "folds": f, "mean": float(np.mean(f)), "std": float(np.std(f, ddof=1)),
                         "n_folds": len(f), "fit_s": r.get("fit_seconds"), "predict_ms": r.get("predict_ms")})
        best = max(rows, key=lambda r: sign * r["mean"])
        cv = self.search["base"]["cv"]
        n_train, n_test = 1.0, st.test_train_ratio(cv, self.search["base"]["cv_folds"])  # only the ratio matters
        pv, bayes = {}, {}
        for r in rows:
            if r["name"] != best["name"]:
                a, b = (best["folds"], r["folds"]) if sign > 0 else (r["folds"], best["folds"])
                pv[r["name"]] = st.nadeau_bengio(a, b, n_train, n_test)["p"]
                bayes[r["name"]] = st.bayes_correlated(a, b, ROPE)
        groups = st.tie_groups(rows, pv, higher_is_better=sign > 0)
        order = sorted(ss.COMPLEXITY, key=lambda f: ss.COMPLEXITY[f]["order"])
        pick = st.one_se_pick(rows, higher_is_better=sign > 0, complexity_order=order)
        hidden = self._hidden_test([self.specs[r["name"]] for r in rows], pm, task)
        table = []
        for r in rows:
            se = r["std"] * np.sqrt(1 / r["n_folds"] + n_test / n_train)
            h = hidden.get(r["name"], {})
            table.append({"name": r["name"], "family": r["family"], "cv_mean": _r(r["mean"]),
                          "ci": [_r(r["mean"] - 1.96 * se), _r(r["mean"] + 1.96 * se)],
                          "p_vs_best": pv.get(r["name"]), "p_equiv_bayes": bayes.get(r["name"], {}).get("p_equiv"),
                          "tie_group": next(i for i, g in enumerate(groups) if r["name"] in g),
                          "hidden": h.get("score"), "hidden_p_vs_best": h.get("p_vs_best"), "ece": h.get("ece"),
                          "dev_to_hidden_gap": _r(h["score"] - r["mean"]) if h.get("score") is not None else None,
                          "fit_s": r["fit_s"], "predict_ms": r["predict_ms"],
                          "big_o": {k: ss.COMPLEXITY.get(r["family"], {}).get(k) for k in ("train", "predict")},
                          "params": self.search["by_name"][r["name"]].get("params")})
        front = _pareto(table, sign)
        for t in table:
            t["pareto"] = t["name"] in front
        table.sort(key=lambda t: -sign * t["cv_mean"])
        self.store.add_candidates([candidate_row(
            self.run_id, "confirm", self.search["by_name"][t["name"]], res[t["name"]], pm, train_rows=n,
            p_vs_best=t["p_vs_best"], tie_with_best=t["tie_group"] == table[0]["tie_group"], hidden_score=t["hidden"],
            hidden_p_vs_best=t["hidden_p_vs_best"], ece=t["ece"]) for t in table])
        self.confirmed = {"pm": pm, "rows": {t["name"]: t for t in table}, "pick": pick}
        out = {"primary_metric": pm, "table": table, "recommendation": pick, "tie_groups": groups,
               "ensemble": self._ensemble([self.specs[r["name"]] for r in rows], pm, task, best["name"]),
               "notes": "p_vs_best: Nadeau-Bengio corrected t on paired folds"
                        + (" (approximate for time-ordered folds, with a conservative walk-forward ratio)"
                           if cv in ("walk_forward", "timeseries", "purged") else "")
                        + "; hidden: one-shot holdout."}
        self._last_confirm = out
        self.emit("confirm", out)
        return out

    def _hidden_test(self, specs: list[dict], pm: str, task: str) -> dict:
        """Fit each spec on the full dev split, score the locked hidden rows once."""
        self._upload_once("hidden_path", self.hidden_df)
        preds = self._predict(specs, self.hidden_path)
        self._hidden_preds = preds
        out = {k: {"score": _r(_hidden_score(pm, p["y_true"], p["pred"], p.get("proba")))} for k, p in preds.items()}
        return self._hidden_stats(out, preds, pm, task)

    def _predict(self, specs: list[dict], path: str) -> dict:
        """Fit each spec on the full dev split and predict the frame at path (one call per spec)."""
        preds = {}
        for s in specs:
            full = {k: v for k, v in s.items() if k != "train_rows"}
            fn = self.fns["predict_holdout_gpu" if s["model"] in GPU_MODELS else "predict_holdout"]
            try:
                preds[s["name"]] = fn.remote(full, path)
            except Exception as e:
                self.emit("status", {"msg": f"prediction failed for {s['name']}: {type(e).__name__}"})
        return preds

    def _hidden_stats(self, out: dict, preds: dict, pm: str, task: str) -> dict:
        if not preds:
            return out
        sign = 1 if HIGHER_IS_BETTER[pm] else -1
        best = max((k for k in out if out[k]["score"] is not None), key=lambda k: sign * out[k]["score"], default=None)
        for k, p in preds.items():
            if p.get("proba") is not None and task == "classification":
                out[k]["ece"] = st.calibration(p["y_true"], p["proba"])["ece"]
            if best and k != best:
                b = preds[best]
                if pm == "roc_auc" and p.get("proba") is not None:
                    out[k]["p_vs_best"] = st.delong(b["y_true"], b["proba"], p["proba"])["p"]
                elif task == "classification":
                    out[k]["p_vs_best"] = st.mcnemar(b["y_true"], b["pred"], p["pred"])["p"]
                else:
                    err = lambda y, q: -float(np.mean(np.abs(np.asarray(y) - np.asarray(q))))
                    out[k]["p_vs_best"] = st.paired_bootstrap(err, b["y_true"], b["pred"], p["pred"])["p"]
        return out

    def _ensemble(self, specs: list[dict], pm: str, task: str, best: str) -> dict | None:
        """Caruana weights on the untouched search_val split, then one comparison on hidden."""
        hidden = getattr(self, "_hidden_preds", {})
        if len(specs) < 2 or len(hidden) < 2 or len(self.val_df) < 30:
            return None
        try:
            val = self._predict(specs, self._upload_once("val_path", self.val_df))
            key = "proba" if task == "classification" else "pred"
            names = [n for n in val if val[n].get(key) is not None and n in hidden and hidden[n].get(key) is not None]
            if len(names) < 2 or (task == "classification" and len(val[names[0]].get("classes") or []) != 2):
                return None
            hib = HIGHER_IS_BETTER[pm]
            fit = ens.caruana({n: np.asarray(val[n][key], float) for n in names},
                              np.asarray(val[names[0]]["y_true"]), pm, hib)
            cmp = ens.compare_on_hidden({n: np.asarray(hidden[n][key], float) for n in names},
                                        np.asarray(hidden[names[0]]["y_true"]), fit["weights"], best, pm, hib)
            out = {"weights": {k: _r(v, 3) for k, v in fit["weights"].items() if v > 0},
                   "val_score": _r(fit["val_score"]), **{k: _r(v) if isinstance(v, float) else v for k, v in cmp.items()}}
            self.emit("status", {"msg": f"ensemble tested on hidden: keep={out.get('keep')}"})
            return out
        except Exception as e:
            self.emit("status", {"msg": f"ensemble skipped: {type(e).__name__}"})
            return None

    def _full(self) -> str:
        """All rows (dev + search_val + hidden) on Modal, for the final model the user receives."""
        return self._upload_once("full_data_path", self.df)

    def report_state(self) -> dict:
        """Everything report.build_report needs, taken from tool results only."""
        conf = getattr(self, "_last_confirm", None)
        return {
            "dataset": {"name": self.dataset_name, "rows": len(self.df), "features": self.df.shape[1] - 1,
                        "target": self.target, "task": self.task, "purpose": self.purpose},
            "split": {"dev": len(self.dev_df), "search_val": len(self.val_df), "hidden_locked": len(self.hidden_df),
                      "method": f"time: latest 20% by {self.time_column}" if self.time_column else "random, stratified"},
            "findings": self.diag["findings"],
            "dropped_columns": (self.search or {}).get("base", {}).get("drop_columns", []),
            "search": getattr(self, "_last_search", None), "confirm": conf, "final": self.final,
            "export": self.export, "external": self.external,
            "usage": {**self.client.ledger.totals(), "modal_usd": self.modal_usd()},
        }

    # ---- external data
    def tool_data_search(self, inp):
        """Tavily search for public datasets; returns candidates only, nothing is claimed."""
        return {"results": xd.search(inp["query"], inp.get("purpose", "enrich"))}

    def _more_rows_spec(self, best: dict, ext, inp) -> tuple[dict, dict]:
        """Align same-schema public rows to dev and upload them as training-only extra rows."""
        rows, rep = xd.align_more_rows(self.dev_df, ext, self.target, inp.get("column_map"))
        if not len(rows):
            raise ValueError("no new rows: every external row already appears in dev")
        spec = {**best, "name": "more_rows", "repeats": CONFIRM_REPEATS, "extra_train_path": upload_dataset(rows)}
        return spec, {"mode": "more_rows", "extra_rows": rep["n_rows"], "coverage": rep["coverage"],
                      "missing_columns": rep["missing_columns"]}

    def tool_data_try(self, inp):
        """Fetch one public file and test the current best on base vs augmented dev with paired folds.
        mode "enrich" (default) left-joins new columns onto dev; "more_rows" adds same-schema rows to the
        training folds only, so every score is still measured on the user's own rows."""
        if not self.search:
            return {"error": "run_search first; data_try compares against the current best config."}
        mode = inp.get("mode", "enrich")
        if mode not in DATA_TRY_SCHEMA_ADDITIONS["mode"]["enum"]:
            return {"error": f"mode must be one of {DATA_TRY_SCHEMA_ADDITIONS['mode']['enum']}, got {mode!r}"}
        links = xd.find_file_links(inp["url"])
        if not links:
            return {"error": "no csv/parquet link found at that url; pass a direct file url."}
        ext = xd.safe_fetch(links[0])
        best = self.specs[self.search["top"][0]["name"]]
        pm = self.search["pm"]
        base_s = {**best, "name": "base", "repeats": CONFIRM_REPEATS}
        if mode == "more_rows":
            aug_s, rep = self._more_rows_spec(best, ext, inp)
        else:
            aug, rep = xd.enrich(self.dev_df, ext, inp["left_key"], inp["right_key"], inp.get("columns"), self.target)
            aug_s = {**best, "name": "enriched", "repeats": CONFIRM_REPEATS, "dataset_path": upload_dataset(aug)}
        res = {r["name"]: r for r in self._map([base_s, aug_s]) if r.get("ok")}
        if len(res) < 2:
            return {"error": "a trial fit failed", "report": rep}
        if mode == "more_rows":  # rows actually trained on (labels the dev split never has are dropped remotely)
            rep["extra_rows"] = res[aug_s["name"]].get("extra_rows", rep["extra_rows"])
        a, b = res[aug_s["name"]]["metrics"][pm]["folds"], res["base"]["metrics"][pm]["folds"]
        if not HIGHER_IS_BETTER[pm]:
            a, b = b, a
        base = self.search["base"]
        test = st.nadeau_bengio(a, b, 1.0, st.test_train_ratio(base["cv"], base["cv_folds"]))
        verdict = "improves" if test["p_one_sided"] < 0.05 else ("worse" if test["p_one_sided"] > 0.95 else "no_gain")
        out = {"source": links[0], "verdict": verdict, "delta": test["mean_diff"], "ci": [test["ci_low"], test["ci_high"]],
               "p_one_sided": test["p_one_sided"], **rep}
        self.external.append(out)
        self.emit("external_trial", out)
        return out

    # ---- export + memory
    def export_user_model(self, name: str) -> dict:
        """Standalone train script + params + model card the end user keeps."""
        row = (self.confirmed or {}).get("rows", {}).get(name, {})
        metrics = {k: row.get(k) for k in ("cv_mean", "ci", "hidden", "p_vs_best", "ece") if row.get(k) is not None}
        spec = {**self.specs[name], "primary_metric": (self.search or {}).get("pm")}
        extra = {"complexity": row.get("big_o")} if row.get("big_o") else None
        self.export = export_bundle(spec, metrics, f"{getattr(self, 'runs_dir', 'runs')}/{self.run_id}_export", purpose=self.purpose,
                                    caveats=[f["finding"] for f in self.diag["findings"][:5]], extra=extra)
        self.emit("export", self.export)
        return self.export

    def remember(self) -> None:
        if self.search:
            winners = [{**self.search["by_name"][t["name"]], "score": t["mean"]} for t in self.search["top"]]
            memory.store({"run_id": self.run_id, "task": self.search["task"], "meta": self.diag["meta"],
                          "dataset_card": {"name": self.dataset_name, "rows": len(self.df)}, "winners": winners},
                         path=self.memory_path)


def compact(messages: list[dict], keep: int = COMPACT_KEEP, limit: int = 400) -> list[dict]:
    """Truncate all but the last `keep` tool results; full data lives in runs/ (Nebius has no prompt cache)."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool_idx[:-keep]) if keep else set(tool_idx)
    return [{**m, "content": m["content"][:limit] + " ...[truncated, full result saved in runs/]"}
            if i in old and len(m["content"]) > limit else m for i, m in enumerate(messages)]
