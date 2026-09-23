"""
Harness tools for FactoryRun: the math decides, the LLM judges.

  diag_summary / diag_run   33 statistical checks on the dev split (hidden rows never touched)
  run_search                thousands of pipelines -> prior-ranked -> racing on Modal with sized subsamples
  confirm_and_test          10-fold paired tests on the top-k, then ONE hidden-holdout test, Pareto ranking
  data_search / data_try    Tavily search, then a measured enrich trial (never a claimed gain)
Mixed into agent.FactoryRun; relies on self.df, self.target, self.emit, self.fns, self.specs.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

import diagnostics as dg
import external_data as xd
import memory
import racing
import sampling
import search_space as ss
import stats_tests as st
from export import export_bundle
from modal_train import upload_dataset

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
COMPACT_KEEP = 2     # tool results kept verbatim in context; older ones are truncated


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


class PipelineTools:
    """Tool methods added to FactoryRun. Each takes the tool input dict and returns a JSON-able dict."""

    def setup_pipeline(self, task: str, purpose: str | None) -> None:
        """Lock the hidden split first, then diagnose the dev split only."""
        self.task, self.purpose = task, purpose
        idx = sampling.split_three(self.df, self.target, task)
        self.dev_df, self.val_df, self.hidden_df = (self.df.iloc[idx[k]] for k in ("dev", "search_val", "hidden"))
        self.ctx = dg.make_context(self.dev_df.reset_index(drop=True), self.target, task)
        self.diag = dg.run_all(self.ctx)
        self.dev_path = self.hidden_path = None
        self.search = self.confirmed = self.export = None
        self.memory_path = None

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
        return out + (list(self.fns["train_candidate_gpu"].map(gpu)) if gpu else [])

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
               "rows": {"dev": len(self.dev_df), "search_val": len(self.val_df), "hidden_locked": len(self.hidden_df)}}
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
        veto = set(inp.get("veto_families") or [])
        meta = self.diag["meta"]
        n_dev, n_feat = len(self.dev_df), self.df.shape[1] - 1 - len(inp.get("drop_columns") or [])
        space = ss.build_space(task, n_dev, n_feat, meta, inp.get("purpose") or {}, budget=SPACE_BUDGET)
        warm = memory.warm_start_configs(meta, task, path=self.memory_path)
        pool, seen = [], set()
        for c in warm + [c for c in space if c.get("family", c.get("model")) not in veto]:
            if c["name"] not in seen and c.get("model") in ss.FAMILIES[task]:
                seen.add(c["name"])
                pool.append(c)
        configs = pool[:RACE_CONFIGS]
        sizing = sampling.choose_n(n_dev, len(configs))
        n_star = max(min(N_MIN, n_dev), sizing["n_star"])
        schedule = racing.rung_schedule(len(configs), min(N_MIN, n_star), n_star, eta=3, top_k=TOP_K)
        base = {"dataset_path": self._dev(), "target": self.target, "task": task, "cv": inp.get("cv", "kfold"),
                "cv_folds": 5, "time_column": inp.get("time_column"), "drop_columns": inp.get("drop_columns") or []}
        by_name = {c["name"]: c for c in configs}

        def evaluate(cfgs, n_rows):
            res = self._map([self._spec(base, c, n_rows) for c in cfgs])
            return [{"name": r["name"], "ok": True, "folds": r["metrics"][pm]["folds"], "fit_seconds": r["fit_seconds"]}
                    if r.get("ok") else {"name": r.get("name"), "ok": False} for r in res]

        self.emit("search_plan", {"space": len(space), "raced": len(configs), "warm_start": len(warm),
                                  "n_star": n_star, "schedule": schedule, "rationale": inp.get("rationale")})
        out = racing.race(configs, evaluate, schedule, higher_is_better=HIGHER_IS_BETTER[pm],
                          budget_fits=MAX_FITS, on_rung=lambda s: self.emit("rung", s))
        for t in out["top"]:
            self.specs[t["name"]] = self._spec(base, by_name[t["name"]], n_star)
        self.search = {"pm": pm, "task": task, "base": base, "n_star": n_star, "top": out["top"], "by_name": by_name}
        top = [{"name": t["name"], "family": by_name[t["name"]].get("family", by_name[t["name"]]["model"]), "mean": _r(t["mean"]), "std": _r(t["std"]),
                "params": by_name[t["name"]].get("params")} for t in out["top"]]
        self.emit("leaderboard", {"round": "race", "primary_metric": pm, "rows": top})
        return {"space_size": len(space), "raced": len(configs), "warm_start": len(warm), "n_star": n_star,
                "justification": sizing["justification"], "schedule": schedule, "fits": out["fits"],
                "stopped": out["stopped"], "rungs": out["rungs"], "top": top}

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
        n_test, n_train = n / 5, n * 4 / 5
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
        self.confirmed = {"pm": pm, "rows": {t["name"]: t for t in table}, "pick": pick}
        out = {"primary_metric": pm, "table": table, "recommendation": pick, "tie_groups": groups,
               "notes": "p_vs_best: Nadeau-Bengio corrected t on 10 paired folds; hidden: one-shot holdout."}
        self.emit("confirm", out)
        return out

    def _hidden_test(self, specs: list[dict], pm: str, task: str) -> dict:
        """Fit each spec on the full dev split, score the locked hidden rows once."""
        if self.hidden_path is None:
            self.hidden_path = upload_dataset(self.hidden_df)
        preds = {}
        for s in specs:
            full = {k: v for k, v in s.items() if k != "train_rows"}
            fn = self.fns["predict_holdout_gpu" if s["model"] in GPU_MODELS else "predict_holdout"]
            try:
                preds[s["name"]] = fn.remote(full, self.hidden_path)
            except Exception as e:
                self.emit("status", {"msg": f"hidden test failed for {s['name']}: {type(e).__name__}"})
        out = {k: {"score": _r(_hidden_score(pm, p["y_true"], p["pred"], p.get("proba")))} for k, p in preds.items()}
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

    # ---- external data
    def tool_data_search(self, inp):
        """Tavily search for public datasets; returns candidates only, nothing is claimed."""
        return {"results": xd.search(inp["query"], inp.get("purpose", "enrich"))}

    def tool_data_try(self, inp):
        """Fetch one public file, left-join new columns onto dev, and test the current best on base vs enriched."""
        if not self.search:
            return {"error": "run_search first; data_try compares against the current best config."}
        links = xd.find_file_links(inp["url"])
        if not links:
            return {"error": "no csv/parquet link found at that url; pass a direct file url."}
        ext = xd.safe_fetch(links[0])
        aug, rep = xd.enrich(self.dev_df, ext, inp["left_key"], inp["right_key"], inp.get("columns"), self.target)
        best = self.specs[self.search["top"][0]["name"]]
        pm = self.search["pm"]
        base_s = {**best, "name": "base", "repeats": CONFIRM_REPEATS}
        aug_s = {**best, "name": "enriched", "repeats": CONFIRM_REPEATS, "dataset_path": upload_dataset(aug)}
        res = {r["name"]: r for r in self._map([base_s, aug_s]) if r.get("ok")}
        if len(res) < 2:
            return {"error": "a trial fit failed", "report": rep}
        a, b = res["enriched"]["metrics"][pm]["folds"], res["base"]["metrics"][pm]["folds"]
        if not HIGHER_IS_BETTER[pm]:
            a, b = b, a
        n = self.search["n_star"]
        test = st.nadeau_bengio(a, b, n * 4 / 5, n / 5)
        verdict = "improves" if test["p_one_sided"] < 0.05 else ("worse" if test["p_one_sided"] > 0.95 else "no_gain")
        out = {"source": links[0], "verdict": verdict, "delta": test["mean_diff"], "ci": [test["ci_low"], test["ci_high"]],
               "p_one_sided": test["p_one_sided"], **rep}
        self.emit("external_trial", out)
        return out

    # ---- export + memory
    def export_user_model(self, name: str) -> dict:
        """Standalone train script + params + model card the end user keeps."""
        row = (self.confirmed or {}).get("rows", {}).get(name, {})
        metrics = {k: row.get(k) for k in ("cv_mean", "ci", "hidden", "p_vs_best", "ece") if row.get(k) is not None}
        spec = {**self.specs[name], "primary_metric": (self.search or {}).get("pm")}
        extra = {"complexity": row.get("big_o")} if row.get("big_o") else None
        self.export = export_bundle(spec, metrics, f"runs/{self.run_id}_export", purpose=self.purpose,
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
