"""Live eval harness: run the agent on the golden and public datasets and score every run from its events.

  uv run --env-file .env evals/run_golden.py [--sets golden|public|all] [--repeat 3] [--only name] [--dry]

Checks per run, all read from the event stream (docs/EVENTS.md):
  finding  the expected check is in the `diagnostics` findings (n/a when none is expected)
  drops    every must_drop column is in drop_columns of the last run_search call (n/a when none)
  cv       that call's cv (default kfold) is one of the allowed schemes
  pick     the shipped model (final_model, else confirm.recommendation.pick) is in tie group 0 or has the
           best hidden score in the `confirm` table
  claims   the report has no "[number removed" marker, so the claim check stripped nothing
A run passes when no check fails; pass^k means all k repeats passed. Tokens and USD come from `usage`.
Each run gets a fresh, empty experience memory (no Supabase, temp file), so repeats are independent and the
eval never writes into the team's memory. --dry uses the scripted client and a fake Modal backend: free, offline,
and only tests this harness. Writes runs/golden_<ts>.md and runs/golden_<ts>.json.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import zlib
from contextlib import contextmanager, nullcontext
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import agent
import factory_tools as ft
import memory
from evals.make_golden import EXPECT
from evals.make_golden import OUT as GOLDEN_DIR
from evals.public_catalog import PUBLIC, PUBLIC_DIR

CHECKS = ("finding", "drops", "cv", "pick", "claims")
REMOVED_MARK = "[number removed"
SETS = ("golden", "public", "all")


# ---------- datasets

def datasets(sets: str) -> dict[str, dict]:
    """name -> {path, target, task, purpose, expect} for the chosen set."""
    golden = {n: {"path": os.path.join(GOLDEN_DIR, f"{n}.csv"), "target": e["target"], "task": e["task"],
                  "purpose": None, "expect": {k: e[k] for k in ("finding", "must_drop", "cv")}}
              for n, e in EXPECT.items()}
    public = {n: {"path": os.path.join(PUBLIC_DIR, f"{n}.csv"), "target": e["target"], "task": e["task"],
                  "purpose": e["purpose"], "expect": e["expect"]} for n, e in PUBLIC.items()}
    return {"golden": golden, "public": public, "all": {**golden, **public}}[sets]


# ---------- scoring

def _allowed(v) -> tuple:
    return tuple(v) if isinstance(v, (list, tuple)) else (v,)


def _last(events: list[dict], kind: str, tool: str | None = None) -> dict | None:
    """Payload of the last event of this kind (for tool_call: of this tool, with parsed dict input)."""
    for e in reversed(events):
        p = e.get("payload") or {}
        if e.get("kind") != kind:
            continue
        if tool is None:
            return p
        if p.get("tool") == tool and isinstance(p.get("input"), dict):
            return p["input"]
    return None


def _pick_ok(confirm: dict | None, final: dict | None) -> bool:
    """Shipped model is statistically tied with the best, or won the one-shot hidden test."""
    table = (confirm or {}).get("table") or []
    name = (final or {}).get("name") or ((confirm or {}).get("recommendation") or {}).get("pick")
    row = next((r for r in table if r.get("name") == name), None)
    if row is None:
        return False
    if row.get("tie_group") == 0:
        return True
    sign = 1 if ft.HIGHER_IS_BETTER.get(confirm.get("primary_metric"), True) else -1
    hidden = [sign * r["hidden"] for r in table if r.get("hidden") is not None]
    return row.get("hidden") is not None and sign * row["hidden"] >= max(hidden)


def score(events: list[dict], expect: dict) -> dict:
    """Score one run; each check is True, False, or None (not applicable). None never fails a run."""
    diag, search = _last(events, "diagnostics"), _last(events, "tool_call", "run_search")
    want, must = expect.get("finding"), expect.get("must_drop") or []
    fired = {f.get("check") for f in (diag or {}).get("findings") or []}
    report = _last(events, "report")
    checks = {
        "finding": None if want is None else any(w in fired for w in _allowed(want)),
        "drops": None if not must else search is not None and set(must) <= set(search.get("drop_columns") or []),
        "cv": search is not None and search.get("cv", "kfold") in _allowed(expect["cv"]),
        "pick": _pick_ok(_last(events, "confirm"), _last(events, "final_model")),
        "claims": report is not None and REMOVED_MARK not in f"{report.get('markdown')}{report.get('spoken_summary')}",
    }
    usage = _last(events, "usage") or {}
    return {"checks": checks, "passed": all(v is not False for v in checks.values()),
            "tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
            "usd": float(usage.get("usd") or 0), "modal_usd": float(usage.get("modal_usd") or 0)}


# ---------- offline backend for --dry (the FakeFn pattern from test_factory_tools.py)

TRUE = {"lightgbm": 0.86, "logreg": 0.855, "ridge": 0.855, "catboost": 0.84, "xgboost": 0.83,
        "random_forest": 0.82, "mlp": 0.78, "tabicl": 0.80}
LOWER_BETTER = {m for m, hib in ft.HIGHER_IS_BETTER.items() if not hib}


def _seed(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode()))


class FakeFn:
    """One fake Modal function: deterministic fold scores per model family, fake fits and predictions."""

    def __init__(self, kind: str, backend: "FakeModal"):
        self.kind, self.backend = kind, backend

    def map(self, specs: list[dict]) -> list[dict]:
        return [self._cv(s) for s in specs]

    def remote(self, spec: dict, *args):
        if self.kind.startswith("predict_holdout"):
            return self.backend.predict(spec, args[0])
        return {"model_path": f"/models/fake/{spec['name']}.joblib", "n_rows": 0, "top_features": []}

    def _cv(self, s: dict) -> dict:
        n = s.get("train_rows") or 2000
        good = TRUE.get(s["model"], 0.8) + _seed(s["name"]).normal(0, 0.02 * np.sqrt(500 / n),
                                                                  s.get("cv_folds", 5) * s.get("repeats", 1))
        metrics = {}
        for m in ft.HIGHER_IS_BETTER:
            folds = (1.2 - good) if m in LOWER_BETTER else good
            metrics[m] = {"mean": float(folds.mean()), "std": float(folds.std()), "folds": folds.round(5).tolist(),
                          "train_mean": float(folds.mean()) + (-0.03 if m in LOWER_BETTER else 0.03)}
        return {"name": s["name"], "ok": True, "fit_seconds": 0.01 * n / 500, "predict_ms": 1.0, "n_rows": n,
                "metrics": metrics}


class FakeModal:
    """Stands in for modal.Function.from_name and upload_dataset; remembers uploaded frames by path."""

    def __init__(self):
        self.frames: dict = {}

    def function(self, app: str, name: str) -> FakeFn:
        return FakeFn(name, self)

    def upload(self, df) -> str:
        path = f"/datasets/fake{len(self.frames)}.parquet"
        self.frames[path] = df
        return path

    def predict(self, spec: dict, path: str) -> dict:
        y, rng = self.frames[path][spec["target"]], _seed(spec["name"])
        err = 1.2 - TRUE.get(spec["model"], 0.8)
        if spec["task"] == "classification":
            classes, codes = np.unique(y.astype(str), return_inverse=True)
            proba = np.clip(0.25 + 0.5 * codes + rng.normal(0, err, len(codes)), 0, 1)
            return {"name": spec["name"], "y_true": codes.tolist(), "pred": (proba > 0.5).astype(int).tolist(),
                    "proba": proba.tolist(), "classes": classes.tolist()}
        y = y.to_numpy(dtype=float)
        return {"name": spec["name"], "y_true": y.tolist(), "pred": (y + rng.normal(0, err * y.std(), len(y))).tolist(),
                "proba": None}


@contextmanager
def dry_backend():
    """Fake Modal functions and uploads, and no Supabase, for the duration of a scripted run."""
    fake = FakeModal()
    with mock.patch.object(agent.modal.Function, "from_name", fake.function), \
            mock.patch.object(ft, "upload_dataset", fake.upload), mock.patch.object(agent, "_maybe_supabase", lambda: None):
        yield fake


# ---------- running

def run_once(entry: dict, dry: bool = False) -> dict:
    """One FactoryRun with a fresh empty memory, scored; a crash is recorded, never raised."""
    run, error = None, None
    try:
        with (dry_backend() if dry else nullcontext()), mock.patch.object(memory, "_supabase", lambda: None), \
                tempfile.TemporaryDirectory() as tmp:
            run = agent.FactoryRun(entry["path"], entry["target"], provider="scripted" if dry else "nebius",
                                   purpose=entry.get("purpose"))
            run.memory_path = os.path.join(tmp, "memory.jsonl")
            run.run()
    except (Exception, SystemExit) as e:  # one broken run must not stop the whole eval
        error = f"{type(e).__name__}: {e}"[:300]
    return {"run_id": getattr(run, "run_id", None), **score(getattr(run, "events", []), entry["expect"]),
            "error": error}


def evaluate(entries: dict[str, dict], repeat: int = 1, dry: bool = False) -> list[dict]:
    """Run each dataset `repeat` times; one summary per dataset with pass^k and cost."""
    out = []
    for name, entry in entries.items():
        if not os.path.exists(entry["path"]):
            runs = [{**score([], entry["expect"]), "run_id": None,
                     "error": f"missing {entry['path']}: run evals/make_golden.py or evals/fetch_public.py"}]
        else:
            runs = [run_once(entry, dry) for _ in range(repeat)]
        out.append({"dataset": name, "k": len(runs), "pass_k": all(r["passed"] for r in runs), "runs": runs,
                    **{k: sum(r[k] for r in runs) for k in ("tokens", "usd", "modal_usd")}})
    return out


# ---------- output

def _cell(values: list) -> str:
    if all(v is None for v in values):
        return "n/a"
    ok = sum(v is not False for v in values)
    return ("pass" if ok else "FAIL") if len(values) == 1 else f"{ok}/{len(values)}"


def to_markdown(results: list[dict], meta: dict) -> str:
    """One row per dataset: each check, pass^k, tokens and cost; then totals and errors."""
    lines = [f"# Golden eval {meta['ts']}", "",
             f"Sets: {meta['sets']}. Repeats: {meta['repeat']}. Provider: {meta['provider']}.", "",
             "| dataset | " + " | ".join(CHECKS) + " | pass^k | tokens | llm usd | modal usd |",
             "|---|" + "---|" * (len(CHECKS) + 4)]
    for r in results:
        cells = [_cell([run["checks"][c] for run in r["runs"]]) for c in CHECKS]
        lines.append(f"| {r['dataset']} | " + " | ".join(cells) + f" | {'PASS' if r['pass_k'] else 'FAIL'} | "
                     f"{r['tokens']} | {r['usd']:.4f} | {r['modal_usd']:.4f} |")
    usd, modal_usd = sum(r["usd"] for r in results), sum(r["modal_usd"] for r in results)
    lines += ["", (f"pass^k: {sum(r['pass_k'] for r in results)}/{len(results)} datasets. "
                   f"Tokens: {sum(r['tokens'] for r in results)}. Cost: ${usd + modal_usd:.4f} "
                   f"(LLM ${usd:.4f}, Modal ${modal_usd:.4f}).")]
    errors = [f"- {r['dataset']} run {run['run_id']}: {run['error']}" for r in results for run in r["runs"]
              if run.get("error")]
    return "\n".join(lines + (["", "Errors:", *errors] if errors else [])) + "\n"


def write(results: list[dict], meta: dict, out_dir: str = "runs") -> tuple[str, str]:
    """Save runs/golden_<ts>.md and .json; return both paths."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"golden_{meta['ts']}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(results, meta))
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({**meta, "results": results}, f, indent=2, default=str)
    return base + ".md", base + ".json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", choices=SETS, default="golden")
    ap.add_argument("--repeat", type=int, default=1, help="runs per dataset; pass^k needs all k to pass")
    ap.add_argument("--only", action="append", help="dataset name (repeatable)")
    ap.add_argument("--dry", action="store_true", help="scripted client + fake Modal: free, tests the harness")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    entries = datasets(args.sets)
    if args.only:
        unknown = sorted(set(args.only) - set(entries))
        if unknown:
            ap.error(f"unknown dataset(s) {unknown} in set {args.sets!r}; known: {sorted(entries)}")
        entries = {n: entries[n] for n in args.only}
    meta = {"ts": time.strftime("%Y%m%d-%H%M%S"), "sets": args.sets, "repeat": args.repeat,
            "provider": "scripted (dry)" if args.dry else "nebius", "datasets": list(entries)}
    results = evaluate(entries, args.repeat, args.dry)
    md, js = write(results, meta, args.out)
    print(to_markdown(results, meta) + f"\nSaved {md} and {js}")
    return 0 if all(r["pass_k"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
