"""Durable run results in Supabase: one `runs` row per run, one `candidates` row per evaluation
(racing rung, confirmation, LLM experiment). Schema: docs/supabase.sql.

Best-effort like the events stream: without a client every call is a no-op, and a failed write is
printed, never raised, so a Supabase outage cannot break a run. runs/<run_id>_* stays the full record.
"""
import json
import math
from datetime import datetime, timezone


def _clean(v):
    """JSON-safe copy: NaN/inf -> None (PostgREST rejects them), numpy scalars -> Python via default=str."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _json(v):
    return _clean(json.loads(json.dumps(v, default=lambda o: o.item() if hasattr(o, "item") else str(o))))


def candidate_row(run_id: str, stage: str, config: dict, result: dict, metric: str, **extra) -> dict:
    """One `candidates` row from a config ({name, model, family?, params, preprocessing}) and a
    train_candidate result. cv_mean/cv_std/train_mean are the run's primary metric; `metrics`
    keeps every metric with its per-fold scores. extra: rung, train_rows, finalist columns."""
    m = (result.get("metrics") or {}).get(metric) or {}
    return {"run_id": run_id, "stage": stage, "name": result.get("name") or config.get("name"),
            "family": config.get("family") or config.get("model"),
            "params": config.get("params") or {}, "preprocessing": config.get("preprocessing") or {},
            "ok": bool(result.get("ok")), "error": result.get("error"),
            "cv_mean": m.get("mean"), "cv_std": m.get("std"), "train_mean": m.get("train_mean"),
            "metrics": result.get("metrics"), "fit_seconds": result.get("fit_seconds"),
            "predict_ms": result.get("predict_ms"), **extra}


class ResultsStore:
    def __init__(self, client, run_id: str):
        self.client, self.run_id = client, run_id

    def _write(self, what: str, fn):
        if not self.client:
            return
        try:
            fn(self.client)
        except Exception as e:
            print(f"supabase {what} failed:", e)

    def start_run(self, **fields):
        """Insert the runs row (status running)."""
        row = _json({"run_id": self.run_id, "status": "running", **fields})
        self._write("runs insert", lambda c: c.table("runs").upsert(row, on_conflict="run_id").execute())

    def add_candidates(self, rows: list[dict]):
        """Bulk-insert candidates rows; one request per call."""
        if rows:
            rows = _json(rows)
            self._write("candidates insert", lambda c: c.table("candidates").insert(rows).execute())

    def finish_run(self, **fields):
        """Set the run's outcome (status, recommendation, report, usage...) and finished_at."""
        row = _json({"finished_at": datetime.now(timezone.utc).isoformat(), **fields})
        self._write("runs update", lambda c: c.table("runs").update(row).eq("run_id", self.run_id).execute())
