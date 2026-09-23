"""
Registers a run in the team's Supabase so the web UI can follow it. Two schemas exist:

  v1 (live today)      datasets -> runs(id) -> events; runs.dataset_id and events.run_id are foreign keys
  v2 (Emre's target)   runs(run_id) -> events, candidates; statuses running / completed / failed

The schema is detected once from the PostgREST OpenAPI description. Best effort: any failure
disables the sync and the run continues (the jsonl replay always works). User rows are never
copied into these tables because the UI reads them with the public key.
"""
import hashlib
import json
import os
import urllib.request
import uuid
from datetime import datetime, timezone

import pandas as pd

V2_STATUS = {"done": "completed", "incomplete": "failed", "failed": "failed", "running": "running"}
CANDIDATE_FIELDS = ("name", "family", "stage", "rung", "train_rows", "params", "preprocessing", "ok", "error",
                    "cv_mean", "cv_std", "train_mean", "metrics", "fit_seconds", "predict_ms", "p_vs_best",
                    "tie_with_best", "hidden_score", "hidden_ci_low", "hidden_ci_high")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_schema(url: str | None, key: str | None) -> str:
    """'v2' if runs is keyed by run_id (Emre's schema), else 'v1'. Defaults to v1 on any error."""
    if not (url and key):
        return "v1"
    try:
        req = urllib.request.Request(url.rstrip("/") + "/rest/v1/", headers={
            "apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/openapi+json"})
        spec = json.load(urllib.request.urlopen(req, timeout=10))
        defs = spec.get("definitions") or spec.get("components", {}).get("schemas", {})
        return "v2" if "run_id" in defs.get("runs", {}).get("properties", {}) else "v1"
    except Exception:
        return "v1"


class SupabaseSync:
    def __init__(self, client, run_id: str, schema: str | None = None):
        self.client, self.run_id, self.ok = client, run_id, False
        self.schema = schema or (detect_schema(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
                                 if client is not None else "v1")

    @property
    def run_key(self) -> str:
        return "run_id" if self.schema == "v2" else "id"

    def start(self, df: pd.DataFrame, path: str, target: str, task: str, purpose: str | None = None,
              llm_model: str | None = None) -> bool:
        """Create the run row (and in v1 its dataset row). Returns True on success."""
        if self.client is None:
            return False
        try:
            if self.schema == "v2":
                self.client.table("runs").insert({
                    "run_id": self.run_id, "status": "running", "dataset_name": os.path.basename(path),
                    "n_rows": len(df), "n_features": df.shape[1] - 1, "target": target, "task": task,
                    "purpose": purpose, "llm_model": llm_model}).execute()
            else:
                with open(path, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
                dataset_id = str(uuid.uuid4())
                self.client.table("datasets").insert({
                    "id": dataset_id, "name": os.path.basename(path), "original_filename": os.path.basename(path),
                    "version_label": "v1", "storage_bucket": "datasets", "source_sha256": sha,
                    "file_size_bytes": os.path.getsize(path), "row_count": len(df), "column_count": df.shape[1],
                    "column_schema": [{"name": str(c), "dtype": str(t)} for c, t in df.dtypes.items()],
                    "preview_rows": [],  # the feed is publicly readable: never copy user rows here
                    "is_demo_fixture": False,
                }).execute()
                self.client.table("runs").insert({
                    "id": self.run_id, "dataset_id": dataset_id, "target": target, "task": task,
                    "status": "running", "started_at": _now(),
                }).execute()
            self.ok = True
        except Exception as e:
            print("supabase run registration failed, live feed disabled:", str(e)[:200])
            self.ok = False
        return self.ok

    def update(self, **fields) -> None:
        """Patch run columns (v2 only: primary_metric, split, profile, ...)."""
        if not (self.ok and self.schema == "v2" and fields):
            return
        try:
            self.client.table("runs").update(fields).eq("run_id", self.run_id).execute()
        except Exception as e:
            print("supabase run update failed:", str(e)[:200])

    def candidates(self, rows: list[dict]) -> None:
        """Batch-insert evaluation rows (v2 only). Unknown keys are dropped, params default to {}."""
        if not (self.ok and self.schema == "v2" and rows):
            return
        clean = [{"run_id": self.run_id, "params": {}, "preprocessing": {},
                  **{k: r[k] for k in CANDIDATE_FIELDS if k in r and r[k] is not None}} for r in rows]
        try:
            self.client.table("candidates").insert(clean).execute()
        except Exception as e:
            print("supabase candidates insert failed:", str(e)[:200])

    def finish(self, status: str, report: dict | None = None, error: str | None = None,
               recommended: str | None = None, model_path: str | None = None, usage: dict | None = None) -> None:
        """Close the run row with its outcome and report."""
        if not self.ok:
            return
        if self.schema == "v2":
            row = {"status": V2_STATUS.get(status, "failed"), "finished_at": _now(), "recommended": recommended,
                   "model_path": model_path, "usage": usage}
            if report:
                row.update(report_md=report.get("markdown"), spoken_summary=report.get("spoken_summary"))
        else:
            row = {"status": status, "finished_at": _now(), "updated_at": _now()}
            if report:
                row.update(report_markdown=report.get("markdown"), spoken_summary=report.get("spoken_summary"))
            if error:
                row["error"] = error[:2000]
        row = {k: v for k, v in row.items() if v is not None}
        try:
            self.client.table("runs").update(row).eq(self.run_key, self.run_id).execute()
        except Exception as e:
            print("supabase run update failed:", str(e)[:200])
