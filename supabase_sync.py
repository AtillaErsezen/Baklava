"""
Registers a run in the team's Supabase schema (datasets -> runs -> events) so the web UI can
follow it. events.run_id is a foreign key to runs.id, and runs.dataset_id to datasets.id, so
both parent rows must exist before the first event. Best effort: any failure disables the sync
and the run continues (jsonl replay always works).
"""
import hashlib
import os
import uuid
from datetime import datetime, timezone

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupabaseSync:
    def __init__(self, client, run_id: str):
        self.client, self.run_id, self.ok = client, run_id, False

    def start(self, df: pd.DataFrame, path: str, target: str, task: str) -> bool:
        """Insert the dataset row, then the run row (status running). Returns True on success."""
        if self.client is None:
            return False
        try:
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            dataset_id = str(uuid.uuid4())
            self.client.table("datasets").insert({
                "id": dataset_id, "name": os.path.basename(path), "original_filename": os.path.basename(path),
                "version_label": "v1", "storage_bucket": "datasets", "source_sha256": sha,
                "file_size_bytes": os.path.getsize(path), "row_count": len(df), "column_count": df.shape[1],
                "column_schema": [{"name": str(c), "dtype": str(t)} for c, t in df.dtypes.items()],
                "preview_rows": [],  # the events feed is publicly readable: never copy user rows here
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

    def finish(self, status: str, report: dict | None = None, error: str | None = None) -> None:
        """Close the run row: done or failed, with the report text."""
        if not self.ok:
            return
        row = {"status": status, "finished_at": _now(), "updated_at": _now()}
        if report:
            row.update(report_markdown=report.get("markdown"), spoken_summary=report.get("spoken_summary"))
        if error:
            row["error"] = error[:2000]
        try:
            self.client.table("runs").update(row).eq("id", self.run_id).execute()
        except Exception as e:
            print("supabase run update failed:", str(e)[:200])
