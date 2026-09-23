"""Provider-independent training history: durable local events plus atomic Supabase delivery."""

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

HIGHER_IS_BETTER = {"roc_auc": True, "f1_macro": True, "accuracy": True, "r2": True, "rmse": False, "mae": False}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def supabase_client(offline=False):
    if offline:
        return None
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url and not key:
        return None
    if not url or not key:
        raise ValueError("Set both SUPABASE_URL and SUPABASE_KEY, or use --offline.")
    from supabase import ClientOptions, create_client
    return create_client(url, key, options=ClientOptions(postgrest_client_timeout=10, storage_client_timeout=30))


def verify_tracking_api(client):
    if client is None:
        return
    try:
        version = client.rpc("apply_tracking_event", {"p_event": None}).execute().data
        if version != 1:
            raise ValueError("Unexpected tracking protocol")
    except Exception:
        raise RuntimeError("Tracking API unavailable. Apply the atomic_training_events migration and check backend credentials before starting training.") from None


def load_tracked_dataset(data_path, dataset_id, client):
    """Read one immutable byte snapshot; only registered files can enter a live run."""
    from modal_train import load_local
    from scripts.register_dataset import prepare_record

    if dataset_id and client is None:
        raise ValueError("--dataset-id requires Supabase access; use a local path for offline execution.")
    if data_path:
        path = Path(data_path)
        raw = path.read_bytes()
        filename = path.name
    elif not dataset_id:
        raise ValueError("Supply a local dataset path or --dataset-id.")
    if client:
        if dataset_id:
            rows = client.table("datasets").select("*").eq("id", dataset_id).execute().data
        else:
            rows = client.table("datasets").select("*").eq("source_sha256", hashlib.sha256(raw).hexdigest()).eq("original_filename", filename).execute().data
            rows = [row for row in rows if row["storage_path"]]
        if len(rows) != 1 or not rows[0]["storage_path"]:
            raise ValueError("Choose one registered Storage dataset with --dataset-id. Register a local file first with scripts.register_dataset.")
        dataset = rows[0]
        if not data_path:
            raw = client.storage.from_(dataset["storage_bucket"]).download(dataset["storage_path"])
            filename = dataset["original_filename"]
        if hashlib.sha256(raw).hexdigest() != dataset["source_sha256"]:
            raise ValueError("Dataset hash mismatch: the file is not the registered version.")
    else:
        dataset, _ = prepare_record(Path(filename), raw, "offline", filename, "local")
        dataset["storage_path"] = None
    # Parse the exact bytes hashed above, even if the original file changes during a run.
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / ("source" + Path(filename).suffix.lower())
        source.write_bytes(raw)
        frame = load_local(str(source))
    if len(frame) != dataset["row_count"]:
        raise ValueError("Dataset row count does not match its registered version.")
    return frame, dataset


class TrackingLog:
    def __init__(self, run_id, dataset, target, task=None, client=None, output_dir="runs"):
        self.client, self.dataset = client, json_safe(dataset)
        self.run = {"id": run_id, "dataset_id": dataset["id"], "target": target, "task": task,
                    "status": "running", "started_at": utc_now(), "finished_at": None}
        self.trainings, self.events = {}, []
        self.pending = deque()
        self.delivery_paused = False
        self.started = False
        self.path = Path(output_dir) / f"{run_id}_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never append a new run to an older run's history by accident.
        self.path.touch(exist_ok=False)

    def emit(self, kind, payload, training_id=None, include_dataset=False, include_run=False):
        changes = {"version": 1}
        if include_dataset:
            changes["dataset"] = self.dataset
        if include_run:
            changes["run"] = self.run
        if training_id:
            changes["training"] = self.trainings[training_id]
        event = json_safe({"event_id": str(uuid.uuid4()), "run_id": self.run["id"],
                          "training_id": training_id, "ts": time.time(), "kind": kind,
                          "payload": {**payload, "_tracking": changes}})
        # Flush before attempting the network so an outage or process failure leaves replay data.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)
        if self.client:
            self.pending.append(event)
            if not self.delivery_paused:
                self.flush()
        print(f"[{kind}] {json.dumps(json_safe(payload), ensure_ascii=False)[:400]}")
        return event

    def flush(self):
        while self.pending:
            for attempt in range(3):
                try:
                    self.client.rpc("apply_tracking_event", {"p_event": self.pending[0]}).execute()
                    self.pending.popleft()
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(0.1 * (2 ** attempt))
            else:
                self.delivery_paused = True
                print(f"Supabase delivery pending; local history is safe in {self.path}. Reconcile this log after connectivity is restored.")
                return False
        self.delivery_paused = False
        return True

    def start(self, profile):
        if not self.started:
            self.emit("run_start", {"dataset_id": self.dataset["id"], "dataset": self.dataset["name"],
                                   "target": self.run["target"], "rows": profile["n_rows"],
                                   "features": profile["n_features"]}, include_dataset=True, include_run=True)
            self.started = True

    def set_task(self, task):
        if self.run["task"] not in (None, task):
            raise ValueError("A run cannot change task type; start a new run.")
        if self.run["task"] is None:
            self.run["task"] = task
            self.emit("run_configured", {"task": task}, include_run=True)

    def queue_training(self, spec, round_number, stage="candidate_cv", parent_id=None):
        training_id = str(uuid.uuid4())
        attempt = 1 + sum(row["round"] == round_number and row["candidate_name"] == spec["name"]
                          and row["stage"] == stage for row in self.trainings.values())
        self.trainings[training_id] = json_safe({
            "id": training_id, "run_id": self.run["id"], "round": round_number,
            "candidate_name": spec["name"], "attempt": attempt, "stage": stage,
            "parent_training_id": parent_id, "model": spec["model"], "requested_config": spec,
            "status": "queued", "data_manifest": {}, "evaluation_config": {}, "warnings": [],
        })
        self.emit("training_queued", {"candidate_name": spec["name"], "stage": stage}, training_id)
        return training_id

    def training_started(self, training_id, started_at):
        row = self.trainings[training_id]
        if row["status"] != "queued":
            raise ValueError("Unexpected training start")
        row.update(status="running", started_at=started_at)
        self.emit("training_started", {"started_at": started_at}, training_id)

    def training_finished(self, training_id, result, primary_metric=None):
        row = self.trainings[training_id]
        if row["status"] in ("succeeded", "failed"):
            raise ValueError("Training already finished")
        result = json_safe(result)
        ok = bool(result.get("ok"))
        metrics = result.get("metrics") if row["stage"] == "candidate_cv" else None
        if ok and row["stage"] == "candidate_cv":
            metric = (metrics or {}).get(primary_metric, {})
            if not all(isinstance(metric.get(key), (int, float)) for key in ("mean", "std", "train_mean")):
                ok = False
                result["error"] = "Worker did not return finite primary validation and training metrics."
        row.update(status="succeeded" if ok else "failed", finished_at=result.get("finished_at") or utc_now(),
                   metrics=metrics if ok else None, error=None if ok else result.get("error", "Training failed"),
                   effective_config=result.get("effective_config"), data_manifest=result.get("data_manifest", {}),
                   evaluation_config=result.get("evaluation_config", {}), duration_seconds=result.get("fit_seconds"),
                   model_path=result.get("model_path"), warnings=result.get("warnings", []))
        if ok and row["stage"] == "candidate_cv":
            metric = metrics[primary_metric]
            higher = HIGHER_IS_BETTER[primary_metric]
            row.update(primary_metric=primary_metric, higher_is_better=higher,
                       overfit_gap=round((metric["train_mean"] - metric["mean"]) * (1 if higher else -1), 5))
            if primary_metric in ("roc_auc", "accuracy", "r2") and metric["mean"] > 0.995:
                row["warnings"].append("suspiciously_high_check_leakage")
        self.emit("training_completed" if ok else "training_failed", {"candidate_name": row["candidate_name"],
                  "status": row["status"], "metrics": row["metrics"], "error": row["error"]}, training_id)
        return ok

    def select(self, training_id):
        row = self.trainings[training_id]
        if row["status"] != "succeeded" or row["stage"] != "candidate_cv":
            raise ValueError("Only a successful evaluated candidate can be selected.")
        self.run["selected_training_id"] = training_id
        self.emit("model_selected", {"training_id": training_id}, include_run=True)

    def report(self, report):
        self.run.update(report_markdown=report["markdown"], spoken_summary=report["spoken_summary"])
        self.emit("report", report, include_run=True)

    def end(self, status, error=None):
        for training_id, row in list(self.trainings.items()):
            if row["status"] in ("queued", "running"):
                self.training_finished(training_id, {"ok": False, "error": "Run stopped before worker completion was confirmed."})
        self.run.update(status=status, finished_at=utc_now(), error=error)
        self.emit("run_end", {"status": status, "reason": error}, include_run=True)
        if self.client:
            self.flush()


def reconcile_log(path, client):
    verify_tracking_api(client)
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            event = json.loads(line)
            if event.get("payload", {}).get("_tracking", {}).get("version") != 1:
                raise ValueError(f"Line {line_number} is not a replayable tracking event.")
            client.rpc("apply_tracking_event", {"p_event": event}).execute()
            count += 1
    return count
