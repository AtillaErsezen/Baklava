"""Exercise tracking and optional Realtime with labelled fixtures, without ML/API charges."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import uuid
import traceback

from supabase import acreate_client, create_client
from tracking import TrackingLog, load_tracked_dataset, reconcile_log, supabase_client, utc_now, verify_tracking_api

SAMPLE_ID = "7beb6dbb-f3a0-5fec-ba12-a9c4132a33e4"


def generate(run_id, dataset, client, output_dir):
    log = TrackingLog(run_id, dataset, "churn", "classification", client, output_dir)
    log.start({"n_rows": dataset["row_count"], "n_features": 2})
    spec = {"name": "tracking_fixture", "model": "logreg", "synthetic_fixture": True}
    candidate = log.queue_training(spec, 1)
    log.training_started(candidate, utc_now())
    log.training_finished(candidate, {
        "ok": True, "finished_at": utc_now(), "fit_seconds": 0,
        "metrics": {"roc_auc": {"mean": .8, "std": .02, "train_mean": .85}},
        "data_manifest": {"dataset_id": dataset["id"], "source_sha256": dataset["source_sha256"],
                          "usable_row_count": 12, "included_columns": ["tenure_months", "monthly_spend"], "synthetic_fixture": True},
        "evaluation_config": {"synthetic_fixture": True},
        "warnings": ["Invented tracking smoke score; no training or evaluation was performed."],
    }, "roc_auc")
    failed = log.queue_training({**spec, "name": "intentional_failure"}, 1)
    log.training_finished(failed, {"ok": False, "error": "Intentional tracking smoke fixture."})
    log.report({"markdown": "Tracking smoke only. Scores are invented fixtures; no model was trained.",
                "spoken_summary": "Tracking smoke only."})
    log.end("incomplete", "Tracking smoke fixture; no final model was fitted.")
    if log.pending:
        raise RuntimeError("Some events were not delivered; reconcile the saved log.")
    return log


async def with_realtime(run_id, dataset, client, output_dir):
    realtime = await acreate_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_PUBLISHABLE_KEY"])
    subscribed, received_end, received_states = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = set()
    training_states = {}

    def training_changed(payload):
        record = payload["data"]["record"]
        old = training_states.get(record["id"])
        if old is None or record["row_version"] > old["row_version"]:
            training_states[record["id"]] = record
        if sorted(row["status"] for row in training_states.values()) == ["failed", "succeeded"]:
            received_states.set()

    def status(state, error=None):
        if state == "SUBSCRIBED":
            subscribed.set()

    def changed(payload):
        record = payload["data"]["record"]
        seen.add(record["event_id"])
        if record["kind"] == "run_end":
            received_end.set()

    channel = realtime.channel("tracking-smoke-" + run_id)
    channel.on_postgres_changes("INSERT", changed, table="events", schema="public", filter=f"run_id=eq.{run_id}")
    channel.on_postgres_changes("*", training_changed, table="trainings", schema="public", filter=f"run_id=eq.{run_id}")
    try:
        await channel.subscribe(status)
        await asyncio.wait_for(subscribed.wait(), timeout=20)
        log = await asyncio.to_thread(generate, run_id, dataset, client, output_dir)
        await asyncio.wait_for(received_end.wait(), timeout=20)
        await asyncio.wait_for(received_states.wait(), timeout=20)
        if seen != {event["event_id"] for event in log.events}:
            raise RuntimeError("Realtime did not deliver the complete event sequence.")
        print(f"PASS Realtime: {len(seen)} distinct events and both terminal training states delivered to the publishable-key subscriber.")
        return log
    finally:
        await realtime.remove_all_channels()
        await realtime.realtime.close()


def verify_live(log, client):
    before = client.table("events").select("event_id").eq("run_id", log.run["id"]).execute().data
    reconcile_log(log.path, client)
    after = client.table("events").select("event_id").eq("run_id", log.run["id"]).execute().data
    if {row["event_id"] for row in before} != {row["event_id"] for row in after} or len(after) != len(log.events):
        raise RuntimeError("Replay duplicated or lost events")
    browser = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_PUBLISHABLE_KEY"])
    rows = browser.table("trainings").select("id,status,metrics").eq("run_id", log.run["id"]).execute().data
    if sorted(row["status"] for row in rows) != ["failed", "succeeded"]:
        raise RuntimeError("Browser history does not match the completed attempts")
    for row in rows:
        if row["status"] == "failed" and row["metrics"] is not None:
            raise RuntimeError("Failed attempt has a fabricated score")
    from postgrest.exceptions import APIError
    try:
        browser.rpc("apply_tracking_event", {"p_event": None}).execute()
    except APIError as exc:
        if exc.code != "42501":
            raise RuntimeError("Browser RPC denial could not be verified") from None
    else:
        raise RuntimeError("Browser can call the backend tracking RPC")
    print("PASS backend writes, browser history, failed-attempt metrics, duplicate replay, and browser RPC rejection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()
    client = supabase_client(args.offline)
    if not args.offline and client is None:
        parser.error("Configure Supabase or explicitly select --offline.")
    if args.realtime and client is None:
        parser.error("Realtime requires Supabase.")
    try:
        verify_tracking_api(client)
        sample = Path(__file__).resolve().parents[1] / "supabase/fixtures/setup_demo.csv"
        _, dataset = load_tracked_dataset(str(sample), SAMPLE_ID if client else None, client)
        if args.offline:
            dataset["is_demo_fixture"] = True
        run_id = "tracking-smoke-" + uuid.uuid4().hex[:12]
        log = asyncio.run(with_realtime(run_id, dataset, client, args.output_dir)) if args.realtime else generate(run_id, dataset, client, args.output_dir)
        if client:
            verify_live(log, client)
        print(json.dumps({"run_id": run_id, "dataset_id": dataset["id"], "events": len(log.events), "log": str(log.path)}))
    except Exception as exc:
        location = traceback.extract_tb(exc.__traceback__)[-1]
        raise SystemExit(f"Tracking smoke failed ({type(exc).__name__} at {Path(location.filename).name}:{location.lineno}). Check setup and saved logs; credentials and raw API errors were withheld.") from None


if __name__ == "__main__":
    main()
