import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import agent
import modal_train
from tracking import TrackingLog, load_tracked_dataset, utc_now

SOURCE = Path(__file__).resolve().parents[1] / "supabase/fixtures/setup_demo.csv"


def result(name="candidate"):
    return {"ok": True, "name": name, "metrics": {"roc_auc": {"mean": .8, "std": .02, "train_mean": .9}},
            "finished_at": utc_now(), "fit_seconds": .1,
            "data_manifest": {"usable_row_count": 12, "included_columns": ["tenure_months", "monthly_spend"]},
            "evaluation_config": {"method": "StratifiedKFold", "n_splits": 3}, "effective_config": {"params": {"C": 1}}}


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        _, self.dataset = load_tracked_dataset(str(SOURCE), None, None)

    def log(self, client=None):
        log = TrackingLog("test-run", self.dataset, "churn", client=client, output_dir=self.tmp.name)
        log.start({"n_rows": 12, "n_features": 2})
        return log

    def test_outage_retains_full_snapshots_and_delivery_order(self):
        client = Mock()
        client.rpc.side_effect = ConnectionError("private error")
        with patch("tracking.time.sleep"):
            log = self.log(client)
        log.emit("thought", {"text": "test"})
        self.assertEqual(client.rpc.call_count, 3)
        saved = [json.loads(line) for line in log.path.read_text().splitlines()]
        self.assertEqual(saved[0]["payload"]["_tracking"]["dataset"]["source_sha256"], self.dataset["source_sha256"])
        client.rpc.side_effect = None
        client.reset_mock()
        self.assertTrue(log.flush())
        self.assertEqual([call.args[1]["p_event"]["event_id"] for call in client.rpc.call_args_list], [e["event_id"] for e in saved])
        self.assertFalse(log.pending)

    def test_final_fit_has_parent_and_never_a_new_cv_score(self):
        log = self.log()
        spec = {"name": "candidate", "model": "logreg"}
        candidate = log.queue_training(spec, 1)
        log.training_started(candidate, utc_now())
        log.training_finished(candidate, result(), "roc_auc")
        final = log.queue_training(spec, 1, "final_fit", candidate)
        log.training_started(final, utc_now())
        log.training_finished(final, {**result(), "model_path": "/models/example"})
        self.assertIsNone(log.trainings[final]["metrics"])
        self.assertEqual(log.trainings[final]["parent_training_id"], candidate)
        with self.assertRaises(ValueError):
            log.training_started(candidate, utc_now())

    def test_nonfinite_metrics_fail_and_unfinished_attempts_close(self):
        log = self.log()
        candidate = log.queue_training({"name": "bad", "model": "logreg"}, 1)
        out = result()
        out["metrics"]["roc_auc"]["mean"] = float("nan")
        self.assertFalse(log.training_finished(candidate, out, "roc_auc"))
        pending = log.queue_training({"name": "pending", "model": "logreg"}, 1)
        log.end("failed", "interrupted")
        self.assertEqual(log.trainings[pending]["status"], "failed")
        self.assertIsNone(log.trainings[candidate]["metrics"])
        self.assertEqual(log.events[-1]["kind"], "run_end")
        json.loads(log.path.read_text().splitlines()[-1])


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.function_patch = patch("agent.modal.Function.from_name")
        self.remote = self.function_patch.start().return_value
        self.addCleanup(self.function_patch.stop)
        self.upload_patch = patch("agent.upload_dataset", return_value={"path": "/fixture.parquet", "sha256": "a" * 64})
        self.upload_patch.start()
        self.addCleanup(self.upload_patch.stop)
        self.run = agent.FactoryRun(str(SOURCE), "churn", dry_run=True, offline=True, output_dir=self.tmp.name)

    def test_results_are_persisted_in_completion_order(self):
        fast_finished = threading.Event()

        def stream(spec, run_id, stage):
            yield {"kind": "started", "started_at": utc_now()}
            if spec["name"] == "slow":
                self.assertTrue(fast_finished.wait(5))
            yield {"kind": "finished", "result": result(spec["name"])}
            if spec["name"] == "fast":
                fast_finished.set()

        self.remote.remote_gen.side_effect = stream
        self.run.tool_run_experiments({"task": "classification", "primary_metric": "roc_auc", "rationale": "test",
                                      "candidates": [{"name": name, "model": "logreg"} for name in ("slow", "fast")]})
        completed = [e["payload"]["candidate_name"] for e in self.run.events if e["kind"] == "training_completed"]
        self.assertEqual(completed, ["fast", "slow"])
        self.assertLess(next(i for i, e in enumerate(self.run.events) if e["kind"] == "training_completed"),
                        next(i for i, e in enumerate(self.run.events) if e["kind"] == "leaderboard"))

    def test_scripted_run_completes_and_final_fit_links_to_cv(self):
        def stream(spec, run_id, stage):
            yield {"kind": "started", "started_at": utc_now()}
            yield {"kind": "finished", "result": {**result(spec["name"]), "model_path": "/models/test.joblib"}}

        self.remote.remote_gen.side_effect = stream
        self.run.run()
        self.assertEqual(self.run.tracker.run["status"], "completed")
        self.assertEqual(len(self.run.tracker.trainings), 6)
        finals = [row for row in self.run.tracker.trainings.values() if row["stage"] == "final_fit"]
        self.assertIsNone(finals[0]["metrics"])
        self.assertEqual(finals[0]["parent_training_id"], self.run.tracker.run["selected_training_id"])

    def test_agent_failure_keeps_partial_log_and_terminal_run(self):
        self.run.client.messages.create = Mock(side_effect=RuntimeError("agent failure"))
        with self.assertRaisesRegex(RuntimeError, "agent failure"):
            self.run.run()
        self.assertEqual(self.run.tracker.run["status"], "failed")
        self.assertEqual(self.run.events[-1]["kind"], "run_end")
        self.assertTrue(Path(self.tmp.name, f"{self.run.run_id}_results.json").exists())

    def test_cleanup_failure_does_not_hide_original_agent_error(self):
        self.run.client.messages.create = Mock(side_effect=RuntimeError("original agent failure"))
        self.run.tracker.end = Mock(side_effect=OSError("disk error"))
        with self.assertRaisesRegex(RuntimeError, "original agent failure"):
            self.run.run()
        self.assertTrue(Path(self.tmp.name, f"{self.run.run_id}_results.json").exists())


class WorkerMetadataTests(unittest.TestCase):
    def test_real_cv_reports_actual_rows_features_splits_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.read_csv(SOURCE)
            frame.loc[len(frame)] = [1, 20, None]
            frame["leak"] = frame.churn
            frame["date"] = pd.date_range("2026-01-01", periods=len(frame))
            path = Path(directory, "sample.parquet")
            frame.to_parquet(path, index=False)
            spec = {"dataset_path": "/sample.parquet", "target": "churn", "task": "classification",
                    "name": "baseline", "model": "logreg", "cv_folds": 3, "drop_columns": ["leak"],
                    "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.object(modal_train, "DATA_DIR", directory), patch.object(modal_train.vol, "reload"):
                out = modal_train.train_candidate.local(spec)
                self.assertTrue(out["ok"], out.get("error"))
                manifest = out["data_manifest"]
                self.assertEqual((manifest["source_row_count"], manifest["usable_row_count"], manifest["excluded_missing_target"]), (13, 12, 1))
                self.assertEqual(manifest["dropped_columns"], ["leak"])
                self.assertEqual(manifest["derived_columns"], ["date_year", "date_month", "date_dow"])
                self.assertEqual(out["evaluation_config"]["fold_sizes"], [{"train": 8, "validation": 4}] * 3)
                self.assertEqual(out["effective_config"]["params"]["max_iter"], 2000)
                self.assertEqual(len(out["metrics"]["roc_auc"]["fold_scores"]), 3)
                spec["prepared_sha256"] = "wrong"
                self.assertFalse(modal_train.train_candidate.local(spec)["ok"])


if __name__ == "__main__":
    unittest.main()
