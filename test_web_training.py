"""Offline tests for CSV validation and the real-agent web adapter."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import web_training as web


class WebTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / "data.csv"
        self.csv.write_text('value,note,target\n' + ''.join(f'{i},"row, {i}",{i % 2}\n' for i in range(100)))

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_quotes_preview_and_target_detection(self):
        data = web.inspect_csv(self.csv)
        self.assertEqual(data["rows"], 100)
        self.assertEqual(data["preview"][0], ["0", "row, 0", "0"])
        self.assertEqual(data["columns"][0]["suggestedTask"], "regression")
        self.assertEqual(web.validate_target(self.csv, "target", "auto"), {"task": "classification"})
        self.assertEqual(web.validate_target(self.csv, "value", "auto"), {"task": "regression"})

    def test_invalid_csv_and_targets(self):
        for contents in ("", "x\n1", "x,x\n", ",y\n", "x,y\n1,2,3\n", 'x,y\n"unterminated,2', "x,y\n1,2"):
            self.csv.write_text(contents)
            with self.assertRaises(ValueError):
                web.inspect_csv(self.csv)
        self.csv.write_text("x,y\n" + "1,\n" * 100)
        with self.assertRaisesRegex(ValueError, "missing"):
            web.validate_target(self.csv, "y", "auto")
        with self.assertRaisesRegex(ValueError, "two different"):
            web.validate_target(self.csv, "x", "auto")
        with self.assertRaisesRegex(ValueError, "target column"):
            web.validate_target(self.csv, "missing", "auto")

    def test_class_counts_and_regression_types(self):
        self.csv.write_text("x,y\n" + "1,cat\n" * 99 + "2,dog\n")
        with self.assertRaisesRegex(ValueError, "15 rows"):
            web.validate_target(self.csv, "y", "classification")
        with self.assertRaisesRegex(ValueError, "numeric"):
            web.validate_target(self.csv, "y", "regression")

    def make_session(self):
        folder = self.root / "runs" / ".web" / "datasets" / "fixture"
        folder.mkdir(parents=True)
        (folder / "dataset.csv").write_text(self.csv.read_text())
        session = self.root / "session.json"
        web.write_json(session, {"id": "fixture", "datasetId": "fixture", "target": "target", "task": "auto", "filename": "original.csv", "purpose": "Test purpose", "createdAt": 1})
        return session

    def test_worker_drives_existing_agent_and_persists_results(self):
        session = self.make_session()
        module = types.ModuleType("agent")
        test = self
        class FakeFactory:
            def __init__(self, path, target, task_hint, provider, purpose):
                test.assertEqual(provider, "nebius")
                test.assertEqual(task_hint, "classification")
                test.assertEqual(purpose, "Test purpose")
                test.assertTrue(Path(path).exists())
                self.run_id = "test-run"
                self.events, self.results, self.profile = [], [], {}
                self.final = self.report = None
            def emit(self, kind, payload):
                self.events.append({"run_id": self.run_id, "ts": 1, "kind": kind, "payload": payload})
            def run(self):
                test.assertEqual(self.dataset_name, "original.csv")
                self.emit("run_start", {"dataset": self.dataset_name})
                self.final = {"name": "TEST FIXTURE", "model_path": "/models/test.joblib"}
                self.emit("final_model", self.final)
                self.report = {"markdown": "TEST FIXTURE", "spoken_summary": "TEST FIXTURE"}
                self.emit("report", self.report)
                self.emit("usage", {"usd": float("nan")})
                self.save()
        module.FactoryRun = FakeFactory
        with patch.object(web, "ROOT", self.root), patch.object(web, "readiness", return_value={"ready": True}), patch.dict(sys.modules, {"agent": module}):
            web.train(session)
        state = json.loads(session.read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["eventCount"], 4)
        events = (self.root / "runs" / state["resultFile"]).read_text()
        self.assertNotIn("NaN", events)
        self.assertEqual(json.loads(events.splitlines()[0])["payload"]["dataset"], "original.csv")
        self.assertTrue((self.root / "runs" / "test-run_results.json").exists())

    def test_missing_setup_fails_without_importing_or_starting_agent(self):
        session = self.make_session()
        with patch.object(web, "ROOT", self.root), patch.object(web, "readiness", return_value={"ready": False, "issues": [{"label": "Nebius API key"}]}), contextlib.redirect_stderr(io.StringIO()):
            web.train(session)
        state = json.loads(session.read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("Nebius API key", state["error"])
        self.assertNotIn("resultFile", state)

    def test_runtime_errors_never_reveal_provider_secrets(self):
        session = self.make_session()
        module = types.ModuleType("agent")
        class Failing:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Provider rejected TOP_SECRET_KEY")
        module.FactoryRun = Failing
        with patch.object(web, "ROOT", self.root), patch.object(web, "readiness", return_value={"ready": True}), patch.dict(sys.modules, {"agent": module}), contextlib.redirect_stderr(io.StringIO()):
            web.train(session)
        state = json.loads(session.read_text())
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("TOP_SECRET", session.read_text())


if __name__ == "__main__":
    unittest.main()
