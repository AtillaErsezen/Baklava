"""Local web adapter for the existing Modal + Nebius FactoryRun.

The web server invokes inspect/check as short-lived commands and train as a
background process. Session state and events survive browser/Vite restarts.
No alternative training engine or mock scores are used by this adapter.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
MAX_ROWS = 100_000
MAX_COLUMNS = 200


def write_json(path: Path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def inspect_csv(path: Path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            headers = next(reader, [])
            if len(headers) < 2:
                raise ValueError("Choose a comma-separated CSV with a header, at least one feature, and a target column.")
            if len(headers) > MAX_COLUMNS:
                raise ValueError(f"CSV files can contain up to {MAX_COLUMNS} columns.")
            if any(len(h) > 200 for h in headers):
                raise ValueError("Keep column names shorter than 200 characters.")
            if any(not h.strip() for h in headers) or len(set(headers)) != len(headers):
                raise ValueError("Every column needs a non-empty, unique header.")
            columns = [{"name": h, "missing": 0, "numeric": True, "integer": True, "values": set()} for h in headers]
            preview, count = [], 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(headers):
                    raise ValueError(f"CSV row {reader.line_num} has {len(row)} cells; expected {len(headers)}.")
                count += 1
                if count > MAX_ROWS:
                    raise ValueError(f"Choose a dataset with no more than {MAX_ROWS:,} rows.")
                if len(preview) < 5:
                    preview.append([v[:200] for v in row])
                for column, value in zip(columns, row):
                    value = value.strip()
                    if not value or value.lower() in {"nan", "na", "n/a", "null", "none"}:
                        column["missing"] += 1
                        continue
                    if len(column["values"]) <= 20:
                        column["values"].add(value)
                    try:
                        number = float(value)
                        if not math.isfinite(number):
                            column["missing"] += 1
                        elif not number.is_integer():
                            column["integer"] = False
                    except ValueError:
                        column["numeric"] = False
            if count < 50:
                raise ValueError("Use at least 50 data rows so the pipeline can split, cross-validate, and test the dataset.")
            for column in columns:
                column["suggestedTask"] = "regression" if column["numeric"] and (not column["integer"] or len(column["values"]) > 20) else "classification"
                column["type"] = "number" if column.pop("numeric") else "text"
                column.pop("integer")
                column["uniqueSample"] = len(column.pop("values"))
            return {"rows": count, "columns": columns, "preview": preview}
    except UnicodeDecodeError as error:
        raise ValueError("Save your CSV with UTF-8 encoding and try again.") from error
    except csv.Error as error:
        raise ValueError("The CSV contains malformed quotes or oversized cells. Check its formatting.") from error


def validate_target(path: Path, target: str, task: str):
    info = inspect_csv(path)
    column = next((c for c in info["columns"] if c["name"] == target), None)
    if column is None:
        raise ValueError("Choose a target column from this dataset.")
    task = column["suggestedTask"] if task == "auto" else task
    if task not in {"classification", "regression"}:
        raise ValueError("Choose classification, regression, or automatic task detection.")
    if column["missing"]:
        raise ValueError("The target column contains missing or non-finite values. Fill or remove those rows before training.")
    if column["uniqueSample"] < 2:
        raise ValueError("The target must contain at least two different values.")
    if task == "regression" and column["type"] != "number":
        raise ValueError("Regression requires a numeric target column.")
    # Match the actual pandas parser used by FactoryRun (NA tokens and numeric coercion).
    import pandas as pd
    series = pd.read_csv(path)[target]
    if series.isna().any():
        raise ValueError("The target column contains missing values. Fill or remove those rows before training.")
    if series.nunique() < 2:
        raise ValueError("The target must contain at least two different values after CSV parsing.")
    if task == "classification":
        from collections import Counter
        counts = Counter(series.astype(str))
        if len(counts) > 100 or min(counts.values()) < 15:
            raise ValueError("Classification requires at least 15 rows per class and no more than 100 classes. For a continuous number, choose regression.")
    return {"task": task}


def readiness():
    issues = []
    packages = ("modal", "pandas", "numpy", "sklearn", "openai", "scipy", "statsmodels", "baycomp", "pyarrow", "tavily")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        issues.append({"label": "Python dependencies", "detail": "Run uv sync in the repository root.", "command": "uv sync"})
    if not os.environ.get("NEBIUS_API_KEY", "").strip():
        issues.append({"label": "Nebius API key", "detail": "Set NEBIUS_API_KEY in the repository’s .env file."})
    modal_ready = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    if not modal_ready and "modal" not in missing:
        try:
            from modal.config import config
            modal_ready = bool(config.get("token_id") and config.get("token_secret"))
        except Exception:
            pass
    if not modal_ready:
        issues.append({"label": "Modal credentials", "detail": "Authenticate Modal from this machine, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.", "command": "uv run modal setup"})
    return {"ready": not issues, "issues": issues, "deployment": "ml-factory", "deploymentNote": "Deploy modal_train.py to your Modal workspace before the first run."}


def train(session_path: Path):
    state = json.loads(session_path.read_text())
    state.update(status="running", stage="Preparing the pipeline", pid=os.getpid(), startedAt=time.time())
    write_json(session_path, state)
    runner = None
    try:
        check = readiness()
        if not check["ready"]:
            raise ValueError("Complete the training setup: " + "; ".join(i["label"] for i in check["issues"]))
        data_path = ROOT / "runs" / ".web" / "datasets" / state["datasetId"] / "dataset.csv"
        task = validate_target(data_path, state["target"], state["task"])["task"]
        from agent import FactoryRun

        class WebRun(FactoryRun):
            def save(self):
                prefix = ROOT / "runs" / self.run_id
                events_path = Path(f"{prefix}_events.jsonl")
                temporary = events_path.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(clean(e), allow_nan=False, default=str) + "\n" for e in self.events), encoding="utf-8")
                temporary.replace(events_path)
                write_json(Path(f"{prefix}_results.json"), clean({"profile": self.profile, "results": self.results, "final": self.final}))
                if self.report:
                    Path(f"{prefix}_report.md").write_text(self.report["markdown"], encoding="utf-8")

            def emit(self, kind, payload):
                # Persist full event payloads before optional cloud event publishing.
                event = clean({"run_id": self.run_id, "ts": time.time(), "kind": kind, "payload": payload})
                path = ROOT / "runs" / f"{self.run_id}_events.jsonl"
                with path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(event, allow_nan=False, default=str) + "\n")
                state.update(runId=self.run_id, eventCount=state.get("eventCount", 0) + 1, updatedAt=time.time())
                stages = {"run_start": "Profiling your dataset", "diagnostics": "Checking data quality", "search_plan": "Searching model candidates", "rung": "Comparing model candidates", "leaderboard": "Evaluating model scores", "confirm": "Testing the finalists", "final_model": "Saving the best-fit model", "report": "Writing your report", "usage": "Finishing up"}
                if kind in stages:
                    state["stage"] = stages[kind]
                if kind == "status":
                    state["message"] = str(payload.get("msg", ""))[:500]
                elif kind == "thought":
                    state["message"] = str(payload.get("text", ""))[:500]
                write_json(session_path, state)
                super().emit(kind, payload)

        runner = WebRun(str(data_path), state["target"], task_hint=task, provider="nebius", purpose=state.get("purpose") or None)
        runner.dataset_name = state["filename"]
        state.update(runId=runner.run_id, task=task)
        write_json(session_path, state)
        runner.run()
        if not runner.final or not runner.report or not runner.final.get("model_path"):
            raise ValueError("The agent ended without a final model and report. Review the partial results, then try again.")
        state.update(status="completed", stage="Your results are ready", resultFile=f"{runner.run_id}_events.jsonl")
    except BaseException as error:
        # Never send provider exception text, keys, or raw logs to the browser.
        traceback.print_exc()
        if runner:
            try:
                runner.save()
            except Exception:
                traceback.print_exc()
        message = str(error) if isinstance(error, ValueError) and (runner is None or "agent ended" in str(error)) else "The training pipeline stopped. Check the session log for provider, deployment, or dataset errors, then retry."
        for value in (os.environ.get(k) for k in ("NEBIUS_API_KEY", "SUPABASE_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "TAVILY_API_KEY")):
            if value:
                message = message.replace(value, "[redacted]")
        state.update(status="failed", stage="Training could not finish", error=message[:600])
        if runner and runner.events:
            state["resultFile"] = f"{runner.run_id}_events.jsonl"
    finally:
        state["finishedAt"] = time.time()
        write_json(session_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "inspect", "validate", "train"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--task", default="auto")
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(args.session)
        return
    try:
        result = readiness() if args.command == "check" else inspect_csv(args.data) if args.command == "inspect" else validate_target(args.data, args.target, args.task)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
