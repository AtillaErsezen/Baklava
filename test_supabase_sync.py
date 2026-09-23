"""Offline checks for supabase_sync.py against a fake client. Run: uv run python test_supabase_sync.py"""
import pandas as pd

from supabase_sync import SupabaseSync

DATASET_REQUIRED = {"name", "original_filename", "version_label", "storage_bucket", "source_sha256",
                    "file_size_bytes", "row_count", "column_count", "column_schema", "preview_rows", "is_demo_fixture"}


class FakeTable:
    def __init__(self, db, name, fail=False):
        self.db, self.name, self.fail, self.op = db, name, fail, None

    def insert(self, row):
        self.op = ("insert", row)
        return self

    def update(self, row):
        self.op = ("update", row)
        return self

    def eq(self, col, val):
        self.op = (*self.op, (col, val))
        return self

    def execute(self):
        if self.fail:
            raise RuntimeError("violates not-null constraint")
        self.db.append((self.name, *self.op))
        return self


class FakeClient:
    def __init__(self, fail_table=None):
        self.db, self.fail_table = [], fail_table

    def table(self, name):
        return FakeTable(self.db, name, fail=name == self.fail_table)


def _df():
    return pd.DataFrame({"a": [1, 2, 3], "y": ["x", "y", "x"]})


def test_start_creates_dataset_then_run_with_required_columns(tmp="runs/_sb.csv"):
    _df().to_csv(tmp, index=False)
    c = FakeClient()
    sb = SupabaseSync(c, "run-1")
    assert sb.start(_df(), tmp, "y", "classification")
    (t1, op1, ds), (t2, op2, run) = c.db[:2]
    assert (t1, op1, t2, op2) == ("datasets", "insert", "runs", "insert")
    assert DATASET_REQUIRED <= set(ds) and ds["preview_rows"] == []  # user rows never go to a readable table
    assert run["id"] == "run-1" and run["dataset_id"] == ds["id"] and run["status"] == "running"


def test_finish_updates_the_run():
    c = FakeClient()
    sb = SupabaseSync(c, "run-2")
    sb.ok = True
    sb.finish("done", report={"markdown": "# r", "spoken_summary": "s"})
    name, op, row, where = c.db[-1]
    assert (name, op, where) == ("runs", "update", ("id", "run-2"))
    assert row["status"] == "done" and row["report_markdown"] == "# r" and "finished_at" in row


def test_failure_disables_sync_without_raising(tmp="runs/_sb.csv"):
    _df().to_csv(tmp, index=False)
    sb = SupabaseSync(FakeClient(fail_table="datasets"), "run-3")
    assert sb.start(_df(), tmp, "y", "classification") is False and sb.ok is False
    sb.finish("done")  # no-op, no exception
    assert SupabaseSync(None, "run-4").start(_df(), tmp, "y", "classification") is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
