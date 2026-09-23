"""Checks for memory.py (local JSONL backend). Run: uv run python test_memory.py (or pytest)."""
import os
import tempfile

import memory

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)


def _tmp_path() -> str:
    return os.path.join(tempfile.mkdtemp(), "experience.jsonl")


def _rec(run_id: str, task: str, meta: dict, winners: list[dict] | None = None) -> dict:
    return {"run_id": run_id, "task": task, "dataset_card": {"name": run_id}, "meta": meta,
            "winners": winners or [{"name": f"lgbm_{run_id}", "model": "lightgbm", "params": {}, "score": 0.8}]}


SMALL = {"n_rows": 1000, "n_features": 10, "imbalance": 0.5, "landmark_linear": 0.70}
MID = {"n_rows": 10000, "n_features": 50, "imbalance": 0.2, "landmark_linear": 0.80}
BIG = {"n_rows": 200000, "n_features": 300, "imbalance": 0.05, "landmark_linear": 0.90}


def test_store_and_nearest_orders_by_similarity():
    path = _tmp_path()
    for rid, meta in (("big", BIG), ("small", SMALL), ("mid", MID)):
        assert memory.store(_rec(rid, "classification", meta), path=path)
    query = {"n_rows": 1200, "n_features": 12, "imbalance": 0.45, "landmark_linear": 0.71}
    out = memory.nearest(query, "classification", k=2, path=path)
    assert [r["run_id"] for r in out] == ["small", "mid"]
    assert out[0]["similarity"] > out[1]["similarity"]
    assert "created_at" in out[0]


def test_other_task_filtered_out():
    path = _tmp_path()
    memory.store(_rec("reg", "regression", SMALL), path=path)
    memory.store(_rec("cls", "classification", BIG), path=path)
    out = memory.nearest(SMALL, "classification", k=3, path=path)
    assert [r["run_id"] for r in out] == ["cls"]


def test_empty_store_returns_empty():
    path = _tmp_path()
    assert memory.nearest(SMALL, "classification", path=path) == []
    assert memory.warm_start_configs(SMALL, "classification", path=path) == []


def test_corrupt_line_skipped():
    path = _tmp_path()
    memory.store(_rec("a", "classification", SMALL), path=path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("{not json\n\n")
    memory.store(_rec("b", "classification", BIG), path=path)
    out = memory.nearest(SMALL, "classification", k=5, path=path)
    assert {r["run_id"] for r in out} == {"a", "b"}


def test_warm_start_prefixes_names():
    path = _tmp_path()
    winners = [{"name": f"m{i}", "model": "lightgbm", "params": {"n_estimators": 100 * i}, "score": 0.9 - i / 100}
               for i in range(5)]
    memory.store(_rec("a", "classification", SMALL, winners), path=path)
    out = memory.warm_start_configs(SMALL, "classification", k=3, per_run=3, path=path)
    assert [c["name"] for c in out] == ["mem_m0", "mem_m1", "mem_m2"]
    assert winners[0]["name"] == "m0"  # stored record not mutated


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
