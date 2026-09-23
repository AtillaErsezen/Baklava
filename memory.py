"""
Experience memory for warm-starting the search.

After each run we store {run_id, created_at, dataset_card, task, meta, winners}. At the warm-start
phase the nearest past runs (cosine on standardized meta-features) donate their winning configs.
Backend: Supabase table `experience` when SUPABASE_URL and SUPABASE_KEY are set (best-effort),
else local JSONL at runs/experience.jsonl.
"""
import json
import logging
import math
import os
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger(__name__)

LOCAL_PATH = "runs/experience.jsonl"
TABLE = "experience"
COUNT_KEYS = {"n_rows", "n_features", "n_classes", "n_numeric", "n_categorical", "n_missing"}


def _supabase():
    """Supabase client or None (env vars unset, package missing, or connection error)."""
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        log.warning("memory: supabase unavailable, using local file: %s", e)
        return None


def _store_local(record: dict, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        return True
    except OSError as e:
        log.warning("memory: local store failed: %s", e)
        return False


def store(record: dict, path: str | None = None) -> bool:
    """Save one run's experience. Never raises; returns True if saved somewhere."""
    rec = {**record, "created_at": record.get("created_at") or datetime.now(timezone.utc).isoformat()}
    client = _supabase()
    if client is not None:
        try:
            client.table(TABLE).upsert(rec, on_conflict="run_id").execute()
            return True
        except Exception as e:
            log.warning("memory: supabase insert failed, using local file: %s", e)
    return _store_local(rec, path or LOCAL_PATH)


def _load_local(task: str, path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("memory: skipping corrupt line %d in %s", i, path)
                continue
            if isinstance(rec, dict) and rec.get("task") == task:
                out.append(rec)
    return out


def _load(task: str, path: str | None) -> list[dict]:
    """All stored records for a task, from Supabase if reachable, else the local file."""
    client = _supabase()
    if client is not None:
        try:
            return client.table(TABLE).select("*").eq("task", task).execute().data or []
        except Exception as e:
            log.warning("memory: supabase select failed, using local file: %s", e)
    return _load_local(task, path or LOCAL_PATH)


def _num(key: str, value) -> float | None:
    """Numeric meta value, log1p-transformed for counts; None if not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if key in COUNT_KEYS or key.startswith("n_"):
        return math.copysign(math.log1p(abs(value)), value)
    return float(value)


def nearest(meta: dict, task: str, k: int = 3, path: str | None = None) -> list[dict]:
    """Top-k stored records of the same task by cosine similarity on z-scored common meta keys."""
    records = [r for r in _load(task, path) if isinstance(r.get("meta"), dict)]
    query = {key: v for key, v in ((key, _num(key, val)) for key, val in meta.items()) if v is not None}
    if not records or not query:
        return []
    stats = {}
    for key in query:
        vals = [v for v in (_num(key, r["meta"].get(key)) for r in records) if v is not None]
        if vals:
            std = float(np.std(vals))
            stats[key] = (float(np.mean(vals)), std if std > 1e-12 else 1.0)

    scored = []
    for r in records:
        common = [key for key in stats if _num(key, r["meta"].get(key)) is not None]
        if not common:
            continue
        q = np.array([(query[key] - stats[key][0]) / stats[key][1] for key in common])
        x = np.array([(_num(key, r["meta"][key]) - stats[key][0]) / stats[key][1] for key in common])
        denom = float(np.linalg.norm(q) * np.linalg.norm(x))
        sim = float(q @ x) / denom if denom > 0 else 0.0
        scored.append({**r, "similarity": round(sim, 6)})
    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:k]


def warm_start_configs(meta: dict, task: str, k: int = 3, per_run: int = 3,
                       path: str | None = None) -> list[dict]:
    """Winner specs from the k nearest runs (top per_run each), renamed mem_<name>, deduplicated."""
    out, seen = [], set()
    for rec in nearest(meta, task, k, path):
        for w in (rec.get("winners") or [])[:per_run]:
            if not isinstance(w, dict) or "name" not in w:
                continue
            name = w["name"] if str(w["name"]).startswith("mem_") else f"mem_{w['name']}"
            if name in seen:
                continue
            seen.add(name)
            out.append({**w, "name": name})
    return out
