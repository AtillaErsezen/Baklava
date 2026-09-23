"""
ML Factory web app: the product API and the static frontend, as a separate Modal app.

  modal deploy web_app.py                    # needs `modal deploy modal_train.py` and the ml-factory-env Secret
  uv run python -m pytest -q test_web_app.py

Flow: POST /api/datasets (csv) -> profile -> POST /api/runs {dataset_id, target} -> run_agent runs FactoryRun
-> the UI follows events live from Supabase, or replays GET /api/runs/<run_id>/events.
"""
import base64
import contextlib
import hmac
import io
import json
import logging
import os
import re
import stat
import time
import uuid
import zipfile
from urllib.parse import urlsplit

import modal
import pandas as pd
from fastapi import FastAPI, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from modal_train import DATA_DIR, check_name, image as train_image, load_local, vol

log = logging.getLogger("web_app")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
FORM_OVERHEAD_BYTES = 64 * 1024  # multipart boundaries and the access_code field
SMALL_BODY_BYTES = 64 * 1024     # every request body except the csv upload
MAX_ROWS, MAX_COLUMNS, MIN_ROWS = 1_000_000, 500, 50
MAX_PURPOSE = 500
PREVIEW_ROWS, PREVIEW_CHARS, EXAMPLE_CHARS = 20, 60, 40
MAX_ACTIVE_RUNS = 3
MAX_RUNS_PER_HOUR = 5            # per client ip
MAX_UPLOADS_PER_HOUR = 20        # per client ip
MAX_FAILED_CODES = 20            # wrong access codes per client ip per hour before a lockout
WINDOW_S = 3600
STALE_S = 2 * 3600               # a queued/running entry older than this died with its container
MAX_EVENTS_BYTES = MAX_EXPORT_BYTES = 20 * 1024 * 1024
MAX_ERROR_CHARS = 200
ACTIVE = ("queued", "running")
TARGET_HINTS = ("target", "label", "churn", "price")
CHUNK = 1024 * 1024


# ------------------------------------------------------------------ helpers (no Modal, testable)

def _err(status: int, msg: str) -> HTTPException:
    return HTTPException(status_code=status, detail=msg)


def _valid_id(value: str) -> str:
    """check_name, as a 400 instead of a ValueError."""
    try:
        return check_name(value)
    except ValueError:
        raise _err(400, "bad id") from None


def _new_id(n_hex: int) -> str:
    return check_name(f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:n_hex]}")


def _recent(store, key: str, now: float) -> list[float]:
    return [t for t in (store.get(key) or []) if now - t < WINDOW_S]


def _hit(store, key: str, now: float) -> None:
    store[key] = _recent(store, key, now) + [now]


def _active_runs(store, now: float) -> int:
    # ponytail: scans the whole Dict per run request; keep an index key if runs reach the thousands
    return sum(1 for _, v in store.items()
               if isinstance(v, dict) and v.get("status") in ACTIVE and now - v.get("ts", 0) < STALE_S)


def _client_ip(request: Request) -> str:
    """The peer address: Modal's proxy puts the real client ip there (its docs point rate limits at it).
    X-Forwarded-For is client-controlled, so trusting it would let anyone dodge the per-ip limits."""
    return request.client.host if request.client else "unknown"


def _authorize(store, request: Request, code: str) -> str:
    """Check the shared access code; return a keyed hash of the client ip for rate-limit keys."""
    expected = os.environ.get("ACCESS_CODE", "")
    if not expected:
        raise _err(503, "server not configured")
    ip = hmac.new(expected.encode(), _client_ip(request).encode(), "sha256").hexdigest()[:32]
    now = time.time()
    if len(_recent(store, f"fail:{ip}", now)) >= MAX_FAILED_CODES:
        raise _err(429, "too many wrong access codes, try again later")
    if not hmac.compare_digest(code.encode(), expected.encode()):
        _hit(store, f"fail:{ip}", now)
        raise _err(401, "wrong access code")
    return ip


def _save_capped(src, dest: str, cap: int) -> None:
    """Copy an upload in chunks, refusing (413) once it passes cap bytes."""
    size = 0
    with open(dest, "wb") as out:
        while chunk := src.read(CHUNK):
            size += len(chunk)
            if size > cap:
                raise _err(413, f"file too large (max {cap:,} bytes)")
            out.write(chunk)


def _load_csv(path: str) -> pd.DataFrame:
    """Parse with the same loader FactoryRun uses, then enforce the shape limits."""
    try:
        df = load_local(path)
    except Exception:
        raise _err(400, "could not parse the file as csv") from None
    if not 2 <= df.shape[1] <= MAX_COLUMNS:
        raise _err(400, f"need 2 to {MAX_COLUMNS} columns")
    if not MIN_ROWS <= len(df) <= MAX_ROWS:
        raise _err(400, f"need {MIN_ROWS} to {MAX_ROWS:,} rows")
    return df


def _store_dataset(upload: UploadFile, uploads: str, cap: int) -> tuple[str, pd.DataFrame]:
    """Save the upload as <uploads>/<dataset_id>.csv (server-chosen name) and validate it; nothing stays on error."""
    if not (upload.filename or "").lower().endswith(".csv"):
        raise _err(400, "upload a .csv file")
    os.makedirs(uploads, exist_ok=True)
    dataset_id = _new_id(16)
    dest = os.path.join(uploads, f"{dataset_id}.csv")
    try:
        _save_capped(upload.file, dest, cap)
        return dataset_id, _load_csv(dest)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.remove(dest)
        raise


def _cell(v, n: int) -> str | None:
    return None if pd.isna(v) else str(v)[:n]


def _suggest_target(cols: list) -> str:
    low = [str(c).lower() for c in cols]
    for hint in TARGET_HINTS:
        for c, name in zip(cols, low):
            if hint in name:
                return c
    return next((c for c, name in zip(cols, low) if name == "y"), cols[-1])


def profile_dataset(df: pd.DataFrame) -> dict:
    """Cheap profile for the target picker (no model fitting). The preview goes only to the uploader."""
    target = _suggest_target(list(df.columns))
    y = df[target]
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    columns = []
    for c in df.columns:
        s = df[c]
        first = s.first_valid_index()
        columns.append({"name": str(c), "dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 1),
                        "n_unique": int(s.nunique()),
                        "example": None if first is None else _cell(s.loc[first], EXAMPLE_CHARS)})
    return {"rows": len(df), "columns": columns,
            "preview": [{str(k): _cell(v, PREVIEW_CHARS) for k, v in row.items()}
                        for row in df.head(PREVIEW_ROWS).to_dict("records")],
            "suggested_target": str(target),
            "suggested_task": "classification" if not numeric or y.nunique() <= 20 else "regression",
            "time_columns": [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]}


def _check_target(path: str, target: str) -> None:
    if target not in pd.read_csv(path, nrows=0).columns:
        raise _err(400, "target column not found in the dataset")
    if pd.read_csv(path, usecols=[target])[target].isna().all():
        raise _err(400, "target column is empty")


def _read_capped(path: str, cap: int) -> bytes:
    """Read a regular file without following symlinks: 404 if absent or not a file, 500 if over cap."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise _err(404, "not found") from None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _err(404, "not found")
        data = f.read(cap + 1)
    if len(data) > cap:
        raise _err(500, "file too large to serve")
    return data


def _jsonl(data: bytes) -> list:
    out = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # blank, or a last line not fully synced yet
    return out


def _zip_dir(folder: str) -> bytes:
    """Zip the regular files directly in folder (no subfolders, no symlinks), MAX_EXPORT_BYTES in total."""
    buf, total = io.BytesIO(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for entry in sorted(os.scandir(folder), key=lambda e: e.name):
            if entry.is_file(follow_symlinks=False):
                data = _read_capped(entry.path, MAX_EXPORT_BYTES - total)
                total += len(data)
                z.writestr(entry.name, data)
    return buf.getvalue()


def _public_key() -> str:
    """SUPABASE_PUBLISHABLE_KEY, or "" if it is really a secret key (a misconfigured Secret)."""
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
    if not key or key == os.environ.get("SUPABASE_KEY") or key.startswith("sb_secret_"):
        if key:
            log.error("SUPABASE_PUBLISHABLE_KEY holds a secret key; not serving it")
        return ""
    try:  # legacy JWT keys: anon is public, service_role is not
        part = key.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (IndexError, ValueError):
        return key
    return "" if isinstance(claims, dict) and claims.get("role") == "service_role" else key


def _csp(supabase_url: str) -> str:
    host = urlsplit(supabase_url or "").hostname or ""
    supabase = f" https://{host} wss://{host}" if supabase_url.startswith("https://") and \
        re.fullmatch(r"[a-z0-9.-]+", host) else ""
    return ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net https://esm.sh; "
            "style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
            f"connect-src 'self'{supabase}; img-src 'self' data: blob:; worker-src blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")


class _BodyCap:
    """Abort a request once its body passes the cap. Starlette spools a multipart upload to disk before the
    handler runs, so this is the only place a cap on the raw stream can live."""

    def __init__(self, app, upload_cap: int):
        self.app, self.upload_cap = app, upload_cap

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cap = self.upload_cap if scope["path"] == "/api/datasets" else SMALL_BODY_BYTES
        seen = 0

        async def capped():
            nonlocal seen
            msg = await receive()
            seen += len(msg.get("body", b""))
            if seen > cap:
                raise _err(413, "request too large")  # FastAPI re-raises HTTPException from body parsing
            return msg

        await self.app(scope, capped, send)


class RunRequest(BaseModel):
    dataset_id: str = Field(max_length=64)
    target: str = Field(min_length=1, max_length=1024)
    purpose: str = Field("", max_length=MAX_PURPOSE)
    access_code: str = Field("", max_length=256)  # missing -> 401 (or 503), not 400


# ------------------------------------------------------------------ the API

def create_app(spawn, status_store, storage_root: str, *, commit=None, refresh=None, web_dir: str | None = None,
               max_upload_bytes: int = MAX_UPLOAD_BYTES) -> FastAPI:
    """spawn(run_id, csv_path, target, purpose) starts a run; status_store is dict-like (a modal.Dict in prod);
    commit/refresh publish and pick up volume writes across containers."""
    store, uploads, runs_dir = status_store, os.path.join(storage_root, "uploads"), os.path.join(storage_root, "runs")
    headers = {"Content-Security-Policy": _csp(os.environ.get("SUPABASE_URL", "")),
               "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(_BodyCap, upload_cap=max_upload_bytes + FORM_OVERHEAD_BYTES)

    @api.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.update(headers)
        return response

    @api.exception_handler(StarletteHTTPException)
    async def http_error(_, exc):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(_, exc):
        fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
        return JSONResponse({"error": f"missing or invalid field: {', '.join(fields) or 'request'}"}, status_code=400)

    @api.exception_handler(Exception)
    async def crashed(_, exc):
        log.error("unhandled error", exc_info=exc)
        return JSONResponse({"error": "internal error"}, status_code=500)

    def _refresh():
        if refresh:
            try:
                refresh()
            except Exception:
                log.warning("volume reload failed", exc_info=True)

    def _dataset_path(dataset_id: str) -> str:
        path = os.path.join(uploads, f"{_valid_id(dataset_id)}.csv")
        _refresh()
        if not os.path.isfile(path):
            raise _err(404, "unknown dataset")
        return path

    @api.get("/api/config")
    def config():
        return {"supabase_url": os.environ.get("SUPABASE_URL", ""), "supabase_publishable_key": _public_key()}

    @api.post("/api/datasets")
    def create_dataset(request: Request, file: UploadFile, access_code: str = Form("", max_length=256)):
        ip = _authorize(store, request, access_code)
        now = time.time()
        if len(_recent(store, f"up:{ip}", now)) >= MAX_UPLOADS_PER_HOUR:
            raise _err(429, f"limit of {MAX_UPLOADS_PER_HOUR} uploads per hour reached")
        dataset_id, df = _store_dataset(file, uploads, max_upload_bytes)
        if commit:
            commit()
        _hit(store, f"up:{ip}", now)
        return {"dataset_id": dataset_id, "profile": profile_dataset(df)}

    @api.get("/api/datasets/{dataset_id:path}")
    def get_dataset(dataset_id: str, request: Request, x_access_code: str = Header("", max_length=256)):
        _authorize(store, request, x_access_code)
        return {"dataset_id": dataset_id, "profile": profile_dataset(_load_csv(_dataset_path(dataset_id)))}

    @api.post("/api/runs")
    def create_run(body: RunRequest, request: Request):
        ip = _authorize(store, request, body.access_code)
        now = time.time()
        if len(_recent(store, f"run:{ip}", now)) >= MAX_RUNS_PER_HOUR:
            raise _err(429, f"limit of {MAX_RUNS_PER_HOUR} runs per hour reached")
        # ponytail: check-then-set across containers, a burst can overshoot by a run or two; fine for a demo
        if _active_runs(store, now) >= MAX_ACTIVE_RUNS:
            raise _err(429, "the factory is busy, try again in a few minutes")
        csv_path = _dataset_path(body.dataset_id)
        _check_target(csv_path, body.target)
        run_id = _new_id(6)
        store[run_id] = {"status": "queued", "ts": now}
        try:
            spawn(run_id, csv_path, body.target, body.purpose.strip() or None)
        except Exception:
            log.exception("spawn failed for run %s", run_id)
            store[run_id] = {"status": "failed", "error": "could not start the run", "ts": now}
            raise _err(503, "could not start the run, try again") from None
        _hit(store, f"run:{ip}", now)
        return {"run_id": run_id}

    @api.get("/api/runs/{run_id}/events")
    def run_events(run_id: str):
        path = os.path.join(runs_dir, f"{_valid_id(run_id)}_events.jsonl")
        _refresh()
        return _jsonl(_read_capped(path, MAX_EVENTS_BYTES))

    @api.get("/api/runs/{run_id}/export.zip")
    def run_export(run_id: str):
        folder = os.path.join(runs_dir, f"{_valid_id(run_id)}_export")
        _refresh()
        if os.path.islink(folder) or not os.path.isdir(folder):
            raise _err(404, "no export for this run")
        return Response(_zip_dir(folder), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}_export.zip"'})

    @api.get("/api/runs/{run_id:path}")  # declared last: the :path form also catches "../x" so it gets a 400
    def run_status(run_id: str):
        entry = store.get(_valid_id(run_id))
        if not isinstance(entry, dict):
            raise _err(404, "unknown run")
        out = {"run_id": run_id, "status": entry.get("status")}
        if out["status"] in ACTIVE and time.time() - entry.get("ts", 0) > STALE_S:
            out.update(status="failed", error="run timed out")
        elif entry.get("error"):
            out["error"] = entry["error"]
        return out

    if web_dir and os.path.isdir(web_dir):
        api.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return api


# ------------------------------------------------------------------ Modal wiring

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("agent", "factory_tools", "providers", "supabase_sync", "modal_train", "diagnostics", "sampling",
           "search_space", "racing", "stats_tests", "memory", "export", "external_data", "purpose", "report",
           "ensemble", "forecasting", "usecases")
SOURCES = tuple(m for m in SOURCES if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{m}.py")))
RUNS_DIR = f"{DATA_DIR}/runs"
WEB_DIR = "/root/web"

app = modal.App("ml-factory-web")
secrets = [modal.Secret.from_name("ml-factory-env")]
runs = modal.Dict.from_name("ml-factory-runs", create_if_missing=True)
image = (
    train_image
    # ponytail: agent deps unpinned past the pyproject floors, pin once a deployed run is green
    .uv_pip_install("fastapi[standard]", "python-multipart", "openai>=3.19.0", "supabase>=2.31.0",
                    "tavily-python>=0.8.4", "baycomp>=1.0.3", "scipy>=1.18.1", "statsmodels>=0.15.0")
    .add_local_python_source(*SOURCES)
    .add_local_dir(os.path.join(HERE, "prompts"), "/root/prompts")
)
if os.path.isdir(os.path.join(HERE, "web")):  # the frontend is optional: the API serves without it
    image = image.add_local_dir(os.path.join(HERE, "web"), WEB_DIR)


@app.function(image=image, secrets=secrets, volumes={DATA_DIR: vol}, cpu=2, memory=4096, timeout=3600)
def run_agent(run_id: str, csv_path: str, target: str, purpose: str | None) -> None:
    """One agent run: status running -> FactoryRun -> done or failed (short error), then commit the volume."""
    check_name(run_id)
    runs[run_id] = {"status": "running", "ts": time.time()}
    try:
        vol.reload()
        from agent import FactoryRun

        FactoryRun(csv_path, target, provider="nebius", purpose=purpose, run_id=run_id, runs_dir=RUNS_DIR).run()
        runs[run_id] = {"status": "done", "ts": time.time()}
    except (Exception, SystemExit) as e:  # FactoryRun raises SystemExit on a missing target
        log.exception("run %s failed", run_id)
        runs[run_id] = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS], "ts": time.time()}
    finally:
        vol.commit()


@app.function(image=image, secrets=secrets, volumes={DATA_DIR: vol}, memory=4096)
@modal.asgi_app()
def web():
    return create_app(run_agent.spawn, runs, DATA_DIR, commit=vol.commit, refresh=vol.reload, web_dir=WEB_DIR)
