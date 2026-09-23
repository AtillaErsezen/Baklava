"""Offline checks of the web API: fake spawn, a plain dict as the status store, files in a tmp dir.
Run: uv run python -m pytest -q test_web_app.py"""
import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

import web_app

CODE = "correct-horse-battery"


def _csv(rows=60) -> bytes:
    return ("x,y\n" + "".join(f"{i},{i % 2}\n" for i in range(rows))).encode()


def _client(tmp_path, monkeypatch, code=CODE, **kw):
    if code is None:
        monkeypatch.delenv("ACCESS_CODE", raising=False)
    else:
        monkeypatch.setenv("ACCESS_CODE", code)
    calls, store = [], {}
    app = web_app.create_app(lambda *a: calls.append(a), store, str(tmp_path), **kw)
    return TestClient(app), calls, store


def _ip(client, ip):
    """The same app seen from another client address."""
    return TestClient(client.app, client=(ip, 50000)) if ip else client


def _upload(client, body=None, name="data.csv", code=CODE, ip=None):
    return _ip(client, ip).post("/api/datasets", files={"file": (name, _csv() if body is None else body, "text/csv")},
                                data={"access_code": code})


def _dataset(client, body=None) -> str:
    r = _upload(client, body)
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def _run(client, dataset_id, target="y", purpose="", code=CODE, ip=None):
    return _ip(client, ip).post("/api/runs", json={"dataset_id": dataset_id, "target": target, "purpose": purpose,
                                                   "access_code": code})


# ---- access code

def test_wrong_access_code_is_401(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch)
    assert _upload(client, code="guess").status_code == 401
    r = _run(client, _dataset(client), code="guess")
    assert r.status_code == 401 and "error" in r.json() and not calls


def test_unset_access_code_refuses_everything_with_503(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch, code=None)
    r = _upload(client, code="")
    assert r.status_code == 503 and r.json() == {"error": "server not configured"}
    assert _run(client, "20260923-120000-abcdef", code="").status_code == 503 and not calls


def test_repeated_wrong_codes_lock_out_that_ip(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    for _ in range(web_app.MAX_FAILED_CODES):
        assert _upload(client, code="guess", ip="6.6.6.6").status_code == 401
    assert _upload(client, ip="6.6.6.6").status_code == 429
    assert _upload(client, ip="7.7.7.7").status_code == 200


# ---- datasets

def test_non_csv_is_400(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    assert _upload(client, name="data.exe").status_code == 400
    assert _upload(client, body=b"\x00\x01\x02 not a table \xff").status_code == 400
    assert not os.listdir(tmp_path / "uploads")


def test_oversize_is_413_and_leaves_no_file(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch, max_upload_bytes=1000)
    assert _upload(client, body=_csv(300)).status_code == 413              # past the file cap, inside the body cap
    assert _upload(client, body=b"1,0\n" * 50_000).status_code == 413      # past the whole-body cap
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not os.listdir(uploads)


def test_small_or_single_column_or_formless_upload_is_400(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    assert _upload(client, body=_csv(10)).status_code == 400
    assert _upload(client, body=b"x\n" + b"1\n" * 60).status_code == 400
    r = client.post("/api/datasets", data={"access_code": CODE})
    assert r.status_code == 400 and "file" in r.json()["error"]           # missing field: 400, not 422
    assert client.post("/api/datasets", files={"file": ("d.csv", _csv(), "text/csv")}).status_code == 401


def test_upload_returns_a_profile_with_suggested_target(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    rows = "".join(f"{i},2024-01-{i % 28 + 1:02d},{'plan ' * 30}{i},{i % 2},{i * 1.5}\n" for i in range(60))
    r = _upload(client, body=("id,signup_date,plan,Churn,tenure\n" + rows).encode())
    assert r.status_code == 200, r.text
    body = r.json()
    p = body["profile"]
    assert p["rows"] == 60 and p["suggested_target"] == "Churn" and p["suggested_task"] == "classification"
    assert p["time_columns"] == ["signup_date"]
    assert [c["name"] for c in p["columns"]] == ["id", "signup_date", "plan", "Churn", "tenure"]
    assert all(len(c["example"]) <= 40 for c in p["columns"])
    assert len(p["preview"]) == 20 and all(len(v) <= 60 for row in p["preview"] for v in row.values())
    assert os.path.isfile(tmp_path / "uploads" / f"{body['dataset_id']}.csv")

    rows = "".join(f"{i},{i * 3.7 + 0.1}\n" for i in range(80))
    p = _upload(client, body=("rooms,amount\n" + rows).encode()).json()["profile"]
    assert (p["suggested_target"], p["suggested_task"]) == ("amount", "regression")


def test_get_dataset_needs_the_access_code_header(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    r = _upload(client)
    dataset_id, profile = r.json()["dataset_id"], r.json()["profile"]
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 401
    assert client.get(f"/api/datasets/{dataset_id}", headers={"X-Access-Code": "guess"}).status_code == 401
    ok = client.get(f"/api/datasets/{dataset_id}", headers={"X-Access-Code": CODE})
    assert ok.status_code == 200 and ok.json() == {"dataset_id": dataset_id, "profile": profile}
    assert len(ok.json()["profile"]["preview"]) <= 20
    assert client.get("/api/datasets/..%2Fx", headers={"X-Access-Code": CODE}).status_code == 400
    assert client.get("/api/datasets/nope", headers={"X-Access-Code": CODE}).status_code == 404


# ---- runs

def test_bad_run_request_is_400_or_404(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch)
    ds = _dataset(client)
    assert _run(client, ds, target="nope").status_code == 400
    assert _run(client, ds, purpose="x" * 501).status_code == 400
    assert _run(client, "../x").status_code == 400
    assert _run(client, "20260923-120000-abcdef").status_code == 404
    assert _run(client, _dataset(client, b"x,y\n" + b"1,\n" * 60)).status_code == 400  # target all null
    r = client.post("/api/runs", json={"access_code": CODE})
    assert r.status_code == 400 and "error" in r.json()
    assert not calls


def test_valid_run_spawns_once_with_a_path_in_uploads(tmp_path, monkeypatch):
    client, calls, store = _client(tmp_path, monkeypatch)
    ds = _dataset(client)
    r = _run(client, ds, purpose="find churners")
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert len(calls) == 1
    got_id, csv_path, target, purpose = calls[0]
    assert got_id == run_id and os.path.dirname(os.path.realpath(csv_path)) == os.path.realpath(tmp_path / "uploads")
    assert open(csv_path, "rb").read() == _csv() and (target, purpose) == ("y", "find churners")
    assert client.get(f"/api/runs/{run_id}").json() == {"run_id": run_id, "status": "queued"}
    store[run_id] = {"status": "failed", "error": "ValueError: boom", "ts": 0}
    assert client.get(f"/api/runs/{run_id}").json()["error"] == "ValueError: boom"


def test_failed_spawn_is_503_and_marks_the_run_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", CODE)
    store = {}
    client = TestClient(web_app.create_app(lambda *a: 1 / 0, store, str(tmp_path)))
    r = _run(client, _dataset(client))
    assert r.status_code == 503 and "error" in r.json()
    assert [v["status"] for v in store.values() if isinstance(v, dict)] == ["failed"]


def test_fourth_concurrent_run_is_429(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch)
    ds = _dataset(client)
    assert [_run(client, ds, ip=f"10.0.0.{i}").status_code for i in range(3)] == [200, 200, 200]
    assert _run(client, ds, ip="10.0.0.9").status_code == 429 and len(calls) == 3


def test_sixth_run_per_ip_per_hour_is_429(tmp_path, monkeypatch):
    client, _, store = _client(tmp_path, monkeypatch)
    ds = _dataset(client)
    for _ in range(web_app.MAX_RUNS_PER_HOUR):
        assert _run(client, ds, ip="1.2.3.4").status_code == 200
        for k in list(store):  # finish runs so the concurrency cap does not trip first
            if isinstance(store[k], dict):
                store[k] = {**store[k], "status": "done"}
    assert _run(client, ds, ip="1.2.3.4").status_code == 429
    spoof = TestClient(client.app, client=("1.2.3.4", 50000), headers={"X-Forwarded-For": "9.9.9.9"})
    assert spoof.post("/api/runs", json={"dataset_id": ds, "target": "y", "access_code": CODE}).status_code == 429
    assert _run(client, ds, ip="5.6.7.8").status_code == 200


def test_bad_run_id_is_400_and_unknown_is_404(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    for path in ("/api/runs/..%2Fx", "/api/runs/..%2Fx/events", "/api/runs/.env/export.zip", "/api/runs/run:abc"):
        r = client.get(path)
        assert r.status_code == 400 and "error" in r.json(), path
    assert client.get("/api/runs/20260923-120000-abcdef").status_code == 404


def test_events_and_export_are_404_then_200(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    rid = "20260923-120000-abcdef"
    assert client.get(f"/api/runs/{rid}/events").status_code == 404
    assert client.get(f"/api/runs/{rid}/export.zip").status_code == 404

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / f"{rid}_events.jsonl").write_text('{"kind": "run_start"}\n{"kind": "usage"}\n', encoding="utf-8")
    r = client.get(f"/api/runs/{rid}/events")
    assert r.status_code == 200 and [e["kind"] for e in r.json()] == ["run_start", "usage"]

    export = runs / f"{rid}_export"
    (export / "sub").mkdir(parents=True)
    (export / "MODEL_CARD.md").write_text("# card", encoding="utf-8")
    (export / "sub" / "nested.txt").write_text("skip me", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak", encoding="utf-8")
    os.symlink(secret, export / "link.txt")
    r = client.get(f"/api/runs/{rid}/export.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert zipfile.ZipFile(io.BytesIO(r.content)).namelist() == ["MODEL_CARD.md"]


# ---- config, headers, static

def test_config_never_returns_the_secret_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_TOPSECRET")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_ok")
    client, _, _ = _client(tmp_path, monkeypatch)
    r = client.get("/api/config")
    assert r.json() == {"supabase_url": "https://proj.supabase.co", "supabase_publishable_key": "sb_publishable_ok"}
    assert "TOPSECRET" not in r.text
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_secret_TOPSECRET")  # misconfigured Secret
    assert "TOPSECRET" not in client.get("/api/config").text


def test_security_headers_and_no_cors(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    client, _, _ = _client(tmp_path, monkeypatch)
    r = client.get("/api/config", headers={"Origin": "https://evil.example"})
    csp = r.headers["content-security-policy"]
    for part in ("default-src 'self'", "script-src 'self' https://cdn.jsdelivr.net https://esm.sh",
                 "style-src 'self' https://fonts.googleapis.com", "https://fonts.gstatic.com",
                 "connect-src 'self' https://proj.supabase.co wss://proj.supabase.co",
                 "img-src 'self' data: blob:", "worker-src blob:"):
        assert part in csp, part
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["referrer-policy"] == "no-referrer"
    assert "access-control-allow-origin" not in r.headers
    assert "content-security-policy" in client.get("/api/runs/nope-404").headers  # errors carry headers too


def test_static_frontend_is_optional(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch, web_dir=str(tmp_path / "missing"))
    assert client.get("/api/config").status_code == 200
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>ML Factory</h1>", encoding="utf-8")
    client, _, _ = _client(tmp_path, monkeypatch, web_dir=str(site))
    r = client.get("/")
    assert r.status_code == 200 and "ML Factory" in r.text and "content-security-policy" in r.headers


# ---- the agent.py hook the web runner relies on

def test_factory_run_takes_a_given_run_id_and_runs_dir(tmp_path, monkeypatch):
    import agent
    from test_factory_tools import FakeFn

    csv = tmp_path / "d.csv"
    csv.write_bytes(_csv(300))
    monkeypatch.setattr(agent.modal.Function, "from_name", lambda app, name: FakeFn(name))
    run = agent.FactoryRun(str(csv), "y", provider="scripted", run_id="web-run_1", runs_dir=str(tmp_path / "out"))
    run.save()
    assert run.run_id == "web-run_1" and (tmp_path / "out" / "web-run_1_events.jsonl").exists()
    with pytest.raises(ValueError):
        agent.FactoryRun(str(csv), "y", provider="scripted", run_id="../escape")
