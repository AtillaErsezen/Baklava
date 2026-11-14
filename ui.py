"""ML Factory — local web UI.

  uv run ui.py        # then open http://127.0.0.1:8000  (PORT=... to change)
"""
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import agent
import train

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = "uploads"
STATE = {"run": None, "thread": None, "error": None, "stop": False}  # one run at a time


class Stopped(BaseException):
    """Not an Exception on purpose: the agent's `except Exception` handlers must not swallow it."""


class UIRun(agent.FactoryRun):
    def emit(self, kind, payload):
        if STATE["stop"]:
            raise Stopped
        super().emit(kind, payload)


def _work(kwargs):
    try:
        STATE["run"] = run = UIRun(**kwargs)
        run.run()
    except Stopped:
        STATE["error"] = "Run stopped. Nothing was saved."
    except BaseException as e:  # SystemExit from a bad target included
        STATE["error"] = f"{type(e).__name__}: {e}"


def _running():
    return bool(STATE["thread"] and STATE["thread"].is_alive())


def _llm_ok():
    """True if something answers at LLM_BASE_URL (an auth error still counts)."""
    try:
        urllib.request.urlopen(agent.BASE_URL.rstrip("/") + "/models", timeout=1)
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # the agent already prints its events; polling would flood the log
        pass

    def _send(self, body, code=200, ctype="application/json", extra=()):
        if not isinstance(body, bytes):
            body = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _local(self):
        """Only this machine's browser, on this page: blocks DNS rebinding and cross-site POSTs."""
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        ok = host.split(":")[0] in ("127.0.0.1", "localhost") and (not origin or urlparse(origin).netloc == host)
        if not ok:
            self._send({"error": "This UI only answers requests from its own page."}, 403)
        return ok

    def do_GET(self):
        if not self._local():
            return
        url = urlparse(self.path)
        run = STATE["run"]
        if url.path == "/":
            with open(os.path.join(HERE, "ui.html"), "rb") as f:
                self._send(f.read(), ctype="text/html; charset=utf-8")
        elif url.path == "/config":
            self._send({"model": agent.MODEL, "base_url": agent.BASE_URL, "llm_ok": _llm_ok(),
                        "has_run": run is not None or _running()})
        elif url.path == "/events":
            since = int(parse_qs(url.query).get("since", ["0"])[0])
            self._send({"events": run.events[since:] if run else [], "running": _running(), "error": STATE["error"]})
        elif url.path == "/model" and run and run.final:
            with open(run.final["model_path"], "rb") as f:
                name = os.path.basename(run.final["model_path"])
                self._send(f.read(), ctype="application/octet-stream",
                           extra=[("Content-Disposition", f'attachment; filename="{name}"')])
        else:
            self._send({"error": "Not found."}, 404)

    def do_POST(self):
        if not self._local():
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/upload":
            name = os.path.basename(unquote(self.headers.get("X-Filename", "")))
            if not name.lower().endswith((".csv", ".parquet")):
                return self._send({"error": "Use a .csv or .parquet file."}, 400)
            os.makedirs(UPLOADS, exist_ok=True)
            path = os.path.join(UPLOADS, name)
            with open(path, "wb") as f:
                f.write(body)
            try:
                df = train.load_local(path)
            except Exception as e:
                return self._send({"error": f"Could not read {name}: {e}"}, 400)
            self._send({"file": name, "rows": len(df), "columns": [str(c) for c in df.columns]})
        elif self.path == "/run":
            if _running():
                return self._send({"error": "A run is already in progress."}, 409)
            try:
                req = json.loads(body)
                path = os.path.join(UPLOADS, os.path.basename(req["file"]))
                kwargs = {"data_path": path, "target": req["target"], "task_hint": req.get("task") or None,
                          "goal": req.get("goal") or None, "no_llm": bool(req.get("no_llm"))}
            except (ValueError, KeyError, TypeError):
                return self._send({"error": "Bad request."}, 400)
            if not os.path.isfile(path) or kwargs["task_hint"] not in (None, "classification", "regression"):
                return self._send({"error": "Add a data file first."}, 400)
            STATE.update(run=None, error=None, stop=False)
            STATE["thread"] = threading.Thread(target=_work, args=(kwargs,), daemon=True)
            STATE["thread"].start()
            self._send({"ok": True})
        elif self.path == "/stop":
            STATE["stop"] = _running()
            self._send({"ok": True})
        else:
            self._send({"error": "Not found."}, 404)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"ML Factory UI → http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
