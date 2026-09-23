"""
End-to-end check of the deployed web app: upload -> dataset profile -> run -> live events -> bundle download
-> the exported script retrains the model. Prints one line per step and exits non-zero on the first failure.

  uv run --env-file .env scripts/live_check.py --url https://<workspace>--ml-factory-web-web.modal.run
  uv run --env-file .env scripts/live_check.py --url ... --csv data/houses.csv --target SalePrice --purpose "price estimates"

Reads ACCESS_CODE from the environment. Costs one real run (Nebius tokens + Modal compute).
"""
import argparse
import io
import os
import subprocess
import sys
import time
import zipfile

import httpx

POLL_S, MAX_WAIT_S = 10, 45 * 60
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--url", required=True, help="deployed web app base URL")
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "churn.csv"))
    ap.add_argument("--target", default="Churn")
    ap.add_argument("--purpose", default="Which customers will churn soon? Explainable is a plus.")
    args = ap.parse_args()
    code = os.environ.get("ACCESS_CODE", "")
    step(bool(code), "ACCESS_CODE set")
    base = args.url.rstrip("/")
    c = httpx.Client(timeout=120)

    cfg = c.get(f"{base}/api/config")
    step(cfg.status_code == 200 and "sb_secret" not in cfg.text, "config is public-safe", cfg.text[:80])
    step(c.post(f"{base}/api/datasets", data={"access_code": "wrong"},
                files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).status_code == 401, "wrong code refused")

    with open(args.csv, "rb") as f:
        up = c.post(f"{base}/api/datasets", data={"access_code": code},
                    files={"file": (os.path.basename(args.csv), f, "text/csv")})
    step(up.status_code == 200, "dataset upload", up.text[:160])
    ds, prof = up.json()["dataset_id"], up.json()["profile"]
    step(prof.get("rows", 0) > 0 and len(prof.get("preview", [])) <= 20, "profile",
         f"{prof.get('rows')} rows, suggested target {prof.get('suggested_target')} ({prof.get('suggested_task')})")

    r = c.post(f"{base}/api/runs", json={"dataset_id": ds, "target": args.target, "purpose": args.purpose,
                                         "access_code": code})
    step(r.status_code == 200, "run started", r.text[:160])
    run_id = r.json()["run_id"]
    step(len(run_id.rsplit("-", 1)[1]) >= 16, "run id is not guessable", run_id)

    t0, status = time.time(), {}
    while time.time() - t0 < MAX_WAIT_S:
        status = c.get(f"{base}/api/runs/{run_id}").json()
        if status.get("status") in ("done", "failed"):
            break
        print(f"      {int(time.time() - t0):4d}s  {status.get('status')}", flush=True)
        time.sleep(POLL_S)
    step(status.get("status") == "done", "run finished", str(status)[:200])

    events = c.get(f"{base}/api/runs/{run_id}/events").json()
    kinds = [e["kind"] for e in events]
    last = {e["kind"]: e["payload"] for e in events}
    step(all(k in kinds for k in ("diagnostics", "search_plan", "rung", "confirm", "final_model", "report", "usage")),
         "event stream complete", f"{len(events)} events")
    plan, conf, use = last["search_plan"], last["confirm"], last["usage"]
    print(f"      funnel: {plan['space']} pipelines -> {plan['raced']} raced, n*={plan['n_star']}, cv={plan.get('cv')}")
    print(f"      pick: {conf['recommendation']['pick']}  ({conf['primary_metric']})")
    for row in conf["table"]:
        print(f"        {row['name']:<20} cv {row['cv_mean']}  ci {row['ci']}  hidden {row['hidden']}  p {row['p_vs_best']}")
    print(f"      cost: {use.get('input_tokens')} tokens in, LLM ${use.get('usd')}, compute ${use.get('modal_usd')}")
    step("[number removed" not in last["report"]["markdown"], "report passes the claim check")

    step(c.get(f"{base}/api/runs/{run_id}/export.zip").status_code == 401, "bundle needs the access code")
    z = c.get(f"{base}/api/runs/{run_id}/export.zip", headers={"X-Access-Code": code})
    step(z.status_code == 200 and z.headers.get("content-type") == "application/zip", "bundle download",
         f"{len(z.content)} bytes")
    out = os.path.join(ROOT, "runs", "live_check", run_id)
    os.makedirs(out, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
    script = next((f for f in os.listdir(out) if f.startswith("train_") and f.endswith(".py")), None)
    step(script is not None, "bundle has the training script", script or "")
    # run it the way a user would: a clean environment with exactly the packages its header asks for
    with open(os.path.join(out, script), encoding="utf-8") as f:
        pip = next((ln.split("pip install", 1)[1].split() for ln in f if "pip install" in ln), ["pandas", "scikit-learn"])
    withs = [a for p in [*pip, "joblib"] for a in ("--with", p)]
    res = subprocess.run(["uv", "run", "--no-project", *withs, "python", os.path.join(out, script), args.csv,
                          "--target", args.target, "--out", os.path.join(out, "model.joblib"), "--cv", "3"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    step(res.returncode == 0, "exported script retrains the model", (res.stdout or res.stderr).strip()[-200:])
    print(f"\nAll checks passed for run {run_id}. Files in {out}")


if __name__ == "__main__":
    main()
