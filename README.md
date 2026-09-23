# Baklava: ML Factory

Upload a tabular dataset, say in plain words what you want from it ("which customers will churn in the next 90
days", "forecast weekly demand per store"), and an open-weight LLM agent on Nebius works out the ML task,
checks the data with 33 statistical diagnostics, races thousands of candidate pipelines on statistically sized
samples on Modal, confirms the finalists with paired tests, scores them once on a locked hidden set, and hands
you a model trained on all of your data plus a script to retrain it yourself.

The LLM judges and narrates; every number comes from deterministic code. Design notes: [docs/AGENT_PLAN.md](docs/AGENT_PLAN.md).
Event contract for UIs: [docs/EVENTS.md](docs/EVENTS.md). Use-case catalog: [docs/USE_CASES.md](docs/USE_CASES.md).

## How a run works

```
goal + CSV
  -> use case match (59 researched use cases, TF-IDF, no tokens)      usecases.py, purpose.py
  -> forecasting goals: leak-free lag / rolling / calendar features   forecasting.py
  -> lock hidden 20% (latest rows if time-ordered)                     sampling.py
  -> 33 diagnostics on dev only: leakage, ids, drift, imbalance ...    diagnostics.py
  -> n* rows from Hoeffding / precision bounds + learning curve         sampling.py
  -> ~3000 pipelines (AutoGluon 2025 portfolio + Sobol), prior-ranked  search_space.py
  -> race top 243 on growing subsamples, drop by corrected t-test      racing.py  (Modal: modal_train.py)
  -> top 3: 10 paired folds, Nadeau-Bengio + Bayesian ROPE, 1-SE pick  stats_tests.py
  -> one hidden test (DeLong / McNemar / bootstrap), calibration, Pareto, Caruana ensemble check (ensemble.py)
  -> refit on ALL rows, export script + params + model card            export.py
  -> report built from tool results only, numbers claim-checked        report.py
```

Models on the menu: logistic/ridge, random forest, LightGBM, XGBoost, CatBoost, MLP (TabICL when the Modal
workspace has GPU billing, `ML_FACTORY_GPU=1`). Metrics: roc_auc, f1_macro, accuracy; rmse, mae, r2.

## Setup (Windows, macOS, Linux)

Needs [uv](https://docs.astral.sh/uv/). Python 3.13 is installed by uv.

```
uv sync
uv run modal setup                  # browser login once (or put MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in .env)
copy .env.example .env              # Windows;  macOS/Linux: cp .env.example .env
```

Fill `.env` (it is gitignored): `NEBIUS_API_KEY`, `TAVILY_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY` (secret, backend
only), `SUPABASE_PUBLISHABLE_KEY` (browser), `ACCESS_CODE` (gates the web app), `AGENT_MODEL`, `ML_FACTORY_GPU`.
Supabase schema: run [docs/supabase.sql](docs/supabase.sql) once in the SQL editor.

## Run

```
# training backend (deploy once per change of modal_train.py)
uv run --env-file .env modal deploy modal_train.py
uv run --env-file .env modal run modal_train.py --csv data/churn.csv --target Churn     # smoke test, every model

# the agent from the command line
uv run --env-file .env agent.py data/churn.csv --target Churn --purpose "flag likely churners early; explainable is a plus"
uv run agent.py data/churn.csv --target Churn --dry-run      # no API keys: scripted LLM, real Modal

# the web app (upload, dataset check, live run, results, download)
uv run --env-file .env modal secret create ml-factory-env --from-dotenv .env --force   # drop MODAL_* lines first
uv run --env-file .env modal deploy web_app.py
```

`AGENT_MODEL` defaults to `Qwen/Qwen3-235B-A22B-Instruct-2507` (chosen by `scripts/probe_models.py`: 10/10 valid tool
calls, lowest known price). Open-weight models only.

## Outputs

| Where | What |
|---|---|
| `runs/<run_id>_events.jsonl` | every event in order (replay mode) |
| `runs/<run_id>_report.md` | the report: dataset card, issues and fixes, search funnel, top models with CI / p / hidden / Big-O, recommendation, n* derivation, caveats, cost |
| `runs/<run_id>_export/` | `train_<model>.py`, `params.json`, `MODEL_CARD.md`: the user's own model, retrainable anywhere |
| Modal volume `/models/<run_id>/<name>.joblib` | the fitted pipeline, trained on every row |
| Supabase `events`, `runs`, `candidates` | live feed, one row per run, one row per evaluated config (results_store.py) |

## Tests

```
uv run evals/make_golden.py          # golden datasets with planted leakage / drift / high cardinality
uv run python -m pytest -q           # ~250 offline tests; live Supabase tests skip without keys
uv run evals/run_golden.py --dry     # scored end-to-end runs with a scripted LLM (no cost)
uv run --env-file .env evals/run_golden.py --sets all --repeat 3   # live: pass^3 over golden + 10 public datasets
```

CI (`.github/workflows/tests.yml`) runs the suite on Windows, macOS and Linux. `test_windows.py` reproduces the
Windows cp1252 console failure mode on any OS.

## Security

Keys only from `.env` / Modal Secrets. The web app requires `ACCESS_CODE`, caps uploads (CSV, 50 MB), limits
concurrent runs and per-IP rates, validates every id, and sends a strict CSP. External data is fetched only from an
allowlist of public dataset hosts with SSRF guards (pinned DNS, redirect re-checks, size caps, no pickle). Model
parameters are allowlisted and size-capped; generated scripts escape all user text. The hidden split never reaches
the LLM, and metric code is outside its reach.
