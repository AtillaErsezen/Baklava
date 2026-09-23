# Baklava — ML Factory

Claude profiles a tabular dataset, plans experiments, cross-validates candidates in parallel on Modal,
picks the winner, and writes a report. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Setup

```powershell
uv sync                                   # creates .venv from uv.lock
uv run modal setup                        # browser login, once
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # bash: export ANTHROPIC_API_KEY=...
uv run make_demo_data.py                  # data/churn.csv (leaky column planted), data/houses.csv
```

## Run

```powershell
# 1) Backend smoke test, no agent: every model in the menu, 3-fold CV
uv run modal run modal_train.py --csv data/churn.csv --target Churn
uv run modal run modal_train.py --csv data/houses.csv --target SalePrice --task regression

# 2) Deploy so the agent can call the named functions
uv run modal deploy modal_train.py

# 3) Agent
uv run agent.py data/churn.csv --target Churn
uv run agent.py data/houses.csv --target SalePrice --task regression
```

Outputs: `runs/<run_id>_events.jsonl`, `_results.json`, `_report.md`. The model lands in the Modal
volume `ml-factory-data` at `/models/<run_id>/<name>.joblib`.

Live UI (optional): create the Supabase `events` table from the plan, set `SUPABASE_URL` / `SUPABASE_KEY`.
