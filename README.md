# Baklava — ML Factory

Give it a tabular dataset and a target column. Claude profiles the data, plans experiments,
cross-validates candidate models in parallel on Modal, picks the winner, and writes a report,
the way a junior ML engineer would. Background, timeline and roles: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## How it works

```
CSV/Parquet ──► agent.py (local) ──────────────────────────► Claude (tool use)
                  │  profile_data: small JSON summary,         │ picks tools, models, params
                  │  never raw rows                            │
                  │◄───────────────────────────────────────────┘
                  │
                  ├─ upload_dataset ──► Modal volume ml-factory-data (/datasets/*.parquet)
                  ├─ train_candidate.map(specs) ──► N containers in parallel, CV each candidate
                  └─ fit_final ──► winner refit on all data ──► /models/<run_id>/<name>.joblib
```

- **Claude never writes training code.** It only picks a model name and constructor params
  from a fixed menu (`ESTIMATORS` in [modal_train.py](modal_train.py)):

  | Task | Models |
  |---|---|
  | classification | `logreg`, `random_forest`, `lightgbm`, `xgboost`, `mlp` |
  | regression | `ridge`, `random_forest`, `lightgbm`, `xgboost`, `mlp` |

- **The budget is enforced in code**: at most 3 experiment rounds, 8 candidates per round,
  and 20 agent steps.
- **Tools** the agent can call: `diag_summary`, `diag_run` (33 statistical checks on the dev split), `run_search` (~3000 pipelines ranked by a benchmark prior, top 243 raced on sized subsamples), `confirm_and_test` (10 paired folds, corrected t / Bayesian tests, 1-SE pick, one hidden-holdout test, Pareto + Big-O), `data_search` / `data_try` (Tavily + measured enrich trial), `inspect_column`, `run_experiments`, `finalize_model` (also exports the user's own training script), `write_report`. Tool errors are returned to the model, never raised, so it can react to them. The hidden 20% split is locked before any diagnostic runs.
- **Metrics**: classification uses `roc_auc`, `f1_macro` and `accuracy`; regression uses
  `rmse`, `mae` and `r2`. Each leaderboard row shows the validation mean ± std, the train
  mean, and an `overfit_gap`. It also warns `suspiciously_high_check_leakage` when
  `roc_auc`, `accuracy` or `r2` is above 0.995.

## Setup

```powershell
uv sync                                   # creates .venv from uv.lock
uv run modal setup                        # browser login, once
# .env (gitignored): NEBIUS_API_KEY=...  TAVILY_API_KEY=...  SUPABASE_URL=...  SUPABASE_KEY=...
uv run make_demo_data.py                  # data/churn.csv, data/houses.csv
```

The Modal image pins the same pandas and pyarrow versions as `uv.lock`, because the parquet
file is written locally and read on Modal. If you bump either one in `pyproject.toml`, bump it
in `modal_train.py` too.

## Run

```powershell
# 1) Backend smoke test, no agent: every model in the menu, default params, 3-fold CV
uv run modal run modal_train.py --csv data/churn.csv --target Churn
uv run modal run modal_train.py --csv data/houses.csv --target SalePrice --task regression

# 2) Deploy: the agent looks up the deployed functions by app name
uv run modal deploy modal_train.py

# 3) Agent
uv run --env-file .env agent.py data/churn.csv --target Churn --purpose "flag likely churners, explainable"
uv run --env-file .env agent.py data/houses.csv --target SalePrice --task regression

# No API key? Scripted stand-in for the LLM; everything else (Modal, events, saving) is real
uv run agent.py data/churn.csv --target Churn --dry-run
```

`AGENT_MODEL` overrides the Nebius model (default `Qwen/Qwen3-235B-A22B-Instruct-2507`). Open-weight models only.

## Outputs

| File | Contents |
|---|---|
| `runs/<run_id>_events.jsonl` | Every event, in order. Use it for replay mode. |
| `runs/<run_id>_results.json` | Data profile, every candidate's CV results, the final model |
| `runs/<run_id>_report.md` | Claude's markdown report |
| Modal volume `/models/<run_id>/<name>.joblib` | `{"pipeline", "classes", "spec"}`, where `pipeline` is a fitted sklearn Pipeline |

## Events (live UI)

Each event is `{run_id, ts, kind, payload}`. It is printed, saved to the jsonl, and inserted
into Supabase if `SUPABASE_URL` / `SUPABASE_KEY` are set:

| kind | payload |
|---|---|
| `run_start` | dataset, target, rows, features |
| `thought` | Claude's plain-language reasoning before each tool call |
| `tool_call` | tool name + input |
| `status` | progress message (upload, final fit) |
| `round_start` | round number, rationale, candidates |
| `leaderboard` | ranked rows for the round, primary metric, wall time |
| `final_model` | winner spec, model path, top features |
| `report` | markdown + `spoken_summary` (≤60 words, for TTS) |

Supabase table:

```sql
create table events (
  id bigserial primary key,
  run_id text, ts double precision, kind text, payload jsonb
);
alter publication supabase_realtime add table events;
```

**Replay mode**: if the live demo fails, play a saved `runs/*_events.jsonl` into the UI in `ts` order.

## Files

- [modal_train.py](modal_train.py) — model menu, preprocessing, CV, and the Modal functions `train_candidate` / `fit_final`, plus the smoke-test entrypoint
- [agent.py](agent.py) — data profiling, tool schemas, the Claude loop, the event stream, `--dry-run`
- [make_demo_data.py](make_demo_data.py) — synthetic demo datasets, including a planted leakage column
