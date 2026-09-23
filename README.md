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
                  ├─ track_training.remote_gen ──► parallel CV workers, start/result packets
                  ├─ track_training(final_fit) ──► winner refit ──► /models/<run_id>/<name>.joblib
                  └─ tracking.py ──► durable JSONL + Supabase dataset/run/training/event records
```

- **Claude never writes training code.** It only picks a model name and constructor params
  from a fixed menu (`ESTIMATORS` in [modal_train.py](modal_train.py)):

  | Task | Models |
  |---|---|
  | classification | `logreg`, `random_forest`, `lightgbm`, `xgboost`, `mlp` |
  | regression | `ridge`, `random_forest`, `lightgbm`, `xgboost`, `mlp` |

- **The budget is enforced in code**: at most 3 experiment rounds, 8 candidates per round,
  and 20 agent steps.
- **Tools** the agent can call: `get_data_profile`, `inspect_column`, `run_experiments`,
  `finalize_model`, `write_report`. Tool errors are returned to Claude, never raised, so it
  can react to them.
- **Metrics**: classification uses `roc_auc`, `f1_macro` and `accuracy`; regression uses
  `rmse`, `mae` and `r2`. Each leaderboard row shows the validation mean ± std, the train
  mean, and an `overfit_gap`. It also warns `suspiciously_high_check_leakage` when
  `roc_auc`, `accuracy` or `r2` is above 0.995.

## Setup

```powershell
uv sync --frozen                          # creates .venv from uv.lock
uv run modal setup                        # browser login, once
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # bash: export ANTHROPIC_API_KEY=...
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

# 3) Agent with local tracking
uv run agent.py data/churn.csv --target Churn --offline
uv run agent.py data/houses.csv --target SalePrice --task regression --offline

# No Anthropic key? Scripted stand-in for Claude; everything else (Modal, events, saving) is real
uv run agent.py data/churn.csv --target Churn --dry-run --offline
```

`CLAUDE_MODEL` overrides the model (default `claude-sonnet-5`).

For shared Supabase history, follow the [handoff](docs/supabase_handoff.md), register an approved
dataset, and load the backend environment for the command:

```sh
uv run --frozen --env-file .env python agent.py --dataset-id <dataset-uuid> --target Churn --dry-run
```

This downloads the private file and verifies its hash before training. A local path also
works if its exact bytes and filename match a registered dataset. The ML teammate must
deploy the updated `modal_train.py`, including `track_training`, first. `--dry-run` still
runs real Modal training; `--offline` disables Supabase tracking only. Deployment steps,
registered demo IDs, and the frontend contract are in the [team handoff](docs/supabase_handoff.md).

## Outputs

| File | Contents |
|---|---|
| `runs/<run_id>_events.jsonl` | Flushed before network delivery; events and state snapshots for replay/reconciliation |
| `runs/<run_id>_results.json` | Dataset, run, training attempts, profile, CV results, final model, pending delivery count |
| `runs/<run_id>_report.md` | Claude's markdown report |
| Modal volume `/models/<run_id>/<name>.joblib` | `{"pipeline", "classes", "spec"}`, where `pipeline` is a fitted sklearn Pipeline |

## Events (live UI)

Each event is `{event_id, run_id, training_id, ts, kind, payload}`. It is printed and
journaled locally before delivery. With `SUPABASE_URL` / `SUPABASE_KEY` configured,
`apply_tracking_event` updates state and inserts its event in one database transaction.
The `payload._tracking` snapshot is for recovery; use the tables for UI state.

| kind | payload |
|---|---|
| `run_start` | dataset, target, rows, features |
| `run_configured` | resolved task type |
| `training_queued` | candidate name and stage; stable training ID in the envelope |
| `training_started` | actual worker start timestamp |
| `training_completed` / `training_failed` | individual result, status, metrics or error |
| `model_selected` | successful CV training ID linked to the final fit |
| `thought` | Claude's plain-language reasoning before each tool call |
| `tool_call` | tool name + input |
| `status` | progress message (upload, final fit) |
| `round_start` | round number, rationale, candidates |
| `leaderboard` | ranked rows for the round, primary metric, wall time |
| `final_model` | winner spec, model path, top features |
| `report` | markdown + `spoken_summary` (≤60 words, for TTS) |
| `run_end` | completed, incomplete, or failed; reason when applicable |

Apply the versioned files in [supabase/migrations](supabase/migrations) for `datasets`,
`runs`, `trainings`, `events`, and the backend-only RPC. The browser has read access to
this shared sample-data history. Raw dataset files stay in a private bucket.

**Recovery:** delivery retries are bounded; pending events remain in the journal. Replay
the whole journal in file order after restoring connectivity; repeated events are deduplicated:

```sh
uv run --frozen --env-file .env python -m scripts.reconcile_tracking runs/<run_id>_events.jsonl
```

**Replay mode:** a UI can play saved events in file order. A browser replay player and
reconnect handling still need to be implemented by the UI owner.

## Files

- [modal_train.py](modal_train.py) — model menu, preprocessing, CV, lineage, `track_training`, and the smoke-test entrypoint
- [agent.py](agent.py) — data profiling, tool schemas, the Claude loop, the event stream, `--dry-run`
- [tracking.py](tracking.py) — dataset verification, per-attempt tracking, durable logs and reconciliation
- [make_demo_data.py](make_demo_data.py) — synthetic demo datasets, including a planted leakage column
