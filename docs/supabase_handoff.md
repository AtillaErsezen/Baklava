# Supabase handoff — ML, agent and UI owners

Updated 2026-09-23 against `main` at `c9f1a0a`. The integration is in this working tree; share/commit the changed files through the team's normal workflow before another machine deploys them.

## Current project and datasets

Project: `bsqmjxaivnuapxzrrcen`, URL `https://bsqmjxaivnuapxzrrcen.supabase.co`. Both migrations are applied in the hosted SQL Editor. Do not rerun them there. The private `datasets` bucket keeps its 30 MiB limit. Database metadata and opted-in previews are public to the demo client.

| File | Dataset ID | Rows | Target | Task |
|---|---|---:|---|---|
| `data/churn.csv` | `728c21d4-1768-50df-9be6-381a6adcf350` | 3,000 | `Churn` | classification |
| `data/houses.csv` | `3ecf209f-2461-5329-b984-7f394f57818b` | 2,000 | `SalePrice` | regression |
| `supabase/fixtures/setup_demo.csv` | `7beb6dbb-f3a0-5fec-ba12-a9c4132a33e4` | 12 | `churn` | setup checks only |

All three files are synthetic, registered as version `v1`, with three public preview rows. `is_demo_fixture` describes the synthetic dataset; it does not imply that every score measured on it is invented. The tracking smoke explicitly marks its own scores as invented in configuration, warnings and report. Churn includes a deliberately leaky `refund_issued` column; inspect which features each actual attempt drops.

Registration verified exact downloaded hashes, public metadata reads and denial of publishable-key file downloads. The source SHA-256 and Storage path are in each dataset row. Editing a file requires registering a new version; matching filenames alone never identify a version.

## Setup reference

The current project is already configured. On a new project, apply the two files in [supabase/migrations](../supabase/migrations) in filename order, then [02_setup_demo.sql](../supabase/02_setup_demo.sql) and [03_verify_setup.sql](../supabase/03_verify_setup.sql). The latter files seed labelled invented scores and verify permissions/publication membership; the seed also supports the SQL regression tests. SQL Editor execution does not record CLI migration history: reconcile that history before adopting `supabase db push` on the existing project.

Create a private `datasets` bucket that accepts the team's CSV/Parquet types and file sizes. Copy [.env.example](../.env.example) to `.env` only if it does not already exist, and fill in the project's URL, publishable key and backend secret privately. Commands use `--env-file .env` to load these values. Keep previews suitable for the public shared demo; they default to zero rows.

Register an approved file with:

```sh
uv run --frozen --env-file .env python -m scripts.register_dataset data/churn.csv --name 'DEMO - synthetic churn' --version v1 --preview-rows 3 --demo-fixture
```

Use `data/houses.csv` with name `DEMO - synthetic houses` for the regression sample. The Realtime smoke also requires registering `supabase/fixtures/setup_demo.csv` with name `SETUP DEMO - synthetic churn (stored)` and the same version/preview/fixture flags. These registrations are complete in the current project. Repeating identical registration preserves the file and dataset ID; conflicting metadata fails without rewriting history. Dataset IDs are project-specific; update `SAMPLE_ID` in the smoke script if testing another project.

## ML teammate: deploy the changed worker

From a checkout containing these changes:

```sh
uv sync --frozen
uv run modal setup                 # only if this machine is not authenticated
uv run modal deploy modal_train.py
```

App name remains `ml-factory`, volume `ml-factory-data`. The agent now calls **`track_training(spec, run_id, stage)`** using `remote_gen`, once per concurrent candidate. It streams `{kind: "started", started_at}` and `{kind: "finished", result}`. Stage is `candidate_cv` or `final_fit`.

Existing `train_candidate` and `fit_final` functions remain available. `upload_dataset(df)` still returns a path; `with_metadata=True` returns `{path, sha256}`. `_load_xy` is internal and now returns `(X, y, classes, manifest)`. Results include source/prepared-data lineage, resolved settings, library versions, split metadata and measured fold scores. Final fit returns no new evaluation metrics.

The worker needs no Supabase key. Only the local/trusted agent writes tracking records. The deployed worker must include the new generator; an older deployment will fail its streams. Pinning all remote ML package versions is still an ML-owner reproducibility follow-up; this patch records versions actually used and preserves the existing image configuration.

## You: run after your teammate confirms deployment

Keep `SUPABASE_URL`, `SUPABASE_KEY` and `SUPABASE_PUBLISHABLE_KEY` in local `.env`. Run from the repository root:

```sh
uv run --frozen --env-file .env python agent.py --dataset-id 728c21d4-1768-50df-9be6-381a6adcf350 --target Churn --dry-run
uv run --frozen --env-file .env python agent.py --dataset-id 3ecf209f-2461-5329-b984-7f394f57818b --target SalePrice --task regression --dry-run
```

`--dry-run` skips LLM reasoning and still executes real Modal training, incurring Modal usage. Remove it when testing the actual agent with its configured provider/key. `--offline` disables Supabase tracking only and requires a local file; it does not disable Modal or LLM calls.

Expected for the current scripted flow: five candidate evaluations plus one final fit, unless a worker fails. The selected training ID references the successful evaluated parent. A run is `completed` only after a successful final fit and nonempty report. Failed candidates stay visible with null metrics. Verify lineage and score fields rather than expecting any particular score in advance.

## Agent/backend owner: preserve these hooks

`FactoryRun(None, target, task_hint=..., dataset_id=...)` downloads the trusted registered object and checks its hash. A local path is allowed if the hash and filename match a registered record. The backend checks the tracking RPC before paid work. `TrackingLog` persists run/task, queued/started/completed attempts, selection, report and run end. Every event is flushed locally before network delivery.

State and event are written together by backend-only `apply_tracking_event(p_event)`. `p_event = null` checks protocol version `1` without mutation. Supply full snapshots as generated by `TrackingLog`; do not write timeline events separately from state. Preserve stable UUIDs when retrying. Finished attempts and their requested specs cannot be rewritten.

The future agent-provider change does not require changing this contract. Racing with new stage names or an experience table does require an agreed migration. The database accepts rounds greater than three; the current agent's budget is still three. No browser upload/run-start endpoint has been added: use the CLI until the backend owner supplies an authorized durable run-start route returning `{run_id, dataset_id}`.

## UI owner: history and live updates

Use the project URL and publishable key only. Fetch:

| Table | Query scope | Display |
|---|---|---|
| `runs` | `id = run_id` | task, target, status, report, selected candidate |
| `datasets` | `id = run.dataset_id` | file/version/hash, source counts/schema/preview |
| `trainings` | `run_id = run_id`, order round then created_at | every attempt, status, model, settings, lineage, scores |
| `events` | `run_id = run_id`, order id ascending, page as needed | timeline |

Subscribe to `events` INSERT and `trainings` INSERT/UPDATE with `run_id=eq.<run_id>`, and `runs` INSERT/UPDATE with `id=eq.<run_id>`. Subscribe before fetching history and buffer changes during loading. Merge mutable records by ID and highest `row_version`; deduplicate events by `event_id`. Refetch complete snapshots on reconnect. These browser behaviors still need implementation and verification. The hosted Python subscription check demonstrates event and training delivery, not browser recovery.

Envelope: `{event_id, run_id, training_id, ts, kind, payload}`. Existing events remain, with lifecycle kinds `run_configured`, `training_queued`, `training_started`, `training_completed`, `training_failed`, `model_selected`, `run_end`. `payload._tracking` holds recovery snapshots; avoid showing it as ordinary timeline text. Unknown kinds should not break rendering.

Read `trainings.metrics[primary_metric]`: validation `mean`, fold `std`, training `train_mean`, and measured fold arrays. Respect `higher_is_better`. Show cross-validation labels, `overfit_gap`, warnings and failure errors. Final-fit rows have null metrics; display their parent's CV result with that label. Use actual usable rows/features from `data_manifest`, not just source counts. There are no epoch curves or independent holdout scores. `model_path` needs a separate trusted download mechanism; it is not a URL.

## Checks and recovery

```sh
# Local tests: no paid API or Modal calls
uv run --frozen python -m unittest discover -s tests -v

# Hosted synthetic writes + Realtime: invented scores, no training
uv run --frozen --env-file .env python -m scripts.supabase_smoke --realtime

# Restore pending tracking; replay the entire journal in file order
uv run --frozen --env-file .env python -m scripts.reconcile_tracking runs/<run_id>_events.jsonl
```

Verified: 15 local Python tests, PostgreSQL atomicity/duplicate/conflict/role checks, live state/event writes, publishable-key history, RPC denial, replay deduplication, eight Realtime events plus successful and failed terminal training states. Latest live fixture run: `tracking-smoke-a7859c7be091`. Its scores are invented. Real local CV was tested; remote Modal deployment and an end-to-end real run remain pending.

Database regression checks are retained in [tracking_constraints.sql](../supabase/tests/tracking_constraints.sql) and [atomic_tracking.sql](../supabase/tests/atomic_tracking.sql). Run as database admin after both migrations and the setup fixture; each test rolls back its records. Both passed together in local PostgreSQL via PGlite.

A caught interruption records unconfirmed work as failed and preserves partial output; it does not cancel already-running remote compute. A hard process kill may leave running rows with no terminal event. Reconciliation restores recorded facts only. Browser replay and reconnect handling are separate UI work. The original SQL setup fixture and old event-only logs are not replayable version-1 journals.
