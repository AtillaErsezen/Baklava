# ML Factory — Implementation Plan

**Event:** Prosus AI House hackathon (one day, doors 8am)
**Pitch:** An agent (Claude) analyzes a company's tabular data and trains the
best-performing ML model for it — choosing techniques, running experiments,
and reporting results like a junior ML engineer would.

## Architecture

| Layer | Tool | Role |
|---|---|---|
| Agent / reasoning | Claude (tool use) | Profiles data, plans experiments, picks models + hyperparameters, decides when to stop |
| Training compute | Modal | Fixed menu of models (LogReg/Ridge, RandomForest, LightGBM, XGBoost, MLP); trains candidates in parallel via `.map()` |
| State / live events | Supabase | Dataset versions, runs, individual training attempts and metrics, plus an `events` feed for the UI |
| Frontend | Lovable (fallback: Streamlit) | Upload CSV → inspect dataset → watch experiments live → compare each attempt's data, settings, and performance → final report |
| Voice | ElevenLabs | ~30s spoken summary of the final report for the demo |

Claude never writes training code directly — it only *selects* a model name and
parameters from a fixed menu (`modal_train.py`). This keeps runs fast, safe, and
reproducible instead of depending on generated code executing correctly.

## Scope lock (decide by 9am, do not revisit)

- Tabular data only (CSV/Parquet). No images, no free text.
- Classification and regression only.
- Model menu fixed at 5 families per task (see `MODEL_MENU` in `modal_train.py`).
- Budget: 3 experiment rounds, max 8 candidates per round.

## Repo layout

```
ml-factory/
├── modal_train.py      # model menu, preprocessing, CV, track_training, final fit
├── agent.py            # data profiling, Claude tool loop, event stream, CLI
├── tracking.py         # durable dataset/run/training tracking and replay
├── supabase/           # database migrations, fixtures, SQL checks
├── pyproject.toml      # dependencies; uv.lock pins the local environment
└── README.md           # setup + run commands
```

## Timeline

| Time | Task |
|---|---|
| 8:00–9:00 | Team forming, roles, lock scope. Pick 2–3 demo datasets (one classification, one regression, one with a deliberately leaky column). |
| 9:00–11:00 | **Modal backend, no agent.** Run `modal run modal_train.py --csv ... --target ...` until every model in the menu trains cleanly. This is the highest-risk piece — finish it first. |
| 11:00–13:00 | **Agent loop.** Wire up Claude + the 5 tools (`get_data_profile`, `inspect_column`, `run_experiments`, `finalize_model`, `write_report`). Get one full run end to end on the CLI. |
| 13:00–15:00 | **UI + training history.** Supabase dataset/run/training records and event feed, Lovable dataset details and per-attempt performance, CSV upload flow. |
| 15:00–16:00 | Report generation polish, ElevenLabs voice summary, pre-run all demo datasets once and cache the results (replay mode). |
| 16:00+ | Pitch prep, rehearse twice. No new features after this point. |

## Roles (3–4 people)

- **Modal / ML** — model menu, preprocessing, cross-validation, final fit.
- **Agent** — system prompt, tool schemas, the Claude loop, data profiling.
- **UI + Supabase** — dataset/run/training records, event feed, CSV upload, dataset details, per-attempt performance comparison, and report rendering.
- **(if 4th)** — pitch narrative, demo data curation, ElevenLabs integration, rehearsal.

Supabase workstream: [team handoff and responsibilities](docs/supabase_handoff.md).

## Setup checklist

```bash
uv sync --frozen
uv run modal setup
export ANTHROPIC_API_KEY=sk-ant-...

# 1) Smoke test the backend alone — do this before touching the agent
uv run modal run modal_train.py --csv data/churn.csv --target Churn

# 2) Deploy so the agent can call the named functions
uv run modal deploy modal_train.py

# 3) Run the agent
uv run agent.py data/churn.csv --target Churn --offline
uv run agent.py data/houses.csv --target SalePrice --task regression --offline
```

Supabase (optional for CLI-only execution; required for the live training-history UI):

The [Supabase handoff](docs/supabase_handoff.md) covers the two migrations,
dataset registration, and live tracking checks. The agent now writes dataset-linked
runs and individual attempts through an atomic RPC, with durable JSONL recovery.
Use `uv run --frozen --env-file .env python agent.py --dataset-id <id> --target <column>`
for live history. `--offline` keeps tracking local but still calls Modal and the LLM;
`--dry-run` replaces only the LLM with a script and still performs real Modal training.
The [team handoff](docs/supabase_handoff.md) contains deployment and UI contracts.

## Agent workflow (what Claude actually does per run)

1. `get_data_profile` → decide task type, primary metric, CV scheme (kfold vs
   timeseries), columns to drop. `inspect_column` for anything ambiguous.
2. **Round 1 — broad sweep:** 4–6 diverse candidates with sensible defaults,
   including a linear baseline.
3. **Round 2 — refine:** 3–6 variants of the top 1–2 families (depth, learning
   rate, regularization, class weighting, encoding).
4. **Round 3 — optional**, only if round 2 changed the ranking or surfaced a
   problem.
5. Watch for a large `overfit_gap` (→ regularize more) or a
   `suspiciously_high_check_leakage` warning (→ `inspect_column`, drop, rerun).
6. `finalize_model` on the winner (ties within one std → pick the simpler one).
7. `write_report` — markdown report + a short spoken summary — then stop.

## Risk mitigation

| Risk | Mitigation |
|---|---|
| Modal cold start / dependency issues eat the morning | Build and test the image first, before any agent code |
| Live demo breaks on stage | Pre-run all demo datasets once, save `events.jsonl`, replay it in the UI if live fails |
| Claude generates broken training code | It doesn't — it only picks from a fixed, pre-tested model menu |
| Token/latency blowup from sending raw data to Claude | Only the small JSON profile is ever sent, never raw rows |
| Scope creep | Hard budget (3 rounds × 8 candidates) enforced in code, not just prompted |

## Demo idea

Seed one dataset with an obvious leakage column. The agent flagging it as
`possible_leakage`, inspecting it, and dropping it live is a strong moment —
it signals "this reasons about the data," not "this just runs AutoML."
