# Agent plan — ML Factory (trimmed)

## Context
Hackathon today (Accel AI Innovate Amsterdam, 23 Sep 2026). A user uploads a tabular dataset, names the target, and
says what the model is for. The agent diagnoses the data, trains and compares candidate models, tests the finalists on
held-out data, and recommends the top 3 with hyperparameters, confidence intervals, p-values, and cost.
Owners: agent = Coflazo, `modal_train.py` = Atilla, Supabase = Emre.

Constraints:
- Open-weight LLM on Nebius only. No Anthropic key or code. Stack: Nebius + Modal + Supabase (+ Tavily, stretch).
- Math and statistics produce every number; the LLM makes judgement calls and narrates.
- **Feature freeze 16:00.** After that: demo runs, pitch, rehearsal.

Starting point: `main` already has the working agent (5 tools, event stream, jsonl replay, `--dry-run`), the model menu
on Modal, uv setup, and demo data (`make_demo_data.py`: leaky churn, houses). We extend it; we don't rewrite it.

## Design decisions
1. **Harness first, LLM judges.** Python runs the phases in a fixed order. The LLM only chooses: task, metric, CV
   scheme, which columns to drop, which configs to try in rounds 2-3, which finalist to recommend, and the report text.
   It never writes training or metric code.
2. **Single agent, same loop as today.** Port the existing tool loop from Anthropic to the OpenAI-compatible API.
   Keep `write_report` as a tool, so we don't depend on JSON-schema output mode.
3. **Hidden holdout.** 20% is split off before anything else and scored only once, at the end.
4. **Rounds, not racing.** Round 1 is a fixed portfolio of about 15 configs and costs no tokens. Rounds 2-3 are
   refinements picked by the LLM. The budget is 3 rounds of up to 8 configs, enforced in code (it already is).
5. **One statistical method per question.** CV comparison: Nadeau–Bengio corrected t-test with Holm correction.
   Hidden set: paired bootstrap. Nothing else.

## Pipeline
```
0 Split     → dev 80 / HIDDEN 20 (stratified; time-ordered data → last 20 %), both uploaded as parquet
1 Diagnose  → existing profile + 3 checks → compact JSON                        ◆ task, metric, CV, drops
2 Round 1   → fixed portfolio (~15 configs, 5-fold CV on dev)                   ◆ read leaderboard
3 Rounds 2-3→ LLM-chosen variants of the top families                           ◆ configs, when to stop
4 Confirm   → top 3: 5×2 repeated CV → Nadeau–Bengio t vs best, Holm → tie groups
5 Hidden    → fit top 3 on dev, predict HIDDEN once → score + 95 % bootstrap CI, paired diff vs best
6 Report    → table: CV mean±std, hidden score [CI], p vs best, fit s, predict ms, Big-O ◆ recommendation + narrative
```

## Components
| File | Change |
|---|---|
| `agent.py` | Port to `openai` client (Nebius base_url); add the split, phases 4-5 and the report table; `ScriptedClient` in OpenAI format |
| `stats.py` (new, ~60 lines) | `nb_corrected_t(scores_a, scores_b, n_train, n_test)`, `holm(pvals)`, `paired_bootstrap(y, pred_a, pred_b, metric)` |
| `modal_train.py` | The small additions below (Atilla) |

### 1. Diagnostics: extend `profile_data`, don't rebuild it
It already flags id-like, constant, high-cardinality, mostly-missing, and numeric leakage (|corr| > 0.95). Add:
- **Categorical leakage**: for columns with ≤ 50 levels, target purity per level > 0.98 → `possible_leakage`.
- **Duplicates across splits**: row hashes in both dev and hidden → warning (it inflates the hidden score).
- **Time order**: a datetime column that is sorted → suggest `timeseries` CV and a time-based split.

### 2. Round-1 portfolio
A plain dict in `agent.py`: 3 configs per family (default, shallow and heavily regularised, deep with a low learning
rate). Written by hand as sklearn/LightGBM/XGBoost constructor kwargs. No external portfolio, no random search.

### 3. Statistics (`stats.py`, numpy + scipy)
- **Nadeau–Bengio** on per-fold scores from 5×2 repeated CV:
  t = mean(d) / sqrt((1/J + n_test/n_train) · var(d)), with df = J − 1 and J = 10 differences. One-sided vs the best,
  then Holm. Configs whose adjusted p ≥ 0.05 are in the best config's tie group.
- **Paired bootstrap** on the hidden set: 2,000 resamples of row indices. Returns each model's score CI and the CI of
  its difference from the best. The same code works for AUC, F1, accuracy, RMSE, MAE and R².
- **Cost**: measured fit seconds and predict ms per 1k rows, plus a static Big-O line per family in the report.

### 4. Nebius provider
- `openai.OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=NEBIUS_API_KEY)`, model from
  `AGENT_MODEL`, temperature 0, `max_tokens` 2k.
- **Choosing the model**: run the churn demo end to end with Qwen3-235B-A22B-Instruct-2507, then with one fallback.
  Keep the first one that finishes with valid tool calls. No separate probe script.
- Check tool-call arguments against the schema and retry once, sending the error back. Don't assume anything about
  caching or constrained decoding; tool results are already small.

### 5. Asks to Atilla (`modal_train.py`), all small
1. Return per-fold test scores in `train_candidate` (needed for Nadeau–Bengio).
2. A `cv_repeats` param (`RepeatedStratifiedKFold` / `RepeatedKFold`).
3. `predict_holdout(spec, hidden_path)`: fit on dev, return hidden predictions (probabilities for classification),
   fit seconds and predict ms.
4. Allow `.map()` over a batch of specs, for round 1's ~15 configs.

## Build order (≈ 10:30 → 16:00)
0. **Sync (5 min)**: merge `main` into `Coflazo-Branch-N`. `uv remove anthropic`, `uv add openai scipy scikit-learn`.
1. **Provider port (1 h)**: OpenAI-format loop; the `--dry-run` scripted replay still passes; one live Nebius run.
2. **Split + diagnostics (1 h)**: 80/20 split, hidden parquet upload, the 3 new checks, round-1 portfolio.
3. **Modal additions (1 h, with Atilla)**: per-fold scores, repeats, `predict_holdout`; smoke test.
4. **Confirm + hidden + stats (1 h)**: `stats.py`, phases 4-5, the report table.
5. **Report + UI events (45 min)**: new event kinds `confirm`, `hidden_test`; check Emre's UI renders them.
6. **Demo cache (30 min)**: run churn and houses live, keep `runs/*_events.jsonl` for replay.

Stretch, only if everything above works by 15:00, in this order:
- **Learning curve**: fit the best config on 25/50/100 % of dev. If the score is still rising, the report says
  "more data would likely help".
- **Tavily suggestions**: search OpenML/UCI/HuggingFace for related datasets and list them as "untested
  suggestions". No automatic fetching or joining.
- **Ensemble**: average the top 3. Chosen on dev CV, never on hidden.

Cut: racing/ASHA, n\* sample sizing, thousands of configs, ~45 diagnostics, TabICLv2/GPU, CatBoost, Supabase
experience memory, automatic external-data joins, baycomp/DeLong/McNemar, context compaction, the model-probe script.
Revisit after the hackathon.

## Verification
- Plain `assert` tests (no network): the split has no overlap and preserves class balance; churn flags
  `refund_issued`; `nb_corrected_t` matches a hand-computed value; bootstrap CI covers a known difference on synthetic
  data; the dry-run loop completes.
- End to end: `uv run agent.py data/churn.csv --target Churn` drops `refund_issued`, reports the top 3 with a hidden
  score and CI, and the recommended model is in the hidden top 3.
- `grep -ri anthropic` returns nothing.

## Git
Commit after each build step on `Coflazo-Branch-N`, then push. Never push to `main`, never force-push. Before
staging, check `git status`: no `.env`, keys, promo codes, or `runs/`. Use `--author` (or a repo-local config in your
own clone) for authorship; don't change anyone's global git identity.

## Security
Keys only from env: `NEBIUS_API_KEY`, `SUPABASE_URL` / `SUPABASE_KEY`, Modal token (`TAVILY_API_KEY` if the stretch
goal is built). `.env` is gitignored. The LLM never executes code.
