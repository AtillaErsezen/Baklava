# Agent development plan - ML Factory (my section), research-backed

## Context
Hackathon today (Accel AI Innovate Amsterdam, 23 Sep 2026). ML Factory: a user uploads a tabular dataset + target +
purpose; an agent diagnoses it with many statistical tools, screens thousands of candidate ML pipelines on
statistically sized random subsamples, trains/tests/backtests the finalists, and recommends the top-k models with
hyperparameters, confidence intervals, p-values, and Big-O / measured cost. My section is the agent.
Other owners: Atilla (`modal_train.py`, Nebius), Emre (Supabase).

Hard constraints from the user:
- **Open-weight LLM on Nebius only. No Anthropic key or code anywhere.** Stack = Nebius + Tavily + Supabase (+ Modal compute).
- Token-efficient; math and statistics do the work; heavy tooling; best possible agent design.
- Tavily: find similar or complementary public data and say, with evidence, whether pulling it improves the model.

Starting point: `origin/main` @ `14656c8` (local branch 1 commit behind, fast-forward). `agent.py` (467 lines):
`FactoryRun`, 5 tools, `ScriptedClient` dry run, `emit()` events, `save()` replay. `modal_train.py`: 5-family menu,
fixed `SEED=42` folds (paired comparisons possible), sklearn `Pipeline` with in-fold preprocessing (leak-safe).

Spine: superpowers (writing-plans → TDD, subagent-driven for independent files). Gates: security-audit,
verification-before-completion.

## What the research changed (3 agents: prior art, community practice, SOTA math)
1. **Harness-driven, LLM-as-judge** (MLZero, AutoKaggle, AIDE): deterministic Python runs the fixed phase order; the
   LLM makes judgement calls on compact summaries and never writes training or metric code. Biggest reliability and
   token win.
2. **Tabular foundation model as the baseline, screener, and landmarker**: TabICLv2 (`tabicl`, BSD-3, open weights) beats
   tuned GBDTs on ~80 % of TabArena and fits 50k×100 in ~10 s on a GPU. TabPFN stays out (non-commercial license)
   unless the organisers confirm the event counts as evaluation.
3. **Rung 0 = the proven zero-shot portfolio**: AutoGluon `zeroshot_portfolio_2025` (20 hand-tuned configs, plain
   dicts, copied in) + TabICLv2 + Sobol random configs. It beats Bayesian-optimisation search on a small budget (TabRepo).
4. **Hidden holdout the agent can never see** (Meta AIRA2): selection by validation alone lost 9-16 points in their tests.
   Top-3 reported, not top-1 (recovers ~10 %).
5. **Nebius has no prompt caching**: every turn re-bills the whole context, so context design is the token budget.
6. **Single agent**; no council (multi-agent hurts in tool-heavy tasks, arXiv 2512.08296). One cheap critic call at most.
7. **Vectorized numpy over C++**: the model libraries are already native. Hot loops (DeLong, paired bootstrap, row hashing)
   are numpy-vectorized (numba has no wheel for this Mac and needs LLVM to build); a C++ port happens only if a benchmark
   shows a loop > 1 s per rung.
8. **Reward hacking is real** (~50 % of episodes when allowed): metric and eval code stay outside the LLM's reach; every
   number comes from a tool return.

## Pipeline (deterministic harness; LLM judges at ◆)
```
0 Ingest      → 3-way split: dev 70 / search-val 10 / HIDDEN 20 (stratified; time data = last 20 %)
1 Diagnose    → ~45 checks + landmarkers (incl. TabICLv2 on 2k rows) → ranked findings      ◆ task/metric/CV/drops
2 Size n*     → pilot → Hoeffding/precision bound + learning-curve knee → verified random sample
3 Warm start  → Supabase experience memory: nearest past datasets by meta-features → their winning configs
4 Space       → portfolio(20) + TFM + memory configs + Sobol randoms (~2-5k) → prior-ranked     ◆ purpose filter, predict-then-verify prune
5 Race        → ASHA η=3 on row fidelity 500→1.5k→4.5k→13.5k→n*, paired early dropping,
                Kendall-τ rank-stability stop                                                ◆ inspect survivors
6 Confirm     → top-k: 10×10 CV, Nadeau-Bengio + baycomp ROPE + Holm; walk-forward/purged CV if temporal
7 External    → Tavily search → fetch → enrich/more-rows paired trial                        ◆ which sources to try
8 Ensemble    → Caruana greedy ensemble of top 3-5 from different families; kept only if it wins
9 Hidden test → one-shot: McNemar/DeLong/paired bootstrap, calibration, dev→hidden gap
10 Report     → top-k table + Pareto (score tie-group × cost) + recommendation                ◆ narrative (final call, no tools, JSON schema)
```

## Components (new files, each < 400 lines)

| File | Role |
|---|---|
| `providers.py` | Nebius OpenAI client, schema-validate + retry-with-error, token/$ ledger, `ScriptedClient` (OpenAI shape) |
| `diagnostics.py` | `@diagnostic` registry; ~45 checks; batched pass; ranked findings |
| `sampling.py` | 3-way split, n\* sizing, stratified draw, representativeness tests |
| `search_space.py` | Portfolio dicts, TFM, Sobol randoms, purpose filters, meta-feature prior |
| `racing.py` | ASHA schedule, paired early dropping, LCCV-style extrapolation with protection, τ stop |
| `stats_tests.py` | Nadeau-Bengio, baycomp wrapper, Holm, McNemar, fast DeLong (vectorized), paired bootstrap, calibration |
| `export.py` | User-model export: fitted pipeline + params JSON + standalone `train_<model>.py` (template, not LLM) + model card |
| `external_data.py` | Tavily search/extract, safe fetch, enrich / more-rows trials |
| `memory.py` | Supabase experience store: meta-features vector → configs/scores; nearest-neighbour warm start |
| `prompts/system.md` | System prompt (6) |
| `agent.py` (modify) | Harness phases, ~8 LLM tools, native OpenAI loop, compaction |

### 1. Diagnostics (~45 checks, one batched pass, ≤ 15 findings to the LLM)
Quality: missing pattern/informative missingness, exact and near duplicates (vectorized row hashing, **cross-split
duplicate check**), constant, id-like, high-cardinality, rare levels, mixed types, MAD/isolation outliers, datetime parse,
text-like. Target: imbalance, entropy, skew + Yeo-Johnson λ, target outliers, kNN label-noise rate, Breusch-Pagan.
Signal: normalised MI, Spearman−Pearson gap, η², single-feature score → **leakage**, monotonicity. Structure: n/p, VIF,
condition number, PCA intrinsic dim, redundant pairs, Friedman H interactions, sparsity. Complexity: **landmarkers**
(linear, stump, depth-3 tree, 1-NN, NB, **TabICLv2**; research says these are the strongest meta-features), RESET,
Fisher F1, N1-MST, class overlap. Time/drift: time order, target autocorrelation, adversarial validation (AUC > 0.6),
per-feature KS/PSI, ACF seasonality, ADF stationarity. Cheap PyMFE groups only (the full set is O(d²)).

### 2. Sample size n\* (random, stratified, verified)
- Pilot n₀ = min(1000, N), stratified on target (quantile bins for regression) × k-means cluster × time bucket.
- Evaluation precision with K configs compared: **n ≥ ln(2K/δ)/(2ε²)** (Hoeffding + union bound); AUC via Hanley-McNeil.
- Training size: fit err(n) = a + b·n^(−c) on 3-4 anchors → n_knee where predicted gain < ε.
- n\* = min(N_dev, max(n_eval, n_knee)); grow stops early if Kendall τ(rank rung i, i+1) ≥ 0.8.
- Representativeness: per-feature KS/χ² with FDR + global adversarial check (sample vs full AUC ≈ 0.5), else redraw.
- The report states n\* with its derivation.

### 3. Search space and prior (thousands of pipelines, zero tokens)
- Families: logreg/ridge, RF, LightGBM, XGBoost, **CatBoost** (new; TabArena's default GBDT), MLP, **TabICLv2** (GPU).
- Preprocessing axes: impute, scale/quantile, onehot/ordinal/target encoding, target transform, class weights.
- Sobol draws per family → ~2-5k configs; purpose filters (interpretability, latency, memory).
- **Prior rules from the benchmarks**: n ≤ 10k & d ≤ 100 → TFM first; 10k-100k → TFM vs CatBoost/LightGBM;
  > 100k or d > 500 → GBDT-led; many high-cardinality categoricals → CatBoost; skewed/irregular/uninformative
  features → GBDT over MLP; smooth signal → MLP allowed with a quantile transform. Spend budget on **family diversity +
  ensembling**, not deep HPO (McElfresh 2023).
- **Predict-then-verify** (FOREAGENT): the LLM gets the findings + the prior-ranked family list once and may re-order or
  veto with reasons. One call, no per-config tokens.

### 4. Racing (statistical guarantees)
- ASHA (async successive halving) η=3 over nested stratified row subsamples, same folds for every config (paired).
  Cost ≈ n·r·log_η n. Guarantee: finds the best arm with budget O(H₂ log n) (Jamieson & Talwalkar 2016).
- Drop config c at a rung if the one-sided paired corrected t vs the leader gives p < α/m, **or** its optimistic learning-curve
  extrapolation can't reach the leader (LCCV). The top 2η per rung are always protected (LCDB: ~15 % of curves are ill-behaved).
- Budget guard: max fits, wall time, and Modal $ per run; every tool result returns budget used/remaining.
- Winner's curse: finalists are re-scored on untouched folds (BBC-CV idea) and on search-val.

### 5. Confirm, backtest, test, rank
- Top-k: 10×10 repeated CV → Nadeau-Bengio corrected t (df = kr−1) + `baycomp` P(better / equivalent / worse)
  with ROPE, Holm vs the best. Tie groups are reported.
- Temporal data: `TimeSeriesSplit(gap=h)` walk-forward + purged k-fold with embargo; CPCV via `skfolio` for a
  score distribution; Mann-Kendall decay test across windows.
- Hidden test, once: McNemar (accuracy), DeLong (AUC), paired bootstrap (regression) vs the best; Brier/ECE
  calibration; permutation importance; 5 % noise robustness; **dev→hidden gap** flag.
- **Cost / Big-O**: theoretical complexity per family + measured fit-time log-log slope across rungs + single-row predict ms.
- **Ranking**: Pareto over (statistical tie group, cost); the purpose picks the operating point. The Caruana ensemble is shown
  only if it beats the best single model on hidden.

### 5b. User purpose in, trainable model out (end-user flow)
The end user gives **dataset + target + purpose** in plain words (e.g. "flag churners early, recall matters, must be
explainable"). Phase 1 turns the purpose into a machine spec via one LLM call with a JSON schema: primary metric,
constraints (interpretability, latency, memory, fairness column), operating point (threshold or top-k %), CV type.
That spec drives the purpose filters, the metric, and the final pick. After the report the user gets:
1. The fitted model (`models/<run_id>/<name>.joblib` on the Modal volume, downloadable) + `params.json`.
2. A standalone `train_<model>.py` generated from a template (not the LLM): the exact sklearn pipeline and hyperparameters,
   so the user retrains on their own full or future data with `uv run train_<model>.py data.csv --target y`.
3. A model card: purpose, metric ± CI, operating threshold, calibration, known caveats, drift checks to rerun.
4. An optional one-click "train on full data" (`fit_final`) when the user accepts the recommendation.
Test: the exported script retrains on churn and reproduces the reported CV score within 1 std.

### 6. System prompt (`prompts/system.md`, ≤ 1.5k tokens, stable prefix, no timestamps)
Original text; section structure inspired by the CL4R1T4S prompt (third-party copy; structure only).
Identity & mission → "math decides, you judge and narrate" → phase map with stopping criteria → tool catalog
(one line each) → defaults over questions → statistical honesty (CI overlap = tie, never cite an untooled number,
"may help, untested") → narration rules (≤ 2 plain sentences per step, for a live audience) → report schema.
2-3 varied worked examples (not identical, to avoid copying). Todo recitation: the harness appends the current phase
checklist at the end of each turn.

### 7. LLM tools (~8, coarse, compact JSON, `response_format: concise|detailed`)
`diag_summary`, `diag_run(check, columns)`, `plan_search(purpose, metric, vetoes, reorder)`,
`screen_status(top_k)`, `screen_inspect(config_id)`, `data_search(query, purpose)`, `data_try(url, mode, keys)`,
`report_submit(...)` (final). Errors name the bad field, the allowed values, and a correct example; no tracebacks.
The tool list stays fixed for the whole run.

### 8. Context and token engineering (Nebius has no caching)
Stable prefix; append-only history with sorted-key JSON; old tool results → 1-line summary + `artifact://` pointer
(full data in `runs/` and Supabase); leaderboard as an AIDE-style compact table; temperature 0; tight `max_tokens`;
ledger of tokens/$ per call in events and the report. Target: < 40k input tokens per full run (measured).

### 9. Nebius provider
- `base_url=https://api.tokenfactory.nebius.com/v1/`, `NEBIUS_API_KEY`, `AGENT_MODEL`.
- **Model pick by measurement**: `scripts/probe_models.py` sends 20 canned tool-call prompts to Qwen3-235B-A22B-Instruct-2507
  ($0.20/$0.60 per 1M) and DeepSeek-V4-Flash ($0.14/$0.28) and counts schema-valid calls + latency; DeepSeek-V4-Pro is the
  fallback for quality. Instruct (non-thinking) models preferred, so no `reasoning_content` round-trip.
- Quirks handled: no tool-call streaming; validate every call against its schema and retry once with the error; the final
  structured report uses a no-tools call with `json_schema` (Nebius rejects tools + constrained decoding together).
- Native OpenAI loop in `agent.py`; `anthropic` dependency removed.

### 10. Experience memory (Supabase)
Table `experience(meta_vector, dataset_card, winning_configs, scores, created_at)`. After every run, store the result;
at phase 3, fetch the 3 nearest runs by cosine on standardized meta-features and inject their winners into rung 0.
Every run is logged as a future distillation dataset (pitch line).

### 11. Tavily external data
Search (`include_domains` openml/huggingface/uci/data.gov/github/zenodo/kaggle-suggest-only) → `extract` to find the
file link → safe fetch (https only, private IPs blocked, 30 s, 50 MB, csv/parquet only) → enrich (left join,
match_rate ≥ 30 %, leakage flags) or more-rows (training folds only) → paired trial on the current best → verdict with
delta ± CI. The learning-curve slope decides whether more-rows is worth searching for.

### 12. Asks to Atilla (`modal_train.py`)
1. Per-fold scores in results. 2. `train_rows` + `subsample_seed` (nested stratified). 3. CatBoost in the menu.
4. `tabicl_candidate` GPU function (`gpu="L4"` or `"A10G"`). 5. `cv: walk_forward|purged` with `gap`/`embargo`.
6. `predict_holdout(spec)` returning per-row predictions. 7. fit time per fold + predict ms.
I can write these as a PR against his file if he prefers.

## Build order (today; every step leaves a working agent)
0. **Sync + plan doc (5 min)**: `git merge --ff-only origin/main` first (keeps it a fast-forward), then
   `docs/AGENT_PLAN.md` → commit → push to `Coflazo-Branch-N`.
1. **Env (30 min)**: git identity = Coflazo; `uv remove anthropic`;
   `uv add openai tavily-python scikit-learn scipy baycomp statsmodels` (done); `! uv run modal setup`; deploy; apply
   the Nebius promo; set up the Supabase `experience` table with Emre.
2. **Provider + model probe (1 h)**: native loop, scripted replay test, pick the model by probe.
3. **Diagnostics + sampling (2 h)**: core 20 checks + n\* + 3-way split; the rest of the catalog as time allows.
4. **Space + racing + stats (2.5 h)**: portfolio, prior, ASHA, tests. TabICLv2 once Atilla's GPU function lands.
5. **Prompt + context + tools (1 h)**: `prompts/system.md`, compaction, ledger, 8 tools.
6. **Confirm/test/rank + report + export (1.5 h)**: `export.py`, model card, purpose spec.
7. **Tavily + memory (1 h)**.
8. **Golden eval + demo cache (45 min)**.
9. Stretch: Caruana ensemble, CAAFE-style LLM features with CV gate (sandboxed on Modal), CPCV, critic call.
Commit per step as Coflazo (`feat: ...`); verify `git log -1 --format='%an <%ae>'`.

## Git cadence (user request: commit + push regularly)
- **Step 0**: copy this plan to `docs/AGENT_PLAN.md` on `Coflazo-Branch-N` → commit `docs: agent development plan` → push.
- After every build step (and at least every ~45 min): run the tests → commit as Coflazo (conventional message) →
  verify the author → `git push origin Coflazo-Branch-N`. The user's own local edits are included in these commits
  after I show `git status`/diff and confirm they are not secrets. Never push to `main`; never force-push.
- Before each commit: `git config user.name/email` = Coflazo; no `.env`, keys, promo codes, or `runs/` artifacts staged.

## Skills map (387 installed; every one that fits this build, by phase; CLAUDE.md allows one process spine)
- **Spine**: `superpowers:writing-plans` → `test-driven-development` → `subagent-driven-development` /
  `dispatching-parallel-agents` (independent files in parallel) → `requesting-code-review` →
  `verification-before-completion` → `finishing-a-development-branch`.
- **Always on**: `karpathy-coder`, `ponytail`, `i-have-adhd`, `caveman` (chat only).
- **Design and decisions**: `architecture-decision-records` (TFM vs GBDT, no council, no caching), `spec`,
  `loop-design-check` (agent loop), `diagram` (architecture diagram for the pitch).
- **Math / ML / algorithms**: `algorithms` (ASHA, racing, MST, Sobol, complexity), `ml-skill` (diagnostics, priors,
  backtesting), `mle-workflow`, `python-patterns`, `error-handling`.
- **LLM / agent engineering**: `cost-aware-llm-pipeline`, `context-budget`, `strategic-compact`,
  `iterative-retrieval` (memory warm start), `benchmark-models` (Nebius probe), `eval-harness` + `ai-regression-testing`
  (golden set, pass^k), `agent-introspection-debugging` (failed runs), `agent-self-evaluation`.
- **Honesty of outputs**: `ai-claim-checker` + `ai-hallucination-fact-check-protocol` (every report number traced
  to a tool result), `stop-slop` + `anti-slop` + `humanizer` (system prompt, report template, narration).
- **Research**: `deep-research`, `research`, `github-trending` (more prior art if time allows).
- **Performance / native**: `benchmark` (hot loops), then only if triggered: `cpp-pro`, `cpp-coding-standards`, `cpp-testing`.
- **Quality**: `plankton-code-quality`, `codehealth-mcp`, `simplify`, `review`, `production-audit`, `delivery-gate`;
  agents `mle-reviewer`, `python-reviewer`, `silent-failure-hunter`, `code-reviewer`.
- **Security**: `security-audit` (focused), `cso`, `careful`/`guard` (destructive commands).
- **Git**: `git-workflow`, `ship` (push cadence), `context-save`/`context-restore` (survive long sessions).
- **Data viz / demo**: `dataviz` (leaderboard, learning curve, Pareto charts for Emre's UI), `brag`/`hyperframes`
  (optional launch video), `make-pdf` (report export).
- **Bugs**: `matt-builder:diagnose` / `investigate` (one bug at a time).
- Not used: education, resume, iOS/Android, frontend-design, and teaching skills (they don't fit this task); `claude-api` (no Anthropic).

## Verification
- Unit tests (plain asserts, no network): diagnostics on synthetic data with known properties (planted leak, linear vs
  `sin`, 95/5 imbalance, drift); n\* formula vs hand calc; biased sample rejected; Nadeau-Bengio vs a hand-computed value;
  DeLong vs a reference; ASHA survivor counts; the racing drop never removes a protected config; SSRF guard rejects
  `http://` / `127.0.0.1` / oversize; OpenAI loop replays a scripted trajectory.
- **Golden eval set** (5 datasets: leaky churn, clean houses, drifting time series, high-cardinality categorical, tiny
  n ≤ 1k): the right diagnostic fires, the recommendation is in the hidden-test top-k, tokens/run and schema retries are
  logged; report **pass^3**.
- End to end: `uv run agent.py data/churn.csv --target Churn --provider nebius` → flags `refund_issued`, races
  thousands of configs, writes a report with top-k, CI/p, Pareto, n\* derivation, and tokens spent.
- `grep -ri anthropic` over the repo returns nothing.
- security-audit focused pass on the fetch and the memory store.

## Security
Keys only from env: `NEBIUS_API_KEY`, `TAVILY_API_KEY`, `SUPABASE_URL`/`SUPABASE_KEY`, Modal token. No Anthropic.
Promo codes from the PDFs never go into the repo. `.env` gitignored. The LLM never executes code on the evaluator path.

## Open questions (default chosen if no answer)
- TabPFN license at the event → default: excluded, TabICLv2 used.
- Big-O reading of "the On" → default: yes.

---

# Finalization scope (added 2026-09-23, afternoon)

## Product flow
1. Upload a dataset (drag and drop) -> 2. Check it (server profile: columns, types, missing %, preview, suggested
target/task, time columns) -> 3. Describe the goal in plain words -> 4. The agent maps the goal to a use case
(`usecases.py`, 40-60 researched B2B / SMB / enterprise use cases, TF-IDF matching, no tokens) and to a spec
(`purpose.py`) -> 5. Forecasting goals get leak-free lag/rolling/calendar features (`forecasting.py`) with walk-forward
CV and naive baselines -> 6. Diagnostics, n* sample, racing, confirmation, hidden test (already live) -> 7. The model is
refit on every row and exported (script, params, model card) for the user to own.

## Frontend
Style and motion of glasa.io (SvelteKit, Unicorn Studio WebGL scenes, DM Sans, dark #0d0d0d with #E8734A accent) and
hockeystack.com (Webflow, GSAP 3.13 ScrollTrigger + SplitText, Swiper, Rive). Ours: static web/, DM Sans, GSAP 3.13
(ScrollTrigger, SplitText) + Lenis from jsDelivr, a self-written WebGL shader hero (Unicorn/Rive need editor assets),
scroll-pinned pipeline explainer, animated funnel, FLIP leaderboard. Style and motion only; no copied brand assets.

## Supabase
Emre's v2 schema (runs keyed by run_id; candidates per rung / confirm / hidden; events; RLS read-only for anon) is
supported by `supabase_sync.py`, which detects v1 vs v2 from the PostgREST OpenAPI description at start.

## Testing on public data
`evals/public_catalog.py` + `evals/fetch_public.py`: classic Kaggle datasets from public mirrors on the allowlisted
domains (Kaggle itself needs auth), fetched through the SSRF-safe `safe_fetch`. `evals/run_golden.py` scores live runs
(finding fired, leaks dropped, CV scheme, pick in tie group, claim check, cost) and reports pass^k.

## ElevenLabs (last step, starts when the key arrives)
Goal: the 60-word `spoken_summary` (already produced by every run) becomes audio on the result screen.
1. Key: `ELEVENLABS_API_KEY` (+ optional `ELEVENLABS_VOICE_ID`, default a neutral stock voice) in `.env` and the
   `ml-factory-env` Modal Secret. Never sent to the browser.
2. `voice.py`: `speak(text, voice_id) -> bytes` via `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
   (header `xi-api-key`, JSON `{text, model_id: "eleven_flash_v2_5" (low latency, cheap) or
   "eleven_multilingual_v2" (quality), voice_settings}`), `Accept: audio/mpeg`, 20 s timeout, one retry on 429/5xx,
   text capped at 400 characters (60 words), em dashes stripped. The claim-checked summary is the only input, so no
   untooled number is ever spoken.
3. Hook: at the end of `FactoryRun.run()`, after `write_report`, best effort (a TTS failure never fails a run):
   write `{runs_dir}/{run_id}_summary.mp3`, emit `{"kind": "audio", "payload": {"path", "chars", "voice_id"}}`,
   add `audio_chars` to the usage ledger (ElevenLabs bills per character; about 350 characters per run).
4. API: `GET /api/runs/{run_id}/summary.mp3` (run_id validated with `check_name`, 404 if absent,
   `Content-Type: audio/mpeg`). CSP: `media-src 'self'`.
5. UI: an audio control on the result screen (play/pause, waveform-style progress bar in the site's motion language,
   transcript shown next to it for accessibility, no autoplay; respects reduced motion).
6. Tests: fake HTTP for `speak` (headers, payload, cap, retry), the endpoint 404/200, and the hook never raising.
7. Verify: one live run produces an mp3 that plays in the browser; cost line shows the characters used.
