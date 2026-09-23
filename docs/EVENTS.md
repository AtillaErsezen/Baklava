# Event contract

Every step of a run is one event row: `{run_id, ts, kind, payload}`. The agent prints it, appends it to
`runs/<run_id>_events.jsonl`, and inserts it into the Supabase `events` table (backend, secret key).
The web UI reads the same rows with the publishable key (RLS: select only) and subscribes to Realtime
inserts filtered by `run_id`. Replay mode plays a stored jsonl in `ts` order.

Payload fields below are the stable contract. Extra fields may appear; the UI should ignore what it
does not know.

| kind | when | payload |
|---|---|---|
| `run_start` | once, first | `dataset`, `target`, `rows`, `features` |
| `diagnostics` | on `diag_summary` | `findings` [{`check`, `severity` 1-3, `finding`}], `rows` {`dev`, `search_val`, `hidden_locked`} |
| `thought` | before tool calls | `text`: the agent's plain-language narration (<= 2 sentences) |
| `tool_call` | each tool call | `tool`, `input` (parsed arguments, or the raw string if invalid) |
| `status` | progress notes | `msg` |
| `search_plan` | start of `run_search` | `space` (pipelines generated), `raced` (configs raced), `warm_start`, `n_star`, `schedule` [{`rung`, `n_rows`, `keep`}], `rationale`, `cv` (scheme used), `time_column` |
| `rung` | after each racing rung | `rung`, `n_rows`, `evaluated`, `survivors`, `dropped_stat`, `dropped_rank`, `leader`, `leader_mean`, `tau` |
| `leaderboard` | after a race or an experiment round | `round` ("race" or a number), `primary_metric`, `rows` [{`name`, `family`?, `mean` or metric, `std`, `params`?}] |
| `round_start` | `run_experiments` only | `round`, `rationale`, `candidates` [{`name`, `model`, `params`}] |
| `confirm` | after `confirm_and_test` | `primary_metric`, `table` (see below), `recommendation` {`pick`, `best`, `threshold`, `within_1se`}, `tie_groups`, `notes` |
| `external_trial` | after `data_try` | `source`, `verdict` (improves / no_gain / worse), `delta`, `ci` [low, high], `p_one_sided`, `match_rate`, `new_columns`, `leakage_flags` |
| `final_model` | after `finalize_model` | `name`, `spec`, `model_path`, `n_rows`, `top_features` [{`feature`, `importance`}] |
| `export` | right after `final_model` | `script`, `params`, `card` (file paths of the user's bundle) |
| `report` | after `write_report` | `markdown`, `spoken_summary` (<= 60 words) |
| `usage` | once, last | `calls`, `input_tokens`, `output_tokens`, `usd` |

`confirm.table` rows: `name`, `family`, `cv_mean`, `ci` [low, high], `p_vs_best` (Nadeau-Bengio on 10 paired
folds; null for the best), `p_equiv_bayes`, `tie_group` (0 = statistically tied with the best), `hidden`
(one-shot holdout score), `hidden_p_vs_best` (DeLong / McNemar / paired bootstrap), `ece`,
`dev_to_hidden_gap`, `fit_s`, `predict_ms`, `big_o` {`train`, `predict`}, `params`, `pareto` (bool).

Screen mapping for the UI:
- Narration stream: `thought`. Activity line: `tool_call`, `status`.
- Diagnostics cards: `diagnostics.findings`, sorted by severity.
- Search funnel: `search_plan.space` -> `search_plan.raced` -> each `rung.survivors` -> `confirm.table` length.
- Leaderboard: latest `leaderboard.rows`. Result screen: `confirm`, `final_model`, `export`, `report`, `usage`.
