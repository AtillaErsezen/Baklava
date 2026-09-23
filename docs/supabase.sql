-- ML Factory Supabase schema. Run once in the Supabase SQL editor (safe to re-run).
-- The backend writes with the SECRET key (bypasses RLS). The browser reads with the
-- PUBLISHABLE key, which RLS limits to reading events, runs and candidates.

-- Live event stream for the UI (see docs/EVENTS.md).
create table if not exists events (
  id bigserial primary key,
  run_id text, ts double precision, kind text, payload jsonb
);
create index if not exists events_run_idx on events (run_id, ts);
do $$ begin
  alter publication supabase_realtime add table events;
exception when duplicate_object then null; end $$;

alter table events enable row level security;
drop policy if exists "events are readable by anyone" on events;
create policy "events are readable by anyone" on events for select to anon, authenticated using (true);
-- no insert/update/delete policy: only the secret key (backend) can write

-- Experience memory for warm-starting the search (memory.py). Backend only.
create table if not exists experience (
  id bigserial primary key,
  run_id text unique,
  created_at timestamptz default now(),
  task text,
  dataset_card jsonb,
  meta jsonb,
  winners jsonb
);
create index if not exists experience_task_idx on experience (task);
alter table experience enable row level security;
-- no policies at all: anon and authenticated cannot read or write; the secret key bypasses RLS

-- Durable run results (results_store.py). One row per run.
-- `create table if not exists` never alters an existing table. If runs/candidates were created by hand with
-- other columns or constraints, drop them once (this deletes their rows), then run this whole file:
--   drop table if exists public.candidates;
--   drop table if exists public.runs;
create table if not exists runs (
  run_id          text primary key,                 -- same id as events.run_id
  created_at      timestamptz default now(),
  finished_at     timestamptz,
  status          text not null default 'running' check (status in ('running', 'completed', 'incomplete')),
  dataset_name    text,
  n_rows          int,
  n_features      int,
  target          text,
  task            text check (task in ('classification', 'regression')),
  primary_metric  text,
  purpose         text,
  llm_model       text,
  split           jsonb,        -- dev / search_val / hidden row counts
  profile         jsonb,        -- diagnostics findings + meta-features
  recommended     text,         -- candidate name of the final model
  model_path      text,         -- joblib path on the Modal volume
  report_md       text,
  spoken_summary  text,
  usage           jsonb         -- LLM calls, tokens, cost
);

-- One row per evaluation of a config: a racing rung, the confirmation, or an LLM experiment round.
create table if not exists candidates (
  id                bigserial primary key,
  run_id            text not null references runs (run_id) on delete cascade,
  created_at        timestamptz default now(),
  stage             text not null check (stage in ('race', 'confirm', 'experiment')),
  rung              int,          -- racing rung or experiment round
  name              text not null,
  family            text,         -- logreg, lightgbm, catboost, ...
  train_rows        int,
  params            jsonb not null default '{}',
  preprocessing     jsonb not null default '{}',
  ok                boolean not null,
  error             text,
  cv_mean           double precision,   -- the run's primary metric
  cv_std            double precision,
  train_mean        double precision,
  metrics           jsonb,              -- every metric, with per-fold scores
  fit_seconds       double precision,
  predict_ms        double precision,
  p_vs_best         double precision,   -- confirm stage only
  tie_with_best     boolean,
  hidden_score      double precision,
  hidden_p_vs_best  double precision,
  ece               double precision
);
create index if not exists candidates_run_idx on candidates (run_id, stage);
create index if not exists candidates_family_idx on candidates (family);

alter table runs enable row level security;
alter table candidates enable row level security;
drop policy if exists "runs are readable by anyone" on runs;
create policy "runs are readable by anyone" on runs for select to anon, authenticated using (true);
drop policy if exists "candidates are readable by anyone" on candidates;
create policy "candidates are readable by anyone" on candidates for select to anon, authenticated using (true);
-- no insert/update/delete policy: only the secret key (backend) can write

-- Table privileges. This project does not grant them automatically, and RLS policies only filter rows on top
-- of them. events stays append-only; the backend needs delete on the results tables for test cleanup.
grant select, insert on public.events to service_role;
grant select, insert, update, delete on public.experience, public.runs, public.candidates to service_role;
grant usage, select on all sequences in schema public to service_role;
grant select on public.events, public.runs, public.candidates to anon, authenticated;
