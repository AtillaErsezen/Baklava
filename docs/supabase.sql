-- ML Factory Supabase schema. Run once in the Supabase SQL editor (safe to re-run).
-- The backend writes with the SECRET key (bypasses RLS). The browser reads with the
-- PUBLISHABLE key, which RLS limits to reading events only.

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
