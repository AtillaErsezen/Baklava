-- ML Factory Supabase schema. Run once in the Supabase SQL editor.

-- Live event stream for the UI (see README.md, "Events").
create table if not exists events (
  id bigserial primary key,
  run_id text, ts double precision, kind text, payload jsonb
);
alter publication supabase_realtime add table events;

-- Experience memory for warm-starting the search (memory.py).
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
