-- Step 1: run this entire file once in a NEW Supabase project's SQL Editor.
-- Shared sample-data demo: browser clients can READ these four tables.
-- Only a trusted backend can write. No existing tables/data are dropped.
-- If a table already exists, the transaction fails instead of replacing it.
begin;

create table public.datasets (
    id uuid primary key default gen_random_uuid(),
    name text not null check (length(btrim(name)) > 0),
    original_filename text not null,
    version_label text not null,
    storage_bucket text not null default 'datasets',
    storage_path text,
    source_sha256 text not null check (source_sha256 ~ '^[0-9a-f]{64}$'),
    file_size_bytes bigint not null check (file_size_bytes >= 0),
    row_count bigint not null check (row_count >= 0),
    column_count integer not null check (column_count > 0),
    column_schema jsonb not null check (jsonb_typeof(column_schema) = 'array'),
    preview_rows jsonb not null default '[]'::jsonb
        check (jsonb_typeof(preview_rows) = 'array'),
    description text,
    is_demo_fixture boolean not null default false,
    created_at timestamptz not null default now(),
    unique (storage_bucket, storage_path)
);

create table public.runs (
    id text primary key check (length(btrim(id)) > 0),
    dataset_id uuid not null references public.datasets(id),
    target text not null check (length(btrim(target)) > 0),
    task text check (task in ('classification', 'regression')),
    status text not null default 'queued'
        check (status in ('queued', 'running', 'completed', 'failed', 'incomplete')),
    selected_training_id uuid,
    report_markdown text,
    spoken_summary text,
    error text,
    started_at timestamptz,
    finished_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    row_version bigint not null default 1 check (row_version > 0),
    check (finished_at is null or started_at is null or finished_at >= started_at),
    check ((status in ('completed', 'failed', 'incomplete')) = (finished_at is not null))
);

create table public.trainings (
    id uuid primary key default gen_random_uuid(),
    run_id text not null references public.runs(id),
    round integer not null check (round between 1 and 3),
    candidate_name text not null check (length(btrim(candidate_name)) > 0),
    attempt integer not null default 1 check (attempt > 0),
    stage text not null default 'candidate_cv'
        check (stage in ('candidate_cv', 'final_fit')),
    parent_training_id uuid,
    model text not null,
    requested_config jsonb not null default '{}'::jsonb
        check (jsonb_typeof(requested_config) = 'object'),
    effective_config jsonb check (jsonb_typeof(effective_config) = 'object'),
    data_manifest jsonb not null default '{}'::jsonb
        check (jsonb_typeof(data_manifest) = 'object'),
    evaluation_config jsonb not null default '{}'::jsonb
        check (jsonb_typeof(evaluation_config) = 'object'),
    status text not null default 'queued'
        check (status in ('queued', 'running', 'succeeded', 'failed')),
    primary_metric text check (
        primary_metric in ('accuracy', 'f1_macro', 'roc_auc', 'rmse', 'mae', 'r2')
    ),
    higher_is_better boolean,
    metrics jsonb check (jsonb_typeof(metrics) = 'object'),
    overfit_gap double precision,
    duration_seconds double precision check (
        duration_seconds >= 0 and duration_seconds < 'Infinity'::double precision
    ),
    warnings jsonb not null default '[]'::jsonb
        check (jsonb_typeof(warnings) = 'array'),
    error text,
    model_path text,
    started_at timestamptz,
    finished_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    row_version bigint not null default 1 check (row_version > 0),
    unique (run_id, id),
    unique (run_id, round, candidate_name, attempt, stage),
    foreign key (run_id, parent_training_id) references public.trainings(run_id, id),
    check (parent_training_id is null or parent_training_id <> id),
    check (
        (stage = 'candidate_cv' and parent_training_id is null)
        or (stage = 'final_fit' and parent_training_id is not null)
    ),
    -- A final fit does not generate a new validation score in the current backend.
    check (stage <> 'final_fit' or metrics is null),
    check (
        status <> 'succeeded' or stage <> 'candidate_cv'
        or (primary_metric is not null and higher_is_better is not null
            and metrics is not null and metrics ? primary_metric)
    ),
    check (status <> 'failed' or nullif(btrim(error), '') is not null),
    check (finished_at is null or started_at is null or finished_at >= started_at),
    check ((status in ('succeeded', 'failed')) = (finished_at is not null)),
    check (
        primary_metric is null or higher_is_better is null
        or higher_is_better = (primary_metric in ('accuracy', 'f1_macro', 'roc_auc', 'r2'))
    )
);

alter table public.runs add constraint runs_selected_training_same_run
    foreign key (id, selected_training_id) references public.trainings(run_id, id);

create table public.events (
    id bigint generated always as identity primary key,
    event_id uuid not null default gen_random_uuid() unique,
    run_id text not null references public.runs(id),
    training_id uuid,
    ts double precision not null default extract(epoch from now())
        check (ts >= 0 and ts < 'Infinity'::double precision),
    kind text not null check (length(btrim(kind)) > 0),
    payload jsonb not null default '{}'::jsonb
        check (jsonb_typeof(payload) = 'object'),
    created_at timestamptz not null default now(),
    foreign key (run_id, training_id) references public.trainings(run_id, id)
);

create index runs_dataset_created_idx on public.runs(dataset_id, created_at);
create index trainings_run_round_created_idx on public.trainings(run_id, round, created_at);
create index events_run_id_idx on public.events(run_id, id);

-- Keep refresh/reconnect merges deterministic, and protect attempt identity.
create function public.guard_tracking_update()
returns trigger
language plpgsql
set search_path = pg_catalog
as $$
begin
    if new.id is distinct from old.id or new.created_at is distinct from old.created_at then
        raise exception 'Tracking identity and creation time are immutable';
    end if;
    if tg_table_name = 'runs' then
        if new.dataset_id is distinct from old.dataset_id
            or new.target is distinct from old.target
            or (old.task is not null and new.task is distinct from old.task) then
            raise exception 'A run must retain its original dataset, target and task';
        end if;
        if old.status in ('completed', 'failed', 'incomplete') and new.status <> old.status then
            raise exception 'A finished run cannot return to an earlier state';
        end if;
        if old.status = 'running' and new.status = 'queued' then
            raise exception 'A running run cannot return to queued';
        end if;
    else
        if new.run_id is distinct from old.run_id
            or new.round is distinct from old.round
            or new.candidate_name is distinct from old.candidate_name
            or new.attempt is distinct from old.attempt
            or new.stage is distinct from old.stage
            or new.parent_training_id is distinct from old.parent_training_id
            or new.model is distinct from old.model
            or new.requested_config is distinct from old.requested_config then
            raise exception 'Create a new attempt to change its dataset or requested configuration';
        end if;
        if old.status in ('succeeded', 'failed') and new is distinct from old then
            raise exception 'A finished training record is immutable; create a new attempt';
        end if;
        if old.status = 'running' and new.status = 'queued' then
            raise exception 'A running training cannot return to queued';
        end if;
    end if;
    new.row_version := old.row_version + 1;
    new.updated_at := clock_timestamp();
    return new;
end;
$$;

create trigger runs_guard_update before update on public.runs
    for each row execute function public.guard_tracking_update();
create trigger trainings_guard_update before update on public.trainings
    for each row execute function public.guard_tracking_update();

-- A final fit and a run's winner must refer to a successful CV candidate.
create function public.check_tracking_selection()
returns trigger
language plpgsql
set search_path = pg_catalog
as $$
declare
    candidate_id uuid;
    candidate_run text;
begin
    if tg_table_name = 'runs' then
        candidate_id := new.selected_training_id;
        candidate_run := new.id;
    else
        candidate_id := new.parent_training_id;
        candidate_run := new.run_id;
    end if;
    if candidate_id is not null and not exists (
        select 1 from public.trainings
        where id = candidate_id and run_id = candidate_run
          and stage = 'candidate_cv' and status = 'succeeded'
    ) then
        raise exception 'Selection must reference a successful CV candidate in the same run';
    end if;
    return new;
end;
$$;

create trigger runs_check_selection before insert or update on public.runs
    for each row execute function public.check_tracking_selection();
create trigger trainings_check_parent before insert or update on public.trainings
    for each row execute function public.check_tracking_selection();

alter table public.datasets enable row level security;
alter table public.runs enable row level security;
alter table public.trainings enable row level security;
alter table public.events enable row level security;

grant usage on schema public to anon, authenticated, service_role;
revoke all on public.datasets, public.runs, public.trainings, public.events
    from public, anon, authenticated;
grant select on public.datasets, public.runs, public.trainings, public.events
    to anon, authenticated;

-- Explicit grants also work in projects with restrictive default privileges.
revoke all on public.datasets, public.runs, public.trainings, public.events from service_role;
grant select, insert on public.datasets, public.events to service_role;
grant select, insert, update on public.runs, public.trainings to service_role;
revoke all on sequence public.events_id_seq from public, anon, authenticated;
grant usage, select on sequence public.events_id_seq to service_role;

create policy demo_read_datasets on public.datasets for select to anon, authenticated using (true);
create policy demo_read_runs on public.runs for select to anon, authenticated using (true);
create policy demo_read_trainings on public.trainings for select to anon, authenticated using (true);
create policy demo_read_events on public.events for select to anon, authenticated using (true);

revoke all on function public.guard_tracking_update() from public, anon, authenticated;
revoke all on function public.check_tracking_selection() from public, anon, authenticated;
grant execute on function public.guard_tracking_update(), public.check_tracking_selection() to service_role;

-- Preserve other publication members; Supabase normally creates this publication.
do $$
declare
    table_name text;
begin
    if not exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
        create publication supabase_realtime;
    end if;
    foreach table_name in array array['runs', 'trainings', 'events'] loop
        if not exists (
            select 1 from pg_publication_tables
            where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = table_name
        ) then
            execute format('alter publication supabase_realtime add table public.%I', table_name);
        end if;
    end loop;
end;
$$;

commit;

