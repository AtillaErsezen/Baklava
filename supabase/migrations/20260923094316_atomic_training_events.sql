begin;

-- Search budgets belong to the agent; later racing rungs may exceed three rounds.
alter table public.trainings drop constraint trainings_round_check;
alter table public.trainings add constraint trainings_round_check check (round >= 1);

create function public.apply_tracking_event(p_event jsonb)
returns integer
language plpgsql
security invoker
set search_path = pg_catalog
as $$
declare
    changes jsonb;
    dataset_state jsonb;
    run_state jsonb;
    training_state jsonb;
    run_key text;
    event_key uuid;
    existing_event public.events;
    existing_dataset public.datasets;
    clock_value timestamptz := clock_timestamp();
begin
    -- Read-only protocol check; still restricted to the backend role.
    if p_event is null then
        return 1;
    end if;
    changes := p_event->'payload'->'_tracking';
    if changes is null or changes->>'version' is distinct from '1' then
        raise exception 'Unsupported tracking event version';
    end if;
    run_key := p_event->>'run_id';
    event_key := (p_event->>'event_id')::uuid;
    if nullif(run_key, '') is null or event_key is null then
        raise exception 'Tracking event needs stable run and event identities';
    end if;
    -- Serialize events within a run, including retries after an uncertain HTTP response.
    perform pg_advisory_xact_lock(hashtextextended(run_key, 0));
    select * into existing_event from public.events where event_id = event_key;
    if found then
        if existing_event.run_id is distinct from run_key
            or existing_event.kind is distinct from p_event->>'kind'
            or existing_event.training_id is distinct from (p_event->>'training_id')::uuid
            or existing_event.ts is distinct from (p_event->>'ts')::double precision
            or existing_event.payload is distinct from p_event->'payload' then
            raise exception 'An event identity cannot be reused for different content';
        end if;
        return 1;
    end if;

    dataset_state := changes->'dataset';
    run_state := changes->'run';
    training_state := changes->'training';
    if dataset_state is not null then
        if run_state is null or dataset_state->>'id' is distinct from run_state->>'dataset_id' then
            raise exception 'Dataset snapshot must match the run snapshot';
        end if;
        select * into existing_dataset from public.datasets where id = (dataset_state->>'id')::uuid;
        if found and not (to_jsonb(existing_dataset) @> (dataset_state - 'created_at')) then
            raise exception 'Dataset snapshot conflicts with the registered version';
        end if;
        insert into public.datasets
        select (jsonb_populate_record(null::public.datasets,
            jsonb_build_object('storage_bucket', 'datasets', 'preview_rows', '[]'::jsonb,
                'is_demo_fixture', false, 'created_at', clock_value) || dataset_state)).*
        on conflict (id) do nothing;
    end if;

    if run_state is not null then
        if run_state->>'id' is distinct from run_key then
            raise exception 'Run snapshot identity does not match event';
        end if;
        insert into public.runs as stored_run
        select (jsonb_populate_record(null::public.runs,
            jsonb_build_object('status', 'queued', 'created_at', clock_value,
                'updated_at', clock_value, 'row_version', 1) || run_state)).*
        on conflict (id) do update set
            dataset_id = excluded.dataset_id, target = excluded.target, task = excluded.task,
            status = excluded.status, selected_training_id = excluded.selected_training_id,
            report_markdown = excluded.report_markdown, spoken_summary = excluded.spoken_summary,
            error = excluded.error, started_at = excluded.started_at, finished_at = excluded.finished_at
        where not (to_jsonb(stored_run) @> run_state);
    end if;

    if training_state is not null then
        if training_state->>'run_id' is distinct from run_key
            or training_state->>'id' is distinct from p_event->>'training_id' then
            raise exception 'Training snapshot identity does not match event';
        end if;
        insert into public.trainings as stored_training
        select (jsonb_populate_record(null::public.trainings,
            jsonb_build_object('attempt', 1, 'stage', 'candidate_cv', 'status', 'queued',
                'requested_config', '{}'::jsonb, 'data_manifest', '{}'::jsonb,
                'evaluation_config', '{}'::jsonb, 'warnings', '[]'::jsonb,
                'created_at', clock_value, 'updated_at', clock_value, 'row_version', 1) || training_state)).*
        on conflict (id) do update set
            run_id = excluded.run_id, round = excluded.round, candidate_name = excluded.candidate_name,
            attempt = excluded.attempt, stage = excluded.stage, parent_training_id = excluded.parent_training_id,
            model = excluded.model, requested_config = excluded.requested_config,
            effective_config = excluded.effective_config, data_manifest = excluded.data_manifest,
            evaluation_config = excluded.evaluation_config, status = excluded.status,
            primary_metric = excluded.primary_metric, higher_is_better = excluded.higher_is_better,
            metrics = excluded.metrics, overfit_gap = excluded.overfit_gap,
            duration_seconds = excluded.duration_seconds, warnings = excluded.warnings, error = excluded.error,
            model_path = excluded.model_path, started_at = excluded.started_at, finished_at = excluded.finished_at
        where not (to_jsonb(stored_training) @> training_state);
    end if;

    insert into public.events(event_id, run_id, training_id, ts, kind, payload)
    values (event_key, run_key, (p_event->>'training_id')::uuid,
        (p_event->>'ts')::double precision, p_event->>'kind', p_event->'payload');
    return 1;
end;
$$;

revoke all on function public.apply_tracking_event(jsonb) from public, anon, authenticated;
grant execute on function public.apply_tracking_event(jsonb) to service_role;

commit;
