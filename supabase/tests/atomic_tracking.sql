-- Run as database admin after BOTH migrations and 02_setup_demo.sql.
-- Uses service_role for writes and browser roles for access checks; rolls back all rows.
begin;
set local role service_role;

do $$
declare
    run_key text := 'atomic-tracking-check';
    training_key uuid := gen_random_uuid();
    initial_event jsonb;
    finished_event jsonb;
    bad_event jsonb;
    state jsonb;
    version_before bigint;
begin
    initial_event := jsonb_build_object(
        'event_id', gen_random_uuid(), 'run_id', run_key, 'training_id', training_key,
        'ts', 1, 'kind', 'training_queued', 'payload', jsonb_build_object('_tracking',
            jsonb_build_object('version', 1, 'run', jsonb_build_object(
                'id', run_key, 'dataset_id', '10000000-0000-4000-8000-000000000001',
                'target', 'churn', 'task', 'classification', 'status', 'running', 'started_at', now()
            ), 'training', jsonb_build_object(
                'id', training_key, 'run_id', run_key, 'round', 4, 'candidate_name', 'baseline',
                'model', 'logreg', 'status', 'queued'
            ))));
    perform public.apply_tracking_event(initial_event);
    state := initial_event #> '{payload,_tracking,training}';
    state := state || jsonb_build_object('status', 'succeeded', 'finished_at', now(),
        'primary_metric', 'roc_auc', 'higher_is_better', true,
        'metrics', '{"roc_auc":{"mean":0.8,"std":0.02,"train_mean":0.85}}'::jsonb);
    finished_event := initial_event || jsonb_build_object('event_id', gen_random_uuid(),
        'kind', 'training_completed', 'payload', jsonb_build_object('_tracking',
            jsonb_build_object('version', 1, 'training', state)));
    perform public.apply_tracking_event(finished_event);
    select row_version into version_before from public.trainings where id = training_key;

    -- Even the earlier queued snapshot must be a no-op when its exact event is replayed.
    perform public.apply_tracking_event(initial_event);
    perform public.apply_tracking_event(finished_event);
    if (select count(*) from public.events where run_id = run_key) <> 2
        or (select count(*) from public.trainings where run_id = run_key) <> 1
        or (select row_version from public.trainings where id = training_key) <> version_before then
        raise exception 'FAIL: duplicate replay changed history or row versions';
    end if;

    begin
        perform public.apply_tracking_event(initial_event || jsonb_build_object('event_id', gen_random_uuid()));
        raise exception using errcode = 'ZX001', message = 'FAIL: stale state was accepted';
    exception when raise_exception then null;
    end;
    begin
        perform public.apply_tracking_event(finished_event || '{"kind":"tampered"}'::jsonb);
        raise exception using errcode = 'ZX001', message = 'FAIL: event identity reused for different content';
    exception when raise_exception then null;
    end;

    -- A valid run update followed by an invalid training must roll back as a whole.
    bad_event := initial_event || jsonb_build_object('event_id', gen_random_uuid(),
        'training_id', gen_random_uuid(), 'kind', 'training_failed');
    state := jsonb_build_object('id', bad_event->>'training_id', 'run_id', run_key,
        'round', 4, 'candidate_name', 'invalid', 'model', 'logreg',
        'status', 'failed', 'finished_at', now()); -- Missing required failure error.
    bad_event := jsonb_set(bad_event, '{payload,_tracking,training}', state);
    bad_event := jsonb_set(bad_event, '{payload,_tracking,run,report_markdown}', '"must roll back"'::jsonb);
    begin
        perform public.apply_tracking_event(bad_event);
        raise exception using errcode = 'ZX001', message = 'FAIL: invalid training was accepted';
    exception when check_violation then null;
    end;
    if (select report_markdown from public.runs where id = run_key) is not null
        or (select row_version from public.runs where id = run_key) <> 1
        or (select count(*) from public.events where run_id = run_key) <> 2 then
        raise exception 'FAIL: partial state/event escaped transaction rollback';
    end if;
end;
$$;

set local role anon;
do $$
begin
    if (select count(*) from public.trainings where run_id = 'atomic-tracking-check') <> 1 then
        raise exception 'FAIL: anonymous history read failed';
    end if;
    begin
        perform public.apply_tracking_event(null);
        raise exception using errcode = 'ZX001', message = 'FAIL: anonymous RPC allowed';
    exception when insufficient_privilege then null;
    end;
end;
$$;

set local role authenticated;
do $$
begin
    if (select count(*) from public.trainings where run_id = 'atomic-tracking-check') <> 1 then
        raise exception 'FAIL: authenticated history read failed';
    end if;
    begin
        perform public.apply_tracking_event(null);
        raise exception using errcode = 'ZX001', message = 'FAIL: authenticated RPC allowed';
    exception when insufficient_privilege then null;
    end;
end;
$$;
rollback;
