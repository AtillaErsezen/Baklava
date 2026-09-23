-- Developer checks after migration + setup fixture. Rolls back all test records.
begin;

-- Backend can create a real lifecycle with monotonically increasing versions.
set local role service_role;
insert into public.runs (id, dataset_id, target, task)
values ('constraint-check', '10000000-0000-4000-8000-000000000001', 'churn', 'classification');
insert into public.trainings (id, run_id, round, candidate_name, model)
values ('40000000-0000-4000-8000-000000000001', 'constraint-check', 1, 'baseline', 'logreg');
update public.trainings set status = 'running', started_at = now()
where id = '40000000-0000-4000-8000-000000000001';
update public.trainings
set status = 'succeeded', primary_metric = 'roc_auc', higher_is_better = true,
    metrics = '{"roc_auc":{"mean":0.8,"std":0.02,"train_mean":0.85}}', finished_at = now()
where id = '40000000-0000-4000-8000-000000000001';
insert into public.events (run_id, training_id, kind, payload)
values ('constraint-check', '40000000-0000-4000-8000-000000000001', 'training_completed', '{}');

do $$
begin
    if (select row_version from public.trainings where id = '40000000-0000-4000-8000-000000000001') <> 3 then
        raise exception 'FAIL: row version did not advance for both lifecycle updates';
    end if;
    begin
        update public.trainings set status = 'running', finished_at = null
        where id = '40000000-0000-4000-8000-000000000001';
        raise exception using errcode = 'ZX001', message = 'FAIL: terminal state regressed';
    exception when raise_exception then null;
    end;
    begin
        update public.runs set target = 'monthly_spend' where id = 'constraint-check';
        raise exception using errcode = 'ZX001', message = 'FAIL: run changed its target';
    exception when raise_exception then null;
    end;
    begin
        update public.runs set selected_training_id = '20000000-0000-4000-8000-000000000001'
        where id = 'constraint-check';
        raise exception using errcode = 'ZX001', message = 'FAIL: selected candidate belongs to another run';
    exception when raise_exception then null;
    end;
    begin
        insert into public.events (run_id, training_id, kind)
        values ('constraint-check', '20000000-0000-4000-8000-000000000001', 'training_completed');
        raise exception using errcode = 'ZX001', message = 'FAIL: event belongs to another training run';
    exception when foreign_key_violation then null;
    end;
    begin
        insert into public.trainings (run_id, round, candidate_name, model, status, finished_at)
        values ('constraint-check', 1, 'missing_score', 'logreg', 'succeeded', now());
        raise exception using errcode = 'ZX001', message = 'FAIL: successful candidate had no score';
    exception when check_violation then null;
    end;
    begin
        insert into public.trainings (
            run_id, round, candidate_name, model, stage, parent_training_id, metrics
        ) values (
            'constraint-check', 1, 'fake_final_score', 'logreg', 'final_fit',
            '40000000-0000-4000-8000-000000000001', '{"roc_auc":{"mean":0.99}}'
        );
        raise exception using errcode = 'ZX001', message = 'FAIL: final fit stored a fabricated validation score';
    exception when check_violation then null;
    end;
end;
$$;

-- A genuine final-fit record has a same-run CV parent and no new CV score.
insert into public.trainings (
    run_id, round, candidate_name, model, stage, parent_training_id
) values (
    'constraint-check', 1, 'final_fit', 'logreg', 'final_fit',
    '40000000-0000-4000-8000-000000000001'
);

set local role anon;
do $$
begin
    if (select count(*) from public.trainings where run_id = 'setup-demo-001') <> 2 then
        raise exception 'FAIL: browser cannot read fixture candidates';
    end if;
    begin
        insert into public.events (run_id, kind) values ('setup-demo-001', 'forged_event');
        raise exception using errcode = 'ZX001', message = 'FAIL: browser inserted an event';
    exception when insufficient_privilege then null;
    end;
    begin
        update public.trainings set model = 'tampered' where run_id = 'setup-demo-001';
        raise exception using errcode = 'ZX001', message = 'FAIL: browser changed training results';
    exception when insufficient_privilege then null;
    end;
    begin
        delete from public.datasets where id = '10000000-0000-4000-8000-000000000001';
        raise exception using errcode = 'ZX001', message = 'FAIL: browser deleted the dataset';
    exception when insufficient_privilege then null;
    end;
end;
$$;

rollback;

