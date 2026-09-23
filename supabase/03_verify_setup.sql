-- Step 3: run after the migration and demo fixture in the Supabase SQL Editor.
-- Assertions fail with a specific error if configuration is incomplete.
-- The final result should contain two rows with validation scores 0.81 and 0.87.
begin;

do $$
declare
    table_name text;
    reader_role text;
begin
    foreach table_name in array array['datasets', 'runs', 'trainings', 'events'] loop
        if not exists (
            select 1 from pg_tables
            where schemaname = 'public' and tablename = table_name and rowsecurity
        ) then
            raise exception 'Table % is missing or RLS is disabled', table_name;
        end if;
        foreach reader_role in array array['anon', 'authenticated'] loop
            if not has_table_privilege(reader_role, 'public.' || table_name, 'SELECT') then
                raise exception '% cannot read %', reader_role, table_name;
            end if;
            if has_table_privilege(reader_role, 'public.' || table_name, 'INSERT,UPDATE,DELETE,TRUNCATE') then
                raise exception '% has unexpected write access to %', reader_role, table_name;
            end if;
        end loop;
        if not has_table_privilege('service_role', 'public.' || table_name, 'INSERT') then
            raise exception 'Trusted backend cannot insert into %', table_name;
        end if;
    end loop;
    foreach table_name in array array['runs', 'trainings', 'events'] loop
        if not exists (
            select 1 from pg_publication_tables
            where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = table_name
        ) then
            raise exception 'Realtime publication is missing %', table_name;
        end if;
    end loop;
end;
$$;

-- Exercise RLS as a real browser role, rather than just the SQL Editor admin.
set local role anon;
do $$
begin
    if (select count(*) from public.datasets where id = '10000000-0000-4000-8000-000000000001') <> 1
        or (select count(*) from public.runs where id = 'setup-demo-001') <> 1
        or (select count(*) from public.trainings where run_id = 'setup-demo-001') <> 2
        or (select count(*) from public.events where run_id = 'setup-demo-001') <> 4 then
        raise exception 'Browser cannot read the expected fixture rows; check that step 2 ran successfully';
    end if;
end;
$$;
commit;

select
    d.name as dataset,
    d.version_label as dataset_version,
    d.is_demo_fixture,
    r.target,
    t.candidate_name,
    t.status,
    (t.data_manifest->>'usable_row_count')::integer as rows_used,
    t.data_manifest->'included_columns' as input_columns,
    t.primary_metric,
    (t.metrics->t.primary_metric->>'mean')::numeric as validation_score,
    (t.metrics->t.primary_metric->>'std')::numeric as validation_std,
    (t.metrics->t.primary_metric->>'train_mean')::numeric as training_score,
    t.duration_seconds
from public.trainings t
join public.runs r on r.id = t.run_id
join public.datasets d on d.id = r.dataset_id
where r.id = 'setup-demo-001'
order by t.candidate_name;

