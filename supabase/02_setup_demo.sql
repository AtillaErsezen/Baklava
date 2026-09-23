-- Step 2: run after the migration. Safe to repeat: fixed IDs prevent duplicates.
-- ALL MODEL SCORES BELOW ARE INVENTED SETUP FIXTURES, NOT TRAINING RESULTS.
-- The CSV is real synthetic sample data in supabase/fixtures/setup_demo.csv.
-- storage_path is null because this script does not upload a file.
begin;

insert into public.datasets (
    id, name, original_filename, version_label, source_sha256, file_size_bytes,
    row_count, column_count, column_schema, preview_rows, description, is_demo_fixture
) values (
    '10000000-0000-4000-8000-000000000001',
    'SETUP DEMO - synthetic churn', 'setup_demo.csv', 'v1',
    '77c298d794de2393c194530e166c369a5e3346dbdd273f95a5d3bd593060a569', 126,
    12, 3,
    '[{"name":"tenure_months","dtype":"integer"},
      {"name":"monthly_spend","dtype":"integer"},
      {"name":"churn","dtype":"integer"}]'::jsonb,
    '[{"tenure_months":2,"monthly_spend":80,"churn":1},
      {"tenure_months":24,"monthly_spend":45,"churn":0},
      {"tenure_months":5,"monthly_spend":95,"churn":1}]'::jsonb,
    'Synthetic setup fixture. Associated model scores are illustrative; no ML job was run.',
    true
) on conflict (id) do nothing;

insert into public.runs (id, dataset_id, target, task, status, started_at)
values (
    'setup-demo-001', '10000000-0000-4000-8000-000000000001',
    'churn', 'classification', 'running', '2026-09-23T10:00:00Z'
) on conflict (id) do nothing;

insert into public.trainings (
    id, run_id, round, candidate_name, model, requested_config,
    data_manifest, evaluation_config, status, primary_metric, higher_is_better,
    metrics, overfit_gap, duration_seconds, started_at, finished_at, warnings
)
select
    fixture.id, 'setup-demo-001', 1, fixture.name, fixture.model, fixture.config,
    '{"source_sha256":"77c298d794de2393c194530e166c369a5e3346dbdd273f95a5d3bd593060a569",
      "target":"churn", "usable_row_count":12, "excluded_missing_target":0,
      "included_columns":["tenure_months","monthly_spend"], "dropped_columns":[],
      "prepared_data_path":null, "synthetic_fixture":true}'::jsonb,
    '{"method":"stratified_kfold", "n_splits":3, "seed":42,
      "fold_sizes":[{"train":8,"validation":4},{"train":8,"validation":4},{"train":8,"validation":4}],
      "synthetic_fixture":true}'::jsonb,
    'succeeded', 'roc_auc', true,
    jsonb_build_object('roc_auc', jsonb_build_object(
        'mean', fixture.val_score, 'std', fixture.val_std, 'train_mean', fixture.train_score
    )),
    fixture.train_score - fixture.val_score, fixture.duration,
    '2026-09-23T10:00:00Z'::timestamptz,
    '2026-09-23T10:00:00Z'::timestamptz + fixture.duration * interval '1 second',
    '["Illustrative setup score; no real training or validation occurred."]'::jsonb
from (values
    ('20000000-0000-4000-8000-000000000001'::uuid,
     'logreg_demo', 'logreg', '{"params":{"C":1.0}}'::jsonb,
     0.81, 0.03, 0.85, 12),
    ('20000000-0000-4000-8000-000000000002'::uuid,
     'random_forest_demo', 'random_forest', '{"params":{"n_estimators":100,"max_depth":4}}'::jsonb,
     0.87, 0.02, 0.94, 15)
) as fixture(id, name, model, config, val_score, val_std, train_score, duration)
on conflict (id) do nothing;

update public.runs
set status = 'incomplete',
    selected_training_id = '20000000-0000-4000-8000-000000000002',
    finished_at = '2026-09-23T10:00:16Z',
    report_markdown = 'SETUP DEMO ONLY: two illustrative candidate results. No real training or final fit was performed.',
    error = 'Setup fixture only; no final model was fitted.'
where id = 'setup-demo-001';

insert into public.events (event_id, run_id, training_id, kind, ts, payload)
values
    ('30000000-0000-4000-8000-000000000001', 'setup-demo-001', null, 'run_start',
     extract(epoch from '2026-09-23T10:00:00Z'::timestamptz),
     '{"dataset_id":"10000000-0000-4000-8000-000000000001", "dataset":"SETUP DEMO - synthetic churn", "target":"churn", "rows":12, "features":2, "synthetic_fixture":true}'),
    ('30000000-0000-4000-8000-000000000002', 'setup-demo-001',
     '20000000-0000-4000-8000-000000000001', 'training_completed',
     extract(epoch from '2026-09-23T10:00:12Z'::timestamptz),
     '{"candidate_name":"logreg_demo", "synthetic_fixture":true}'),
    ('30000000-0000-4000-8000-000000000003', 'setup-demo-001',
     '20000000-0000-4000-8000-000000000002', 'training_completed',
     extract(epoch from '2026-09-23T10:00:15Z'::timestamptz),
     '{"candidate_name":"random_forest_demo", "synthetic_fixture":true}'),
    ('30000000-0000-4000-8000-000000000004', 'setup-demo-001', null, 'run_end',
     extract(epoch from '2026-09-23T10:00:16Z'::timestamptz),
     '{"status":"incomplete", "reason":"Setup fixture; no final model was fitted.", "synthetic_fixture":true}')
on conflict (event_id) do nothing;

commit;

