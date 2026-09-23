// "Recent runs" on the start screen, read from the Supabase `runs` table with the publishable key.
// Loads when the app section comes near the viewport; stays hidden if Supabase is unreachable.

import { h, $, icon, validId } from './dom.js';
import { getSupabase } from './supa.js';

const LIMIT = 8;
const STATES = {
  running: ['Running', 'clock', 'tag-accent'],
  completed: ['Done', 'circleCheck', 'tag-good'],
  incomplete: ['Incomplete', 'triangle', ''],
};
const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

function ago(iso) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return '';
  const s = (t - Date.now()) / 1000;
  const steps = [[60, 'second'], [3600, 'minute'], [86400, 'hour'], [604800, 'day'], [Infinity, 'week']];
  const [, unit] = steps.find(([lim]) => Math.abs(s) < lim);
  const div = { second: 1, minute: 60, hour: 3600, day: 86400, week: 604800 }[unit];
  return rtf.format(Math.round(s / div), unit);
}

async function fetchRuns(client) {
  const cols = 'id, run_id, dataset_name, target, status, recommended, created_at';
  for (const [select, order] of [[cols, 'created_at'], ['*', 'created_at'], ['*', null]]) {
    let q = client.from('runs').select(select).limit(LIMIT);
    if (order) q = q.order(order, { ascending: false });
    const { data, error } = await q;
    if (!error) return data || [];
  }
  return null;
}

function row(r) {
  const id = String(r.run_id ?? r.id ?? '');
  if (!validId(id)) return null;
  const [label, ic, cls] = STATES[r.status] || [String(r.status || 'Unknown'), 'info', ''];
  const href = r.status === 'running' ? `?run=${encodeURIComponent(id)}` : `?replay=${encodeURIComponent(id)}`;
  return h('li', null, h('a', { class: 'recent-item', href },
    h('span', { class: 'recent-main' }, h('code', { text: r.dataset_name || 'dataset' }), h('span', { class: 'recent-target', text: r.target ? `predicting ${r.target}` : '' })),
    h('span', { class: 'recent-side' },
      h('span', { class: `tag ${cls}` }, icon(ic), label),
      r.recommended ? h('code', { class: 'recent-pick', text: typeof r.recommended === 'object' ? r.recommended.name || JSON.stringify(r.recommended) : String(r.recommended) }) : null,
      h('time', { datetime: r.created_at || '', text: ago(r.created_at) }))));
}

export function initRecentRuns() {
  const box = $('#recent');
  const target = $('#app');
  if (!box || !target) return;
  let started = false;
  const load = async () => {
    if (started) return;
    started = true;
    const client = await getSupabase();
    if (!client) return;
    let runs = null;
    try { runs = await fetchRuns(client); } catch { runs = null; }
    if (!runs) return;
    const items = runs.map(row).filter(Boolean);
    $('#recent-list').replaceChildren(...(items.length ? items : [h('li', { class: 'empty', text: 'No runs yet. Yours will be the first.' })]));
    box.hidden = false;
  };
  new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); load(); }
  }, { rootMargin: '600px 0px' }).observe(target);
}
