// Where events come from: Supabase realtime (with a REST backfill), polling the recorded jsonl
// when Supabase is unreachable or behind, and the replay player for recorded runs.

import { $, icon, fmtClock, getJSON } from './dom.js';
import { getSupabase } from './supa.js';

const POLL_MS = 3000;
const STATUS_MS = 5000;
const STALE_MS = 15000;

function normalize(row) {
  if (!row || typeof row !== 'object' || typeof row.kind !== 'string') return null;
  let { ts } = row;
  if (typeof ts === 'string') ts = /^\d+(\.\d+)?$/.test(ts) ? parseFloat(ts) : Date.parse(ts) / 1000;
  if (!Number.isFinite(ts)) return null;
  let { payload } = row;
  if (typeof payload === 'string') { try { payload = JSON.parse(payload); } catch { payload = {}; } }
  return { ts, kind: row.kind, payload: payload && typeof payload === 'object' ? payload : {}, run_id: row.run_id, id: row.event_id ?? row.id ?? null };
}

/** Dedupes by event_id / id and by (ts, kind), keeps ts order, and says when a late event forces a rebuild. */
export class EventStore {
  constructor() { this.list = []; this.keys = new Set(); }

  add(rows) {
    const added = [];
    let rebuild = false;
    for (const raw of rows || []) {
      const ev = normalize(raw);
      if (!ev) continue;
      const k = `${ev.ts.toFixed(6)}|${ev.kind}`;
      const idKey = ev.id != null ? `id:${ev.id}` : null;
      if (this.keys.has(k) || (idKey && this.keys.has(idKey))) continue;
      this.keys.add(k);
      if (idKey) this.keys.add(idKey);
      const last = this.list.at(-1);
      if (last && ev.ts < last.ts) rebuild = true;
      this.list.push(ev);
      added.push(ev);
    }
    if (rebuild) this.list.sort((a, b) => a.ts - b.ts);
    else added.sort((a, b) => a.ts - b.ts);
    return { added, rebuild };
  }
}

const STATUS_LABELS = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed' };

/** Follow a live run. Returns stop(). */
export function startLive(runId, view) {
  const store = new EventStore();
  let stopped = false, pollTimer = 0, statusTimer = 0, clock = 0, client = null, channel = null;
  let lastEventAt = Date.now(), active = true, firstBatch = true;
  view.mode = 'live';
  view.runId = runId;
  view.setStatus('queued', 'Connecting');
  view.setBusy(true);

  const ingest = (rows) => {
    const { added, rebuild } = store.add(rows);
    if (!added.length) return 0;
    lastEventAt = Date.now();
    if (rebuild || firstBatch) {
      view.reset();
      store.list.forEach((e) => view.apply(e, { animate: false }));
    } else added.forEach((e) => view.apply(e, { animate: true }));
    firstBatch = false;
    return added.length;
  };

  const pollOnce = async () => {
    try { return ingest(await getJSON(`/api/runs/${encodeURIComponent(runId)}/events`)); } catch { return 0; }
  };
  const startPolling = () => {
    if (pollTimer || stopped) return;
    const tick = async () => {
      await pollOnce();
      if (!stopped) pollTimer = setTimeout(tick, POLL_MS);
    };
    pollTimer = setTimeout(tick, 0);
  };

  const checkStatus = async () => {
    try {
      const s = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
      const state = STATUS_LABELS[s.status] ? s.status : 'running';
      view.setStatus(state, STATUS_LABELS[state]);
      active = state === 'queued' || state === 'running';
      if (state === 'queued') view.setActivity('Waiting for a worker to pick up the run');
      if (state === 'failed') view.setActivity(`The run failed: ${s.error || 'no reason given'}`);
      if (!active) {
        view.setBusy(false);
        await pollOnce(); // the recorded jsonl is the source of truth for the final events
        stop({ keepStatus: true });
        return;
      }
      // Supabase can be silent while the jsonl still grows (sync disabled on the backend): fall back to polling.
      if (!pollTimer && Date.now() - lastEventAt > STALE_MS && (await pollOnce()) > 0) startPolling();
    } catch (e) {
      if (e.status === 404) {
        view.setStatus('failed', 'Unknown run');
        view.setActivity('No run with this id exists on the server.');
        view.setBusy(false);
        stop({ keepStatus: true });
        return;
      }
    }
    if (!stopped) statusTimer = setTimeout(checkStatus, STATUS_MS);
  };

  const connect = async () => {
    client = await getSupabase();
    if (stopped) return;
    if (!client) { startPolling(); return; }
    try {
      // Subscribe before the backfill so nothing inserted in between is missed; the store dedupes the overlap.
      channel = client.channel(`events-${runId}`)
        .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'events', filter: `run_id=eq.${runId}` }, (msg) => ingest([msg.new]))
        .subscribe((status) => { if (status === 'CHANNEL_ERROR' || status === 'TIMED_OUT') startPolling(); });
      const { data, error } = await client.from('events').select('*').eq('run_id', runId).order('ts');
      if (error) throw error;
      ingest(data || []);
      if (!data?.length) await pollOnce();
    } catch {
      startPolling();
    }
  };

  function stop({ keepStatus } = {}) {
    stopped = true;
    clearTimeout(pollTimer);
    clearTimeout(statusTimer);
    clearInterval(clock);
    if (channel && client) client.removeChannel(channel);
    if (!keepStatus) view.setBusy(false);
  }

  clock = setInterval(() => { if (active) view.tickElapsed(Date.now() / 1000); }, 1000);
  connect();
  checkStatus();
  return stop;
}

/** Replay recorded events with their original relative timing at 1x, 4x or 16x, with a scrubber. */
export function startReplay(rows, view, { speed = 4, mode = 'replay', runId = null } = {}) {
  const store = new EventStore();
  store.add(rows);
  const events = store.list;
  view.mode = mode;
  view.runId = runId || events[0]?.run_id || null;
  const box = $('#replay');
  const btn = $('#replay-play');
  const scrub = $('#replay-scrub');
  const time = $('#replay-time');
  box.hidden = false;
  if (!events.length) {
    view.setStatus('failed', 'No events');
    view.setActivity('This recording has no events.');
    return () => {};
  }
  const t0 = events[0].ts;
  const duration = Math.max(0.1, events.at(-1).ts - t0);
  scrub.max = String(duration);
  let t = 0, idx = 0, playing = false, raf = 0, last = 0, rate = speed, resumeAfterScrub = false;
  for (const input of box.querySelectorAll('input[name="speed"]')) {
    input.checked = +input.value === rate;
    input.onchange = () => { rate = +input.value; if (playing) setButton(); };
  }

  const label = () => {
    time.textContent = `${fmtClock(t)} / ${fmtClock(duration)}`;
    scrub.value = String(t);
    scrub.setAttribute('aria-valuetext', `${fmtClock(t)} of ${fmtClock(duration)}`);
  };
  const setButton = () => {
    const ended = t >= duration;
    btn.replaceChildren(icon(playing ? 'pause' : ended ? 'replay' : 'play'));
    btn.setAttribute('aria-label', playing ? 'Pause' : ended ? 'Play again' : 'Play');
    view.setStatus(playing ? 'replaying' : ended ? 'done' : 'paused', playing ? `Replay ${rate}x` : ended ? 'Replay finished' : 'Replay paused');
    view.setBusy(playing);
  };
  const flush = (animate) => {
    while (idx < events.length && events[idx].ts - t0 <= t + 1e-9) view.apply(events[idx++], { animate });
    view.tickElapsed(t0 + t);
  };
  const tick = (now) => {
    const dt = Math.min(1, (now - last) / 1000); // a background tab resumes without a long jump
    last = now;
    t = Math.min(duration, t + dt * rate);
    flush(true);
    label();
    if (t >= duration) { playing = false; setButton(); return; }
    raf = requestAnimationFrame(tick);
  };
  const play = () => {
    if (t >= duration) seek(0);
    playing = true;
    last = performance.now();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(tick);
    setButton();
  };
  const pause = () => { playing = false; cancelAnimationFrame(raf); setButton(); };
  function seek(to) {
    t = Math.min(duration, Math.max(0, to));
    idx = 0;
    view.reset();
    flush(false);
    label();
  }

  btn.onclick = () => (playing ? pause() : play());
  scrub.oninput = () => {
    if (playing) { resumeAfterScrub = true; pause(); }
    seek(+scrub.value);
    setButton();
  };
  scrub.onchange = () => { if (resumeAfterScrub) { resumeAfterScrub = false; play(); } };

  seek(0);
  play();
  return () => { pause(); box.hidden = true; };
}
