// The live run view. Events arrive one at a time (live, polling or replay) and each kind updates
// its own section. reset() + apply(..., {animate: false}) rebuilds any point in time for the scrubber.

import { h, $, icon, fmtInt, fmtNum, fmtClock, tidy, richText, sevChip, prefersReducedMotion, gsapReady } from './dom.js';
import { Funnel, splitBar, tokenMeter } from './charts.js';
import { renderResult } from './result.js';

const code = (t) => h('code', { text: String(t ?? '') });

const TOOL_LABELS = {
  diag_summary: () => 'Reading the data checks',
  diag_run: () => 'Running the statistical checks',
  inspect_column: (i) => ['Inspecting column ', code(i.column)],
  run_search: () => 'Planning the search and racing pipelines',
  run_experiments: () => 'Running an experiment round',
  confirm_and_test: (i) => `Confirming the top ${i.k || 3} with paired folds, then one hidden test`,
  data_search: () => 'Searching for outside data',
  data_try: (i) => ['Trying outside data from ', i.source || i.url || 'a source'],
  finalize_model: (i) => ['Training the final model ', code(i.candidate_name || i.name)],
  write_report: () => 'Writing the report',
};

const CHECK_LABELS = {
  leakage: 'Leakage', id_like: 'ID-like column', missing_pattern: 'Missing values', mad_outliers: 'Outliers',
  imbalance: 'Class imbalance', duplicates: 'Duplicates', drift: 'Drift', constant: 'Constant column',
  high_cardinality: 'High cardinality', target_outliers: 'Target outliers', label_noise: 'Label noise',
  use_case_leakage: 'Use-case leakage', naive_baseline: 'Naive baseline',
};
const humanize = (s) => String(s || 'check').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

export class RunView {
  constructor({ onResultReady, onRunEnd } = {}) {
    this.onResultReady = onResultReady || (() => {});
    this.onRunEnd = onRunEnd || (() => {});
    this.el = {
      meta: $('#run-meta'), status: $('#run-status'), elapsed: $('#run-elapsed'),
      notes: $('#notes-log'), activity: $('#activity'), activityText: $('#activity-text'),
      diag: $('#diag'), diagSub: $('#diag-sub'), search: $('#funnel'), searchSub: $('#search-sub'),
      board: $('#board'), boardSub: $('#board-sub'), usage: $('#usage'), result: $('#result'),
    };
    this.funnel = new Funnel(this.el.search);
    this.mode = 'live';
    this.reset();
  }

  reset() {
    this.st = { start: null, t0: null, lastTs: null, diag: null, plan: null, rungs: [], board: null, confirm: null,
      confirmK: 3, externals: [], final: null, exp: null, report: null, usage: null, notes: 0, ended: false };
    this.resultAnnounced = false;
    this.el.meta.textContent = 'Waiting for the first event';
    this.el.elapsed.textContent = '';
    this.el.notes.replaceChildren(h('p', { class: 'notes-empty', text: 'The agent explains each step here before it calls a tool.' }));
    this.el.activityText.replaceChildren('Waiting for the agent');
    this.el.diag.replaceChildren(h('p', { class: 'empty', text: 'Findings appear after the first pass over the dev split.' }));
    this.el.diagSub.textContent = '';
    this.funnel.reset();
    this.el.search.append(h('p', { class: 'empty', text: 'The search plan appears when racing starts: pipelines generated, raced, and cut rung by rung.' }));
    this.el.searchSub.textContent = '';
    this.el.board.replaceChildren(h('p', { class: 'empty', text: 'Ranked pipelines appear after the race.' }));
    this.el.boardSub.textContent = '';
    this.el.usage.replaceChildren(h('p', { class: 'empty', text: 'Token counts arrive when the run ends.' }));
    this.el.result.replaceChildren(h('p', { class: 'empty', text: 'Results appear once the finalists are confirmed.' }));
    this.boardRows = new Map();
  }

  setStatus(state, label) {
    const icons = { queued: 'clock', running: 'clock', replaying: 'play', paused: 'pause', done: 'circleCheck', failed: 'x' };
    this.el.status.dataset.state = state;
    this.el.status.replaceChildren(icon(icons[state] || 'clock'), label);
  }

  setBusy(busy) { this.el.activity.classList.toggle('is-busy', !!busy); }

  tickElapsed(nowTs) {
    if (this.st.t0 == null) return;
    this.el.elapsed.textContent = `${fmtClock((nowTs ?? this.st.lastTs) - this.st.t0)} elapsed`;
  }

  apply(ev, { animate = true } = {}) {
    const p = ev.payload || {};
    if (this.st.t0 == null) this.st.t0 = ev.ts;
    this.st.lastTs = ev.ts;
    switch (ev.kind) {
      case 'run_start': this.st.start = p; this.st.t0 = ev.ts; this.renderMeta(); break;
      case 'thought': this.addNote(ev, animate); break;
      case 'tool_call': this.setActivity(TOOL_LABELS[p.tool] ? TOOL_LABELS[p.tool](typeof p.input === 'object' && p.input ? p.input : {}) : ['Calling ', code(p.tool)]);
        if (p.tool === 'confirm_and_test' && p.input?.k) this.st.confirmK = +p.input.k || 3;
        break;
      case 'status': this.setActivity(tidy(p.msg)); break;
      case 'diagnostics': this.st.diag = p; this.renderDiag(animate); break;
      case 'search_plan': this.st.plan = p; this.renderFunnel(animate ? ['raced'] : []); break;
      case 'rung': this.st.rungs = [...this.st.rungs.filter((r) => r.rung !== p.rung), p]; this.renderFunnel(animate ? [`rung-${p.rung}`] : []); break;
      case 'leaderboard': this.st.board = p; this.renderBoard(animate); break;
      case 'round_start': this.setActivity(`Experiment round ${p.round}: ${(p.candidates || []).length} candidates`); break;
      case 'confirm': this.st.confirm = p; this.renderFunnel(animate ? ['confirm', 'pick'] : []); this.renderResult(animate); break;
      case 'external_trial': this.st.externals = [...this.st.externals, p]; this.renderResult(false); break;
      case 'final_model': this.st.final = p; this.renderResult(false); break;
      case 'export': this.st.exp = p; this.renderResult(false); break;
      case 'report': this.st.report = p; this.renderResult(false); break;
      case 'usage': this.st.usage = p; this.st.ended = true; this.renderUsage(); this.renderResult(false);
        this.setActivity('Run finished'); this.setBusy(false); if (animate) this.onRunEnd(); break;
      default: break; // the contract allows new kinds; unknown ones are ignored
    }
    this.tickElapsed();
  }

  renderMeta() {
    const s = this.st.start || {};
    this.el.meta.replaceChildren(code(s.dataset || 'dataset'), `, predicting `, code(s.target || 'target'),
      ` from ${fmtInt(s.features)} features, ${fmtInt(s.rows)} rows`);
  }

  setActivity(content) { this.el.activityText.replaceChildren(...[content].flat()); }

  addNote(ev, animate) {
    const log = this.el.notes;
    if (!this.st.notes) log.replaceChildren();
    this.st.notes += 1;
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
    const note = h('div', { class: `note${animate ? ' is-new' : ''}` },
      h('time', { text: fmtClock(ev.ts - (this.st.t0 ?? ev.ts)) }),
      h('p', null, richText(ev.payload?.text)));
    log.append(note);
    if (nearBottom || !animate) log.scrollTo({ top: log.scrollHeight, behavior: animate && !prefersReducedMotion() ? 'smooth' : 'auto' });
  }

  renderDiag(animate) {
    const d = this.st.diag || {};
    const findings = [...(d.findings || [])].sort((a, b) => (b.severity || 0) - (a.severity || 0));
    this.el.diagSub.textContent = findings.length ? `${findings.length} findings on the dev split, most severe first` : 'No findings';
    const cards = findings.map((f) => h('article', { class: `finding${f.severity >= 3 ? ' sev-3-card' : ''}` },
      h('div', { class: 'finding-head' }, sevChip(f.severity >= 3 ? 3 : f.severity >= 2 ? 2 : 1),
        h('b', { text: CHECK_LABELS[f.check] || humanize(f.check) }), code(f.check)),
      h('p', null, richText(f.finding))));
    const grid = h('div', { class: 'findings' }, cards);
    this.el.diag.replaceChildren(findings.length ? grid : h('p', { class: 'empty', text: 'The checks found nothing to act on.' }));
    if (d.rows) this.el.diag.append(h('div', { class: 'rows-split' }, splitBar(d.rows)));
    if (animate && cards.length && gsapReady() && !prefersReducedMotion()) {
      window.gsap.from(cards, { opacity: 0, y: 10, duration: 0.45, ease: 'power2.out', stagger: 0.05, clearProps: 'opacity,transform' });
    }
  }

  funnelRows() {
    const { plan, rungs, confirm } = this.st;
    const rows = [];
    const space = +plan.space || 0, raced = +plan.raced || 0;
    rows.push({ key: 'space', label: 'Generated', detail: plan.warm_start ? `${fmtInt(plan.warm_start)} warm-started from past runs` : 'ranked by a benchmark prior', total: space, count: space });
    rows.push({ key: 'raced', label: 'Raced', detail: 'most promising by the prior', total: space, count: raced, drop: space > raced ? `${fmtInt(space - raced)} left out by the prior` : '' });
    const sched = [...(plan.schedule || [])];
    for (const r of rungs) if (!sched.some((x) => x.rung === r.rung)) sched.push({ rung: r.rung, n_rows: r.n_rows, keep: r.survivors });
    sched.sort((a, b) => a.rung - b.rung);
    let prev = raced;
    for (const plannedRung of sched) {
      const done = rungs.find((x) => x.rung === plannedRung.rung);
      const label = `Rung ${plannedRung.rung + 1}`;
      if (done) {
        const drops = [];
        if (done.dropped_stat) drops.push(`${fmtInt(done.dropped_stat)} cut by paired tests`);
        if (done.dropped_rank) drops.push(`${fmtInt(done.dropped_rank)} by rank`);
        rows.push({ key: `rung-${done.rung}`, label, total: done.evaluated, count: done.survivors,
          detail: `${fmtInt(done.n_rows)} rows each${done.leader ? `, leader ${done.leader} ${fmtNum(done.leader_mean, 3)}` : ''}`,
          drop: drops.length ? drops.join(', ') : '' });
        prev = done.survivors;
      } else {
        rows.push({ key: `rung-${plannedRung.rung}`, label, total: prev, count: plannedRung.keep, pending: true, detail: `${fmtInt(plannedRung.n_rows)} rows each, planned` });
        prev = plannedRung.keep;
      }
    }
    const k = confirm ? (confirm.table || []).length : this.st.confirmK;
    rows.push({ key: 'confirm', label: 'Confirmed', detail: confirm ? '10 paired folds, then one hidden test' : '10 paired folds, planned', total: prev, count: k, pending: !confirm });
    rows.push({ key: 'pick', label: 'Picked', detail: confirm ? confirm.recommendation?.pick || '' : 'by the 1-SE rule', total: k, count: 1, pending: !confirm, pick: !!confirm });
    return rows;
  }

  renderFunnel(animateKeys) {
    if (!this.st.plan) return;
    this.el.search.querySelector('.empty')?.remove();
    const plan = this.st.plan;
    const cv = plan.cv ? `, ${String(plan.cv).replace(/_/g, ' ')} CV${plan.time_column ? ` on ${plan.time_column}` : ''}` : '';
    this.el.searchSub.textContent = `n* = ${fmtInt(plan.n_star)} rows at most per pipeline${cv}`;
    this.funnel.update(this.funnelRows(), +plan.space || 1, new Set(animateKeys));
  }

  renderBoard(animate) {
    const b = this.st.board;
    const metric = b.primary_metric || 'score';
    this.el.boardSub.textContent = `${b.round === 'race' ? 'after the race' : `round ${b.round}`}, ${metric}`;
    let table = this.el.board.querySelector('table');
    if (!table) {
      table = h('table', { class: 'lb' },
        h('thead', null, h('tr', null, h('th', { scope: 'col', text: '#' }), h('th', { scope: 'col', text: 'Pipeline' }),
          h('th', { scope: 'col', text: 'Family' }), h('th', { scope: 'col', class: 'num', text: `${metric}, mean ± sd` }))),
        h('tbody'));
      this.el.board.replaceChildren(table);
    }
    const tbody = table.tBodies[0];
    const before = new Map([...this.boardRows].map(([name, tr]) => [name, tr.getBoundingClientRect().top]));
    const next = new Map();
    (b.rows || []).forEach((r, i) => {
      const name = String(r.name ?? `row ${i + 1}`);
      const tr = this.boardRows.get(name) || h('tr');
      const mean = r.mean ?? r[metric] ?? r.cv_mean;
      tr.className = i === 0 ? 'is-top' : '';
      tr.replaceChildren(h('td', { class: 'mono', text: String(i + 1) }), h('td', { class: 'mono', text: name }),
        h('td', { text: r.family || '' }), h('td', { class: 'num', text: `${fmtNum(mean, 4)} ± ${fmtNum(r.std, 4)}` }));
      next.set(name, tr);
      tbody.append(tr);
    });
    for (const [name, tr] of this.boardRows) if (!next.has(name)) tr.remove();
    this.boardRows = next;
    if (!animate || !gsapReady() || prefersReducedMotion()) return;
    // Layout animation (FLIP): rows that changed rank slide from their old position; new rows fade in.
    for (const [name, tr] of next) {
      const old = before.get(name);
      if (old == null) window.gsap.from(tr, { opacity: 0, duration: 0.35, ease: 'power2.out', clearProps: 'opacity' });
      else {
        const dy = old - tr.getBoundingClientRect().top;
        if (Math.abs(dy) > 1) window.gsap.from(tr, { y: dy, duration: 0.45, ease: 'power2.inOut', clearProps: 'transform' });
      }
    }
  }

  renderUsage() {
    this.el.usage.replaceChildren(tokenMeter(this.st.usage || {}));
  }

  renderResult(animate) {
    if (!this.st.confirm) return;
    renderResult(this.el.result, this.st, { animate: animate && !this.resultAnnounced, mode: this.mode, runId: this.runId });
    if (!this.resultAnnounced) {
      this.resultAnnounced = true;
      this.onResultReady();
    }
  }
}
