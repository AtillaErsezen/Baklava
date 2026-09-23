// The app flow: 1 upload, 2 check the data, 3 describe the goal, 4 run, 5 results.
// Server errors map to inline messages next to the field they concern.

import { h, $, $$, icon, fmtInt, fmtPct, fmtBytes, getJSON, validId, prefersReducedMotion, gsapReady } from './dom.js';

export const STEPS = ['upload', 'check', 'goal', 'run', 'result'];
const MAX_BYTES = 50 * 1024 * 1024;
const CODE_KEY = 'mlf-access';

const TYPE_NAMES = [
  [/^(int|uint)/, 'integer'], [/^float/, 'decimal'], [/^bool/, 'true / false'], [/^datetime|^date/, 'date'],
  [/^(object|str|string|category)/, 'text'],
];
const typeName = (dtype) => (TYPE_NAMES.find(([re]) => re.test(String(dtype))) || [null, String(dtype)])[1];

function setError(id, message, input) {
  const el = $(`#${id}`);
  if (el) el.replaceChildren(...(message ? [message] : []));
  if (input) input.setAttribute('aria-invalid', message ? 'true' : 'false');
}

function formError(id, message) {
  const el = $(`#${id}`);
  el.replaceChildren(...(message ? [icon('octagon'), h('span', { text: message })] : []));
}

const capitalize = (s) => String(s || '').replace(/^./, (c) => c.toUpperCase());

export class Flow {
  constructor({ fixture, onStartRun }) {
    this.fixture = fixture;
    this.onStartRun = onStartRun;
    this.current = 'upload';
    this.done = new Set();
    this.file = null;
    this.dataset = null; // {id, fileName, profile}
    this.code = sessionStorage.getItem(CODE_KEY) || '';
    this.bindStepper();
    this.bindUpload();
    this.bindCheck();
    this.bindGoal();
    $('#sample-btn').hidden = !fixture;
    if (this.code) $('#access').value = this.code;
  }

  // ---------- stepper ----------
  bindStepper() {
    for (const btn of $$('#stepper button')) btn.addEventListener('click', () => this.show(btn.dataset.step));
  }

  enable(step, on = true) { $(`#stepper [data-step="${step}"]`).disabled = !on; }

  lockInputs() { ['upload', 'check', 'goal'].forEach((s) => this.enable(s, false)); }

  /** Direction-aware transition: forward slides in from the right, back from the left. */
  show(step, { focus = true } = {}) {
    const from = STEPS.indexOf(this.current);
    const to = STEPS.indexOf(step);
    this.current = step;
    for (const btn of $$('#stepper button')) {
      const i = STEPS.indexOf(btn.dataset.step);
      if (btn.dataset.step === step) btn.setAttribute('aria-current', 'step');
      else btn.removeAttribute('aria-current');
      btn.classList.toggle('is-done', this.done.has(btn.dataset.step) && i !== to);
    }
    let panel = null;
    for (const p of $$('[data-panel]')) {
      p.hidden = p.dataset.panel !== step;
      if (!p.hidden) panel = p;
    }
    if (panel && from !== to && gsapReady() && !prefersReducedMotion()) {
      window.gsap.fromTo(panel, { opacity: 0, x: 16 * Math.sign(to - from) }, { opacity: 1, x: 0, duration: 0.4, ease: 'expo.out', clearProps: 'opacity,transform' });
    }
    if (focus && panel) {
      const heading = panel.querySelector('.panel-title');
      heading?.focus({ preventScroll: true });
      const top = $('#stepper').getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight * 0.6) $('#stepper').scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    }
    window.ScrollTrigger?.refresh();
  }

  complete(step) { this.done.add(step); }

  // ---------- 1. upload ----------
  bindUpload() {
    const drop = $('#drop');
    const input = $('#file');
    $('.drop-icon', drop).append(icon('upload'));
    input.addEventListener('change', () => this.setFile(input.files[0]));
    for (const type of ['dragenter', 'dragover']) drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add('is-over'); });
    for (const type of ['dragleave', 'dragend']) drop.addEventListener(type, () => drop.classList.remove('is-over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('is-over');
      this.setFile(e.dataTransfer?.files?.[0]);
    });
    $('#upload-form').addEventListener('submit', (e) => { e.preventDefault(); this.upload(); });
    $('#sample-btn').addEventListener('click', () => this.loadSample());
  }

  setFile(file) {
    const drop = $('#drop');
    const chip = $('#file-chip');
    setError('file-error', '');
    drop.classList.remove('is-invalid');
    this.file = null;
    chip.hidden = true;
    if (!file) return;
    let problem = '';
    if (!/\.csv$/i.test(file.name)) problem = 'Choose a .csv file. Save spreadsheets as CSV first.';
    else if (file.size === 0) problem = 'This file is empty.';
    else if (file.size > MAX_BYTES) problem = `This file is ${fmtBytes(file.size)}. The limit is 50 MB, so upload a sample of the rows.`;
    if (problem) {
      setError('file-error', problem);
      drop.classList.add('is-invalid');
      return;
    }
    this.file = file;
    const remove = h('button', { class: 'btn btn-secondary btn-sm', type: 'button', 'aria-label': `Remove ${file.name}` }, icon('x'), 'Remove');
    remove.addEventListener('click', () => { $('#file').value = ''; this.setFile(null); $('#file').focus(); });
    chip.replaceChildren(icon('file'), h('b', { text: file.name }), h('span', { text: fmtBytes(file.size) }), remove);
    chip.hidden = false;
    if (gsapReady() && !prefersReducedMotion()) window.gsap.from(chip, { opacity: 0, scale: 0.98, duration: 0.3, ease: 'expo.out', clearProps: 'opacity,transform' });
  }

  upload() {
    const access = $('#access');
    formError('upload-error', '');
    setError('access-error', '', access);
    let ok = true;
    if (!this.file) { setError('file-error', 'Choose a CSV file first.'); ok = false; }
    if (!access.value.trim()) { setError('access-error', 'Enter the access code to upload.', access); ok = false; }
    if (!ok) { (this.file ? access : $('#file')).focus(); return; }
    this.code = access.value.trim();
    const form = new FormData();
    form.append('file', this.file, this.file.name);
    form.append('access_code', this.code);
    const btn = $('#upload-btn');
    const progress = $('#upload-progress');
    const bar = $('.progress-bar', progress);
    btn.disabled = true;
    btn.textContent = 'Uploading';
    progress.hidden = false;
    bar.style.transform = 'scaleX(0)';
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/datasets');
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.style.transform = `scaleX(${(e.loaded / e.total).toFixed(3)})`; };
    xhr.upload.onload = () => { btn.textContent = 'Profiling the columns'; };
    const finish = () => { btn.disabled = false; btn.textContent = 'Upload and profile'; progress.hidden = true; };
    xhr.onerror = () => { finish(); formError('upload-error', 'Could not reach the server. Check the connection and try again.'); };
    xhr.onload = () => {
      finish();
      const body = xhr.response || {};
      if (xhr.status >= 200 && xhr.status < 300 && body.dataset_id && body.profile) {
        sessionStorage.setItem(CODE_KEY, this.code);
        this.openDataset({ id: body.dataset_id, fileName: this.file.name, profile: body.profile });
        return;
      }
      const msg = body.error || '';
      if (xhr.status === 401) { setError('access-error', 'That access code was not accepted.', access); access.focus(); }
      else if (xhr.status === 413) setError('file-error', 'The file is over the 50 MB upload limit.');
      else if (xhr.status === 400) setError('file-error', `The server could not use this file: ${msg || 'it did not parse as CSV'}.`);
      else if (xhr.status === 429) formError('upload-error', `${capitalize(msg || 'too many uploads')}. Try again later.`);
      else if (xhr.status === 503) formError('upload-error', 'The server is not taking uploads right now. Try again in a minute.');
      else formError('upload-error', `The upload failed (${xhr.status || 'no response'}). Try again.`);
    };
    xhr.send(form);
  }

  async loadSample() {
    formError('upload-error', '');
    try {
      const body = await getJSON('fixtures/sample_profile.json');
      this.openDataset({ id: body.dataset_id, fileName: body.file_name, profile: body.profile, sample: true });
    } catch {
      formError('upload-error', 'Could not load the sample dataset.');
    }
  }

  /** ?dataset=<id> on reload: fetch the profile again with the code this tab already used. */
  async reopen(id) {
    if (!validId(id)) return;
    if (!this.code) {
      formError('upload-error', 'Enter the access code and upload the file again to continue.');
      return;
    }
    try {
      const body = await getJSON(`/api/datasets/${encodeURIComponent(id)}`, { headers: { Accept: 'application/json', 'X-Access-Code': this.code } });
      this.openDataset({ id, fileName: 'your dataset', profile: body.profile }, { push: false });
    } catch (e) {
      formError('upload-error', e.status === 404 ? 'That dataset is no longer on the server. Upload it again.' : 'Could not reopen the dataset. Upload it again.');
    }
  }

  openDataset(ds, { push = true } = {}) {
    this.dataset = ds;
    if (push && !ds.sample) history.pushState(null, '', `?dataset=${encodeURIComponent(ds.id)}#app`);
    this.renderCheck();
    this.complete('upload');
    this.enable('check');
    this.enable('goal');
    this.show('check');
  }

  // ---------- 2. check the data ----------
  bindCheck() {
    $('#target').addEventListener('change', () => this.renderTarget());
    $('#check-next').addEventListener('click', () => { this.complete('check'); this.renderGoalSummary(); this.show('goal'); });
    $('#check-back').addEventListener('click', () => this.show('upload'));
  }

  renderCheck() {
    const { profile, fileName } = this.dataset;
    const cols = profile.columns || [];
    $('#check-summary').replaceChildren(h('b', { text: fileName }), `: ${fmtInt(profile.rows)} rows, ${fmtInt(cols.length)} columns. Suggested target `,
      h('code', { text: profile.suggested_target || '' }), profile.suggested_task ? ` for ${profile.suggested_task}.` : '.');
    const select = $('#target');
    select.replaceChildren(...cols.map((c) => h('option', { value: c.name, text: c.name })));
    select.value = profile.suggested_target || cols.at(-1)?.name || '';
    const tbody = $('#columns-table tbody');
    tbody.replaceChildren(...cols.map((c) => h('tr', { dataset: { col: c.name } },
      h('td', null, h('div', { class: 'col-name' }, h('code', { text: c.name }))),
      h('td', { text: typeName(c.dtype) }),
      h('td', { class: 'num' }, h('span', { class: 'miss' }, c.missing_pct > 0 ? h('span', { class: 'miss-bar', 'aria-hidden': 'true' }, h('i', { vars: { width: `${Math.min(100, Math.max(4, c.missing_pct))}%` } })) : null, fmtPct(c.missing_pct))),
      h('td', { class: 'num', text: fmtInt(c.n_unique) }),
      h('td', { class: c.example == null ? 'cell-null' : 'mono', text: c.example == null ? 'empty' : String(c.example) }))));
    const names = cols.map((c) => c.name);
    const preview = profile.preview || [];
    $('#preview-title').textContent = `Preview, first ${fmtInt(preview.length)} of ${fmtInt(profile.rows)} rows`;
    $('#preview-table thead').replaceChildren(h('tr', null, names.map((n) => h('th', { scope: 'col', text: n }))));
    $('#preview-table tbody').replaceChildren(...preview.map((row) => h('tr', null, names.map((n, i) => {
      const v = Array.isArray(row) ? row[i] : row?.[n];
      return v == null ? h('td', { class: 'cell-null', text: 'empty' }) : h('td', { text: String(v), title: String(v) });
    }))));
    this.renderTarget();
  }

  renderTarget() {
    const name = $('#target').value;
    const { profile } = this.dataset;
    const col = (profile.columns || []).find((c) => c.name === name) || {};
    const isNumeric = /^(int|uint|float)/.test(String(col.dtype));
    const task = name === profile.suggested_target && profile.suggested_task
      ? profile.suggested_task : (!isNumeric || col.n_unique <= 20 ? 'classification' : 'regression');
    const parts = [task === 'classification' ? `Classification with ${fmtInt(col.n_unique)} classes.` : 'Regression on a numeric column.'];
    if (col.missing_pct > 0) parts.push(`${fmtPct(col.missing_pct)} of rows have no value here.`);
    if (col.n_unique === profile.rows) parts.push('Every value is unique, which looks like an ID rather than an outcome.');
    $('#target-hint').textContent = parts.join(' ');
    for (const tr of $$('#columns-table tbody tr')) {
      const on = tr.dataset.col === name;
      tr.classList.toggle('is-target', on);
      const cell = tr.querySelector('.col-name');
      cell.querySelector('.tag')?.remove();
      if (on) cell.append(h('span', { class: 'tag tag-accent', text: 'target' }));
    }
    this.task = task;
  }

  // ---------- 3. goal ----------
  bindGoal() {
    const purpose = $('#purpose');
    for (const chip of $$('.chip')) {
      chip.addEventListener('click', () => {
        purpose.value = chip.textContent;
        setError('purpose-error', '', purpose);
        purpose.focus();
        purpose.setSelectionRange(purpose.value.length, purpose.value.length);
      });
    }
    $('#goal-form').addEventListener('submit', (e) => { e.preventDefault(); this.start(); });
    $('#goal-back').addEventListener('click', () => this.show('check'));
  }

  renderGoalSummary() {
    const { profile, fileName } = this.dataset;
    $('#goal-summary').replaceChildren('Predicting ', h('code', { text: $('#target').value }), ` (${this.task}) from ${fmtInt((profile.columns || []).length - 1)} columns of ${fileName}, ${fmtInt(profile.rows)} rows.`);
    $('#goal-access-field').hidden = !!this.code || !!this.dataset.sample;
  }

  async start() {
    const purpose = $('#purpose');
    const access = $('#goal-access');
    formError('goal-error', '');
    setError('purpose-error', '', purpose);
    const text = purpose.value.trim();
    if (text.length < 8) {
      setError('purpose-error', 'Describe the goal in a sentence: what to predict, and what a mistake costs.', purpose);
      purpose.focus();
      return;
    }
    if (!$('#goal-access-field').hidden) {
      if (!access.value.trim()) { setError('goal-access-error', 'Enter the access code to start the run.', access); access.focus(); return; }
      this.code = access.value.trim();
    }
    if (this.dataset.sample) { this.complete('goal'); this.onStartRun('fixture'); return; }
    const btn = $('#start-btn');
    btn.disabled = true;
    btn.textContent = 'Starting';
    try {
      const body = await getJSON('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ dataset_id: this.dataset.id, target: $('#target').value, purpose: text, access_code: this.code }),
      });
      if (!validId(body?.run_id)) throw Object.assign(new Error('bad run id'), { status: 500 });
      sessionStorage.setItem(CODE_KEY, this.code);
      this.complete('goal');
      this.onStartRun(body.run_id);
    } catch (e) {
      const msg = e.message || '';
      if (e.status === 401) {
        $('#goal-access-field').hidden = false;
        setError('goal-access-error', 'That access code was not accepted.', access);
        access.focus();
      } else if (e.status === 404) formError('goal-error', 'This dataset is no longer on the server. Go back to step 1 and upload it again.');
      else if (e.status === 400) formError('goal-error', `The server refused the run: ${msg}.`);
      else if (e.status === 429) formError('goal-error', `${capitalize(msg)}.`);
      else if (e.status === 503) formError('goal-error', 'Could not start the run. Try again in a minute.');
      else formError('goal-error', 'Could not reach the server. Check the connection and try again.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Start the run';
    }
  }
}
