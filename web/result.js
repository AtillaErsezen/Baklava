// Results: recommendation, confirm table with CI whiskers, Pareto scatter, n*, outside data,
// the model bundle and the agent's report. Rebuilt from state whenever a result event lands.

import { h, icon, fmtInt, fmtNum, fmtSigned, fmtP, fmtParam, tidy, richText, prefersReducedMotion, gsapReady } from './dom.js';
import { ciDomain, ciSvg, ciAxis, revealCi, scatter, featureBars } from './charts.js';
import { renderMarkdown, sectionText } from './markdown.js';

const code = (t) => h('code', { text: String(t ?? '') });
const list = (names) => names.map((n, i) => [i ? (i === names.length - 1 ? ' and ' : ', ') : '', code(n)]);
const basename = (p) => String(p || '').split(/[\\/]/).pop();

/** POSIX shell quoting for the copy-paste command: plain words stay bare, anything else is single-quoted. */
export const shq = (s) => (/^[\w@%+=:,./-]+$/.test(String(s)) ? String(s) : `'${String(s).replace(/'/g, `'\\''`)}'`);

function whyText(c, pick, best) {
  const rec = c.recommendation || {};
  const within = rec.within_1se || [];
  const thr = fmtNum(rec.threshold, 4);
  const out = [];
  if (pick.name === best.name) {
    out.push(h('p', null, code(pick.name), ` has the highest cross-validated ${c.primary_metric}. `,
      within.length > 1 ? ['The 1-SE rule looks at every model scoring at or above ', thr, ' (one standard error below the best): ', list(within), '. It keeps the cheapest of them, which here is the best model itself.']
        : `No other model is within one standard error of it (threshold ${thr}).`));
  } else {
    out.push(h('p', null, code(best.name), ` has the highest mean (${fmtNum(best.cv_mean, 4)}). `, code(pick.name),
      ` scores ${fmtNum(pick.cv_mean, 4)}, inside one standard error of it (threshold ${thr}), and costs less to train and serve, so the 1-SE rule prefers it.`));
  }
  const tie = (c.tie_groups || []).find((g) => g.includes(pick.name) && g.length > 1);
  if (tie) out.push(h('p', null, 'Paired tests find no real difference between ', list(tie), '. Any of them is a defensible choice; the pick is the one that costs least.'));
  if (pick.pareto) out.push(h('p', null, 'It sits on the Pareto front: no other finalist is both more accurate and faster to fit.'));
  return out;
}

function recCard(st, pick, best) {
  const c = st.confirm;
  const hiddenRows = st.diag?.rows?.hidden_locked;
  const half = Array.isArray(pick.ci) ? (pick.ci[1] - pick.ci[0]) / 2 : null;
  const params = st.final?.spec?.params || pick.params || {};
  return h('article', { class: 'rec', 'aria-labelledby': 'rec-title' },
    h('div', { class: 'rec-head' }, h('span', { class: 'tag tag-accent', text: 'Recommended' }),
      pick.family ? h('span', { class: 'tag', text: pick.family }) : null,
      pick.pareto ? h('span', { class: 'tag tag-good' }, icon('check'), 'Pareto front') : null),
    h('h4', { class: 'rec-name', id: 'rec-title', text: pick.name }),
    h('div', { class: 'rec-scores' },
      h('div', { class: 'score' }, h('b', { text: `${fmtNum(pick.cv_mean, 3)} ± ${fmtNum(half, 3)}` }),
        h('span', { text: `${c.primary_metric}, 95% CI ${fmtNum(pick.ci?.[0], 3)} to ${fmtNum(pick.ci?.[1], 3)} over 10 paired folds` })),
      h('div', { class: 'score' }, h('b', { text: fmtNum(pick.hidden, 3) }),
        h('span', { text: `hidden test on ${hiddenRows ? `${fmtInt(hiddenRows)} locked rows` : 'the locked rows'}, scored once; gap to CV ${fmtSigned(pick.dev_to_hidden_gap, 3)}` }))),
    h('div', { class: 'rec-why' }, whyText(c, pick, best)),
    Object.keys(params).length ? [h('h5', { class: 'sub-title', text: 'Parameters' }),
      h('dl', { class: 'params' }, Object.entries(params).map(([k, v]) => h('div', null, h('dt', { text: k }), h('dd', { text: fmtParam(v) }))))] : null);
}

function takeAway(st, { mode, runId }) {
  const exp = st.exp;
  const script = basename(exp?.script) || (st.final ? `train_${st.final.name}.py` : '');
  const data = st.start?.dataset || 'data.csv';
  const target = st.start?.target || st.final?.spec?.target;
  const cmd = script ? `python ${shq(script)} ${shq(data)}${target ? ` --target ${shq(target)}` : ''} --out model.joblib` : '';
  const liveId = mode === 'fixture' ? null : runId;
  const copyBtn = h('button', { type: 'button', 'aria-label': 'Copy the retrain command' }, icon('copy'), h('span', { text: 'Copy' }));
  const copied = h('span', { class: 'visually-hidden', role: 'status' });
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      copyBtn.lastChild.textContent = 'Copied';
      copied.textContent = 'Command copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(copyBtn.previousSibling);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      copyBtn.lastChild.textContent = 'Press Ctrl+C';
    }
    setTimeout(() => { copyBtn.lastChild.textContent = 'Copy'; copied.textContent = ''; }, 2000);
  });
  const download = exp && liveId
    ? downloadButton(liveId)
    : h('button', { class: 'btn btn-primary', type: 'button', disabled: true }, icon('download'), exp ? 'Download works on live runs' : 'Bundle is being prepared');
  return h('section', { class: 'block take', 'aria-labelledby': 'take-title' },
    h('h4', { class: 'block-title', id: 'take-title', text: 'Your model' }),
    download,
    exp ? h('ul', { class: 'files', 'aria-label': 'Files in the bundle' },
      [exp.script, exp.params, exp.card].filter(Boolean).map((f) => h('li', null, icon('file'), code(basename(f))))) : null,
    cmd ? [h('p', { class: 'block-sub', text: 'Retrain it on new data with the bundled script:' }),
      h('div', { class: 'cmd' }, h('code', { text: cmd }), copyBtn), copied] : null,
    st.final ? [h('h5', { class: 'sub-title', text: 'What drives it' }), featureBars(st.final.top_features)] : null);
}

// The bundle holds the user's model and retrain script, so the API asks for the access code (header, not URL):
// fetch it with the code kept in this tab's sessionStorage, then hand the browser a blob to save.
function downloadButton(runId) {
  const status = h('span', { class: 'visually-hidden', role: 'status' });
  const btn = h('button', { class: 'btn btn-primary', type: 'button' }, icon('download'), 'Download the model bundle');
  btn.addEventListener('click', async () => {
    let code = '';
    try { code = sessionStorage.getItem('mlf-access') || ''; } catch { /* storage blocked */ }
    btn.disabled = true;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/export.zip`, { headers: { 'X-Access-Code': code } });
      if (!res.ok) {
        status.textContent = res.status === 401 ? 'Enter the access code on the start screen first' : 'Download failed, try again';
        btn.lastChild.textContent = status.textContent;
        return;
      }
      const url = URL.createObjectURL(await res.blob());
      const a = h('a', { href: url, download: `${runId}_export.zip` });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      status.textContent = 'Download started';
    } catch {
      status.textContent = 'Download failed, try again';
    } finally {
      btn.disabled = false;
    }
  });
  return h('span', { class: 'download' }, btn, status);
}

function confirmTable(st, pickName, bestName) {
  const c = st.confirm;
  const rows = c.table || [];
  const thr = c.recommendation?.threshold;
  const domain = ciDomain(rows, thr);
  const metric = c.primary_metric;
  return h('section', { class: 'block confirm-wrap', 'aria-labelledby': 'confirm-title' },
    h('div', { class: 'block-head' }, h('h4', { class: 'block-title', id: 'confirm-title', text: 'Finalists, 10 paired folds' }),
      h('p', { class: 'block-sub', text: `Line: 95% CI of ${metric}. Tick: 1-SE threshold ${fmtNum(thr, 4)}.` })),
    h('div', { class: 'table-wrap', 'data-lenis-prevent': '' },
      h('table', { class: 'data-table confirm-table' },
        h('thead', null, h('tr', null,
          h('th', { scope: 'col', text: 'Model' }),
          h('th', { scope: 'col' }, h('span', { text: `CV ${metric}` }), ciAxis(domain, thr)),
          h('th', { scope: 'col', class: 'num', text: 'p vs best' }),
          h('th', { scope: 'col', class: 'num', text: 'Hidden' }),
          h('th', { scope: 'col', class: 'num', text: 'Fit' }),
          h('th', { scope: 'col', class: 'num', text: 'Predict' }),
          h('th', { scope: 'col', text: 'Big-O, train / predict' }),
          h('th', { scope: 'col', text: 'Pareto' }))),
        h('tbody', null, rows.map((r) => {
          const isPick = r.name === pickName;
          const tags = [isPick ? h('span', { class: 'tag tag-accent', text: 'pick' }) : null,
            r.name === bestName && !isPick ? h('span', { class: 'tag', text: 'best mean' }) : null,
            r.tie_group === 0 && r.name !== bestName ? h('span', { class: 'tag', text: 'tied with best' }) : null];
          return h('tr', { class: isPick ? 'is-pick' : '' },
            h('td', null, h('div', { class: 'col-name' }, code(r.name), tags), h('small', { class: 'block-sub', text: r.family || '' })),
            h('td', { class: 'ci-cell' }, h('div', { class: 'col-name' },
              ciSvg({ low: r.ci?.[0], high: r.ci?.[1], mean: r.cv_mean, domain, threshold: thr,
                label: `${r.name}: ${fmtNum(r.cv_mean, 4)}, 95% CI ${fmtNum(r.ci?.[0], 4)} to ${fmtNum(r.ci?.[1], 4)}` }),
              h('span', { class: 'mono', text: fmtNum(r.cv_mean, 4) }))),
            h('td', { class: 'num', text: r.p_vs_best == null ? 'best' : fmtP(r.p_vs_best) }),
            h('td', { class: 'num', text: fmtNum(r.hidden, 4) }),
            h('td', { class: 'num', text: r.fit_s != null ? `${fmtNum(r.fit_s, 1)} s` : '-' }),
            h('td', { class: 'num', text: r.predict_ms != null ? `${fmtNum(r.predict_ms, 1)} ms` : '-' }),
            h('td', null, h('small', { class: 'mono', text: `${r.big_o?.train || '-'} / ${r.big_o?.predict || '-'}` })),
            h('td', null, r.pareto ? h('span', { class: 'tag tag-good' }, icon('check'), 'yes') : h('span', { class: 'block-sub', text: 'no' })));
        })))),
    c.notes ? h('p', { class: 'block-sub', text: tidy(c.notes) }) : null,
    c.ensemble ? ensembleNote(c.ensemble, metric) : null);
}

/** The harness tries a weighted ensemble of the finalists and keeps it only if it wins on the hidden set. */
function ensembleNote(e, metric) {
  const weights = Array.isArray(e.weights)
    ? e.weights.map((w, i) => (typeof w === 'object' && w ? `${w.name ?? `model ${i + 1}`} ${fmtNum(w.weight ?? w.w, 2)}` : fmtNum(w, 2)))
    : Object.entries(e.weights || {}).map(([k, w]) => `${k} ${fmtNum(w, 2)}`);
  const facts = [
    e.ensemble_score != null ? `ensemble ${fmtNum(e.ensemble_score, 4)} against ${fmtNum(e.best_single_score, 4)} for the best single model on the hidden set` : null,
    e.delta != null ? `change ${fmtSigned(e.delta, 4)}` : null,
    e.p != null ? `p ${fmtP(e.p)}` : null,
    e.val_score != null ? `validation ${metric} ${fmtNum(e.val_score, 4)}` : null,
  ].filter(Boolean);
  return h('div', { class: 'ensemble' },
    h('span', { class: `tag ${e.keep ? 'tag-good' : ''}` }, icon(e.keep ? 'check' : 'flat'), e.keep ? 'Ensemble tested, kept' : 'Ensemble tested, not kept'),
    h('p', null, facts.length ? `${capitalize(facts.join(', '))}.` : '',
      weights.length ? ` Weights: ${weights.join(', ')}.` : '',
      e.keep ? '' : ' The single model stays, since the ensemble has to win on the hidden set to replace it.'));
}

const capitalize = (s) => String(s).replace(/^./, (ch) => ch.toUpperCase());

function paretoBlock(st, pickName) {
  const rows = st.confirm.table || [];
  const box = h('div');
  scatter(box, rows.map((r) => ({ name: r.name, x: r.fit_s, y: r.cv_mean, front: !!r.pareto, pick: r.name === pickName,
    detail: r.predict_ms != null ? `predict ${fmtNum(r.predict_ms, 1)} ms` : '' })),
  { xLabel: 'Fit time, seconds (log scale)', yLabel: `CV ${st.confirm.primary_metric}` });
  return h('section', { class: 'block', 'aria-labelledby': 'pareto-title' },
    h('div', { class: 'block-head' }, h('h4', { class: 'block-title', id: 'pareto-title', text: 'Score against fit time' }),
      h('p', { class: 'block-sub', text: 'Blue: Pareto front. Ring: the pick.' })), box);
}

const VERDICT = {
  improves: ['Improves', 'up', 'tag-good'],
  no_gain: ['No gain', 'flat', ''],
  worse: ['Worse', 'down', 'tag-crit'],
};

function sideBlock(st) {
  const plan = st.plan || {};
  const md = st.report?.markdown || '';
  const why = sectionText(md, /n\*|sample size/i);
  const rungs = (plan.schedule || []).map((r) => fmtInt(r.n_rows));
  const ext = st.externals;
  return h('section', { class: 'block', 'aria-labelledby': 'nstar-title' },
    h('h4', { class: 'block-title', id: 'nstar-title', text: 'Why these sample sizes' }),
    h('p', { class: 'nstar-big', text: `n* = ${fmtInt(plan.n_star)} rows` }),
    rungs.length ? h('p', { class: 'block-sub', text: `Rungs used ${rungs.slice(0, -1).join(', ')}${rungs.length > 1 ? ' and ' : ''}${rungs.at(-1)} rows.` }) : null,
    why ? h('p', { class: 'md' }, richText(why)) : null,
    h('h4', { class: 'block-title sub-title', text: 'Outside data' }),
    ext.length ? h('ul', { class: 'verdicts' }, ext.map((x) => {
      const [label, ic, cls] = VERDICT[x.verdict] || [tidy(x.verdict || 'unknown'), 'info', ''];
      const facts = [
        x.delta != null ? `change ${fmtSigned(x.delta, 4)}` : null,
        Array.isArray(x.ci) ? `95% CI ${fmtSigned(x.ci[0], 4)} to ${fmtSigned(x.ci[1], 4)}` : null,
        x.p_one_sided != null ? `one-sided p ${fmtP(x.p_one_sided)}` : null,
        x.match_rate != null ? `${Math.round(x.match_rate * 100)}% of rows matched` : null,
        Array.isArray(x.new_columns) ? `${x.new_columns.length} new columns` : null,
      ].filter(Boolean);
      return h('li', null,
        h('div', { class: 'v-head' }, h('span', { class: `tag ${cls}` }, icon(ic), label), h('span', { text: tidy(x.source) })),
        h('p', { text: facts.join(', ') }),
        (x.leakage_flags || []).length ? h('p', null, h('span', { class: 'sev sev-2' }, icon('triangle'), 'Leakage flags'), ' ', (x.leakage_flags || []).map(String).join(', ')) : null);
    })) : h('p', { class: 'empty', text: 'No outside data was tried in this run.' }));
}

function reportBlock(st) {
  const r = st.report;
  if (!r) return h('section', { class: 'block' }, h('h4', { class: 'block-title', text: 'Agent report' }), h('p', { class: 'empty', text: 'The agent writes its report after the final fit.' }));
  const md = h('div', { class: 'md' });
  md.append(renderMarkdown(r.markdown));
  return h('section', { class: 'block', 'aria-labelledby': 'report-title' },
    h('div', { class: 'block-head' }, h('h4', { class: 'block-title', id: 'report-title', text: 'Agent report' })),
    r.spoken_summary ? h('p', { class: 'spoken' }, richText(r.spoken_summary)) : null,
    h('details', { class: 'report', open: true }, h('summary', { text: 'Read the full report' }), md));
}

export function renderResult(root, st, { animate, mode, runId }) {
  const c = st.confirm;
  const rows = c.table || [];
  const rec = c.recommendation || {};
  const pick = rows.find((r) => r.name === rec.pick) || rows[0];
  const best = rows.find((r) => r.name === rec.best) || rows[0];
  if (!pick) {
    root.replaceChildren(h('p', { class: 'empty', text: 'The confirm step returned no finalists.' }));
    return;
  }
  const card = recCard(st, pick, best);
  root.replaceChildren(
    h('div', { class: 'result-grid' }, card, takeAway(st, { mode, runId })),
    confirmTable(st, pick.name, best.name),
    h('div', { class: 'result-grid' }, paretoBlock(st, pick.name), sideBlock(st)),
    reportBlock(st));
  if (animate && gsapReady() && !prefersReducedMotion()) {
    window.gsap.from(card, { opacity: 0, scale: 0.98, duration: 0.5, ease: 'expo.out', clearProps: 'opacity,transform' });
    revealCi(root);
  }
}
