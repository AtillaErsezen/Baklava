// Hand-built charts: funnel bars (HTML, clip-path widths), CI whiskers and the Pareto scatter (SVG),
// split bar, feature bars and the token meter. Color carries emphasis only; every value is also text.

import { h, s, icon, fmtInt, fmtNum, fmtUsd, prefersReducedMotion, gsapReady } from './dom.js';

// ---------- funnel ----------
/** Log scale: 3,000 pipelines and 3 finalists both stay readable. 1 maps to a visible sliver, not zero. */
export const funnelScale = (max) => {
  const top = Math.log10(Math.max(2, max) + 1);
  return (v) => Math.min(1, Math.max(0, Math.log10(Math.max(0, v) + 1) / top));
};

const TICKS = [1, 10, 100, 1000, 10000, 100000];

export function funnelAxis(max) {
  const scale = funnelScale(max);
  const ticks = TICKS.filter((t) => t <= max * 1.01);
  return h('div', { class: 'funnel-axis', 'aria-hidden': 'true' },
    h('span', { text: 'log scale' }),
    h('div', { class: 'funnel-ticks' }, ticks.map((t) => h('span', { text: fmtInt(t), vars: { left: `${(scale(t) * 100).toFixed(2)}%` } }))),
    h('span'));
}

function setBarWidth(el, w) { el.style.setProperty('--w', String(w)); }

/** Keyed update of a funnel: rows keep their element, so a new rung animates without redrawing the rest. */
export class Funnel {
  constructor(root, { labelWidth } = {}) {
    this.root = root;
    this.rows = new Map();
    this.max = 0;
    if (labelWidth) root.style.setProperty('--fl', labelWidth);
  }

  reset() {
    this.root.replaceChildren();
    this.rows.clear();
    this.max = 0;
  }

  /** rows: [{key, label, detail, total, count, pending, pick, drop}]; animate: keys whose bar should narrow. */
  update(rows, max, animate = new Set()) {
    if (max !== this.max) {
      this.max = max;
      this.root.querySelector('.funnel-axis')?.remove();
      this.root.prepend(funnelAxis(max));
    }
    const scale = funnelScale(max);
    for (const row of rows) {
      let el = this.rows.get(row.key);
      if (!el) {
        el = h('div', { class: 'f-row', dataset: { key: row.key } },
          h('div', { class: 'f-label' }, h('b'), h('small')),
          h('div', { class: 'f-track', 'aria-hidden': 'true' }, h('span', { class: 'f-ghost' }), h('span', { class: 'f-bar' })),
          h('div', { class: 'f-count' }),
          h('div', { class: 'f-drop' }));
        this.rows.set(row.key, el);
        this.root.append(el);
      }
      const [label, track, count, drop] = el.children;
      label.children[0].textContent = row.label;
      label.children[1].textContent = row.detail || '';
      el.classList.toggle('is-pending', !!row.pending);
      el.classList.toggle('is-pick', !!row.pick);
      drop.textContent = row.drop || '';
      drop.hidden = !row.drop;
      const ghost = track.children[0];
      const bar = track.children[1];
      const from = scale(row.total ?? row.count ?? 0);
      const to = scale(row.count ?? 0);
      setBarWidth(ghost, from);
      if (!row.pending && animate.has(row.key) && gsapReady() && !prefersReducedMotion()) {
        // Continuity transition: the bar starts at the width it was handed (evaluated) and narrows to the survivors.
        const n = { v: row.total ?? 0 };
        setBarWidth(bar, from);
        count.textContent = fmtInt(n.v);
        window.gsap.to(bar, { '--w': to, duration: 0.9, ease: 'power3.inOut' });
        window.gsap.to(n, { v: row.count ?? 0, duration: 0.9, ease: 'power3.inOut', onUpdate: () => { count.textContent = fmtInt(n.v); } });
      } else {
        setBarWidth(bar, to);
        count.textContent = row.count != null ? fmtInt(row.count) : '';
      }
    }
  }
}

// ---------- CI whisker (one table cell) ----------
const CI_W = 180;
const CI_PAD = 6;

export function ciDomain(rows, threshold) {
  const lows = rows.map((r) => r.ci?.[0]).filter(Number.isFinite);
  const highs = rows.map((r) => r.ci?.[1]).filter(Number.isFinite);
  let lo = Math.min(...lows, Number.isFinite(threshold) ? threshold : Infinity);
  let hi = Math.max(...highs, Number.isFinite(threshold) ? threshold : -Infinity);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  const pad = (hi - lo) * 0.08 || 0.01;
  return [lo - pad, hi + pad];
}

export function ciSvg({ low, high, mean, domain, threshold, label }) {
  const [d0, d1] = domain;
  const x = (v) => CI_PAD + ((v - d0) / (d1 - d0)) * (CI_W - 2 * CI_PAD);
  const g = s('svg', { viewBox: `0 0 ${CI_W} 26`, width: CI_W, height: 26, role: 'img', 'aria-label': label },
    Number.isFinite(threshold) ? s('line', { class: 'ci-thr', x1: x(threshold), x2: x(threshold), y1: 3, y2: 23 }) : null,
    Number.isFinite(low) && Number.isFinite(high) ? s('line', { class: 'ci-line', x1: x(low), x2: x(high), y1: 13, y2: 13 }) : null,
    Number.isFinite(mean) ? s('circle', { class: 'ci-dot', cx: x(mean), cy: 13, r: 4.5 }) : null);
  return g;
}

export function ciAxis(domain, threshold) {
  const [d0, d1] = domain;
  const x = (v) => CI_PAD + ((v - d0) / (d1 - d0)) * (CI_W - 2 * CI_PAD);
  const mid = (d0 + d1) / 2;
  return s('svg', { class: 'ci-axis', viewBox: `0 0 ${CI_W} 16`, width: CI_W, height: 16, 'aria-hidden': 'true' },
    [d0 + (d1 - d0) * 0.08, mid, d1 - (d1 - d0) * 0.08].map((v, i) =>
      s('text', { x: x(v), y: 12, 'text-anchor': i === 0 ? 'start' : i === 2 ? 'end' : 'middle' }, v.toFixed(3))),
    Number.isFinite(threshold) ? s('line', { class: 'ci-thr', x1: x(threshold), x2: x(threshold), y1: 14, y2: 16 }) : null);
}

/** One-time reveal: each interval grows out of its point estimate. */
export function revealCi(root) {
  if (!gsapReady() || prefersReducedMotion()) return;
  root.querySelectorAll('.ci-line').forEach((line, i) => {
    const dot = line.parentNode.querySelector('.ci-dot');
    const cx = dot ? dot.getAttribute('cx') : line.getAttribute('x1');
    window.gsap.from(line, { attr: { x1: cx, x2: cx }, duration: 0.7, delay: 0.08 * i, ease: 'expo.out' });
  });
}

// ---------- Pareto scatter ----------
const niceLogTicks = (lo, hi) => {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
};

const niceTicks = (lo, hi, n = 4) => {
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
};

const fmtSec = (v) => (v >= 10 ? String(Math.round(v)) : String(Number(v.toPrecision(2))));

/** points: [{name, x, y, front, pick, detail}] with x = fit seconds (log), y = CV score. */
export function scatter(root, points, { xLabel, yLabel }) {
  root.replaceChildren();
  const valid = points.filter((p) => p.x > 0 && Number.isFinite(p.y));
  if (!valid.length) {
    root.append(h('p', { class: 'empty', text: 'Fit times were not reported for these models.' }));
    return;
  }
  // A narrower viewBox on phones keeps the 11px labels near 11px on screen instead of shrinking with the chart.
  const W = window.innerWidth < 640 ? 380 : 560, H = 300, M = { l: 56, r: 24, t: 16, b: 48 };
  const xs = valid.map((p) => p.x), ys = valid.map((p) => p.y);
  const x0 = Math.min(...xs) / 1.8, x1 = Math.max(...xs) * 1.8;
  const ySpan = Math.max(Math.max(...ys) - Math.min(...ys), 0.01);
  const y0 = Math.min(...ys) - ySpan * 0.35, y1 = Math.max(...ys) + ySpan * 0.35;
  const X = (v) => M.l + ((Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))) * (W - M.l - M.r);
  const Y = (v) => H - M.b - ((v - y0) / (y1 - y0)) * (H - M.t - M.b);
  const tip = h('div', { class: 'tip', role: 'status', hidden: true });
  const show = (p, cx, cy) => {
    tip.replaceChildren(h('b', { text: p.name }), h('br'), `${yLabel} ${fmtNum(p.y, 4)}`, h('br'), `fit ${fmtSec(p.x)} s`, p.detail ? [h('br'), p.detail] : null);
    tip.hidden = false;
    const box = root.getBoundingClientRect();
    const scaleX = box.width / W;
    tip.style.left = `${cx * scaleX}px`;
    tip.style.top = `${cy * scaleX}px`;
  };
  const hide = () => { tip.hidden = true; };

  const front = valid.filter((p) => p.front).sort((a, b) => a.x - b.x);
  const svg = s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': `${yLabel} against fit time for ${valid.length} models` },
    niceTicks(y0, y1).map((t) => s('g', null,
      s('line', { class: 'grid', x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }),
      s('text', { x: M.l - 8, y: Y(t) + 4, 'text-anchor': 'end' }, t.toFixed(3)))),
    niceLogTicks(x0, x1).map((t) => s('text', { x: X(t), y: H - M.b + 18, 'text-anchor': 'middle' }, fmtSec(t))),
    s('line', { class: 'axis', x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }),
    s('text', { class: 'axis-title', x: (M.l + W - M.r) / 2, y: H - 8, 'text-anchor': 'middle' }, xLabel),
    s('text', { class: 'axis-title', x: -(M.t + H - M.b) / 2, y: 14, transform: 'rotate(-90)', 'text-anchor': 'middle' }, yLabel),
    front.length > 1 ? s('polyline', { class: 'front', points: front.map((p) => `${X(p.x)},${Y(p.y)}`).join(' ') }) : null,
    valid.map((p) => {
      const cx = X(p.x), cy = Y(p.y);
      const right = cx > W - 150;
      const g = s('g', { tabindex: '0', role: 'img', 'aria-label': `${p.name}: ${yLabel} ${fmtNum(p.y, 4)}, fit ${fmtSec(p.x)} seconds${p.front ? ', on the Pareto front' : ''}${p.pick ? ', recommended' : ''}` },
        s('circle', { class: 'hit', cx, cy, r: 16 }),
        p.pick ? s('circle', { class: 'pt-ring', cx, cy, r: 10 }) : null,
        s('circle', { class: `pt${p.front ? ' is-front' : ''}`, cx, cy, r: 5.5 }),
        s('text', { class: 'pt-label', x: right ? cx - 14 : cx + 14, y: cy + 4, 'text-anchor': right ? 'end' : 'start' }, p.pick ? `${p.name} (pick)` : p.name));
      g.addEventListener('pointerenter', () => show(p, cx, cy));
      g.addEventListener('pointerleave', hide);
      g.addEventListener('focus', () => show(p, cx, cy));
      g.addEventListener('blur', hide);
      return g;
    }));
  root.append(h('div', { class: 'chart-wrap' }, svg, tip));
}

// ---------- rows split (dev / search-val / hidden) ----------
export function splitBar(rows) {
  const parts = [
    { k: 'dev', label: 'Dev', n: rows?.dev },
    { k: 'search_val', label: 'Search-val', n: rows?.search_val },
    { k: 'hidden', label: 'Hidden, locked', n: rows?.hidden_locked },
  ].filter((p) => Number.isFinite(p.n) && p.n > 0);
  const total = parts.reduce((a, p) => a + p.n, 0) || 1;
  return h('div', { class: 'split' },
    h('div', { class: 'split-bar', role: 'img', 'aria-label': parts.map((p) => `${p.label} ${fmtInt(p.n)} rows`).join(', ') },
      parts.map((p) => h('span', { class: 'split-seg', dataset: { k: p.k }, vars: { 'flex-grow': String(p.n) } }))),
    h('div', { class: 'split-legend' }, parts.map((p) => h('div', null,
      h('b', { text: fmtInt(p.n) }),
      h('span', null, p.k === 'hidden' ? icon('lock') : null, `${p.label}, ${Math.round((p.n / total) * 100)}%`)))));
}

// ---------- feature importance ----------
export function featureBars(features) {
  const list = (features || []).filter((f) => Number.isFinite(f.importance)).slice(0, 8);
  if (!list.length) return h('p', { class: 'empty', text: 'This model did not report feature importances.' });
  const max = Math.max(...list.map((f) => f.importance)) || 1;
  const total = list.reduce((a, f) => a + f.importance, 0) || 1;
  return h('div', { class: 'bars' }, list.map((f) => {
    const name = String(f.feature).replace(/^(num|cat|bool|txt|dt)__/, '');
    return h('div', { class: 'bar-row' },
      h('code', { text: name, title: String(f.feature) }),
      h('i', { vars: { '--w': (f.importance / max).toFixed(4) }, 'aria-hidden': 'true' }),
      h('span', { text: `${Math.round((f.importance / total) * 100)}%` }));
  }));
}

// ---------- token meter ----------
export function tokenMeter(usage) {
  const inT = usage.input_tokens || 0, outT = usage.output_tokens || 0;
  const llm = Number.isFinite(usage.usd) ? usage.usd : null;
  const compute = Number.isFinite(usage.modal_usd) ? usage.modal_usd : null;
  const total = (llm ?? 0) + (compute ?? 0);
  return h('div', { class: 'meter' },
    h('div', null,
      h('span', { class: 'meter-cost', text: fmtUsd(total) }),
      h('span', { class: 'block-sub', text: ` for ${fmtInt(usage.calls)} model calls` }),
      h('p', { class: 'meter-split', text: [llm != null ? `LLM ${fmtUsd(llm)}` : null, compute != null ? `compute ${fmtUsd(compute)}` : null].filter(Boolean).join(' \u00b7 ') })),
    h('div', { class: 'meter-bar', role: 'img', 'aria-label': `${fmtInt(inT)} input tokens, ${fmtInt(outT)} output tokens` },
      h('i', { vars: { 'flex-grow': String(Math.max(inT, 1)) } }), h('i', { vars: { 'flex-grow': String(Math.max(outT, 1)) } })),
    h('dl', null,
      h('dt', null, h('span', { class: 'swatch' }), 'Input tokens'), h('dd', { text: fmtInt(inT) }),
      h('dt', null, h('span', { class: 'swatch is-light' }), 'Output tokens'), h('dd', { text: fmtInt(outT) })));
}
