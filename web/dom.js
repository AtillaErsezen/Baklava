// DOM and formatting helpers. Every string goes in through textContent or createTextNode:
// nothing from the network or a CSV is ever parsed as HTML.

const SVG_NS = 'http://www.w3.org/2000/svg';

function setProps(el, props) {
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === 'class') el.setAttribute('class', value);
    else if (key === 'text') el.textContent = String(value);
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key === 'vars') for (const [name, v] of Object.entries(value)) el.style.setProperty(name, v);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? '' : String(value));
  }
}

export function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  setProps(el, props);
  return append(el, kids);
}

export function s(tag, props, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  setProps(el, props);
  return append(el, kids);
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const prefersReducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** GSAP is loaded from the CDN with defer; the app still works (without motion) if it is blocked. */
export const gsapReady = () => typeof window.gsap !== 'undefined';

// ---------- formatting ----------
const intFmt = new Intl.NumberFormat('en-US');
const MINUS = '−';

export const isNum = (n) => typeof n === 'number' && Number.isFinite(n);
export const fmtInt = (n) => (isNum(+n) && n !== null && n !== '' ? intFmt.format(Math.round(+n)) : '-');
export const fmtNum = (n, digits = 3) => (isNum(n) ? n.toFixed(digits).replace('-', MINUS) : '-');
export const fmtSigned = (n, digits = 3) => (isNum(n) ? (n > 0 ? '+' : n < 0 ? MINUS : '') + Math.abs(n).toFixed(digits) : '-');
export const fmtP = (p) => (isNum(p) ? (p < 0.001 ? '< 0.001' : p.toFixed(3)) : '-');
export const fmtPct = (n, digits = 1) => (isNum(n) ? `${n.toFixed(digits)}%` : '-');
export const fmtUsd = (n) => (isNum(n) ? `$${n < 0.1 ? n.toFixed(4) : n.toFixed(2)}` : '-');
export const fmtBytes = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(1)} MB`);

export function fmtClock(seconds) {
  const t = Math.max(0, Math.floor(seconds || 0));
  const m = Math.floor(t / 60);
  return `${m}:${String(t % 60).padStart(2, '0')}`;
}

/** Parameter values: 3 significant digits for floats, integers and strings as they are. */
export function fmtParam(v) {
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return intFmt.format(v);
    return String(Number(v.toPrecision(3)));
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (v == null) return 'none';
  return typeof v === 'object' ? JSON.stringify(v) : String(v);
}

/** House style for agent text: plain hyphens instead of long dashes, trimmed. */
export const tidy = (text) => String(text ?? '').replace(/[–—]/g, '-').trim();

/** Inline code spans in agent narration (`name`) become <code>; everything else stays text. */
export function richText(text) {
  const parts = tidy(text).split(/(`[^`\n]+`)/g);
  return parts.filter(Boolean).map((p) => (p.startsWith('`') && p.endsWith('`') ? h('code', { text: p.slice(1, -1) }) : p));
}

// ---------- icons (paths from Tabler Icons, MIT) ----------
const PATHS = {
  upload: ['M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2', 'M7 9l5 -5l5 5', 'M12 4l0 12'],
  download: ['M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2', 'M7 11l5 5l5 -5', 'M12 4l0 12'],
  file: ['M14 3v4a1 1 0 0 0 1 1h4', 'M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z'],
  check: ['M5 12l5 5l10 -10'],
  x: ['M18 6l-12 12', 'M6 6l12 12'],
  octagon: ['M12.802 2.165l5.575 2.389c.48 .206 .863 .589 1.07 1.07l2.388 5.574c.22 .512 .22 1.092 0 1.604l-2.389 5.575c-.206 .48 -.589 .863 -1.07 1.07l-5.574 2.388c-.512 .22 -1.092 .22 -1.604 0l-5.575 -2.389a2.036 2.036 0 0 1 -1.07 -1.07l-2.388 -5.574a2.036 2.036 0 0 1 0 -1.604l2.389 -5.575c.206 -.48 .589 -.863 1.07 -1.07l5.574 -2.388a2.036 2.036 0 0 1 1.604 0z', 'M12 8v4', 'M12 16h.01'],
  triangle: ['M12 9v4', 'M10.363 3.591l-8.106 13.534a1.914 1.914 0 0 0 1.636 2.871h16.214a1.914 1.914 0 0 0 1.636 -2.87l-8.106 -13.536a1.914 1.914 0 0 0 -3.274 0z', 'M12 16h.01'],
  info: ['M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0', 'M12 9h.01', 'M11 12h1v4h1'],
  lock: ['M5 13a2 2 0 0 1 2 -2h10a2 2 0 0 1 2 2v6a2 2 0 0 1 -2 2h-10a2 2 0 0 1 -2 -2v-6z', 'M11 16a1 1 0 1 0 2 0a1 1 0 0 0 -2 0', 'M8 11v-4a4 4 0 1 1 8 0v4'],
  play: ['M7 4v16l13 -8z'],
  pause: ['M6 6a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1z', 'M14 6a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1z'],
  replay: ['M19.95 11a8 8 0 1 0 -.5 4m.5 5v-5h-5'],
  copy: ['M7 9.667a2.667 2.667 0 0 1 2.667 -2.667h8.666a2.667 2.667 0 0 1 2.667 2.667v8.666a2.667 2.667 0 0 1 -2.667 2.667h-8.666a2.667 2.667 0 0 1 -2.667 -2.667z', 'M4.012 16.737a2.005 2.005 0 0 1 -1.012 -1.737v-10c0 -1.1 .9 -2 2 -2h10c.75 0 1.158 .385 1.5 1'],
  up: ['M3 17l6 -6l4 4l8 -8', 'M14 7l7 0l0 7'],
  down: ['M3 7l6 6l4 -4l8 8', 'M21 10l0 7l-7 0'],
  flat: ['M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0', 'M9 12l6 0'],
  circleCheck: ['M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0', 'M9 12l2 2l4 -4'],
  clock: ['M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0', 'M12 7v5l3 3'],
};

export function icon(name, label) {
  const el = s('svg', { class: 'icon', viewBox: '0 0 24 24', 'aria-hidden': label ? null : 'true', role: label ? 'img' : null, 'aria-label': label || null },
    (PATHS[name] || []).map((d) => s('path', { d })));
  return el;
}

// ---------- severity vocabulary (icon + text label, never color alone) ----------
export const SEVERITY = {
  3: { label: 'Must act', icon: 'octagon' },
  2: { label: 'Should act', icon: 'triangle' },
  1: { label: 'Note', icon: 'info' },
};

export function sevChip(level) {
  const lv = SEVERITY[level] ? level : 1;
  return h('span', { class: `sev sev-${lv}` }, icon(SEVERITY[lv].icon), SEVERITY[lv].label);
}

// ---------- network ----------
export async function getJSON(url, init) {
  const res = await fetch(url, { headers: { Accept: 'application/json' }, ...init });
  let body = null;
  try { body = await res.json(); } catch { body = null; }
  if (!res.ok) {
    const err = new Error((body && body.error) || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

/** Ids come from the URL, so they are checked against the server's own rule before use in paths or filters. */
export const ID_RE = /^(?!\.)[A-Za-z0-9_.-]{1,64}$/;
export const validId = (id) => typeof id === 'string' && ID_RE.test(id) && id !== '..';
