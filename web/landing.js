// Landing: WebGL shader behind the hero, SplitText headline reveal, Lenis smooth scroll synced to
// ScrollTrigger, scroll reveals, and the pinned explainer where the search funnel narrows on scroll.
// Everything degrades: no GSAP means static content, reduced motion means no Lenis, no pin, no shader loop.

import { h, $, $$, icon, fmtInt, sevChip, prefersReducedMotion } from './dom.js';
import { funnelScale } from './charts.js';

// ---------- shader ----------
const VERT = 'attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}';
const FRAG = `
precision highp float;
uniform vec2 u_res;
uniform float u_time;
float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1.,0.)),u.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),u.x),u.y);}
float fbm(vec2 p){float v=0.,a=.5;mat2 m=mat2(1.6,1.2,-1.2,1.6);for(int i=0;i<5;i++){v+=a*noise(p);p=m*p;a*=.5;}return v;}
void main(){
  vec2 uv=gl_FragCoord.xy/u_res;
  vec2 p=vec2(uv.x*u_res.x/u_res.y,uv.y);
  float t=u_time*.035;
  vec2 q=vec2(fbm(p*1.3+vec2(0.,t)),fbm(p*1.3+vec2(5.2,-t)));
  vec2 r=vec2(fbm(p*1.7+3.*q+vec2(1.7,9.2)+t*.6),fbm(p*1.7+3.*q+vec2(8.3,2.8)-t*.5));
  float f=fbm(p*1.1+2.4*r);
  vec3 col=vec3(.051,.051,.059);
  float right=smoothstep(.2,1.,uv.x)*(.5+.5*uv.y);
  float low=smoothstep(.75,0.,uv.y)*smoothstep(.05,.7,uv.x);
  col=mix(col,vec3(.24,.39,1.),smoothstep(.42,.95,f)*right*.8);
  col=mix(col,vec3(.29,.61,.91),smoothstep(.6,1.,r.x)*right*.35);
  col=mix(col,vec3(.91,.45,.29),smoothstep(.5,.95,q.y*f*1.7)*low*.5);
  col+=(hash(gl_FragCoord.xy+fract(u_time))-.5)*.02;
  gl_FragColor=vec4(col,1.);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) || 'shader compile failed');
  return sh;
}

/** Idle animation: a slow domain-warped noise field. Runs only while the hero is on screen and the tab is visible. */
export function initShader(canvas) {
  const gl = canvas?.getContext('webgl', { antialias: false, alpha: false, depth: false, powerPreference: 'low-power' });
  if (!gl) { canvas?.classList.add('is-fallback'); return; }
  let uRes, uTime;
  try {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link failed');
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'a');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    uRes = gl.getUniformLocation(prog, 'u_res');
    uTime = gl.getUniformLocation(prog, 'u_time');
  } catch {
    canvas.classList.add('is-fallback');
    return;
  }
  // The field is soft, so it renders at a fraction of device pixels and the browser scales it up.
  const QUALITY = 0.5;
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr * QUALITY));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr * QUALITY));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uRes, canvas.width, canvas.height);
    draw(lastT);
  };
  const START = 18; // a pleasant frame to open on, and the still frame under reduced motion
  let lastT = START, raf = 0, onScreen = true, t0 = performance.now();
  const draw = (t) => { gl.uniform1f(uTime, t); gl.drawArrays(gl.TRIANGLES, 0, 3); };
  const loop = (now) => { lastT = START + (now - t0) / 1000; draw(lastT); raf = requestAnimationFrame(loop); };
  const sync = () => {
    cancelAnimationFrame(raf);
    raf = 0;
    if (onScreen && !document.hidden && !prefersReducedMotion()) {
      t0 = performance.now() - (lastT - START) * 1000;
      raf = requestAnimationFrame(loop);
    } else draw(lastT);
  };
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; sync(); }).observe(canvas);
  document.addEventListener('visibilitychange', sync);
  matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', sync);
  resize();
  sync();
}

// ---------- explainer content (values from recorded run 20260923-121753-88d3) ----------
const RUN = {
  split: { dev: 2099, search_val: 301, hidden_locked: 600 },
  checks: [
    [3, 'leakage', 'refund_issued predicts the target on its own; dropped.'],
    [2, 'id_like', 'customer_id is an identifier; dropped.'],
    [1, 'missing_pattern', 'last_login_days is missing in 11% of rows.'],
    [1, 'mad_outliers', 'Outliers in last_login_days and support_tickets.'],
    [1, 'imbalance', 'Churn is imbalanced 3.9 to 1; metric f1_macro.'],
  ],
  nstar: { needed: 11478, dev: 2099, rungs: [500, 1500, 2099] },
  funnel: [
    ['space', 'Generated', 3000, 3000],
    ['raced', 'Raced', 3000, 243],
    ['r1', 'Rung 1', 243, 81],
    ['r2', 'Rung 2', 81, 27],
    ['r3', 'Rung 3', 27, 6],
    ['confirm', 'Confirmed', 6, 3],
    ['pick', 'Picked', 3, 1],
  ],
};

const READOUTS = [
  ['3,000', 'rows uploaded'],
  ['33', 'statistical checks, 5 findings'],
  ['2,099', 'rows per pipeline at most, n*'],
  [null, 'pipelines still in the race'],
  ['3', 'finalists, 10 paired folds each'],
  ['0.657', 'f1_macro on 600 hidden rows, scored once'],
  ['1', 'model, with the script that trains it'],
];
const RACE = [3000, 243, 81, 27, 6];

function buildPlates() {
  const split = $('#how-split');
  const total = RUN.split.dev + RUN.split.search_val + RUN.split.hidden_locked;
  split.append(
    h('div', { class: 'split-bar' }, [['dev', RUN.split.dev], ['search_val', RUN.split.search_val], ['hidden', RUN.split.hidden_locked]]
      .map(([k, n]) => h('span', { class: 'split-seg', dataset: { k }, vars: { 'flex-grow': String(n) } }))),
    h('div', { class: 'split-legend' },
      h('div', null, h('b', { text: fmtInt(RUN.split.dev) }), h('span', { text: `Dev, ${Math.round((RUN.split.dev / total) * 100)}%` })),
      h('div', null, h('b', { text: fmtInt(RUN.split.search_val) }), h('span', { text: 'Search-val, 10%' })),
      h('div', null, h('b', { text: fmtInt(RUN.split.hidden_locked) }), h('span', null, icon('lock'), 'Hidden, locked, 20%'))));

  $('#how-checks').append(...RUN.checks.map(([sev, check, text]) =>
    h('li', null, sevChip(sev), h('code', { text: check }), h('p', { text: text }))));

  const nstar = $('#how-nstar');
  const max = RUN.nstar.needed;
  const row = (label, n, ghost) => h('div', { class: 'nstar-row' },
    h('span', { text: label }),
    h('span', { class: `nstar-bar${ghost ? ' is-ghost' : ''}`, vars: { '--w': String(n / max) } }),
    h('span', { class: 'mono', text: fmtInt(n) }));
  nstar.append(
    row('Hoeffding bound', RUN.nstar.needed, true),
    row('Dev rows', RUN.nstar.dev),
    ...RUN.nstar.rungs.map((n, i) => row(`Rung ${i + 1}`, n)),
    h('p', { class: 'nstar-note', text: 'For a 0.02 error at 95% confidence across 243 configs the bound asks for 11,478 rows. The dev split has 2,099, so n* is capped there and the last rung uses all of them.' }));

  const funnel = $('#how-funnel');
  const rows = RUN.funnel.map(([key, label, total, count]) => {
    const el = h('div', { class: 'f-row', dataset: { key } },
      h('div', { class: 'f-label' }, h('b', { text: label })),
      h('div', { class: 'f-track' }, h('span', { class: 'f-ghost' }), h('span', { class: 'f-bar' })),
      h('div', { class: 'f-count', text: fmtInt(count) }));
    funnel.append(el);
    return { el, total, count, key };
  });
  const extra = h('div', { class: 'funnel-extra' },
    h('span', { class: 'tag tag-accent', dataset: { stage: '5' } }, icon('lock'), 'Hidden test 0.657, gap −0.021'),
    ...['train_lgbm_p02.py', 'params.json', 'MODEL_CARD.md'].map((f) => h('span', { class: 'tag', dataset: { stage: '6' } }, icon('file'), f)));
  funnel.append(extra);
  return { rows, extra };
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const clamp01 = (t) => Math.min(1, Math.max(0, t));

function makeRenderer(how, plates) {
  const stages = $$('.stage', how);
  const plateEls = $$('.plate', how);
  const num = $('#readout-num');
  const label = $('#readout-label');
  const scale = funnelScale(3000);
  let current = -1;

  return function render(progress) {
    const N = stages.length;
    const pos = clamp01(progress) * N;
    const stage = Math.min(N - 1, Math.floor(pos));
    const local = stage === N - 1 && progress >= 1 ? 1 : pos - stage;
    if (stage !== current) {
      current = stage;
      stages.forEach((el, i) => el.classList.toggle('is-active', i === stage));
      const plate = Math.min(stage, 3);
      plateEls.forEach((el, i) => el.classList.toggle('is-active', i === plate));
      label.textContent = READOUTS[stage][1];
      plates.extra.querySelectorAll('.tag').forEach((t) => t.classList.toggle('is-on', +t.dataset.stage <= stage));
    }
    // Number ticker: during the race the count moves on a log scale, so every rung gets equal scroll.
    if (stage === 3) {
      const seg = Math.min(RACE.length - 1.0001, local * (RACE.length - 1));
      const k = Math.floor(seg);
      const v = Math.exp(Math.log(RACE[k]) + (Math.log(RACE[k + 1]) - Math.log(RACE[k])) * easeInOut(seg - k));
      num.textContent = fmtInt(v);
    } else num.textContent = READOUTS[stage][0];

    // Funnel rows: each bar is handed its parent's width and narrows to its own (continuity transition).
    plates.rows.forEach((row, j) => {
      let t;
      if (j === 0) t = 1;
      else if (j <= 4) t = stage < 3 ? 0 : stage > 3 ? 1 : clamp01(local * 4 - (j - 1));
      else if (j === 5) t = stage < 4 ? 0 : stage > 4 ? 1 : clamp01(local * 2);
      else t = stage < 5 ? 0 : stage > 5 ? 1 : clamp01(local * 2);
      row.el.classList.toggle('is-hidden', t <= 0);
      const e = easeInOut(t);
      const w = scale(row.total) + (scale(row.count) - scale(row.total)) * e;
      row.el.children[1].children[0].style.setProperty('--w', String(scale(row.total)));
      row.el.children[1].children[1].style.setProperty('--w', String(w));
      row.el.children[2].textContent = fmtInt(Math.exp(Math.log(row.total) + (Math.log(row.count) - Math.log(row.total)) * e));
      row.el.classList.toggle('is-pick', row.key === 'pick' && t >= 1);
    });
  };
}

function initExplainer(gsap, ScrollTrigger) {
  const how = $('#how');
  if (!how) return;
  const plates = buildPlates();
  const render = makeRenderer(how, plates);
  render(1);
  if (!gsap || !ScrollTrigger) return;
  const mm = gsap.matchMedia();
  mm.add('(min-width: 961px) and (min-height: 700px) and (prefers-reduced-motion: no-preference)', () => {
    how.classList.add('is-pinned');
    render(0);
    const st = ScrollTrigger.create({
      trigger: '.how-inner',
      start: 'center center',
      end: () => `+=${Math.round(window.innerHeight * 4.2)}`,
      pin: true,
      anticipatePin: 1,
      invalidateOnRefresh: true,
      onUpdate: (self) => render(self.progress),
    });
    return () => { st.kill(); how.classList.remove('is-pinned'); render(1); };
  });
  // Narrow or short screens: no pin; the race plays once when the funnel scrolls into view.
  mm.add('(max-width: 960px) and (prefers-reduced-motion: no-preference), (max-height: 699px) and (prefers-reduced-motion: no-preference)', () => {
    const proxy = { p: 3 / 7 };
    render(proxy.p);
    const tween = gsap.to(proxy, { p: 1, duration: 3.6, ease: 'none', paused: true, onUpdate: () => render(proxy.p) });
    // A separate trigger: onEnter fires during creation when the page already sits past the funnel,
    // and an inline scrollTrigger would then call tween.play() before `tween` is assigned.
    const st = ScrollTrigger.create({ trigger: '#how-funnel', start: 'top 80%', once: true, onEnter: () => tween.play() });
    return () => { st.kill(); tween.kill(); render(1); };
  });
}

// ---------- hero, reveals, nav, smooth scroll ----------
function initHero(gsap, SplitText) {
  const root = document.documentElement;
  if (!gsap || prefersReducedMotion()) { root.classList.add('is-static'); return; }
  const start = () => {
    root.classList.add('is-ready');
    const title = $('.hero-title');
    if (SplitText && title) {
      SplitText.create(title, {
        type: 'lines,words', mask: 'lines', linesClass: 'line', autoSplit: true,
        onSplit(self) {
          gsap.set(title, { opacity: 1 });
          return gsap.from(self.words, { yPercent: 115, duration: 1.1, ease: 'expo.out', stagger: 0.035 });
        },
      });
    } else gsap.fromTo(title, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.9, ease: 'expo.out' });
    gsap.fromTo('[data-hero-fade]', { opacity: 0, y: 18 }, { opacity: 1, y: 0, duration: 0.9, ease: 'expo.out', stagger: 0.1, delay: 0.35 });
  };
  // Split after fonts load, or the lines break at the fallback font's widths.
  Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 1200))]).then(start);
}

function initReveals(gsap, ScrollTrigger) {
  if (!gsap || !ScrollTrigger || prefersReducedMotion()) return;
  const items = $$('[data-reveal]');
  gsap.set(items, { opacity: 0, y: 24 });
  ScrollTrigger.batch(items, {
    start: 'top 88%',
    once: true,
    onEnter: (batch) => gsap.to(batch, { opacity: 1, y: 0, duration: 0.8, ease: 'expo.out', stagger: 0.08, overwrite: true }),
  });
}

export function initNav({ solid } = {}) {
  const nav = $('#nav');
  const hero = $('#top');
  if (solid || !hero || getComputedStyle(hero).display === 'none') { nav.classList.add('is-solid'); return; }
  new IntersectionObserver(([e]) => nav.classList.toggle('is-solid', !e.isIntersecting), { rootMargin: '-64px 0px 0px 0px' }).observe(hero);
  // Over the hero the bar stays dark, but once content scrolls under it the bar gets a backdrop.
  const sentinel = h('div', { class: 'scroll-sentinel', 'aria-hidden': 'true' });
  hero.prepend(sentinel);
  new IntersectionObserver(([e]) => nav.classList.toggle('is-scrolled', !e.isIntersecting)).observe(sentinel);
}

let lenis = null;
export const getLenis = () => lenis;

function initSmoothScroll(gsap, ScrollTrigger) {
  const Lenis = window.Lenis;
  if (!Lenis || !gsap || prefersReducedMotion()) return;
  lenis = new Lenis({ lerp: 0.1, smoothWheel: true });
  if (ScrollTrigger) lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
}

/** In-page links scroll smoothly and move keyboard focus to the target section. */
export function scrollToTarget(target) {
  if (!target) return;
  if (lenis) lenis.scrollTo(target, { offset: -64, duration: 1.1 });
  else target.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
  target.focus({ preventScroll: true });
}

export function initLanding() {
  const { gsap, ScrollTrigger, SplitText } = window;
  if (gsap && ScrollTrigger) gsap.registerPlugin(ScrollTrigger, ...(SplitText ? [SplitText] : []));
  initShader($('#shader'));
  initSmoothScroll(gsap, ScrollTrigger);
  initHero(gsap, SplitText);
  initExplainer(gsap, ScrollTrigger);
  initReveals(gsap, ScrollTrigger);
  initNav();
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a || a.getAttribute('href').length < 2) return;
    const target = document.getElementById(a.getAttribute('href').slice(1));
    if (!target || getComputedStyle(target).display === 'none') return;
    e.preventDefault();
    history.replaceState(null, '', a.getAttribute('href'));
    scrollToTarget(target);
  });
}
