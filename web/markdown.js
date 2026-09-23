// A small markdown renderer for the agent's report. It builds DOM nodes directly, so the text
// is never parsed as HTML. Supports headings, paragraphs, lists, tables, fenced code, quotes,
// rules, and inline code / bold / italic / links (http, https and mailto only).
// ponytail: one list level only; nested lists render flat. Add indentation tracking if reports start nesting.

import { h, tidy } from './dom.js';

// Underscore emphasis is left out on purpose: column names like customer_id and refund_issued would turn italic.
// The harness replaces figures the tools never produced with "[number removed: not in tool results]".
const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\s][^*\n]*\*)|(\[number removed[^\]\n]*\])|(\[[^\]\n]+\]\([^)\s]+\))/g;
const SAFE_URL = /^(https?:\/\/|mailto:)/i;

export function inline(text) {
  const out = [];
  let last = 0;
  const src = tidy(text);
  for (const m of src.matchAll(INLINE)) {
    if (m.index > last) out.push(src.slice(last, m.index));
    const [tok] = m;
    if (m[1]) out.push(h('code', { text: tok.slice(1, -1) }));
    else if (m[2]) out.push(h('strong', null, inline(tok.slice(2, -2))));
    else if (m[3]) out.push(h('em', null, inline(tok.slice(1, -1))));
    else if (m[4]) out.push(h('span', { class: 'tag removed', title: 'The report tried to state a number that no tool returned, so the harness removed it.', text: tok.slice(1, -1) }));
    else {
      const [, label, url] = tok.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      out.push(SAFE_URL.test(url) ? h('a', { href: url, rel: 'noopener noreferrer', target: '_blank' }, inline(label)) : label);
    }
    last = m.index + tok.length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

const splitRow = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
const isTableRule = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line || '');
const LIST_ITEM = /^\s*([-*+]|\d+[.)])\s+(.*)$/;

/** Render markdown to a DocumentFragment. headingOffset shifts # levels so a report h1 becomes h3. */
export function renderMarkdown(src, headingOffset = 2) {
  const frag = document.createDocumentFragment();
  const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
  let para = [];
  const flush = () => {
    if (para.length) frag.append(h('p', null, inline(para.join(' '))));
    para = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*(```|~~~)/);
    if (fence) {
      flush();
      const code = [];
      while (++i < lines.length && !lines[i].trim().startsWith(fence[1])) code.push(lines[i]);
      frag.append(h('pre', null, h('code', { text: code.join('\n') })));
      continue;
    }
    if (!line.trim()) { flush(); continue; }
    const heading = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) {
      flush();
      const level = Math.min(6, heading[1].length + headingOffset);
      frag.append(h(`h${level}`, null, inline(heading[2])));
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); frag.append(h('hr')); continue; }
    if (line.trim().startsWith('|') && isTableRule(lines[i + 1])) {
      flush();
      const head = splitRow(line);
      const body = [];
      i += 1;
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith('|')) body.push(splitRow(lines[++i]));
      frag.append(h('table', null,
        h('thead', null, h('tr', null, head.map((c) => h('th', { scope: 'col' }, inline(c))))),
        h('tbody', null, body.map((r) => h('tr', null, head.map((_, k) => h('td', null, inline(r[k] ?? ''))))))));
      continue;
    }
    const item = line.match(LIST_ITEM);
    if (item) {
      flush();
      const ordered = /\d/.test(item[1]);
      const list = h(ordered ? 'ol' : 'ul');
      let m = item;
      while (m) {
        list.append(h('li', null, inline(m[2])));
        m = i + 1 < lines.length ? lines[i + 1].match(LIST_ITEM) : null;
        if (m && /\d/.test(m[1]) !== ordered) break;
        if (m) i += 1;
      }
      frag.append(list);
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const quote = [line.replace(/^\s*>\s?/, '')];
      while (i + 1 < lines.length && /^\s*>/.test(lines[i + 1])) quote.push(lines[++i].replace(/^\s*>\s?/, ''));
      frag.append(h('blockquote', null, h('p', null, inline(quote.join(' ')))));
      continue;
    }
    para.push(line.trim());
  }
  flush();
  return frag;
}

/** The body text of the first section whose heading matches re, as plain text (for pull-outs like n*). */
export function sectionText(src, re) {
  const lines = String(src || '').split(/\r?\n/);
  const start = lines.findIndex((l) => /^#{1,6}\s/.test(l) && re.test(l));
  if (start < 0) return '';
  const body = [];
  for (let i = start + 1; i < lines.length && !/^#{1,6}\s/.test(lines[i]); i++) body.push(lines[i]);
  return tidy(body.join(' ').replace(/\s+/g, ' '));
}
