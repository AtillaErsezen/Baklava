// Entry point. URL state: ?dataset=<id> (step 2), ?run=<id> (live run), ?replay=<id> (recorded run;
// "fixture" plays web/fixtures/sample_events.json without the API). ?fixture=1 adds the sample dataset button.

import { $, getJSON, validId } from './dom.js';
import { initLanding, initNav, scrollToTarget } from './landing.js';
import { Flow } from './flow.js';
import { RunView } from './run.js';
import { startLive, startReplay } from './sources.js';
import { initRecentRuns } from './recent.js';

const params = new URLSearchParams(location.search);
const runId = params.get('run');
const replayId = params.get('replay');
const datasetId = params.get('dataset');
const fixture = params.has('fixture') || replayId === 'fixture';

let stopSource = () => {};

const view = new RunView({
  onResultReady: () => flow.enable('result'),
  onRunEnd: () => { flow.complete('run'); if (flow.current === 'run') flow.show('result'); },
});

const flow = new Flow({ fixture, onStartRun: (id) => beginRun(id) });

function enterRunMode() {
  document.body.classList.add('is-run-mode');
  initNav({ solid: true });
  const cta = $('.nav-cta');
  cta.textContent = 'New run';
  cta.setAttribute('href', './#app');
  flow.lockInputs();
  flow.enable('run');
  flow.show('run', { focus: false });
  window.ScrollTrigger?.getAll().forEach((st) => st.kill());
  window.scrollTo(0, 0);
}

async function beginReplay(id) {
  enterRunMode();
  view.setStatus('queued', 'Loading');
  try {
    const url = id === 'fixture' ? 'fixtures/sample_events.json' : `/api/runs/${encodeURIComponent(id)}/events`;
    const rows = await getJSON(url);
    stopSource = startReplay(rows, view, { mode: id === 'fixture' ? 'fixture' : 'replay', runId: id === 'fixture' ? null : id });
  } catch (e) {
    view.setStatus('failed', 'Not found');
    view.setActivity(e.status === 404 ? 'No recording with this id exists on the server.' : 'Could not load the recording.');
  }
}

function beginRun(id) {
  stopSource();
  if (id === 'fixture') {
    history.pushState(null, '', '?replay=fixture');
    beginReplay('fixture');
    return;
  }
  history.pushState(null, '', `?run=${encodeURIComponent(id)}`);
  enterRunMode();
  stopSource = startLive(id, view);
}

window.addEventListener('popstate', () => location.reload());

if (runId && validId(runId)) {
  enterRunMode();
  stopSource = startLive(runId, view);
} else if (replayId && (replayId === 'fixture' || validId(replayId))) {
  beginReplay(replayId);
} else {
  initLanding();
  initRecentRuns();
  if (datasetId && validId(datasetId)) {
    flow.reopen(datasetId);
    requestAnimationFrame(() => scrollToTarget($('#app')));
  } else if (location.hash === '#app') {
    requestAnimationFrame(() => scrollToTarget($('#app')));
  }
}
