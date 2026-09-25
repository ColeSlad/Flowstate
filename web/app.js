'use strict';
const $ = id => document.getElementById(id);
const number = (value, digits = 0) => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const percent = value => value == null ? 'Unavailable' : `${number(value * 100, 0)}%`;
const policies = {
  cpu_latency: ['CPU latency', 'New searches run on the CPU, with no intentional batch delay.'],
  gpu_immediate: ['GPU immediate', 'New searches run on the GPU as soon as it is available.'],
  gpu_batch: ['GPU batching', 'New searches collect into GPU batches as traffic allows.'],
  balanced: ['Balanced', 'New requests split evenly between the CPU and immediate GPU execution.']
};
let latest, lastEvent = 0, lastPolicy, pending = false, changedAt = 0;
let history = [], transitions = [];
let controls = {traffic: 'low', top_k: 10, contention: false};
function connection(state) {
  if (document.body.dataset.connection === state) return;
  document.body.dataset.connection = state;
  $('connection-label').textContent = state === 'live' ? 'Runtime connected' : 'Telemetry paused · reconnecting';
  $('controls').disabled = state !== 'live' || pending;
}
function renderControls() {
  document.querySelectorAll('[data-traffic]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.traffic === controls.traffic)));
  if (![...$('top-k').options].some(option => Number(option.value) === controls.top_k)) {
    const option = new Option(`${controls.top_k} results`, String(controls.top_k));
    const before = [...$('top-k').options].find(value => Number(value.value) > controls.top_k);
    $('top-k').insertBefore(option, before || null);
  }
  $('top-k').value = controls.top_k;
  $('contention').checked = controls.contention;
  $('contention').disabled = !latest?.gpu_available || Boolean(latest?.contention.error);
  $('contention-label').textContent = latest?.contention.error ? 'Competing workload stopped' : controls.contention ? 'Competing workload on' : 'Competing workload off';
  $('contention-note').textContent = latest?.gpu_available ? 'Runs a real competing CUDA kernel' : 'Requires an available CUDA GPU';
  for (const option of $('top-k').options) option.disabled = latest && Number(option.value) > latest.vectors;
}
async function updateControls(change) {
  if (pending) return;
  pending = true; $('controls').disabled = true; $('control-feedback').textContent = '';
  try {
    const response = await fetch('/api/controls', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...controls, ...change})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The control change could not be applied.');
    controls = result; changedAt = Date.now(); renderControls();
  } catch (error) {
    $('control-feedback').textContent = error.message;
    renderControls();
  } finally {
    pending = false; $('controls').disabled = document.body.dataset.connection !== 'live';
  }
}
document.querySelectorAll('[data-traffic]').forEach(button => button.addEventListener('click', () => updateControls({traffic: button.dataset.traffic})));
$('top-k').addEventListener('change', event => updateControls({top_k: Number(event.target.value)}));
$('contention').addEventListener('change', event => updateControls({contention: event.target.checked}));

function chart() {
  const width = Math.max(260, $('latency-chart').clientWidth), height = $('latency-chart').clientHeight;
  const left = 47, right = width - 12, top = 12, bottom = height - 28;
  $('latency-chart').setAttribute('viewBox', `0 0 ${width} ${height}`);
  $('chart-empty').setAttribute('x', width / 2); $('chart-empty').setAttribute('y', height / 2);
  const end = latest?.uptime_ms || 0, start = Math.max(0, end - 60000);
  const samples = history.filter(value => value.time >= start);
  const values = samples.flatMap(value => [value.p95, value.p99]).filter(value => value != null);
  const max = Math.max(20, ...values) * 1.12;
  const x = time => left + (time - start) / Math.max(60000, end - start) * (right - left);
  const y = value => bottom - value / max * (bottom - top);
  const ns = 'http://www.w3.org/2000/svg';
  $('chart-grid').replaceChildren();
  for (let i = 0; i <= 3; i++) {
    const value = max * i / 3, yy = y(value);
    const line = document.createElementNS(ns, 'line');
    for (const [key, val] of Object.entries({x1: left, x2: right, y1: yy, y2: yy})) line.setAttribute(key, val);
    const label = document.createElementNS(ns, 'text'); label.setAttribute('x', left - 9); label.setAttribute('y', yy + 3); label.setAttribute('text-anchor', 'end'); label.textContent = number(value);
    $('chart-grid').append(line, label);
  }
  for (let i = 0; i <= 4; i++) {
    const label = document.createElementNS(ns, 'text'); label.setAttribute('x', left + (right - left) * i / 4); label.setAttribute('y', height - 6); label.setAttribute('text-anchor', i === 0 ? 'start' : i === 4 ? 'end' : 'middle');
    label.textContent = `${number((start + i * 15000) / 1000)}s`; $('chart-grid').append(label);
  }
  const path = key => {
    let drawing = false, previousTime = 0;
    return samples.map(value => {
      if (value.time - previousTime > 1500) drawing = false;
      previousTime = value.time;
      if (value[key] == null) { drawing = false; return ''; }
      const command = drawing ? 'L' : 'M'; drawing = true;
      return `${command}${x(value.time).toFixed(2)},${y(value[key]).toFixed(2)}`;
    }).join(' ');
  };
  $('p95-line').setAttribute('d', path('p95')); $('p99-line').setAttribute('d', path('p99'));
  $('slo-line').setAttribute('d', `M${left},${y(15)} L${right},${y(15)}`);
  $('chart-empty').setAttribute('visibility', values.length ? 'hidden' : 'visible');
  $('chart-description').textContent = `Live latency over the last 60 seconds. Current p95 ${number(latest?.p95_ms, 1)} ms; p99 ${number(latest?.p99_ms, 1)} ms.`;
}
function render(state) {
  if (latest && state.uptime_ms < latest.uptime_ms) { history = []; transitions = []; lastPolicy = undefined; }
  latest = state; lastEvent = Date.now(); connection('live');
  if (!pending && Date.now() - changedAt > 1000) { controls = state.controls; renderControls(); }
  $('dataset-shape').textContent = `${number(state.vectors)} vectors × ${number(state.dimension)} dimensions`;
  $('dataset-backends').textContent = state.gpu_available ? `${state.cpu_backend.toUpperCase()} + CUDA · deterministic dataset` : `${state.cpu_backend.toUpperCase()} · CPU-only runtime`;
  for (const [id, key] of Object.entries({arrival: 'arrival_rate', throughput: 'throughput', queue: 'queue_depth', 'gpu-pending': 'pending_gpu_jobs', rejected: 'rejected', failures: 'failed', violations: 'slo_violations', 'change-count': 'policy_changes'})) $(id).textContent = number(state[key]);
  for (const metric of ['p95', 'p99']) { $(metric).textContent = number(state[`${metric}_ms`], 1); $(metric).classList.toggle('over-target', state[`${metric}_ms`] > state.slo_ms); }
  $('slo').textContent = state.slo_ms;
  const total = state.cpu_completed + state.gpu_completed;
  for (const backend of ['cpu', 'gpu']) {
    const count = state[`${backend}_completed`], share = total ? count / total * 100 : 0;
    $(`${backend}-share`).textContent = total ? `${number(share)}%` : '—';
    $(`${backend}-bar`).style.width = `${share}%`;
    $(`${backend}-requests`).textContent = number(count);
    $(`${backend}-util`).textContent = percent(state[`${backend}_utilization`]);
  }
  $('cpu-kind').textContent = state.cpu_backend.toUpperCase(); $('gpu-kind').textContent = state.gpu_available ? 'CUDA' : 'UNAVAILABLE';
  $('batch-size').textContent = number(state.mean_batch_size, 1);
  $('gpu-memory').textContent = state.gpu_memory_free_bytes == null ? 'Unavailable' : `${number(state.gpu_memory_free_bytes / 1073741824, 1)} GiB`;
  $('controller-kind').textContent = state.controller === 'laya' ? 'LOCAL LAYA · 2,000 MS CONTROL' : 'NATIVE HEURISTIC · 500 MS CONTROL';
  const [name, description] = policies[state.policy] || ['Unknown', 'Waiting for a policy decision.'];
  $('policy-name').textContent = name; $('policy-description').textContent = description;
  if (state.policy !== lastPolicy) {
    transitions.unshift({name, time: state.uptime_ms}); transitions = transitions.slice(0, 6); lastPolicy = state.policy;
    $('transitions').replaceChildren(...transitions.map(value => {
      const row = document.createElement('li'), label = document.createElement('span'), time = document.createElement('time');
      label.textContent = value.name; time.textContent = `${Math.floor(value.time / 60000)}:${String(Math.floor(value.time / 1000) % 60).padStart(2, '0')}`;
      row.append(label, time); return row;
    }));
  }
  $('model-status').hidden = state.controller !== 'laya';
  $('confidence').textContent = state.laya.confidence == null ? 'Unavailable' : `${number(state.laya.confidence * 100, 1)}%`;
  const status = state.laya.status.replaceAll('_', ' ');
  $('model-detail').textContent = `${state.laya.fallback ? 'Heuristic fallback' : status === 'accepted' ? 'Model policy applied' : 'Model status'} · ${status}${state.laya.latency_ms == null ? '' : ` · ${number(state.laya.latency_ms)} ms`} · ${number(state.laya.fallbacks)} fallbacks`;
  if (state.contention.error) $('control-feedback').textContent = `GPU workload stopped: ${state.contention.error}`;
  $('runtime-note').textContent = `${number(state.total_completed)} searches completed · whole-system utilization`;
  if (!history.length || history.at(-1).time !== state.uptime_ms) {
    history.push({time: state.uptime_ms, p95: state.p95_ms, p99: state.p99_ms});
    history = history.filter(value => value.time >= state.uptime_ms - 60000).slice(-125);
  }
  chart();
}
let events, reconnectTimer, retryDelay = 1000, pageHidden = false;
function connect() {
  if (pageHidden) return;
  clearTimeout(reconnectTimer);
  events?.close();
  const stream = new EventSource('/events');
  events = stream;
  stream.onmessage = event => {
    if (stream !== events) return;
    retryDelay = 1000;
    try { render(JSON.parse(event.data)); } catch { connection('offline'); }
  };
  stream.onerror = () => {
    if (stream !== events || pageHidden) return;
    connection('offline');
    // Native EventSource retries broken connections, but closes permanently on HTTP errors.
    if (stream.readyState === EventSource.CLOSED) {
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, retryDelay);
      retryDelay = Math.min(10000, retryDelay * 2);
    }
  };
}
connect();
setInterval(() => { if (lastEvent && Date.now() - lastEvent > 2500) connection('offline'); }, 1000);
window.addEventListener('pagehide', () => {
  pageHidden = true; clearTimeout(reconnectTimer); events?.close(); connection('offline');
});
window.addEventListener('pageshow', () => {
  pageHidden = false;
  if (!events || events.readyState === EventSource.CLOSED) connect();
});

async function comparison() {
  try {
    const response = await fetch('/api/comparison'); if (!response.ok) throw new Error();
    const data = await response.json();
    $('comparison-setup').textContent = `${data.hardware} · ${number(data.vectors)} × ${data.dimension} · ${data.date}`;
    const names = {cpu_latency: 'Static CPU (AVX2)', gpu_immediate: 'Static GPU', gpu_batch: 'Static GPU batching', heuristic: 'Heuristic', laya: 'Local Laya (gated)'};
    $('comparison-rows').replaceChildren(...data.modes.map(value => {
      const row = document.createElement('tr'); row.dataset.mode = value.mode;
      for (const text of [names[value.mode], number(value.queries_per_second, 1), `${number(value.p99_ms, 1)} ms`, number(value.rejected), number(value.slo_violations)]) {
        const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
      }
      return row;
    }));
  } catch { $('comparison-setup').textContent = 'Recorded comparison unavailable.'; }
}
comparison();
