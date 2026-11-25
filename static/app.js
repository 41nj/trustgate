'use strict';
// TrustGate front end. All metrics come from the Python engine (/api/metrics); this file only
// renders them. The one client-side computation is the cost-vs-threshold curve (Figure 4), which
// applies the engine's rule (accept if confidence >= t) on a 1-point grid for plotting.

const $ = id => document.getElementById(id);
const state = { rows: [], info: {}, metrics: null, serial: 0, timer: null, liveAvailable: false };

const pct = (x, d = 1) => x == null ? 'N/A' : `${(x * 100).toFixed(d).replace(/\.0+$/, '')}%`;
const pp = x => `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(1)} pp`;
const dec = x => x == null ? 'N/A' : x.toFixed(3);
const num = x => x.toLocaleString('en-US', { maximumFractionDigits: 1 });
const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// 95% Wilson score interval for a proportion; stays inside [0, 1] and behaves well for small n.
function wilson(k, n, z = 1.96) {
  const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

const active = () => { const c = $('category').value; return c ? state.rows.filter(r => r.category === c) : state.rows; };
const settings = () => ({ threshold: +$('threshold').value / 100, wrong_cost: +$('wrong').value, defer_cost: +$('defer').value });

async function request(path, body) {
  const res = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw Error(data.error || 'Request failed');
  return data;
}

function message(text, isError = false) { $('message').textContent = text; $('message').className = isError ? 'error' : ''; }

async function refresh() {
  if (!state.rows.length) return;
  const token = ++state.serial;
  try {
    const m = await request('/api/metrics', { rows: active(), ...settings() });
    if (token !== state.serial) return; // a newer request superseded this one
    state.metrics = m;
    render();
  } catch (e) { if (token === state.serial) message(e.message, true); }
}

function schedule() { labels(); clearTimeout(state.timer); state.timer = setTimeout(refresh, 40); }

function labels() {
  const s = settings();
  $('threshold-out').textContent = pct(s.threshold, 0);
  $('wrong-out').textContent = num(s.wrong_cost);
  $('defer-out').textContent = num(s.defer_cost);
}

// Loading data

function assign(data, kind) {
  state.rows = data.rows;
  state.info = { ...data, kind };
  document.body.classList.remove('no-data');
  const cats = [...new Set(state.rows.map(r => r.category).filter(Boolean))].sort();
  $('category').innerHTML = '<option value="">All categories</option>' + cats.map(c => `<option>${esc(c)}</option>`).join('');
  $('category-wrap').hidden = !cats.length;
  labels();
  return refresh();
}

function showEmpty() {
  state.rows = []; state.metrics = null; state.info = {};
  document.body.classList.add('no-data');
  $('title').textContent = 'TrustGate';
  $('subtitle').textContent = '';
}

async function loadIndex() {
  const status = await request('/api/status');
  state.liveAvailable = status.live_available;
  if (status.default_model) $('model').value = status.default_model;
  for (const id of ['live', 'model', 'count']) $(id).disabled = !state.liveAvailable;
  if (state.liveAvailable) plan();
  else $('live-plan').textContent = 'No API provider configured. Set TRUSTGATE_API_BASE and TRUSTGATE_API_KEY, then restart the server. Saved evaluations remain available offline.';
  const evals = status.evaluations || [];
  if (!evals.length) { $('preset').innerHTML = '<option value="">None saved</option>'; $('preset').disabled = true; return showEmpty(); }
  $('preset').innerHTML = '<option value="" hidden>Current run or import</option>' +
    evals.map(e => `<option value="${esc(e.name)}">${esc(String(e.model).split('/').pop())} · ${esc(String(e.dataset).split('/').pop())} (${e.n})</option>`).join('');
  $('preset').value = evals[0].name;
  return loadPreset();
}

async function loadPreset() {
  try {
    await assign(await request('/api/evaluations/' + encodeURIComponent($('preset').value)), 'real');
    message('');
  } catch (e) { message(e.message, true); }
}

// Running an evaluation

// Ask the server how many answers for this model and subset are already cached.
let planToken = 0;
async function plan() {
  const token = ++planToken;
  try {
    const p = await request('/api/live/plan', { model: $('model').value.trim(), count: +$('count').value });
    if (token !== planToken) return;
    $('live-plan').textContent = p.new
      ? `${p.cached} of ${p.total} answers cached; ${p.new} new requests will be sent.`
      : `All ${p.total} answers are cached; no requests will be sent.`;
    return p;
  } catch (e) { if (token === planToken) $('live-plan').textContent = e.message; }
}

// Figures

const M = { l: 58, r: 22, t: 22, b: 52 };
let W = 640, H = 380, IW = W - M.l - M.r, IH = H - M.t - M.b;

function draw(id, fn) {
  const el = $(id);
  if (!el.offsetParent) return;
  W = Math.max(320, Math.round(el.clientWidth));
  H = Math.round(Math.min(520, Math.max(170, W * (+el.dataset.aspect || 0.6))));
  IW = W - M.l - M.r; IH = H - M.t - M.b;
  el.innerHTML = fn();
}

const ticks = (max, n) => Array.from({ length: n + 1 }, (_, i) => +(max * i / n).toFixed(10));

// Clean scale: step from {1, 2, 5} × 10^k, about four intervals, max rounded up to a whole step.
function scale(value, minimum) {
  const v = Math.max(value, minimum), raw = v / 4, exp = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(f => f * exp).find(s => s >= raw);
  const max = Math.ceil(v / step - 1e-9) * step;
  return { max, ticks: ticks(max, Math.round(max / step)) };
}

function plot({ xMin = 0, xMax = 1, yMax = 1, xTicks = ticks(1, 5), yTicks = ticks(1, 5), xFmt = v => pct(v, 0), yFmt = v => pct(v, 0), xLabel, yLabel }) {
  const x = v => M.l + (v - xMin) / (xMax - xMin) * IW, y = v => M.t + IH - v / yMax * IH;
  let g = '';
  for (const t of yTicks) g += `<line class="grid" x1="${M.l}" x2="${M.l + IW}" y1="${y(t)}" y2="${y(t)}"/><text class="tick" x="${M.l - 8}" y="${y(t) + 4}" text-anchor="end">${yFmt(t)}</text>`;
  for (const t of xTicks) g += `<line class="axis" x1="${x(t)}" x2="${x(t)}" y1="${M.t + IH}" y2="${M.t + IH + 4}"/><text class="tick" x="${x(t)}" y="${M.t + IH + 18}" text-anchor="middle">${xFmt(t)}</text>`;
  g += `<line class="axis" x1="${M.l}" x2="${M.l + IW}" y1="${M.t + IH}" y2="${M.t + IH}"/>`;
  g += `<text class="label" x="${M.l + IW / 2}" y="${H - 8}" text-anchor="middle">${xLabel}</text>`;
  g += `<text class="label" transform="translate(14 ${M.t + IH / 2}) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
  return { x, y, g };
}
const svg = (body, label) => `<svg class="chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(label)}">${body}</svg>`;
const tip = text => `data-tip="${esc(text)}"`;

function rule(p, t, xMin = 0) {
  if (t < xMin) return `<text class="annot" x="${M.l + 6}" y="${M.t + 8}">t = ${pct(t, 0)} (below axis): everything shown is accepted</text>`;
  const X = p.x(t), right = t > xMin + .8 * (1 - xMin);
  return `<line class="rule" x1="${X}" x2="${X}" y1="${M.t - 4}" y2="${M.t + IH}"/><text class="annot" x="${X + (right ? -6 : 6)}" y="${M.t + 8}" text-anchor="${right ? 'end' : 'start'}">t = ${pct(t, 0)}</text>`;
}

// Bar with a small rounded data end and a square base.
function bar(x, yBase, yTop, w, round) {
  const h = yBase - yTop; if (h <= 0) return '';
  const r = round ? Math.min(3, w / 2, h) : 0;
  return `M${x} ${yBase}V${yTop + r}${r ? `Q${x} ${yTop} ${x + r} ${yTop}` : ''}H${x + w - r}${r ? `Q${x + w} ${yTop} ${x + w} ${yTop + r}` : ''}V${yBase}Z`;
}

// Group predictions by exact stated value when there are few (verbalized confidence usually is
// coarse), otherwise by 5-point bins. Distinct stated values are never merged if avoidable.
function confidenceGroups(rows) {
  const exact = new Set(rows.map(r => r.confidence)).size <= 21;
  const groups = new Map();
  for (const r of rows) {
    const key = exact ? r.confidence : Math.min(19, Math.floor(r.confidence * 20));
    const g = groups.get(key) || { key, center: exact ? key : key / 20 + .025, label: exact ? pct(key) : `${key * 5}–${key * 5 + 5}%`, ok: 0, bad: 0 };
    g[r.correct ? 'ok' : 'bad']++;
    groups.set(key, g);
  }
  return { exact, groups: [...groups.values()].sort((a, b) => a.key - b.key) };
}

function histogram(rows, t) {
  const { exact, groups } = confidenceGroups(rows);
  // Start the axis at the first decile below the lowest stated value so the bars are readable.
  const lowest = Math.min(...rows.map(r => r.confidence));
  const xMin = lowest >= .3 ? Math.max(0, Math.floor(lowest * 10) / 10 - .1) : 0;
  const s = scale(Math.max(...groups.map(b => b.ok + b.bad)), 4);
  const steps = Math.round((1 - xMin) * 20);
  const xTicks = Array.from({ length: steps + 1 }, (_, i) => +(xMin + i * .05).toFixed(2)).filter((v, i) => steps <= 10 || i % 2 === 0);
  const p = plot({ xMin, yMax: s.max, yTicks: s.ticks, yFmt: num, xTicks, xLabel: exact ? 'Stated confidence' : 'Stated confidence (5-point bins)', yLabel: 'Predictions' });
  let g = p.g;
  const bw = Math.min(exact ? 22 : 24, IW * .05 / (1 - xMin) * .6);
  for (const b of groups) {
    const x = Math.min(p.x(b.center) - bw / 2, M.l + IW - bw), base = p.y(0), mid = p.y(b.ok), top = p.y(b.ok + b.bad);
    if (b.ok) g += `<path fill="var(--accent)" d="${bar(x, base, mid, bw, !b.bad)}"/>`;
    if (b.bad) g += `<path fill="var(--wrong)" d="${bar(x, b.ok ? mid - 1.5 : base, top, bw, true)}"/>`;
    g += `<rect x="${x - 6}" y="${M.t}" width="${bw + 12}" height="${IH}" fill="transparent" ${tip(`${b.label}: ${b.ok + b.bad} predictions\n${b.ok} correct, ${b.bad} wrong`)}/>`;
  }
  const mode = groups.reduce((a, b) => (b.ok + b.bad > a.ok + a.bad ? b : a));
  const mx = Math.min(p.x(mode.center), M.l + IW - bw / 2) - bw / 2 - 10, my = Math.max(p.y(mode.ok + mode.bad) + 12, M.t + 30);
  g += `<text class="annot" x="${mx}" y="${my}" text-anchor="end">${mode.ok + mode.bad} at ${mode.label}</text>`;
  if (mode.bad) g += `<text class="annot-muted" x="${mx}" y="${my + 16}" text-anchor="end">${mode.bad} of them wrong</text>`;
  g += rule(p, t, xMin);
  const note = xMin > 0 ? `<span>Axis starts at ${pct(xMin, 0)}; no predictions below ${pct(lowest)}.</span>` : '';
  return svg(g, 'Confidence distribution') +
    `<div class="legend"><span><i style="background:var(--accent)"></i>Correct</span><span><i style="background:var(--wrong)"></i>Wrong</span>${note}</div>`;
}

function reliability(m) {
  const p = plot({ xLabel: 'Mean stated confidence in bin', yLabel: 'Observed accuracy' });
  let g = p.g + `<line class="ref" x1="${p.x(0)}" y1="${p.y(0)}" x2="${p.x(1)}" y2="${p.y(1)}"/>`;
  const angle = -Math.atan2(IH, IW) * 180 / Math.PI;
  g += `<text class="annot-muted" transform="translate(${p.x(.3)} ${p.y(.3) - 7}) rotate(${angle})">perfect calibration</text>`;
  for (const b of m.bins) {
    if (!b.count) continue;
    const cx = p.x(b.confidence), cy = p.y(b.accuracy), r = Math.max(4, Math.sqrt(b.count / m.n) * 16);
    g += `<line stroke="var(--ink-2)" stroke-width="1" x1="${cx}" x2="${cx}" y1="${p.y(b.confidence)}" y2="${cy}"/>`;
    g += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="var(--accent)" fill-opacity=".85" stroke="var(--paper)" stroke-width="2"/>`;
    g += `<circle cx="${cx}" cy="${cy}" r="${Math.max(13, r + 4)}" fill="transparent" ${tip(`Bin ${pct(b.low, 0)}–${pct(b.high, 0)}, n = ${b.count}\nmean confidence ${pct(b.confidence)}\naccuracy ${pct(b.accuracy)}\ngap ${pp(b.accuracy - b.confidence)}`)}/>`;
    if (b.count / m.n >= .2) g += `<text class="annot" x="${cx}" y="${cy + r + 16}" text-anchor="middle">n = ${b.count}</text>`;
  }
  return svg(g, 'Reliability diagram');
}

function riskCoverage(m) {
  const pts = m.curve.filter(c => c.risk !== null);
  const s = scale(Math.max(0, ...pts.map(c => c.risk)), .1);
  const p = plot({ yMax: s.max, yTicks: s.ticks, xLabel: 'Coverage (share of questions answered)', yLabel: 'Selective risk (error rate)' });
  let g = p.g;
  if (pts.length > 1) g += `<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" points="${pts.map(c => `${p.x(c.coverage)},${p.y(c.risk)}`).join(' ')}"/>`;
  for (const c of pts) {
    g += `<circle cx="${p.x(c.coverage)}" cy="${p.y(c.risk)}" r="${pts.length > 60 ? 0 : 3.5}" fill="var(--accent)" stroke="var(--paper)" stroke-width="1.5"/>`;
    g += `<circle cx="${p.x(c.coverage)}" cy="${p.y(c.risk)}" r="11" fill="transparent" ${tip(`t ≥ ${pct(c.threshold)}\ncoverage ${pct(c.coverage)} (${c.accepted} answered)\nrisk ${pct(c.risk)} (${c.errors} wrong)\ncost ${num(c.cost)}`)}/>`;
  }
  const sel = m.selected, t = settings().threshold;
  if (sel.risk !== null) {
    const right = sel.coverage > .75;
    g += `<circle cx="${p.x(sel.coverage)}" cy="${p.y(sel.risk)}" r="7" fill="none" stroke="var(--ink)" stroke-width="1.25"/>`;
    g += `<text class="annot" x="${p.x(sel.coverage) + (right ? -12 : 12)}" y="${p.y(sel.risk) - 10}" text-anchor="${right ? 'end' : 'start'}">t = ${pct(t, 0)}: ${pct(sel.risk)} risk</text>`;
  } else g += `<text class="empty" x="${M.l + IW / 2}" y="${M.t + 18}" text-anchor="middle">No questions answered at t = ${pct(t, 0)}</text>`;
  if (pts.length === 1) g += `<text class="empty" x="${M.l + IW / 2}" y="${M.t + IH / 2}" text-anchor="middle">One distinct confidence value: a single non-empty policy</text>`;
  return svg(g, 'Risk–coverage curve');
}

function costCurve(rows) {
  const { wrong_cost: wc, defer_cost: dc, threshold } = settings();
  const series = Array.from({ length: 101 }, (_, i) => {
    const t = i / 100; let cost = 0, acc = 0;
    for (const r of rows) if (r.confidence >= t) { acc++; if (!r.correct) cost += wc; } else cost += dc;
    return { t, cost, acc };
  });
  const s = scale(Math.max(...series.map(x => x.cost)), 1);
  const p = plot({ yMax: s.max, yTicks: s.ticks, yFmt: num, xLabel: 'Threshold t', yLabel: 'Total cost' });
  let g = p.g + `<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" points="${series.map(x => `${p.x(x.t)},${p.y(x.cost)}`).join(' ')}"/>`;
  const best = series.reduce((a, b) => (b.cost < a.cost ? b : a));
  g += `<circle cx="${p.x(best.t)}" cy="${p.y(best.cost)}" r="4" fill="var(--ink)" stroke="var(--paper)" stroke-width="1.5"/>`;
  g += `<text class="annot" x="${p.x(best.t) - 8}" y="${p.y(best.cost) + 18}" text-anchor="end">minimum ${num(best.cost)}</text>`;
  g += rule(p, threshold);
  const step = IW / 101;
  for (const x of series) g += `<rect x="${p.x(x.t) - step / 2}" y="${M.t}" width="${step}" height="${IH}" fill="transparent" ${tip(`t = ${pct(x.t, 0)}\n${x.acc} answered, ${rows.length - x.acc} deferred\ncost ${num(x.cost)}`)}/>`;
  return svg(g, 'Policy cost versus threshold');
}

function holdoutChart(h, wrongTest) {
  const items = [
    { name: `Fitted threshold (${h.threshold === null ? 'defer all' : '≥ ' + pct(h.threshold)})`, cost: h.test.cost, fill: 'var(--accent)', detail: `${h.test.accepted} answered, ${h.test.errors} wrong` },
    { name: 'Answer all', cost: h.test_answer_all_cost, fill: 'var(--neutral)', detail: `${h.n_test} answered, ${wrongTest} wrong` },
    { name: 'Defer all', cost: h.test_defer_all_cost, fill: 'var(--neutral)', detail: `${h.n_test} deferred` },
  ];
  const left = 190, s = scale(Math.max(...items.map(i => i.cost)), 1), x = v => left + v / s.max * (W - left - M.r - 40);
  const rowH = 38, top = 8;
  H = top + items.length * rowH + 44;
  let g = '';
  for (const t of s.ticks) g += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${top + items.length * rowH}"/><text class="tick" x="${x(t)}" y="${top + items.length * rowH + 18}" text-anchor="middle">${num(t)}</text>`;
  items.forEach((it, i) => {
    const y = top + i * rowH + 9, bh = 20;
    g += `<text class="label" x="${left - 12}" y="${y + 15}" text-anchor="end">${esc(it.name)}</text>`;
    if (it.cost > 0) g += `<path fill="${it.fill}" d="M${x(0)} ${y}H${x(it.cost) - 3}Q${x(it.cost)} ${y} ${x(it.cost)} ${y + 3}V${y + bh - 3}Q${x(it.cost)} ${y + bh} ${x(it.cost) - 3} ${y + bh}H${x(0)}Z"/>`;
    g += `<text class="annot" x="${x(it.cost) + 8}" y="${y + 15}">${num(it.cost)}</text>`;
    g += `<rect x="0" y="${y - 6}" width="${W}" height="${bh + 12}" fill="transparent" ${tip(`${it.name}\ncost ${num(it.cost)} on ${h.n_test} test questions\n${it.detail}`)}/>`;
  });
  g += `<line class="axis" x1="${x(0)}" x2="${x(0)}" y1="${top}" y2="${top + items.length * rowH}"/>`;
  g += `<text class="label" x="${x(s.max / 2)}" y="${H - 6}" text-anchor="middle">Cost on the test half (lower is better)</text>`;
  return svg(g, 'Held-out policy cost');
}

// Text and tables

function lede(m) {
  const { kind, run, dataset, counts, source } = state.info, cat = $('category').value;
  $('title').textContent = run?.model
    ? `${run.model} on ${dataset?.dataset || 'an imported dataset'}${cat ? `: ${cat}` : ''}`
    : `${source || 'Imported results'}${cat ? `: ${cat}` : ''}`;
  $('subtitle').textContent = kind === 'real'
    ? `${m.n} multiple-choice questions${cat ? ' in this category' : ''}, answered by the model and graded by exact match against the answer key. Confidence is the model's own stated probability of being correct.`
    : `${m.n} imported prediction records. Correctness labels are taken as supplied and have not been verified.`;
  const distinct = new Set(active().map(r => r.confidence)).size;
  const correct = Math.round(m.accuracy * m.n), [lo, hi] = wilson(correct, m.n);
  const items = [
    ['Accuracy', pct(m.accuracy), `${correct}/${m.n} correct · 95% CI ${pct(lo)}–${pct(hi)}`],
    ['Mean stated confidence', pct(m.mean_confidence), `${pp(m.mean_confidence - m.accuracy)} relative to accuracy`],
    ['Calibration error (ECE)', dec(m.ece), '10 equal-width bins · lower is better'],
    ['AUROC', dec(m.auroc), m.auroc == null ? 'undefined: only one outcome class' : distinct === 1 ? 'constant confidence: no ranking' : 'ranking of right vs. wrong · 0.5 = chance'],
  ];
  $('headline').innerHTML = items.map(([k, v, sub]) => `<div><dt>${k}</dt><dd class="${v === 'N/A' ? 'na' : ''}">${v}</dd><small>${esc(sub)}</small></div>`).join('');

  // Values are HTML: everything from the data is escaped here.
  const code = v => `<span class="code">${esc(v)}</span>`, d = [];
  if (dataset?.dataset) d.push(['Dataset', esc(`${dataset.dataset}${dataset.split ? `, ${dataset.split} split` : ''}`)]);
  if (dataset?.revision) d.push(['Revision', code(dataset.revision)]);
  if (dataset?.license) d.push(['License', esc(dataset.license)]);
  if (dataset?.sampling) d.push(['Sampling', esc(dataset.sampling)]);
  if (run?.model) d.push(['Model', code(run.model)]);
  if (run) d.push(['Decoding', esc(`temperature ${run.temperature}, max_tokens ${run.max_tokens}, prompt ${run.prompt_version}`)]);
  if (run?.started) d.push(['Run started', esc(run.started.replace('T', ' '))]);
  if (counts) d.push(['Responses', esc(`${counts.questions} asked, ${counts.valid} parsed, ${counts.unparseable} unparseable, ${counts.failed} failed. Unparseable and failed items are excluded from all metrics and listed in the run file.`)]);
  if (counts?.cached !== undefined) d.push(['This run', esc(`${counts.cached} answers reused from the local cache, ${counts.new} newly generated.`)]);
  if (kind !== 'real') d.push(['Source', esc(source || 'imported file')]);
  $('details-list').innerHTML = d.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function distribution() {
  const rows = active(), { exact, groups } = confidenceGroups(rows), distinct = new Set(rows.map(r => r.confidence));
  const mode = groups.reduce((a, b) => (b.ok + b.bad > a.ok + a.bad ? b : a)), modeN = mode.ok + mode.bad;
  const rest = rows.filter(r => (exact ? r.confidence !== mode.key : Math.min(19, Math.floor(r.confidence * 20)) !== mode.key));
  let text;
  if (distinct.size === 1) text = `Every answer states the same confidence (${pct([...distinct][0])}). Confidence carries no information about which answers are wrong.`;
  else {
    text = `The model used ${distinct.size} distinct confidence values${exact ? '' : ' (shown in 5-point bins)'}. The most frequent, ${mode.label}, accounts for ${modeN} of ${rows.length} answers (${pct(modeN / rows.length, 0)}), of which ${mode.bad} ${mode.bad === 1 ? 'is' : 'are'} wrong.`;
    if (rest.length) text += ` The remaining ${rest.length} answers are correct ${pct(rest.filter(r => r.correct).length / rest.length, 0)} of the time.`;
  }
  $('dist-text').textContent = text;
  $('conf-table').tBodies[0].innerHTML = groups.slice().reverse().map(b =>
    `<tr><td>${b.label}</td><td class="r">${b.ok + b.bad}</td><td class="r">${b.bad}</td><td class="r">${pct(b.ok / (b.ok + b.bad), 0)}</td></tr>`).join('');
}

function calibration(m) {
  const gap = m.mean_confidence - m.accuracy;
  $('cal-stats').innerHTML = [['ECE', dec(m.ece)], ['Brier score', dec(m.brier)], ['Mean gap', pp(gap)]]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
  $('cal-text').textContent = Math.abs(gap) < .02
    ? 'On average, stated confidence matches accuracy. Individual bins may still deviate.'
    : `The model is ${gap > 0 ? 'overconfident' : 'underconfident'}: its average stated confidence is ${Math.abs(gap * 100).toFixed(1)} percentage points ${gap > 0 ? 'above' : 'below'} its accuracy. Bins with few predictions are noisy.`;
  $('bins-table').tBodies[0].innerHTML = m.bins.filter(b => b.count).reverse().map(b =>
    `<tr><td>${pct(b.low, 0)}–${pct(b.high, 0)}</td><td class="r">${b.count}</td><td class="r">${pct(b.confidence)}</td><td class="r">${pct(b.accuracy)}</td><td class="r">${pp(b.accuracy - b.confidence)}</td></tr>`).join('');
}

function selective(m) {
  const s = m.selected, t = settings().threshold, none = m.curve[0], all = m.curve[m.curve.length - 1];
  const distinct = new Set(active().map(r => r.confidence)).size;
  let text = s.accepted
    ? `At t = ${pct(t, 0)} the policy answers ${s.accepted} of ${m.n} questions (${pct(s.coverage)}) with ${pct(1 - s.risk)} selective accuracy and a cost of ${num(s.cost)}, compared with ${num(all.cost)} for answering everything and ${num(none.cost)} for deferring everything.`
    : `At t = ${pct(t, 0)} no question is answered; the cost equals deferring everything (${num(s.cost)}).`;
  text += m.auroc == null
    ? ' AUROC is undefined because all answers in this view are correct or all are wrong.'
    : distinct === 1
      ? ' All answers share one confidence value, so AUROC is exactly 0.5 and no threshold can separate right from wrong answers.'
      : ` AUROC = ${dec(m.auroc)}: a randomly chosen correct answer receives higher stated confidence than a randomly chosen wrong one in ${pct(m.auroc, 0)} of pairs (ties count half).`;
  $('sel-text').textContent = text;
  $('theory').textContent = pct(m.theoretical_threshold, 0);
  const rule = th => th === null ? '—' : `≥ ${pct(th)}`;
  const row = (name, th, p, emph) => `<tr class="${emph ? 'emph' : ''}"><td>${name}</td><td class="r">${th}</td><td class="r">${pct(p.coverage)}</td><td class="r">${p.errors}</td><td class="r">${p.risk == null ? 'N/A' : pct(p.risk)}</td><td class="r">${num(p.cost)}</td></tr>`;
  $('policy-table').tBodies[0].innerHTML = row('Answer all', '≥ 0%', all) + row('Defer all', '—', none) +
    row('Current threshold', `≥ ${pct(t, 0)}`, s, true) + row('Retrospective optimum', rule(m.best.threshold), m.best);
}

function holdout(m) {
  const h = m.holdout;
  $('holdout-body').hidden = !h; $('holdout-missing').hidden = !!h;
  if (!h) return;
  const { wrong_cost: wc, defer_cost: dc } = settings();
  const wrongTest = active().filter(r => r.split === 'test' && !r.correct).length;
  const rule = h.threshold === null ? 'defer all' : `≥ ${pct(h.threshold)}`;
  $('holdout-steps').innerHTML = [
    `<b>Fit.</b> On the calibration half (n = ${h.n_calibration}), the cheapest policy is <b>${rule}</b>${h.calibration.risk == null ? '' : `, with ${pct(h.calibration.risk)} selective risk`} and cost ${num(h.calibration.cost)}.`,
    `<b>Apply.</b> The same rule on the unseen test half (n = ${h.n_test}) answers ${pct(h.test.coverage)} of questions${h.test.risk == null ? '' : ` with ${pct(h.test.risk)} selective risk`} and costs ${num(h.test.cost)}.`,
    `<b>Compare</b> against the two fixed policies on the same test questions.`,
  ].map(x => `<li>${x}</li>`).join('');
  const fitted = h.test.cost, all = h.test_answer_all_cost, none = h.test_defer_all_cost;
  let verdict;
  if (h.threshold === null) verdict = `The calibration half favours deferring everything. On unseen questions that costs ${num(none)}, against ${num(all)} for answering everything.`;
  else if (fitted < Math.min(all, none)) verdict = `On unseen questions the fitted threshold is the cheapest policy (${num(fitted)} vs. ${num(all)} for answering and ${num(none)} for deferring everything).`;
  else if (fitted === Math.min(all, none)) verdict = `On unseen questions the fitted threshold costs the same as ${fitted === none ? 'deferring' : 'answering'} everything (${num(fitted)}).`;
  else verdict = `On unseen questions ${none < all ? 'deferring' : 'answering'} everything is cheaper (${num(Math.min(all, none))}) than the fitted threshold (${num(fitted)}): at these costs, the threshold learned on calibration data does not pay off.`;
  $('holdout-verdict').textContent = verdict;
  $('holdout-caption').innerHTML = `<b>Figure 5.</b> Cost of three policies on the ${h.n_test} test questions with <i>C</i><sub>w</sub> = ${num(wc)} and <i>C</i><sub>d</sub> = ${num(dc)}. The threshold was chosen on the calibration half only. With n = ${h.n_test}, differences of a few errors are within noise.`;
  draw('holdout-chart', () => holdoutChart(h, wrongTest));
  const row = (name, data, th, p, emph) => `<tr class="${emph ? 'emph' : ''}"><td>${name}</td><td>${data}</td><td class="r">${th}</td><td class="r">${pct(p.coverage)}</td><td class="r">${p.errors}</td><td class="r">${p.risk == null ? 'N/A' : pct(p.risk)}</td><td class="r">${num(p.cost)}</td></tr>`;
  $('holdout-table').tBodies[0].innerHTML =
    row('Fitted threshold', `calibration (n = ${h.n_calibration})`, rule, h.calibration) +
    row('Fitted threshold', `test (n = ${h.n_test})`, rule, h.test, true) +
    row('Answer all', `test (n = ${h.n_test})`, '≥ 0%', { coverage: 1, errors: wrongTest, risk: wrongTest / h.n_test, cost: h.test_answer_all_cost }) +
    row('Defer all', `test (n = ${h.n_test})`, '—', { coverage: 0, errors: 0, risk: null, cost: h.test_defer_all_cost });
}

function predictions() {
  const t = settings().threshold, f = $('filter').value;
  const keep = r => f === 'all' || (f === 'wrong' && !r.correct) || (f === 'accepted' && r.confidence >= t) || (f === 'deferred' && r.confidence < t) || (f === 'wrong-accepted' && !r.correct && r.confidence >= t) || (f === r.origin);
  const list = active().filter(keep).sort((a, b) => b.confidence - a.confidence || a.correct - b.correct);
  const opt = (r, k) => {
    const i = typeof k === 'string' && /^[A-J]$/.test(k) ? k.charCodeAt(0) - 65 : -1;
    return r.options && r.options[i] !== undefined ? `<span class="clamp"><span class="letter">${k}</span>${esc(r.options[i])}</span>` : esc(k ?? '—');
  };
  const hasOrigin = active().some(r => r.origin);
  for (const o of ['new', 'cached']) $('filter').querySelector(`[value="${o}"]`).hidden = !hasOrigin;
  if (!hasOrigin && (f === 'new' || f === 'cached')) $('filter').value = 'all';
  $('pred-count').textContent = `${list.length} of ${active().length} shown`;
  $('pred-table').tBodies[0].innerHTML = list.map(r => {
    const accepted = r.confidence >= t;
    return `<tr><td>${esc(r.question_id ?? r.id ?? '')}</td>
      <td>${r.category ? `<span class="cat">${esc(r.category)}${r.split ? ` · ${esc(r.split)}` : ''}${r.origin ? ` · ${r.origin === 'new' ? 'newly generated' : 'cached'}` : ''}</span>` : ''}<span class="clamp" title="${esc(r.question)}">${esc(r.question ?? '—')}</span></td>
      <td>${opt(r, r.prediction)}</td><td>${r.prediction === r.answer ? '<span class="letter">' + esc(r.answer) + '</span><span class="deferred">same</span>' : opt(r, r.answer)}</td>
      <td class="r">${pct(r.confidence, 0)}</td>
      <td><span class="mark ${r.correct ? 'ok' : 'bad'}"></span>${r.correct ? 'correct' : 'wrong'}</td>
      <td class="${accepted ? '' : 'deferred'}">${accepted ? 'answer' : 'defer'}</td></tr>`;
  }).join('') || '<tr><td colspan="7" class="muted">No predictions match this filter.</td></tr>';
}

function drawFigures() {
  const m = state.metrics, rows = active(), t = settings().threshold;
  if (!m) return;
  draw('hist', () => histogram(rows, t));
  draw('rel', () => reliability(m));
  draw('rc', () => riskCoverage(m));
  draw('cost-curve', () => costCurve(rows));
}

function render() {
  const m = state.metrics;
  lede(m);
  distribution();
  calibration(m);
  selective(m);
  holdout(m);
  predictions();
  drawFigures();
}

// Interaction

document.addEventListener('pointermove', e => {
  const el = e.target.closest && e.target.closest('[data-tip]'), box = $('tip');
  if (!el) { box.hidden = true; return; }
  box.textContent = el.getAttribute('data-tip');
  box.hidden = false;
  box.style.left = Math.min(e.clientX + 14, innerWidth - box.offsetWidth - 8) + 'px';
  box.style.top = Math.min(e.clientY + 14, innerHeight - box.offsetHeight - 8) + 'px';
});
let resizeTimer;
addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawFigures(); if (state.metrics) holdout(state.metrics); }, 120); });

for (const id of ['threshold', 'wrong', 'defer']) $(id).addEventListener('input', schedule);
$('filter').addEventListener('change', predictions);
$('category').addEventListener('change', () => { message(''); refresh(); });
$('preset').addEventListener('change', loadPreset);
$('reset').addEventListener('click', () => { $('threshold').value = 80; $('wrong').value = 10; $('defer').value = 1; schedule(); });
$('apply-theory').addEventListener('click', () => { if (state.metrics) { $('threshold').value = Math.ceil(state.metrics.theoretical_threshold * 100 - 1e-9); schedule(); } });

for (const id of ['model', 'count']) $(id).addEventListener('input', () => { clearTimeout(state.planDelay); state.planDelay = setTimeout(plan, 300); });

$('live').addEventListener('click', async () => {
  const p = await plan(); if (!p) return;
  if (p.new && !confirm(`Send ${p.new} new requests to the configured provider? ${p.cached} cached answers will be reused. Paid providers may charge.`)) return;
  for (const id of ['live', 'import', 'preset']) $(id).disabled = true;
  message(p.new ? `Generating ${p.new} new answers…` : 'Loading cached answers…');
  try {
    const data = await request('/api/live', { model: $('model').value.trim(), count: +$('count').value });
    await assign({ ...data, source: `Run: ${data.run.model}` }, 'real');
    $('preset').value = '';
    message(`${data.counts.questions} questions: ${data.counts.cached} from cache, ${data.counts.new} newly generated. Saved under results/.`);
    $('live').closest('details').open = false;
    plan();
  } catch (e) { message(e.message, true); }
  finally { $('live').disabled = !state.liveAvailable; $('import').disabled = false; $('preset').disabled = $('preset').options.length < 2; }
});

$('import').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', async () => {
  try {
    const file = $('file').files[0]; if (!file) return;
    if (file.size > 5e6) throw Error('File too large.');
    const data = JSON.parse(await file.text());
    const rows = Array.isArray(data) ? data : data.rows;
    await request('/api/metrics', { rows, ...settings() }); // validate before switching
    await assign({ ...(Array.isArray(data) ? {} : data), rows, source: `Imported: ${file.name}` }, 'imported');
    $('preset').value = '';
    message(`Imported ${rows.length} records from ${file.name}.`);
  } catch (e) { message(e.message, true); }
  finally { $('file').value = ''; }
});

$('export').addEventListener('click', async () => {
  if (!state.rows.length) return;
  await refresh(); if (!state.metrics) return;
  const { source, kind, dataset, run, counts } = state.info;
  const blob = new Blob([JSON.stringify({ source, kind, dataset, run, counts, rows: state.rows, category: $('category').value || null, policy: settings(), metrics: state.metrics }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = 'trustgate-results.json'; a.click(); URL.revokeObjectURL(url);
});

loadIndex().catch(e => { showEmpty(); message(e.message, true); });
