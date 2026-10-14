/* Model Quantization Planner — zero-dependency client app (UI, import flow, tooltips, URL state). */
import * as Q from './quant.js';
import {
  importHF, importFiles, importUrl, importConfigText, importManual, listLocalModels, importLocalServer,
  setToken, net, isAbort, UserError, fmtBytes, fmtCount, ensureTensors, readTensorSample, planSample,
  scanEstimate, listQuantized, repoSizes, Budget, DTYPE_BITS, MAX_SAMPLE,
} from './import.js';
import { READABLE_DTYPES } from './parsers.js';
import { drawHist, drawStair, drawErr, drawClip, drawCompare, drawHeat, chartPNG, reportPNG, fmtNum } from './charts.js';

/* =====================================================================
 * Helpers & state
 * ===================================================================== */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtGB = (g) => (g >= 100 ? g.toFixed(0) : g >= 10 ? g.toFixed(1) : g.toFixed(2)) + ' GB';
const fmtDb = (s) => (Number.isFinite(s) ? s.toFixed(1) : s > 0 ? '∞' : '−∞');
const fmtBits = (b) => String(+b.toFixed(3));
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const pct = (x, d = 1) => (x >= 99 && x < 100 ? (Math.floor(x * 10) / 10).toFixed(1) : x >= 10 || x === 0 ? x.toFixed(0) : x.toFixed(d)) + '%';
const debounce = (fn, ms) => { let t = 0; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

const S = {
  profile: null,
  wsrc: 'gen',
  gen: { dist: 'laplace', sigma: 0.02, count: 65536, outlierPct: 0.02, outlierMul: 15, seed: 1 },
  cfg: { ...Q.DEFAULT_CFG },
  real: null,          // { key, tensor, cols, rows, n, bytes, blocks }
  result: null,        // last worker result
  resultCfg: null,
  sizeOpts: { headroom: 0.85, keepEmb: true },
  readyMade: null,
  scan: null,          // { rows, perTensor, bytes, plan }
  localListing: null,
  log: { hist: true, err: true },
};
let busy = null;       // AbortController of the running import / load / scan

/* =====================================================================
 * Hover help (tooltips) — plain-English explanations for everything
 * ===================================================================== */
const TIPS = {
  params: 'The total number of weights (numbers) in the model.\nMore parameters = more memory. Memory = parameters × bits per weight ÷ 8.',
  active: 'Mixture-of-experts models only use some of their experts for each word (token).\nActive parameters set the speed; TOTAL parameters set the memory you need, because all experts must be loaded.',
  layers: 'The number of transformer blocks stacked on top of each other.',
  hidden: 'The width of the model: how many numbers describe each token inside the model (hidden_size).',
  context: 'The longest input + output (in tokens) the model was built for. Longer contexts need more memory for the KV cache, which is not included in the size table.',
  dtype: 'The number format the original weights are stored in. Most new models use BF16 (16 bits per weight).',
  origSize: 'How much memory the weights take in their original format.\nparameters × bits ÷ 8.',
  quant: 'Whether this checkpoint is already quantized (compressed), and how.',
  arch: 'The model family / code class, from config.json.',
  vocab: 'How many different tokens (word pieces) the model knows. The embedding table has vocab × hidden weights.',
  source: 'Where the parameter count came from. “Exact” values were counted from the real weight files; “estimated” values were computed from config.json.',
  headroom: 'How much of each GPU\'s memory you allow the weights to use. The rest is left for the KV cache, activations and the framework.\n85% is a reasonable default for short prompts; use less for long contexts or many users at once.',
  keepEmb: 'GPTQ, AWQ, bitsandbytes and most INT4 / INT8 tools leave the token embeddings and the output layer in 16-bit, because they are sensitive and only a small part of big models.\nTurn off to see the size if everything were quantized.',
  bitsCol: 'Effective bits per weight: the bits of each weight plus its share of the scales (and zero-points) stored with it.\nExample: INT4 with one FP16 scale per 128 weights = 4 + 16 ÷ 128 = 4.125 bits.',
  sizeCol: 'Memory needed for the weights in this format (GB = 10⁹ bytes).\nsize = parameters × bits per weight ÷ 8.',
  savedCol: 'How much smaller than the original format.',
  fitCol: 'Does the model fit on one GPU of this size?\n✓ = yes, within the usable share you set.\n×N = you would need N of these GPUs (the model split across them).\nKV cache and activations need extra memory on top.',
  sqnr: 'SQNR (signal-to-quantization-noise ratio): how big the weights are compared with the rounding error, in decibels (dB).\nHigher is better. Every extra bit adds about 6 dB.\n\nRough guide: above 30 dB = excellent (INT8 / FP8 are usually 31–41 dB), 20–30 = good, 12–20 = moderate — typical 4-bit (INT4 group 128 measured 15–19 dB on real Qwen3-8B layers), below 12 = heavy damage.',
  mse: 'Mean squared error: the average of (original − quantized)² over all weights.',
  rmse: 'Root mean squared error: the typical size of the rounding error, in the same units as the weights.\nCompare it with the spread (σ) of the weights.',
  maxErr: 'The single largest error of any weight. Large values usually come from clipping.',
  cosine: 'Cosine similarity between the original and quantized weights (as one long vector).\n1 = pointing exactly the same way. 0.999 is typical for good 4-bit.',
  clipped: 'Share of weights outside the clip range, which were cut to the edge value.',
  zeroed: 'Share of non-zero weights that became exactly 0 after quantization.\nHigh values mean the steps are too coarse for the many small weights.',
  levels: 'How many of the format\'s possible values actually appear, and how crowded the busiest ones are.\nIf most weights share 2–3 values, a lot of information is lost.',
  bits: 'Effective bits per weight, including the scales (and zero-points) shared by each group.',
  groups: 'How many separate scales were used (one per group, row or tensor).',
  std: 'Standard deviation (σ): the typical size of a weight.',
  amax: 'The largest weight by absolute value. Symmetric quantization sets the scale from this number (absmax), so one huge weight makes the steps coarse for everyone.',
  outlierRatio: 'Largest |weight| ÷ the 99.9th percentile of |weight|.\nAbout 1–2 = no real outliers. Above 3 = a few weights are much bigger than the rest and will stretch the scale.',
  kurtosis: 'How heavy the tails are. Normal (Gaussian) = 3, Laplace = 6. Higher = more extreme values.',
  insights: 'Automatic observations from these results, in plain English. They are hints worth testing, not guarantees.',
  scanSample: 'How many weights to read from every tensor (as 4 evenly spaced blocks of rows). More = more accurate ranking, more data to read.',
  scanEmb: 'Leave the embedding and output layers out of the INT8 / INT4 plan (they stay 16-bit), like most quantization tools do.',
  planN: 'How many of the most sensitive tensors to keep in INT8. Everything else becomes INT4 (group 128).\nMore INT8 = better quality, bigger model.',
};
const tip = (key) => ` data-tip="${esc(TIPS[key] ?? key)}"`;
const tipText = (s) => ` data-tip="${esc(s)}"`;
const qi = '<i class="tip-i" aria-hidden="true">?</i>';

const tipEl = Object.assign(document.createElement('div'), { className: 'tooltip' });
tipEl.setAttribute('role', 'tooltip');
document.body.appendChild(tipEl);
let tipTarget = null, tipTimer = 0, mouse = { x: 0, y: 0 };
function hideTip() { clearTimeout(tipTimer); tipEl.classList.remove('show'); }
function placeTip() {
  const r = tipEl.getBoundingClientRect(), gap = 16;
  let x = mouse.x + gap, y = mouse.y + gap;
  if (x + r.width > innerWidth - 8) x = mouse.x - r.width - gap;
  if (y + r.height > innerHeight - 8) y = mouse.y - r.height - gap;
  tipEl.style.left = Math.max(8, x) + 'px';
  tipEl.style.top = Math.max(8, y) + 'px';
}
document.addEventListener('mousemove', (e) => {
  mouse = { x: e.clientX, y: e.clientY };
  const t = e.target.closest?.('[data-tip]') || null;
  if (t !== tipTarget) {
    tipTarget = t;
    hideTip();
    if (t) tipTimer = setTimeout(() => {
      if (tipTarget !== t || !t.isConnected) return;
      tipEl.textContent = t.dataset.tip;
      tipEl.classList.add('show');
      placeTip();
    }, 350);
  } else if (tipEl.classList.contains('show')) placeTip();
});
for (const ev of ['pointerdown', 'wheel', 'keydown', 'scroll']) document.addEventListener(ev, () => { tipTarget = null; hideTip(); }, { capture: true, passive: true });
document.documentElement.addEventListener('mouseleave', () => { tipTarget = null; hideTip(); });

/* =====================================================================
 * Worker
 * ===================================================================== */
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let msgId = 0;
const pending = new Map();
worker.onmessage = (e) => {
  const m = e.data, p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  m.type === 'error' ? p.reject(new Error(m.message)) : p.resolve(m);
};
worker.onerror = (e) => { console.error('Worker error', e); toast('Background worker failed — reload the page.'); };
function call(msg, transfer = []) {
  const id = ++msgId;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ ...msg, id }, transfer); });
}

/* =====================================================================
 * Import panel
 * ===================================================================== */
const srcVal = () => $('input[name=src]:checked').value;
function showSource(v) {
  $(`input[name=src][value=${v}]`).checked = true;
  $$('.src-body').forEach((b) => { b.hidden = b.dataset.for !== v; });
  $('#btn-import').textContent = v === 'local' && !$('#local-model').value ? 'Connect & list models' : 'Import';
}
$('#sources').addEventListener('change', () => showSource(srcVal()));

const EXAMPLES = [
  ['Qwen/Qwen2.5-0.5B-Instruct', 'A small model (0.5B) — quick to scan every layer.'],
  ['Qwen/Qwen2.5-7B-Instruct', 'A popular 7B model, 4 weight shards.'],
  ['mistralai/Mixtral-8x7B-v0.1', 'A mixture-of-experts model: 46.7B total, 12.9B active parameters.'],
  ['meta-llama/Llama-3.1-70B-Instruct', 'A gated model: the parameter count works without a token; config and weights need one.'],
  ['TheBloke/Llama-2-7B-GPTQ', 'An already-quantized GPTQ repo: packed 4-bit weights.'],
];
$('#hf-examples').innerHTML = '<span class="muted">Try:</span> ' + EXAMPLES.map(([id, t]) => `<button class="chip" data-example="${esc(id)}"${tipText(t + '\nClick to import it.')}>${esc(id)}</button>`).join('');
$('#hf-examples').addEventListener('click', (e) => {
  const b = e.target.closest('[data-example]');
  if (!b) return;
  $('#hf-id').value = b.dataset.example;
  showSource('hf');
  doImport();
});

// Smart box: detect what was pasted and pick the matching option
function detectInput(raw) {
  const t = raw.trim();
  if (!t) return null;
  if (t.startsWith('{')) return { src: 'config', fill: () => { $('#config-input').value = t; } };
  if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/i.test(t)) return { src: 'local', fill: () => { $('#local-url').value = t; } };
  if (/^https?:\/\//i.test(t)) {
    if (/\.(safetensors|gguf|json)(\?.*)?$/i.test(t)) return { src: 'url', fill: () => { $('#url-input').value = t; } };
    if (/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//i.test(t)) return { src: 'hf', fill: () => { $('#hf-id').value = t; } };
    return { src: 'url', fill: () => { $('#url-input').value = t; } };
  }
  if (/^\d[\d.,_]*\s*([kmbt]|e\d+)?$/i.test(t)) return { src: 'manual', fill: () => { $('#man-params').value = t; } };
  if (/^[\w.-]+\/[\w.-]+$/.test(t)) return { src: 'hf', fill: () => { $('#hf-id').value = t; } };
  return null;
}
function applySmart() {
  const d = detectInput($('#smart-box').value);
  if (!d) return false;
  d.fill(); showSource(d.src);
  return true;
}
$('#smart-box').addEventListener('input', debounce(applySmart, 150));
$('#smart-box').addEventListener('keydown', (e) => { if (e.key === 'Enter') { if (applySmart()) doImport(); else showMsg(new UserError('Couldn\'t tell what that is.', 'Pick an option below and fill in its box.', 'warn')); } });
$('#hf-id').addEventListener('keydown', (e) => { if (e.key === 'Enter') doImport(); });
$('#url-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') doImport(); });
$('#man-params').addEventListener('keydown', (e) => { if (e.key === 'Enter') doImport(); });
$('#hf-token').addEventListener('input', (e) => setToken(e.target.value));

// Files: picker + drag and drop
let pickedFiles = [];
const dz = $('#dropzone');
dz.addEventListener('click', () => $('#file-input').click());
dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file-input').click(); } });
$('#file-input').addEventListener('change', (e) => { setFiles(e.target.files); });
for (const ev of ['dragenter', 'dragover']) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('over'); });
for (const ev of ['dragleave', 'drop']) dz.addEventListener(ev, () => dz.classList.remove('over'));
dz.addEventListener('drop', (e) => { e.preventDefault(); setFiles(e.dataTransfer.files); });
// Dropping anywhere on the import panel switches to "File on my computer"
$('#import-panel').addEventListener('dragover', (e) => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); showSource('file'); } });
$('#import-panel').addEventListener('drop', (e) => { if (e.target.closest('#dropzone')) return; if (e.dataTransfer?.files?.length) { e.preventDefault(); showSource('file'); setFiles(e.dataTransfer.files); } });
function setFiles(list) {
  pickedFiles = [...list];
  $('#file-list').textContent = pickedFiles.map((f) => `${f.name} (${fmtBytes(f.size)})`).join(' · ');
  if (pickedFiles.length) { showSource('file'); doImport(); }
}

// Local server help
function localHelp() {
  const origin = location.origin;
  $('#local-help').innerHTML = `<b>Allow this page to talk to your server</b> (it is your own computer — nothing leaves it):
    <div><b>Ollama</b>: pages on <code>http://localhost</code> work out of the box. For this page's address, set <code>OLLAMA_ORIGINS</code> and restart Ollama:</div>
    <pre>Windows (PowerShell):  setx OLLAMA_ORIGINS "${esc(origin)}"
macOS / Linux:         OLLAMA_ORIGINS="${esc(origin)}" ollama serve</pre>
    <div style="margin-top:6px"><b>LM Studio</b>: Developer tab → Settings → turn on <i>Enable CORS</i>. <b>vLLM</b>: <code>vllm serve &lt;model&gt; --allowed-origins '["${esc(origin)}"]'</code></div>
    <div class="muted" style="margin-top:4px">Your browser may also ask for permission to access devices on your local network — allow it for this page.</div>`;
}
localHelp();
$('#local-url').addEventListener('input', () => { S.localListing = null; $('#local-model').innerHTML = '<option value="">(press Connect to list models)</option>'; showSource('local'); });
$('#local-model').addEventListener('change', () => showSource('local'));

function setBusy(label) {
  busy?.abort();
  busy = label ? new AbortController() : null;
  $('#btn-import').disabled = !!label;
  $('#btn-cancel').hidden = !label;
  $('#import-progress').hidden = !label;
  $('#import-progress').className = 'progress indeterminate';
  $('#import-status').textContent = label || '';
  return busy?.signal;
}
function progress(text, frac) {
  $('#import-status').textContent = text;
  const bar = $('#import-progress');
  if (frac === undefined) bar.className = 'progress indeterminate';
  else { bar.className = 'progress'; bar.firstElementChild.style.width = Math.round(frac * 100) + '%'; }
}
$('#btn-cancel').addEventListener('click', () => { busy?.abort(); });

function msgHtml(level, text, next) {
  const ico = { error: '!', warn: '!', info: 'i', ok: '✓' }[level] || 'i';
  return `<div class="msg ${level}"><span class="ico">${ico}</span><span>${esc(text)}${next ? `<b class="next">${esc(next)}</b>` : ''}</span></div>`;
}
function showMsg(e) {
  if (!e) { $('#import-msg').innerHTML = ''; return; }
  if (isAbort(e)) { $('#import-msg').innerHTML = msgHtml('info', 'Cancelled.'); return; }
  const level = e.level || 'error';
  $('#import-msg').innerHTML = msgHtml(level, e.message, e.next || (e instanceof UserError ? '' : 'Something unexpected went wrong. Please try another option.'));
  if (!(e instanceof UserError)) console.error(e);
}

async function doImport() {
  if (busy) return;
  const src = srcVal();
  showMsg(null);
  const signal = setBusy('Working…');
  try {
    let p;
    if (src === 'hf') p = await importHF($('#hf-id').value, { signal, progress });
    else if (src === 'file') {
      if (!pickedFiles.length) throw new UserError('Choose a file first.', 'Click the box or drop .safetensors / .gguf / config.json files on it.', 'warn');
      p = await importFiles(pickedFiles, { signal, progress });
    } else if (src === 'url') p = await importUrl($('#url-input').value, { signal, progress });
    else if (src === 'config') p = importConfigText($('#config-input').value);
    else if (src === 'manual') p = importManual({ params: $('#man-params').value, dtype: $('#man-dtype').value, layers: $('#man-layers').value, hidden: $('#man-hidden').value });
    else if (src === 'local') {
      const base = $('#local-url').value;
      const chosen = $('#local-model').value;
      if (!S.localListing || !chosen) {
        progress('Asking the server for its models…');
        S.localListing = await listLocalModels(base, { signal });
        $('#local-model').innerHTML = S.localListing.models.map((m) => `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join('');
        showSource('local');
        $('#import-msg').innerHTML = msgHtml('ok', `Found ${S.localListing.models.length} model${S.localListing.models.length === 1 ? '' : 's'} on the ${S.localListing.kind === 'ollama' ? 'Ollama' : 'OpenAI-compatible'} server.`, 'Pick one in the Model list, then press Import.');
        return;
      }
      progress('Reading model details from the server…');
      p = await importLocalServer(base, chosen, S.localListing, { signal });
    }
    setProfile(p);
  } catch (e) {
    showMsg(e);
  } finally {
    setBusy(null);
  }
}
$('#btn-import').addEventListener('click', doImport);

net.onChange = () => {
  const hosts = [...net.hosts.entries()].map(([h, n]) => `${h} (${n})`).join(', ');
  const b = $('#net-badge');
  b.textContent = `Network: ${net.requests} request${net.requests === 1 ? '' : 's'} · ${fmtBytes(net.bytes)}`;
  b.dataset.tip = `Every network request this page has made, and where to.\nIt only ever talks to the model source you picked.\n\n${hosts || 'None yet.'}`;
};

/* =====================================================================
 * Model overview
 * ===================================================================== */
const sourceLabels = { hf: 'Hugging Face', file: 'Local file', url: 'Direct link', local: 'Local server', config: 'config.json', manual: 'Manual entry' };

function featureList(p) {
  const real = p.weightAccess !== 'none';
  const why = p.weightNote || ({ local: 'Local servers only share summary information, not the weights.', config: 'A config.json has no weights in it.', manual: 'There are no weights to read for a model entered by hand.' }[p.source] || 'Weights are not available from this source.');
  const how = 'To enable: import from Hugging Face, open the .safetensors / F16 .gguf file, or paste a direct link to it.';
  return [
    ['Size of every format + GPU fit', true, 'Works for every source: it only needs the parameter count.'],
    ['Model details', !!(p.layers || p.hiddenSize || p.contextLength), 'Layers, width and context length.' + (p.source === 'manual' ? '\nNot available for manual entry — import config.json for details.' : '')],
    ['Ready-made quantized versions', p.source === 'hf', p.source === 'hf' ? 'Existing quantized versions listed on Hugging Face.' : 'Only for Hugging Face models: it searches the Hub for versions made from this model.\nTo enable: import the model by its Hugging Face ID.'],
    ['Precision-loss charts (generated weights)', true, 'Always available: uses random weights shaped like a real LLM layer.'],
    ['Real precision loss on a chosen layer', real, real ? 'You can analyse a sample of any weight matrix of this model.' : `${why}\n\n${how}`],
    ['Layer sensitivity scan + mixed-precision plan', real, real ? 'You can scan every weight matrix and get a mixed-precision plan.' : `${why}\n\n${how}`],
  ];
}

function renderOverview() {
  const p = S.profile;
  const el = $('#overview');
  const origBits = DTYPE_BITS[p.originalDtype] || 16;
  const origGB = Q.sizeGB(p.paramsTotal, origBits);
  const stat = (label, value, tipKey, sub = '', cls = '', extra = '') => `<div class="stat ${cls}"${tipText((TIPS[tipKey] || tipKey) + (extra ? `\n\n${extra}` : ''))}><div class="stat-label">${label}${qi}</div><div class="stat-value">${value}</div>${sub ? `<div class="stat-sub">${sub}</div>` : ''}</div>`;
  const link = p.hf ? `<a href="https://huggingface.co/${esc(p.hf.id)}" target="_blank" rel="noopener noreferrer">${esc(p.name)}</a>` : esc(p.name);
  const q = p.quantization;
  el.innerHTML = `
    <div class="panel-title"><h2${tipText('Everything the tool found out about the model, normalised into one summary. All features below use it.')}>Model overview${qi}</h2>
      <span class="small muted">${esc(sourceLabels[p.source] || p.source)}</span></div>
    <div class="ov-head">
      <div><div class="ov-name">${link}</div>
        <div class="ov-src"${tip('source')}>Source: <b>${esc(p.paramSource)}</b>${p.paramExact ? ' ✓' : ''} · ${esc(p.sourceDetail)} ${qi}</div></div>
      ${p.hf?.gated ? `<span class="tag fp8"${tipText('This model is gated: you must accept its licence on huggingface.co. The parameter count still works without a token; config.json and weight samples need your token.')}>gated</span>` : ''}
    </div>
    <div class="stats">
      ${stat('Parameters', fmtCount(p.paramsTotal), 'params', fmtInt(p.paramsTotal), '', `Here: ${fmtInt(p.paramsTotal)} weights.\nSource: ${p.paramSource}.`)}
      ${p.paramsActive && p.paramsActive !== p.paramsTotal ? stat('Active per token', fmtCount(p.paramsActive), 'active', `${p.experts} experts, ${p.expertsUsed} used`, '', `Here: ${fmtInt(p.paramsActive)} of ${fmtInt(p.paramsTotal)} weights are used for each token (${pct((100 * p.paramsActive) / p.paramsTotal)}).`) : ''}
      ${stat('Original size', fmtGB(origGB), 'origSize', `${p.originalDtype}`, '', `Here: ${fmtCount(p.paramsTotal)} × ${origBits} bits ÷ 8 = ${fmtGB(origGB)}.`)}
      ${stat('Original data type', esc(p.originalDtype || '?'), 'dtype')}
      ${stat('Existing quantization', q ? esc(q.label) : 'none', 'quant', q ? 'already compressed' : 'full precision', q ? 'mid' : '')}
      ${p.layers ? stat('Layers', fmtInt(p.layers), 'layers') : ''}
      ${p.hiddenSize ? stat('Hidden size', fmtInt(p.hiddenSize), 'hidden', p.heads ? `${p.heads} heads${p.kvHeads && p.kvHeads !== p.heads ? `, ${p.kvHeads} KV` : ''}` : '') : ''}
      ${p.contextLength ? stat('Context length', fmtInt(p.contextLength) + ' tokens', 'context') : ''}
      ${p.vocabSize ? stat('Vocabulary', fmtInt(p.vocabSize), 'vocab') : ''}
      ${p.architecture ? stat('Architecture', esc(p.architecture), 'arch') : ''}
    </div>
    <div class="features">${featureList(p).map(([name, on, t]) => `<span class="feat ${on ? '' : 'off'}"${tipText(t)}>${on ? '✓' : '✕'} ${esc(name)}</span>`).join('')}</div>
    ${p.weightNote ? msgHtml(p.weightAccess === 'none' ? 'info' : 'warn', p.weightNote) : ''}
    ${p.analyseBase ? `<div style="margin-top:8px"><button data-analyse-base="${esc(p.analyseBase)}"${tipText(`Import the original model ${p.analyseBase} so its real weights can be analysed.`)}>Analyse the original model: ${esc(p.analyseBase)}</button></div>` : ''}
    ${p.suggestHf ? `<div style="margin-top:8px"><button data-analyse-base="${esc(p.suggestHf)}"${tipText('Import this model from Hugging Face for exact details and real-weight analysis.')}>Import ${esc(p.suggestHf)} from Hugging Face</button></div>` : ''}
    ${(p.warnings || []).map((w) => msgHtml(w.level || 'warn', w.text, w.next)).join('')}
    ${p.assumptions?.length ? `<details class="assume"><summary>How these numbers were worked out (${p.assumptions.length} note${p.assumptions.length === 1 ? '' : 's'})</summary><ul>${p.assumptions.map((a) => `<li>${esc(a)}</li>`).join('')}</ul></details>` : ''}`;
}
$('#overview').addEventListener('click', (e) => {
  const b = e.target.closest('[data-analyse-base]');
  if (!b) return;
  $('#hf-id').value = b.dataset.analyseBase;
  showSource('hf');
  window.scrollTo({ top: $('#import-panel').offsetTop - 70, behavior: 'smooth' });
  doImport();
});

/* =====================================================================
 * Size & GPU-fit table
 * ===================================================================== */
function sizeRows(p) {
  const opts = { embParams: p.embParams || 0, keepEmb: S.sizeOpts.keepEmb, hidden: p.hiddenSize || 4096 };
  return Q.SIZE_ROWS.map((row) => ({ row, ...Q.rowSize(row, p.paramsTotal, opts) }));
}
function formulaFor(r, p) {
  const P = fmtCount(p.paramsTotal);
  if (r.keptEmb) {
    const e = p.embParams;
    return `Here: (${fmtCount(p.paramsTotal - e)} × ${fmtBits(r.bits)} bits + ${fmtCount(e)} embedding/output weights × 16 bits) ÷ 8 = ${fmtGB(r.gb)}.\nAverage: ${fmtBits(r.avgBits)} bits per weight.`;
  }
  return `Here: ${P} × ${fmtBits(r.bits)} bits ÷ 8 = ${fmtGB(r.gb)}.`;
}

function renderSizes() {
  const p = S.profile;
  const el = $('#sizes');
  const rows = sizeRows(p);
  const origBits = DTYPE_BITS[p.originalDtype] || 16;
  const origGB = Q.sizeGB(p.paramsTotal, origBits);
  const maxGB = Math.max(...rows.map((r) => r.gb));
  const hr = S.sizeOpts.headroom;
  const fam = { float: 'Floating point', quant: 'Quantized for GPU libraries (vLLM, Transformers, TGI…)', gguf: 'GGUF for llama.cpp / Ollama / LM Studio (approximate)' };
  let lastFam = '';
  const body = rows.map((r) => {
    const isOrig = r.row.id === { BF16: 'bf16', FP16: 'fp16', FP32: 'fp32', FP8: 'fp8' }[p.originalDtype];
    const head = r.row.family !== lastFam ? `<tr class="group-head"><td colspan="${5 + Q.GPU_SIZES.length}">${fam[r.row.family]}</td></tr>` : '';
    lastFam = r.row.family;
    const saved = 1 - r.gb / origGB;
    const fits = Q.GPU_SIZES.map((g) => {
      const usable = (g * Q.GIB * hr) / 1e9;
      if (Q.fitsOn(r.gb, g, hr)) {
        return `<td class="num"${tipText(`${g} GB GPU (${(g * Q.GIB / 1e9).toFixed(1)} × 10⁹ bytes); usable at ${Math.round(hr * 100)}% = ${fmtGB(usable)}.\nWeights ${fmtGB(r.gb)} → fits, leaving ${fmtGB(g * Q.GIB / 1e9 - r.gb)} for the KV cache and activations.`)}><span class="fit yes">✓</span></td>`;
      }
      const n = Q.gpusNeeded(r.gb, g, hr);
      return `<td class="num"${tipText(`Weights ${fmtGB(r.gb)} > ${fmtGB(usable)} usable on one ${g} GB GPU.\nWould need ${n} of these GPUs (${n} × ${fmtGB(usable)} = ${fmtGB(n * usable)}), with the model split across them (tensor or pipeline parallel).`)}><span class="fit no">×${n > 99 ? '99+' : n}</span></td>`;
    }).join('');
    return `${head}<tr class="${isOrig ? 'orig' : ''}">
      <td${tipText(`${r.row.label}\n${r.row.note}${r.row.family === 'gguf' ? '\n\nBits per weight measured by llama.cpp on Llama-3.1-8B (whole file). Other models differ a little — treat as approximate.' : ''}${isOrig ? '\n\nThis is the model\'s original format.' : ''}`)}>${esc(r.row.label)}${isOrig ? ' <span class="small muted">(original)</span>' : ''}</td>
      <td class="num"${tipText(`${TIPS.bitsCol}${r.keptEmb ? `\n\nHere: ${fmtBits(r.bits)} bits for most weights; ${fmtBits(r.avgBits)} on average with the embeddings kept at 16 bits.` : ''}`)}>${fmtBits(r.bits)}${r.row.family === 'gguf' ? '<span class="muted">≈</span>' : ''}${r.keptEmb ? `<div class="small muted">avg ${r.avgBits.toFixed(2)}</div>` : ''}</td>
      <td class="num"${tipText(`${TIPS.sizeCol}\n\n${formulaFor(r, p)}`)}><b>${fmtGB(r.gb)}</b></td>
      <td${tipText(`Size compared with the largest format in this table.`)}><div class="sizebar"><span style="width:${(100 * r.gb) / maxGB}%"></span></div></td>
      <td class="num"${tipText(`${TIPS.savedCol}\n\nHere: 1 − ${fmtGB(r.gb)} ÷ ${fmtGB(origGB)} = ${pct(saved * 100)}${saved < 0 ? ' (bigger than the original)' : ''}.`)}>${saved > 0.005 ? '−' + pct(saved * 100) : saved < -0.005 ? '+' + pct(-saved * 100) : '—'}</td>
      ${fits}</tr>`;
  }).join('');
  el.innerHTML = `
    <div class="panel-title"><h2${tipText('How big the model is in every common format, and which GPUs it fits on.\nHover any number to see how it was calculated.')}>Size &amp; GPU fit — ${esc(fmtCount(p.paramsTotal))} parameters${qi}</h2></div>
    <div class="table-tools">
      <label class="field"${tip('headroom')}><span>Usable GPU memory <b id="hr-out">${Math.round(hr * 100)}%</b>${qi}</span><input type="range" id="headroom" min="50" max="100" step="1" value="${Math.round(hr * 100)}" /></label>
      ${p.embParams ? `<label class="toggle"${tip('keepEmb')}><input type="checkbox" id="keep-emb" ${S.sizeOpts.keepEmb ? 'checked' : ''} /> Keep embeddings &amp; output layer in 16-bit (${fmtCount(p.embParams)} weights) ${qi}</label>` : `<span class="small muted"${tipText('The embedding size is unknown for this source, so every weight is counted at the format\'s bit width.')}>All weights counted at the format's bit width ${qi}</span>`}
    </div>
    <div class="table-wrap"><table class="data">
      <thead><tr><th${tipText('The storage format. Grouped by the software that runs it.')}>Format</th><th class="num"${tip('bitsCol')}>Bits / weight</th><th class="num"${tip('sizeCol')}>Size</th><th${tipText('Bar: size compared with the largest format in this table.')}></th><th class="num"${tip('savedCol')}>Saved</th>
        ${Q.GPU_SIZES.map((g) => `<th class="num"${tipText(`${TIPS.fitCol}\n\nColumn: one GPU with ${g} GB of memory${{ 8: ' (e.g. RTX 4060)', 12: ' (e.g. RTX 4070)', 16: ' (e.g. RTX 4080, T4)', 24: ' (e.g. RTX 4090, L4, A10)', 48: ' (e.g. L40S, A6000)', 80: ' (e.g. A100 / H100 80 GB)' }[g]}.`)}>${g} GB</th>`).join('')}</tr></thead>
      <tbody>${body}</tbody></table></div>
    <p class="kv-note">⚠ Weights only. <b>KV cache and activations need extra memory</b> — more for long contexts and many users. <a href="#" onclick="return false" class="muted"${tipText('A VRAM calculator for KV cache and activations is planned as a companion tool.')}>VRAM calculator (coming soon)</a></p>`;
  $('#headroom').addEventListener('input', (e) => { S.sizeOpts.headroom = e.target.value / 100; renderSizes(); renderResultsPanel(); });
  $('#keep-emb')?.addEventListener('change', (e) => { S.sizeOpts.keepEmb = e.target.checked; renderSizes(); renderResultsPanel(); });
}

/* =====================================================================
 * Ready-made quantized versions (Hugging Face)
 * ===================================================================== */
async function loadReadyMade() {
  const p = S.profile, el = $('#readymade');
  if (p.source !== 'hf') { el.hidden = true; return; }
  el.hidden = false;
  const base = p.quantizedCheckpoint && p.hf.baseModel ? p.hf.baseModel : p.hf.id;
  el.innerHTML = `<div class="panel-title"><h2>Ready-made quantized versions</h2></div><p class="empty-state">Searching huggingface.co for quantized versions of ${esc(base)}…</p>`;
  let list;
  try { list = await listQuantized(base); }
  catch (e) { el.innerHTML = `<div class="panel-title"><h2>Ready-made quantized versions</h2></div>${msgHtml('warn', `Couldn't load the list: ${e.message}`, e.next)}`; return; }
  if (S.profile !== p) return;
  S.readyMade = { base, list };
  renderReadyMade();
  // Fetch file sizes for the most downloaded few (3 at a time)
  const top = list.slice(0, 12);
  let i = 0;
  await Promise.all([0, 1, 2].map(async () => {
    while (i < top.length) {
      const r = top[i++];
      try { r.sizes = await repoSizes(r.id); } catch { r.sizes = { error: true }; }
      if (S.profile === p) renderReadyMade();
    }
  }));
}
function sizesCell(r) {
  if (!r.sizes) return `<button class="mini" data-sizes="${esc(r.id)}"${tipText('Look up this repo\'s file sizes (one small request to huggingface.co).')}>show</button>`;
  if (r.sizes.error) return '<span class="muted">?</span>';
  if (r.sizes.gguf) return r.sizes.gguf.slice(0, 14).map(([lv, b]) => `<span class="chip"${tipText(`${lv}: ${fmtBytes(b)} file.\nA bigger file = more bits per weight = closer to the original.`)}>${esc(lv)} ${(b / 1e9).toFixed(1)}</span>`).join(' ') + (r.sizes.gguf.length > 14 ? ' …' : '') + ' <span class="muted small">GB</span>';
  return r.sizes.total ? `<b>${fmtBytes(r.sizes.total)}</b> <span class="muted small">(${r.sizes.files} file${r.sizes.files === 1 ? '' : 's'})</span>` : '<span class="muted">—</span>';
}
function renderReadyMade() {
  const el = $('#readymade');
  const { base, list } = S.readyMade;
  el.innerHTML = `
    <div class="panel-title"><h2${tipText('Quantized versions other people have already published on Hugging Face, made from this model (the “Quantizations” list on the model page).\nIf the format you want exists, you can download it instead of quantizing yourself.')}>Ready-made quantized versions of ${esc(base)} (${list.length}${list.length >= 40 ? '+' : ''})${qi}</h2>
      <a class="small" href="https://huggingface.co/models?other=base_model:quantized:${encodeURIComponent(base)}" target="_blank" rel="noopener noreferrer">see all on huggingface.co ↗</a></div>
    ${list.length ? `<div class="table-wrap"><table class="data">
      <thead><tr><th${tipText('The repository on huggingface.co. Click to open it.')}>Repository</th><th${tipText('How the weights are stored. GGUF runs in llama.cpp / Ollama / LM Studio; GPTQ, AWQ, bitsandbytes and FP8 run in GPU libraries like vLLM and Transformers.')}>Format</th><th${tipText('Bits and group size from the repo\'s quantization_config, when it has one.')}>Details</th><th${tipText('Size of the weight files. GGUF repos usually hold several levels — each chip is one level and its file size in GB.')}>Size</th><th class="num"${tipText('Downloads in the last 30 days — a rough sign of how well-tested a version is.')}>Downloads</th></tr></thead>
      <tbody>${list.map((r) => `<tr>
        <td class="mono"><a href="https://huggingface.co/${esc(r.id)}" target="_blank" rel="noopener noreferrer">${esc(r.id)}</a></td>
        <td><span class="tag ${r.format.key}">${esc(r.format.label)}</span></td>
        <td class="small">${r.quant ? esc(r.quant.label) : r.ggufTotal ? `${fmtCount(r.ggufTotal)} params` : '<span class="muted">—</span>'}</td>
        <td>${sizesCell(r)}</td>
        <td class="num">${fmtInt(r.downloads)}</td></tr>`).join('')}</tbody></table></div>`
      : '<p class="empty-state">No quantized versions are linked to this model on Hugging Face yet. Look for repos named like “…-GGUF”, “…-AWQ” or “…-GPTQ”, or make one yourself.</p>'}`;
}
$('#readymade').addEventListener('click', async (e) => {
  const b = e.target.closest('[data-sizes]');
  if (!b) return;
  const r = S.readyMade.list.find((x) => x.id === b.dataset.sizes);
  b.disabled = true;
  try { r.sizes = await repoSizes(r.id); } catch { r.sizes = { error: true }; }
  renderReadyMade();
});

/* =====================================================================
 * Profile loaded
 * ===================================================================== */
function setProfile(p) {
  S.profile = p;
  S.scan = null;
  S.readyMade = null;
  $('#model-results').hidden = false;
  renderOverview();
  renderSizes();
  loadReadyMade();
  if (S.real && S.wsrc === 'real') { S.real = null; setWsrc('gen'); }
  updateRealAvailability();
  renderScan();
  renderResultsPanel();
  $('#import-msg').innerHTML = msgHtml('ok', `Loaded ${p.name}: ${fmtCount(p.paramsTotal)} parameters (${p.paramExact ? 'exact' : p.source === 'manual' ? 'entered by hand' : 'estimate'}).`);
  setTimeout(() => window.scrollTo({ top: $('#model-results').offsetTop - 70, behavior: 'smooth' }), 50);
}

/* =====================================================================
 * Explorer controls
 * ===================================================================== */
const PRESETS = {
  typical: { gen: { dist: 'laplace', sigma: 0.02, count: 65536, outlierPct: 0.02, outlierMul: 15, seed: 1 }, cfg: { format: 'int4', sym: true, granularity: 'group', groupSize: 128, clip: 'none', scaleDtype: 'fp16', fullRange: false } },
  clean: { gen: { dist: 'normal', sigma: 0.02, count: 65536, outlierPct: 0, outlierMul: 15, seed: 1 }, cfg: { format: 'int4', sym: true, granularity: 'tensor', groupSize: 128, clip: 'none', scaleDtype: 'fp16', fullRange: false } },
  disaster: { gen: { dist: 'laplace', sigma: 0.02, count: 65536, outlierPct: 0.05, outlierMul: 40, seed: 1 }, cfg: { format: 'int4', sym: true, granularity: 'tensor', groupSize: 128, clip: 'none', scaleDtype: 'fp16', fullRange: false } },
};
PRESETS.compare = PRESETS.typical;

function readControls() {
  S.gen = {
    dist: $('#g-dist').value, sigma: +$('#g-sigma').value, count: 2 ** +$('#g-count').value,
    outlierPct: +$('#g-opct').value, outlierMul: +$('#g-omul').value, seed: Math.max(1, Math.floor(+$('#g-seed').value || 1)),
  };
  const g = $('#q-gran').value;
  S.cfg = {
    format: $('#q-format').value, sym: $('input[name=sym]:checked').value === '1', fullRange: $('#q-full').checked,
    granularity: g.startsWith('g') ? 'group' : g, groupSize: g.startsWith('g') ? +g.slice(1) : S.cfg.groupSize,
    scaleDtype: $('#q-sdt').value, clip: $('#q-clip').value, clipPct: +$('#q-pct').value,
  };
}
function writeControls() {
  $('#g-dist').value = S.gen.dist; $('#g-sigma').value = S.gen.sigma; $('#g-count').value = Math.round(Math.log2(S.gen.count));
  $('#g-opct').value = S.gen.outlierPct; $('#g-omul').value = S.gen.outlierMul; $('#g-seed').value = S.gen.seed;
  $('#q-format').value = S.cfg.format;
  $(`input[name=sym][value="${S.cfg.sym ? 1 : 0}"]`).checked = true;
  $('#q-full').checked = !!S.cfg.fullRange;
  $('#q-gran').value = S.cfg.granularity === 'group' ? `g${S.cfg.groupSize}` : S.cfg.granularity;
  $('#q-sdt').value = S.cfg.scaleDtype; $('#q-clip').value = S.cfg.clip; $('#q-pct').value = S.cfg.clipPct;
}
function updateLabels() {
  $('#o-sigma').textContent = S.gen.sigma.toFixed(3);
  $('#o-count').textContent = fmtInt(S.gen.count);
  $('#o-opct').textContent = `${S.gen.outlierPct.toFixed(2)}% (${fmtInt((S.gen.count * S.gen.outlierPct) / 100)})`;
  $('#o-omul').textContent = `${S.gen.outlierMul}× σ ≈ ±${fmtNum(S.gen.outlierMul * S.gen.sigma, 2)}`;
  $('#o-pct').textContent = S.cfg.clipPct.toFixed(2) + '%';
  const f = Q.FORMATS[S.cfg.format];
  const cast = f.kind === 'cast', isInt = f.kind === 'int';
  // Options that do not apply are greyed out, and their tooltip says why
  const grey = (el, why) => {
    el.dataset.tipOn ??= el.dataset.tip;
    el.classList.toggle('disabled-box', !!why);
    el.dataset.tip = why || el.dataset.tipOn;
  };
  const noScale = S.cfg.granularity === 'none';
  grey($('#q-sym-row'), isInt ? '' : cast ? 'Not used: FP16 / BF16 are a plain cast with no scale or zero-point.' : 'Only integer formats can be asymmetric. Float formats and NF4 are always symmetric around zero.');
  grey($('#q-gran').closest('.field'), cast ? 'Not used: FP16 / BF16 are a plain cast with no scale to share.' : '');
  grey($('#q-sdt').closest('.field'), cast || noScale ? 'Not used: there is no scale with this format / setting.' : '');
  grey($('#q-clip').closest('.field'), cast || noScale ? 'Not used: with no scale there is nothing to clip — the range of the format is fixed.' : '');
  $('#q-gran option[value=none]').disabled = !(f.kind === 'float' || cast);
  if (S.cfg.granularity === 'none' && !(f.kind === 'float' || cast)) { $('#q-gran').value = 'g128'; readControls(); }
  $('#q-pct-row').hidden = S.cfg.clip !== 'percentile' || cast;
  $('#q-full-row').hidden = !(isInt && S.cfg.sym);
}

const runAnalysis = debounce(analyzeNow, 100);
let lastReq = 0;
async function analyzeNow() {
  readControls();
  updateLabels();
  const source = S.wsrc === 'real' && S.real ? { kind: 'real', key: S.real.key } : { kind: 'gen', params: S.gen };
  const id = ++lastReq;
  const cfg = { ...S.cfg };
  const t0 = performance.now();
  try {
    const r = await call({ type: 'analyze', source, cfg });
    if (id !== lastReq) return; // a newer request is on its way
    S.result = r; S.resultCfg = cfg;
    renderCharts();
    renderResultsPanel();
    $('#calc-time').textContent = `${fmtInt(r.n)} weights · worker ${r.ms.toFixed(0)} ms · total ${(performance.now() - t0).toFixed(0)} ms`;
  } catch (e) {
    if (id === lastReq) $('#results').innerHTML = msgHtml('error', e.message);
  }
}
$('#controls').addEventListener('input', (e) => { if (e.target.closest('#real-controls')) return; readControls(); updateLabels(); runAnalysis(); });
$('#controls').addEventListener('change', (e) => { if (e.target.closest('#real-controls')) return; $('#preset').value = ''; readControls(); updateLabels(); runAnalysis(); });
$('#preset').addEventListener('change', (e) => {
  const pr = PRESETS[e.target.value];
  if (!pr) return;
  S.gen = { ...pr.gen }; S.cfg = { ...S.cfg, ...pr.cfg };
  setWsrc('gen');
  writeControls(); updateLabels(); analyzeNow();
  if (e.target.value === 'compare') {
    const fig = $('[data-chart=cmp]');
    fig.scrollIntoView({ behavior: 'smooth', block: 'center' });
    fig.style.boxShadow = '0 0 0 3px #bfdbfe';
    setTimeout(() => { fig.style.boxShadow = ''; }, 1600);
  }
});

/* ---------- Weight source: generated / real layer ---------- */
function setWsrc(v) {
  S.wsrc = v;
  $(`input[name=wsrc][value=${v}]`).checked = true;
  $('#gen-controls').hidden = v !== 'gen';
  $('#real-controls').hidden = v !== 'real';
}
$('#wsrc').addEventListener('change', async () => {
  const v = $('input[name=wsrc]:checked').value;
  if (v === 'real' && (!S.profile || S.profile.weightAccess === 'none')) { setWsrc('gen'); toast('Load a model with readable weights first (see the tooltip).'); return; }
  setWsrc(v);
  if (v === 'real') { await prepareTensorList(); if (S.real) analyzeNow(); }
  else analyzeNow();
});

function updateRealAvailability() {
  const p = S.profile;
  const ok = p && p.weightAccess !== 'none';
  const lab = $('#wsrc-real-label');
  lab.classList.toggle('disabled', !ok);
  lab.dataset.tip = ok ? 'A sample of a real weight matrix from the model you loaded.'
    : !p ? 'Not available yet: import a model first (Hugging Face, a .safetensors / F16 .gguf file, or a direct link).'
      : `Not available for this model: ${p.weightNote || 'its weights can\'t be read from this source.'}\n\nTo enable: import from Hugging Face, open the .safetensors file, or paste a direct link to it.`;
  $('#real-controls').querySelector('#t-info').textContent = '';
  if (ok) { $('#t-select').innerHTML = ''; tensorOptions(); }
}

function readableTensors() {
  return (S.profile?.tensors || []).filter((t) => READABLE_DTYPES.has(t.dtype) && t.numel >= 2);
}
async function prepareTensorList() {
  const p = S.profile;
  if (!p.allTensorsLoaded && p.weightFiles.length > (p.loadedFiles?.size || 0)) {
    $('#t-info').textContent = `Reading ${p.weightFiles.length} weight-file headers…`;
    try { await ensureTensors(p, { progress: (t) => { $('#t-info').textContent = t; } }); }
    catch (e) { $('#t-info').innerHTML = msgHtml('error', e.message, e.next); return; }
  }
  tensorOptions();
}
function tensorOptions() {
  const all = readableTensors();
  const words = $('#t-search').value.toLowerCase().split(/\s+/).filter(Boolean);
  const list = all.filter((t) => words.every((w) => t.name.toLowerCase().includes(w)));
  const prev = $('#t-select').value;
  $('#t-select').innerHTML = list.slice(0, 600).map((t) => `<option value="${esc(t.name)}">${esc(t.name)}  [${t.shape.join('×')}] ${t.dtype}</option>`).join('') + (list.length > 600 ? `<option disabled>… ${list.length - 600} more — type to narrow down</option>` : '');
  const pick = list.find((t) => t.name === prev) || list.find((t) => /layers?\.0\.|blk\.0\./.test(t.name) && /down_proj|ffn_down|mlp\.c_proj|dense_4h/.test(t.name)) || list.find((t) => t.shape.length >= 2) || list[0];
  if (pick) $('#t-select').value = pick.name;
  if (!all.length) $('#t-info').textContent = 'No readable (F32 / F16 / BF16) tensors found.';
  else tensorInfo();
}
function selectedTensor() { return readableTensors().find((t) => t.name === $('#t-select').value); }
function tensorInfo() {
  const t = selectedTensor();
  if (!t) return;
  const plan = planSample(t, +$('#t-sample').value);
  const bytes = plan.ranges.reduce((a, r) => a + r.end - r.start, 0);
  const src = S.profile.fileSources[t.file];
  $('#t-info').innerHTML = `<b>${t.shape.join(' × ')}</b> · ${fmtInt(t.numel)} weights · ${t.dtype}<br>Will read ${fmtInt(plan.rows * plan.cols)} weights (${fmtBytes(bytes)}) in ${plan.ranges.length} evenly spaced block${plan.ranges.length === 1 ? '' : 's'} of rows${src.kind === 'file' ? ' from your file (no network)' : ''}.${S.real?.tensor === t.name ? '<br><span style="color:var(--ok)">✓ Loaded</span>' : ''}`;
}
$('#t-search').addEventListener('input', debounce(tensorOptions, 120));
$('#t-select').addEventListener('change', tensorInfo);
$('#t-sample').addEventListener('change', tensorInfo);
$('#t-select').addEventListener('dblclick', () => $('#btn-load-tensor').click());
$('#btn-load-tensor').addEventListener('click', async () => {
  const t = selectedTensor();
  if (!t) return toast('Pick a tensor first');
  const btn = $('#btn-load-tensor');
  btn.disabled = true;
  const ctl = new AbortController();
  try {
    $('#t-info').textContent = 'Reading…';
    const s = await readTensorSample(S.profile, t, Math.min(MAX_SAMPLE, +$('#t-sample').value), {
      signal: ctl.signal, budget: new Budget(100e6), onProgress: (f) => { $('#t-info').textContent = `Reading… ${Math.round(f * 100)}%`; },
    });
    const key = `${t.file}:${t.name}:${s.weights.length}`;
    await call({ type: 'setReal', key, weights: s.weights, cols: s.cols }, [s.weights.buffer]);
    S.real = { key, tensor: t.name, shape: t.shape, cols: s.cols, rows: s.rows, n: s.rows * s.cols, bytes: s.bytes, blocks: s.blocks, partialRows: s.partialRows };
    tensorInfo();
    analyzeNow();
  } catch (e) {
    $('#t-info').innerHTML = msgHtml(e.level || 'error', e.message, e.next);
  } finally { btn.disabled = false; }
});

/* =====================================================================
 * Charts
 * ===================================================================== */
const hovers = {};
function renderCharts() {
  const r = S.result;
  if (!r) return;
  const cfg = S.resultCfg;
  const cv = (k) => $(`[data-chart=${k}] canvas`);
  const levelsNote = r.groups > 1 ? `of a typical group (median scale ${fmtNum(r.hist.repScale, 3)})` : '';
  hovers.hist = drawHist(cv('hist'), { ...r.hist, levelsNote }, { log: S.log.hist });
  hovers.stair = drawStair(cv('stair'), r.stair);
  hovers.err = drawErr(cv('err'), r.errHist, { log: S.log.err });
  hovers.clip = drawClip(cv('clip'), r.sweep, { sqnr: r.metrics.sqnr });
  hovers.cmp = drawCompare(cv('cmp'), r.compare, { label: `${cfgLabel(cfg)} (yours)`, bits: r.bits, sqnr: r.metrics.sqnr }, r.compareSubset);
  hovers.heat = drawHeat(cv('heat'), r.heat);
}
const chartTip = $('#chart-tip');
$('#charts').addEventListener('mousemove', (e) => {
  const fig = e.target.closest('[data-chart]');
  const cv = e.target.closest('canvas');
  if (!fig || !cv || !hovers[fig.dataset.chart]) { chartTip.style.display = 'none'; return; }
  const b = cv.getBoundingClientRect();
  const h = hovers[fig.dataset.chart](e.clientX - b.left, e.clientY - b.top);
  if (fig.dataset.chart === 'heat' && h) inspectCell(h.k);
  if (!h) { chartTip.style.display = 'none'; return; }
  chartTip.textContent = h.text;
  chartTip.style.display = 'block';
  const r = chartTip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  chartTip.style.left = x + 'px'; chartTip.style.top = y + 'px';
});
$('#charts').addEventListener('mouseleave', () => { chartTip.style.display = 'none'; });
$('#charts').addEventListener('change', (e) => {
  const lg = e.target.dataset.log;
  if (lg) { S.log[lg] = e.target.checked; renderCharts(); }
});
const CHART_TITLES = { hist: 'A · Weights & quantization levels', stair: 'B · Original vs quantized', err: 'C · Error histogram', clip: 'E · Clipping trade-off', cmp: 'D · All formats compared', heat: 'F · 32×32 slice: original · quantized · error' };
$('#charts').addEventListener('click', (e) => {
  const k = e.target.closest('[data-png]')?.dataset.png;
  if (!k || !S.result) return;
  const png = chartPNG($(`[data-chart=${k}] canvas`), CHART_TITLES[k], subtitle());
  png.toBlob((b) => downloadBlob(`quant-${k}.png`, b));
});
addEventListener('resize', debounce(renderCharts, 150));

function cfgLabel(c) {
  const f = Q.FORMATS[c.format];
  if (f.kind === 'cast') return f.label;
  const g = c.granularity === 'group' ? `g${c.groupSize}` : c.granularity === 'channel' ? 'per channel' : c.granularity === 'none' ? 'no scale' : 'per tensor';
  return `${f.label} ${g}${f.kind === 'int' && !c.sym ? ' asym' : ''}${c.clip !== 'none' ? ` clip ${c.clip === 'auto' ? 'auto' : c.clipPct + '%'}` : ''}`;
}
function weightsLabel() {
  if (S.wsrc === 'real' && S.real) return `${S.real.tensor} (${fmtInt(S.real.n)} of ${fmtInt(S.real.shape.reduce((a, b) => a * b, 1))} weights, ${S.real.blocks} blocks)`;
  const g = S.gen;
  return `generated ${g.dist}, σ ${g.sigma}, ${fmtInt(g.count)} weights, outliers ${g.outlierPct}% × ${g.outlierMul}, seed ${g.seed}`;
}
const subtitle = () => `${cfgLabel(S.resultCfg)} · ${weightsLabel()}${S.profile ? ' · ' + S.profile.name : ''}`;

/* =====================================================================
 * Results panel: metrics + insights + exports
 * ===================================================================== */
function verdict(s) {
  if (s >= 30) return { t: 'Excellent', c: 'good', col: '#16a34a' };
  if (s >= 20) return { t: 'Good', c: 'good', col: '#16a34a' };
  if (s >= 12) return { t: 'Moderate (typical 4-bit)', c: 'mid', col: '#d97706' };
  return { t: 'Heavy damage', c: 'bad', col: '#dc2626' };
}

function insights(r, cfg) {
  const out = [];
  const m = r.metrics, st = r.stats, f = Q.FORMATS[cfg.format], a = r.alt || {};
  const L = m.levelsAvail;
  // 1. Outliers squeezing weights into few levels
  if (st.outlierRatio > 3 && m.top3Share > 60 && cfg.granularity !== 'group' && f.kind !== 'cast') {
    out.push({ level: 'error', text: `Outliers make the scale ${st.outlierRatio.toFixed(0)}× larger than 99.9% of weights need, so ${pct(m.top3Share)} of weights fall into just 3 of ${L} ${f.label} levels${a.g128 ? ' — try group size 128' : ''}.`,
      tip: `Largest |w| = ${fmtNum(st.amax)}, 99.9th percentile = ${fmtNum(st.p999)} → ratio ${st.outlierRatio.toFixed(1)}.\nThe scale is set by the largest value, so the step between levels is huge compared with typical weights.` });
  }
  // 2. Group size effect
  // Alternatives are computed without clipping, so compare with this setting's no-clipping value (chart E at 100%)
  const noClip = cfg.clip === 'none' || !r.sweep ? m.sqnr : r.sweep.sqnr[0];
  const nc = cfg.clip === 'none' ? '' : ' (both without clipping)';
  if ((a.g128 || a.tensor) && cfg.granularity !== 'none') {
    if (a.g128) {
      if (a.g128.sqnr - noClip > 2) out.push({ level: 'info', text: `Group size 128 would raise SQNR from ${fmtDb(noClip)} dB to ${fmtDb(a.g128.sqnr)} dB for only ${fmtBits(a.g128.bits - r.bits)} extra bits per weight${nc}.`, tip: `With one scale per 128 weights, one outlier only affects its own group.\nCost: ${fmtBits(a.g128.bits)} instead of ${fmtBits(r.bits)} bits per weight.` });
    } else if (a.tensor.sqnr < noClip - 2) {
      out.push({ level: 'ok', text: `Group size ${cfg.groupSize} raised SQNR from ${fmtDb(a.tensor.sqnr)} dB (one scale per tensor) to ${fmtDb(noClip)} dB for only ${fmtBits(r.bits - a.tensor.bits)} extra bits per weight${nc}.`, tip: `Per tensor: ${fmtBits(a.tensor.bits)} bits, ${fmtDb(a.tensor.sqnr)} dB.\nGroup ${cfg.groupSize}: ${fmtBits(r.bits)} bits, ${fmtDb(noClip)} dB.` });
    }
  }
  // 3. Clipping
  const sw = r.sweep;
  if (sw && cfg.clip === 'none' && sw.best > 0 && sw.sqnr[sw.best] - m.sqnr > 1) out.push({ level: 'info', text: `Clipping at ${Math.round(sw.ratios[sw.best] * 100)}% of the max would raise SQNR from ${fmtDb(m.sqnr)} to ${fmtDb(sw.sqnr[sw.best])} dB — try Clipping: Auto.`, tip: 'Cutting the very largest values makes the scale smaller, so every other weight gets finer steps.\nSee chart E: the lowest point of the curve is the best clip level.' });
  if (sw && cfg.clip !== 'none' && sw.sqnr[0] - m.sqnr > 1) out.push({ level: 'warn', text: `Clipping is hurting here: without it SQNR would be about ${fmtDb(sw.sqnr[0])} dB instead of ${fmtDb(m.sqnr)} dB. The cut-off values are too important (heavy tails).`, tip: `${pct(m.clippedPct, 2)} of weights were clipped. Try a higher percentile, Auto, or None.` });
  else if (cfg.clip !== 'none' && m.clippedPct > 2) out.push({ level: 'warn', text: `${pct(m.clippedPct)} of weights are clipped. Clipping too hard damages the largest (often most important) weights.`, tip: 'Try a higher percentile or Auto.' });
  // 4. Zeroed weights
  if (m.zeroedPct > 25) out.push({ level: 'warn', text: `${pct(m.zeroedPct)} of non-zero weights were rounded to exactly zero: the steps are too coarse for the many small weights.`, tip: cfg.granularity === 'none' ? 'With no scale, FP8 / FP4 cannot represent small values — their smallest step is fixed. Use a scale.' : 'Smaller groups, clipping or more bits make the steps finer.' });
  // 5. More bits
  if (a.int8 && r.bits <= 5 && m.sqnr < 20) out.push({ level: 'info', text: `INT8 with the same scaling would give ${fmtDb(a.int8.sqnr)} dB (+${(a.int8.sqnr - m.sqnr).toFixed(0)} dB) at ${fmtBits(a.int8.bits)} bits per weight.`, tip: 'Each extra bit adds about 6 dB. (INT8 shown without clipping.)' });
  // 6. Model size
  const p = S.profile;
  if (p) {
    const bits = r.bits, gb = Q.sizeGB(p.paramsTotal, bits), fp16 = Q.sizeGB(p.paramsTotal, 16);
    const one = Q.GPU_SIZES.find((g) => Q.fitsOn(gb, g, S.sizeOpts.headroom));
    out.push({ level: 'info', text: `${p.name.split('/').pop()} drops from ${fmtGB(fp16)} (FP16) to ${fmtGB(gb)} (${cfgLabel(cfg)})${one ? ` — enough for one ${one} GB GPU` : ` — needs ${Q.gpusNeeded(gb, 80, S.sizeOpts.headroom)} × 80 GB GPUs`}.`, tip: `${fmtCount(p.paramsTotal)} × ${fmtBits(bits)} bits ÷ 8 = ${fmtGB(gb)} (all weights at this setting; see the size table for embeddings kept at 16-bit).\nKV cache and activations need extra memory.` });
  }
  if (!out.length || (out.length === 1 && p)) {
    const v = verdict(m.sqnr);
    out.unshift({ level: v.c === 'good' ? 'ok' : v.c === 'mid' ? 'warn' : 'error', text: `${v.t}: SQNR ${fmtDb(m.sqnr)} dB. ${m.sqnr >= 20 ? 'The rounding error is small compared with the weights.' : 'Try more bits, smaller groups or clipping.'}`, tip: TIPS.sqnr });
  }
  return out.slice(0, 4);
}

function renderResultsPanel() {
  const r = S.result, el = $('#results');
  if (!r) { el.innerHTML = '<p class="empty-state">Calculating…</p>'; return; }
  const cfg = S.resultCfg, m = r.metrics, st = r.stats, f = Q.FORMATS[cfg.format];
  const v = verdict(m.sqnr);
  const ex = (key, here) => tipText(`${TIPS[key]}\n\nHere: ${here}`);
  const per = cfg.granularity === 'group' ? cfg.groupSize : cfg.granularity === 'channel' ? r.cols : null;
  const sb = cfg.scaleDtype === 'fp32' ? 32 : 16, zb = f.kind === 'int' && !cfg.sym ? f.bits : 0;
  const bitsHere = f.kind === 'cast' || !per ? `${f.bits} bits${cfg.granularity === 'tensor' ? ' (one scale for the whole tensor adds almost nothing)' : ''}.` : `${f.bits} + (${sb}${zb ? ` + ${zb}` : ''}) ÷ ${per} = ${fmtBits(r.bits)} bits.`;
  const ms = m.rmsW ** 2;
  const lvl = m.levelsUsed !== null ? `${m.levelsUsed} / ${m.levelsAvail}` : '—';
  const p = S.profile;
  const gb = p ? Q.sizeGB(p.paramsTotal, r.bits) : null;
  el.innerHTML = `
    <h4${tip('sqnr')}>Result · ${esc(cfgLabel(cfg))}${qi}</h4>
    <div class="big-sqnr"${ex('sqnr', `average w² = ${fmtNum(ms, 3)}, average error² (MSE) = ${fmtNum(m.mse, 3)}\nSQNR = 10 · log10(${fmtNum(ms, 3)} ÷ ${fmtNum(m.mse, 3)}) = ${fmtDb(m.sqnr)} dB.`)}>
      <span class="v" style="color:${v.col}">${fmtDb(m.sqnr)}</span><span class="u">dB SQNR</span><span class="verdict stat ${v.c}" style="padding:1px 8px">${v.t}</span></div>
    <div class="metric-box">
      <div class="metric-line"${ex('bits', bitsHere)}><span>Bits per weight${qi}</span><b>${fmtBits(r.bits)}</b></div>
      ${p ? `<div class="metric-line"${tipText(`Size of ${p.name} if every weight used this setting.\n\nHere: ${fmtCount(p.paramsTotal)} × ${fmtBits(r.bits)} bits ÷ 8 = ${fmtGB(gb)}.`)}><span>Model size at this setting${qi}</span><b>${fmtGB(gb)}</b></div>` : ''}
      <div class="metric-line"${ex('rmse', `√${fmtNum(m.mse, 3)} = ${fmtNum(m.rmse, 3)}, compared with weight spread σ = ${fmtNum(st.std, 3)} (${pct((100 * m.rmse) / (st.std || 1))} of σ).`)}><span>RMS error${qi}</span><b>${fmtNum(m.rmse, 3)}</b></div>
      <div class="metric-line"${ex('mse', `${fmtNum(m.mse, 3)} (average over ${fmtInt(m.n)} weights).`)}><span>MSE${qi}</span><b>${fmtNum(m.mse, 3)}</b></div>
      <div class="metric-line"${ex('maxErr', `${fmtNum(m.maxErr, 3)}, which is ${(m.maxErr / (st.std || 1)).toFixed(1)} × σ.`)}><span>Max error${qi}</span><b>${fmtNum(m.maxErr, 3)}</b></div>
      <div class="metric-line"${ex('cosine', `${m.cosine.toFixed(6)}.`)}><span>Cosine similarity${qi}</span><b>${m.cosine.toFixed(5)}</b></div>
      <div class="metric-line"${ex('clipped', `${fmtInt((m.clippedPct * m.n) / 100)} of ${fmtInt(m.n)} weights.`)}><span>Clipped${qi}</span><b>${pct(m.clippedPct, 2)}</b></div>
      <div class="metric-line"${ex('zeroed', `${pct(m.zeroedPct, 2)} of the non-zero weights became 0.`)}><span>Rounded to zero${qi}</span><b>${pct(m.zeroedPct, 2)}</b></div>
      <div class="metric-line"${ex('levels', m.levelsUsed !== null ? `${m.levelsUsed} of ${m.levelsAvail} possible values are used. The 3 most common hold ${pct(m.top3Share)} of all weights.` : 'Not counted for 16-bit formats (65,536 possible values).')}><span>Levels used${qi}</span><b>${lvl}${m.top3Share !== null ? ` <span class="muted">· top 3: ${pct(m.top3Share)}</span>` : ''}</b></div>
      <div class="metric-line"${ex('groups', `${fmtInt(r.groups)} scale${r.groups === 1 ? '' : 's'} for ${fmtInt(r.n)} weights (${r.rows} rows × ${fmtInt(r.cols)}).`)}><span>Scales used${qi}</span><b>${fmtInt(r.groups)}</b></div>
    </div>
    <h4${tipText('Basic statistics of the weights being quantized.')}>Weights${qi}</h4>
    <div class="metric-box">
      <div class="metric-line"${ex('std', `σ = ${fmtNum(st.std, 4)}, mean ${fmtNum(st.mean, 3)}.`)}><span>Spread σ${qi}</span><b>${fmtNum(st.std, 4)}</b></div>
      <div class="metric-line"${ex('amax', `${fmtNum(st.amax, 4)} = ${(st.amax / (st.std || 1)).toFixed(1)} × σ.`)}><span>Largest |w| (absmax)${qi}</span><b>${fmtNum(st.amax, 4)}</b></div>
      <div class="metric-line"${ex('outlierRatio', `${fmtNum(st.amax, 4)} ÷ ${fmtNum(st.p999, 4)} = ${st.outlierRatio.toFixed(2)}.`)}><span>Outlier ratio${qi}</span><b style="color:${st.outlierRatio > 3 ? 'var(--danger)' : 'inherit'}">${st.outlierRatio.toFixed(2)}</b></div>
      <div class="metric-line"${ex('kurtosis', `${st.kurtosis.toFixed(1)}.`)}><span>Kurtosis${qi}</span><b>${st.kurtosis.toFixed(1)}</b></div>
    </div>
    <h4${tip('insights')}>Insights${qi}</h4>
    <ul class="findings">${insights(r, cfg).map((i) => `<li class="finding ${i.level}"${tipText(i.tip || '')}><span class="ico">${{ error: '!', warn: '!', info: 'i', ok: '✓' }[i.level]}</span><span>${esc(i.text)}</span></li>`).join('')}</ul>
    <p class="honest">Weight error is a strong hint, not a measure of answer quality. Real GPTQ / AWQ use calibration data and usually do somewhat better than plain rounding shown here.</p>
    <h4${tipText('Save or share what you see. Nothing is uploaded: files are created in your browser.')}>Export &amp; share${qi}</h4>
    <div class="btn-col">
      <button data-exp="ai" class="primary"${tipText('Copy a plain-text summary (model, sizes, measured results, recommendation) to paste into ChatGPT, Claude or any assistant.\nContains no token and no file data.')}>Copy summary for an AI assistant</button>
      <button data-exp="link"${tipText('Copy a link that reopens this exact setup: settings, generated weights and the Hugging Face model ID.\nNever includes your token or local files.')}>Copy shareable link</button>
      <button data-exp="md"${tipText('Download a Markdown report with the model, size table, results, insights and layer scan.')}>Download report (.md)</button>
      <button data-exp="png"${tipText('Download one PNG image with the key numbers and all charts.')}>Download report (.png)</button>
      <button data-exp="csv"${tipText('Download the metrics, the all-formats comparison and the layer scan as a CSV spreadsheet.')}>Download metrics (.csv)</button>
    </div>`;
}
$('#results').addEventListener('click', (e) => {
  const k = e.target.closest('[data-exp]')?.dataset.exp;
  if (!k) return;
  ({ ai: () => copyText(aiSummary(), 'Summary copied — paste it into your AI assistant'), link: copyLink, md: () => download('quantization-report.md', reportMarkdown(), 'text/markdown'), png: exportReportPNG, csv: () => download('quantization-metrics.csv', metricsCSV(), 'text/csv') })[k]();
});

/* =====================================================================
 * Bit inspector
 * ===================================================================== */
const bitBoxes = (s, cls) => `<span class="bits">${s.split('').map((b) => `<span class="${cls}">${b}</span>`).join('')}</span>`;
function floatBitsHtml(v, fmt) {
  const b = Q.floatBits(v, fmt);
  const e = b.subnormal ? 1 - fmt.bias : b.expField - fmt.bias;
  const mant = b.subnormal ? `${b.man}/${2 ** fmt.man}` : `1 + ${b.man}/${2 ** fmt.man}`;
  return `<span class="bits"><span class="s">${b.sign}</span></span> ${bitBoxes(b.expBits, 'e')} ${bitBoxes(b.manBits, 'm')}
    <div class="bits-legend"><b style="color:#991b1b">sign</b> · <b style="color:#1e40af">exponent</b> ${b.expField}${b.subnormal ? ' (subnormal)' : ` − bias ${fmt.bias} = ${e}`} · <b style="color:#166534">mantissa</b> ${b.man} → value = ${b.sign ? '−' : ''}2^${e} × (${mant}) = ${fmtNum(v, 6)}</div>`;
}
function fmtOf(cfg) { const f = Q.FORMATS[cfg.format]; return f.fmt; }
function inspectCell(k) {
  const r = S.result, cfg = S.resultCfg, h = r.heat, f = Q.FORMATS[cfg.format];
  const w = h.w[k], q = h.q[k], code = h.codes[k], seg = h.segInfo[h.seg[k]];
  const head = `<b>Row ${h.r0 + Math.floor(k / h.nc)}, column ${h.c0 + (k % h.nc)}</b>: original <b>${fmtNum(w, 6)}</b> → stored <b>${fmtNum(q, 6)}</b> (error ${fmtNum(w - q, 3)})<br>`;
  let body = '';
  const scaleBits = seg && cfg.scaleDtype === 'fp16' && seg.scale > 0 ? ` <span class="muted">— stored as FP16:</span> ${floatBitsHtml(seg.scale, Q.FLOATS.fp16).split('<div')[0]}` : '';
  if (f.kind === 'int') {
    const b = f.bits, bitsStr = ((code + 2 ** b) % 2 ** b).toString(2).padStart(b, '0');
    body = cfg.sym
      ? `Stored integer q = <b>${code}</b> → ${bitBoxes(bitsStr, 'q')} <span class="muted">(${b}-bit two's complement)</span><br>Group scale = ${fmtNum(seg.scale, 5)}${scaleBits}<br>value = q × scale = ${code} × ${fmtNum(seg.scale, 5)} = ${fmtNum(q, 6)}`
      : `Stored integer q = <b>${code}</b> → ${bitBoxes(bitsStr, 'q')} <span class="muted">(unsigned ${b}-bit)</span><br>Group scale = ${fmtNum(seg.scale, 5)}, zero-point = ${seg.zp}${scaleBits}<br>value = (q − zp) × scale = (${code} − ${seg.zp}) × ${fmtNum(seg.scale, 5)} = ${fmtNum(q, 6)}`;
  } else if (f.kind === 'float') {
    body = `Stored ${f.label} number: ${floatBitsHtml(code, f.fmt)}${cfg.granularity !== 'none' ? `Then × group scale ${fmtNum(seg.scale, 5)}${scaleBits}<br>= ${fmtNum(code, 5)} × ${fmtNum(seg.scale, 5)} = ${fmtNum(q, 6)}` : ''}`;
  } else if (f.kind === 'nf4') {
    body = `Stored 4-bit index <b>${code}</b> → ${bitBoxes(code.toString(2).padStart(4, '0'), 'q')} → NF4 level ${Q.NF4_LEVELS[code].toFixed(4)}<br>× group absmax ${fmtNum(seg.scale, 5)}${scaleBits}<br>= ${fmtNum(q, 6)}`;
  } else {
    body = `Stored as ${f.label}: ${floatBitsHtml(q, fmtOf(cfg))}`;
  }
  $('#bit-inspector').innerHTML = head + body;
}

/* =====================================================================
 * Layer sensitivity scan
 * ===================================================================== */
const isEmbName = (n) => /embed|wte|wpe|lm_head|token_embd|^output\.weight$|embed_out/.test(n);
function renderScan() {
  const el = $('#scan-body'), p = S.profile;
  if (!p) { el.innerHTML = '<p class="empty-state">Import a model with readable weights (Hugging Face, a local .safetensors / F16 .gguf file, or a direct link) to scan its layers.</p>'; return; }
  if (p.weightAccess === 'none') {
    el.innerHTML = `<div class="disabled-box" style="opacity:.6"><p class="empty-state">Not available for this model.</p></div>${msgHtml('info', p.weightNote || 'This source does not give access to the weights.', 'To enable: import from Hugging Face, open the .safetensors / F16 .gguf file, or paste a direct link to it.')}`;
    return;
  }
  const sc = S.scan;
  el.innerHTML = `
    <div class="scan-actions">
      <label class="field" style="margin:0;flex-direction:row;align-items:center;gap:8px"${tip('scanSample')}><span>Sample per tensor${qi}</span>
        <select id="scan-sample"><option value="16384">16,384 weights</option><option value="65536" selected>65,536 weights</option><option value="131072">131,072 weights</option></select></label>
      <button id="scan-start" class="primary"${tipText('Work out how much would be read, then start. You are asked before anything above 100 MB.')}>${sc?.rows ? 'Scan again' : 'Estimate & start scan'}</button>
      <button id="scan-cancel" hidden>Cancel</button>
      <div class="progress" id="scan-progress" hidden><span></span></div>
      <span class="small muted" id="scan-status">${p.allTensorsLoaded === false ? `${p.weightFiles.length} weight files — their headers are read first.` : `${fmtInt((p.tensors || []).length)} tensors in the model.`}</span>
    </div>
    <div id="scan-results"></div>`;
  if (sc?.perTensor) $('#scan-sample').value = sc.perTensor;
  $('#scan-start').addEventListener('click', startScan);
  $('#scan-cancel').addEventListener('click', () => busy?.abort());
  if (sc?.rows) renderScanResults();
}

async function startScan() {
  if (busy) return toast('Something else is running — wait or cancel it first.');
  const p = S.profile;
  const per = +$('#scan-sample').value;
  busy = new AbortController();
  const signal = busy.signal;
  const status = (t) => { $('#scan-status').textContent = t; };
  const bar = $('#scan-progress');
  $('#scan-start').disabled = true; $('#scan-cancel').hidden = false; bar.hidden = false;
  try {
    if (!p.allTensorsLoaded) { status('Reading weight-file headers…'); await ensureTensors(p, { signal, progress: (t) => status(t) }); }
    const est = scanEstimate(p, per, 4);
    if (!est.tensors.length) throw new UserError('No weight matrices stored as F32 / F16 / BF16 were found.', 'This checkpoint may already be quantized.');
    const remote = Object.values(p.fileSources).some((s) => s.kind === 'url');
    if (remote && est.bytes > 100e6) {
      const ok = await confirmModal('This scan reads more than 100 MB', `Scanning ${fmtInt(est.tensors.length)} tensors at ${fmtInt(per)} weights each would download about <b>${fmtBytes(est.bytes)}</b> in ${fmtInt(est.requests)} requests (never the whole model, which is ${fmtGB(Q.sizeGB(p.paramsTotal, DTYPE_BITS[p.originalDtype] || 16))}).<br><br>Continue, or cancel and pick a smaller sample size?`, `Download ${fmtBytes(est.bytes)}`);
      if (!ok) throw Object.assign(new Error('Cancelled'), { name: 'AbortError' });
    }
    const budget = remote ? new Budget(est.bytes * 1.02 + 1e6) : null;
    const rows = [];
    let done = 0, bytes = 0, idx = 0;
    const t0 = performance.now();
    status(`Scanning 0 of ${est.tensors.length} tensors (≈ ${fmtBytes(est.bytes)}${remote ? ' to download' : ' to read from your file'})…`);
    await Promise.all(Array.from({ length: 4 }, async () => {
      while (idx < est.tensors.length) {
        if (signal.aborted) throw Object.assign(new Error('Cancelled'), { name: 'AbortError' });
        const t = est.tensors[idx++];
        const s = await readTensorSample(p, t, per, { signal, budget, blocks: 4 });
        bytes += s.bytes;
        const r = await call({ type: 'scan', weights: s.weights, cols: s.cols }, [s.weights.buffer]);
        rows.push({ name: t.name, shape: t.shape, numel: t.numel, dtype: t.dtype, cols: s.cols, int4: r.int4, fp8: r.fp8, int8: r.int8, ...r.stats });
        done++;
        bar.firstElementChild.style.width = Math.round((100 * done) / est.tensors.length) + '%';
        if (done % 3 === 0 || done === est.tensors.length) status(`Scanned ${done} of ${est.tensors.length} tensors · ${fmtBytes(bytes)} read · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      }
    }));
    rows.sort((a, b) => a.int4 - b.int4 || b.outlierRatio - a.outlierRatio);
    S.scan = { rows, perTensor: per, bytes, ms: performance.now() - t0, N: null };
    S.scan.N = defaultN(rows);
    renderScan();
    status(`Done: ${rows.length} tensors, ${fmtBytes(bytes)} read in ${(S.scan.ms / 1000).toFixed(1)} s.`);
    $('#scan-status').textContent = `Done: ${rows.length} tensors, ${fmtBytes(bytes)} read in ${(S.scan.ms / 1000).toFixed(1)} s.`;
  } catch (e) {
    $('#scan-results').innerHTML = isAbort(e) ? msgHtml('info', 'Scan cancelled.') : msgHtml(e.level || 'error', e.message, e.next);
    if (!(e instanceof UserError) && !isAbort(e)) console.error(e);
  } finally {
    busy = null;
    if ($('#scan-start')) { $('#scan-start').disabled = false; $('#scan-cancel').hidden = true; bar.hidden = true; }
  }
}

function defaultN(rows) {
  const s = rows.map((r) => r.int4).sort((a, b) => a - b);
  const med = s[Math.floor(s.length / 2)];
  const n = rows.filter((r) => !(S.sizeOpts.keepEmb && isEmbName(r.name)) && (r.int4 < med - 2 || r.outlierRatio > 4)).length;
  return Math.min(n, Math.max(1, Math.round(rows.length * 0.1)));
}

function planOf(rows, N) {
  const p = S.profile;
  const keep = S.sizeOpts.keepEmb;
  const ranked = rows.filter((r) => !(keep && isEmbName(r.name)));
  const int8 = new Set(ranked.slice(0, N).map((r) => r.name));
  const allParams = (p.tensors || []).reduce((a, t) => a + t.numel, 0) || p.paramsTotal;
  let bitsPlan = 0, bits4 = 0, scanned = 0;
  for (const r of rows) {
    const emb = keep && isEmbName(r.name);
    const b8 = 8 + 16 / r.cols;
    bitsPlan += r.numel * (emb ? 16 : int8.has(r.name) ? b8 : 4.125);
    bits4 += r.numel * (emb ? 16 : 4.125);
    scanned += r.numel;
  }
  const other = Math.max(0, allParams - scanned) * 16; // norms, biases and non-float tensors stay 16-bit
  return { int8: [...int8], gb: (bitsPlan + other) / 8e9, gb4: (bits4 + other) / 8e9, gb16: (allParams * 16) / 8e9, params: allParams };
}

function renderScanResults() {
  const sc = S.scan, el = $('#scan-results');
  const rows = sc.rows;
  const best = Math.max(...rows.map((r) => r.int4)), worst = Math.min(...rows.map((r) => r.int4));
  const plan = planOf(rows, sc.N);
  const maxN = rows.filter((r) => !(S.sizeOpts.keepEmb && isEmbName(r.name))).length;
  const one = Q.GPU_SIZES.find((g) => Q.fitsOn(plan.gb, g, S.sizeOpts.headroom));
  el.innerHTML = `
    <div class="scan-table-wrap"><table class="data">
      <thead><tr><th class="num"${tipText('Rank: 1 = hurt most by INT4.')}>#</th><th${tipText('The weight tensor (matrix) name.')}>Tensor</th><th${tipText('Rows × columns of the matrix.')}>Shape</th>
        <th class="num"${tipText('Number of weights in this tensor.')}>Weights</th>
        <th class="num"${tipText(`SQNR after INT4 with one FP16 scale per 128 weights (4.125 bits). Lower = more damage. The table is sorted by this column.\n\n${TIPS.sqnr}`)}>INT4 g128</th>
        <th class="num"${tipText('SQNR after FP8 E4M3 with one scale per tensor (8 bits).')}>FP8</th>
        <th class="num"${tipText('SQNR after INT8 with one scale per row (≈ 8 bits).')}>INT8</th>
        <th class="num"${tip('outlierRatio')}>Outliers</th>
        <th${tipText('How much INT4 damages this tensor compared with the others: full bar = the worst tensor in this model.')}>Damage</th></tr></thead>
      <tbody>${rows.map((r, i) => {
        const d = best > worst ? (best - r.int4) / (best - worst) : 0;
        const col = d > 0.66 ? 'var(--danger)' : d > 0.33 ? 'var(--warn)' : 'var(--ok)';
        const inPlan = plan.int8.includes(r.name);
        return `<tr${tipText(`${r.name}\nINT4 g128: ${fmtDb(r.int4)} dB · FP8: ${fmtDb(r.fp8)} dB · INT8: ${fmtDb(r.int8)} dB\nOutlier ratio ${r.outlierRatio.toFixed(2)} (largest ${fmtNum(r.amax, 3)} vs 99.9th percentile ${fmtNum(r.p999, 3)}), kurtosis ${r.kurtosis.toFixed(1)}.\nSampled ${fmtInt(r.n)} of ${fmtInt(r.numel)} weights.${inPlan ? '\n\nIn the plan: kept in INT8.' : isEmbName(r.name) && S.sizeOpts.keepEmb ? '\n\nEmbedding / output layer: kept in 16-bit.' : ''}`)}>
          <td class="num">${i + 1}</td><td class="mono">${inPlan ? '<b>' : ''}${esc(r.name)}${inPlan ? '</b> <span class="tag gptq">INT8</span>' : isEmbName(r.name) && S.sizeOpts.keepEmb ? ' <span class="tag other">16-bit</span>' : ''}</td>
          <td class="mono">${r.shape.join('×')}</td><td class="num">${fmtCount(r.numel)}</td>
          <td class="num"><b style="color:${col}">${fmtDb(r.int4)}</b></td><td class="num">${fmtDb(r.fp8)}</td><td class="num">${fmtDb(r.int8)}</td>
          <td class="num" style="color:${r.outlierRatio > 3 ? 'var(--danger)' : 'inherit'}">${r.outlierRatio.toFixed(1)}</td>
          <td><div class="dmg"><span style="width:${Math.max(3, d * 100)}%;background:${col}"></span></div></td></tr>`;
      }).join('')}</tbody></table></div>
    <div class="plan">
      <div class="plan-box">
        <h3${tipText('A mixed-precision plan: the most sensitive tensors stay in INT8, everything else becomes INT4 with group size 128.\nSizes use the exact tensor sizes from the weight files. Norms, biases and other small tensors stay 16-bit.')}>Mixed-precision advice${qi}</h3>
        <p style="margin:4px 0 8px">Keep <b>${sc.N}</b> most sensitive tensor${sc.N === 1 ? '' : 's'} in INT8, rest INT4 g128 → <b>${fmtGB(plan.gb)}</b>${one ? ` (fits one ${one} GB GPU)` : ''}.</p>
        <div class="metric-line"${tipText(`All tensors INT4 g128 (4.125 bits)${S.sizeOpts.keepEmb ? ', embeddings 16-bit' : ''}.`)}><span>All INT4${qi}</span><b>${fmtGB(plan.gb4)}</b></div>
        <div class="metric-line"${tipText(`Extra memory for keeping ${sc.N} tensor${sc.N === 1 ? '' : 's'} in INT8.`)}><span>Cost of the INT8 layers${qi}</span><b>+${fmtGB(plan.gb - plan.gb4)}</b></div>
        <div class="metric-line"${tipText(`Everything in 16-bit: ${fmtCount(plan.params)} × 16 bits ÷ 8.`)}><span>FP16 / BF16${qi}</span><b>${fmtGB(plan.gb16)}</b></div>
        <label class="field" style="margin-top:8px"${tip('planN')}><span>Tensors kept in INT8: <b>${sc.N}</b>${qi}</span><input type="range" id="plan-n" min="0" max="${maxN}" step="1" value="${sc.N}" /></label>
      </div>
      <div>
        <h4 style="margin-top:0"${tipText('The tensors the plan keeps in INT8 — use these names with your quantization tool (e.g. a “modules_to_not_convert” / ignore list, or llama.cpp --tensor-type overrides).')}>Kept in INT8${qi}</h4>
        <div class="plan-list">${plan.int8.map(esc).join('<br>') || '<span class="muted">none — everything INT4</span>'}</div>
        <p class="small muted">Ranking uses INT4 SQNR on a ${fmtInt(sc.perTensor)}-weight sample of each tensor (outlier ratio breaks ties). It is a guide to which layers to protect, not a quality guarantee.</p>
      </div>
    </div>`;
  $('#plan-n').addEventListener('input', (e) => { S.scan.N = +e.target.value; renderScanResults(); });
}

/* =====================================================================
 * Exports: summary, Markdown, PNG, CSV, shareable link
 * ===================================================================== */
function download(name, content, mime) { downloadBlob(name, new Blob([content], { type: mime })); }
function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function copyText(text, msg = 'Copied to clipboard') {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
  }
  toast(msg);
}

function profileLines(p) {
  if (!p) return ['Model: none loaded (generated weights only)'];
  const l = [`Model: ${p.name}`, `Parameters: ${fmtInt(p.paramsTotal)} (${fmtCount(p.paramsTotal)}) — ${p.paramSource}`];
  if (p.paramsActive && p.paramsActive !== p.paramsTotal) l.push(`Active per token: ${fmtCount(p.paramsActive)} (${p.experts} experts, ${p.expertsUsed} per token)`);
  if (p.architecture) l.push(`Architecture: ${p.architecture}`);
  if (p.layers) l.push(`Layers: ${p.layers}${p.hiddenSize ? `, hidden size ${p.hiddenSize}` : ''}${p.contextLength ? `, context ${fmtInt(p.contextLength)} tokens` : ''}`);
  l.push(`Original data type: ${p.originalDtype}${p.quantization ? `; existing quantization: ${p.quantization.label}` : ''}`);
  return l;
}
const bitsText = (r) => fmtBits(r.bits) + (r.keptEmb ? ` (avg ${r.avgBits.toFixed(2)} with 16-bit embeddings)` : r.row.family === 'gguf' ? ' (approx.)' : '');
function sizeLines(p) {
  return sizeRows(p).map((r) => {
    const fits = Q.GPU_SIZES.filter((g) => Q.fitsOn(r.gb, g, S.sizeOpts.headroom));
    return `${r.row.label}: ${bitsText(r)} bits/weight, ${fmtGB(r.gb)}, ${fits.length ? `fits one ${fits[0]} GB GPU` : `needs ${Q.gpusNeeded(r.gb, 80, S.sizeOpts.headroom)} × 80 GB GPUs`}`;
  });
}
function resultLines() {
  const r = S.result, m = r.metrics, cfg = S.resultCfg;
  return [
    `Setting: ${cfgLabel(cfg)} (${fmtBits(r.bits)} bits per weight, scales stored as ${cfg.scaleDtype.toUpperCase()})`,
    `Weights: ${weightsLabel()}`,
    `SQNR ${fmtDb(m.sqnr)} dB (${verdict(m.sqnr).t}), RMS error ${fmtNum(m.rmse, 3)}, max error ${fmtNum(m.maxErr, 3)}, cosine ${m.cosine.toFixed(5)}`,
    `Clipped ${pct(m.clippedPct, 2)}, rounded to zero ${pct(m.zeroedPct, 2)}${m.levelsUsed !== null ? `, levels used ${m.levelsUsed}/${m.levelsAvail} (top 3 hold ${pct(m.top3Share)})` : ''}`,
    `Weight stats: σ ${fmtNum(r.stats.std, 4)}, absmax ${fmtNum(r.stats.amax, 4)}, outlier ratio ${r.stats.outlierRatio.toFixed(2)}, kurtosis ${r.stats.kurtosis.toFixed(1)}`,
  ];
}
function recommendation() {
  const p = S.profile;
  if (!p) return null;
  const rows = sizeRows(p);
  const target = Q.GPU_SIZES.map((g) => ({ g, best: rows.filter((r) => r.row.family !== 'float' || r.row.id !== 'fp32').filter((r) => Q.fitsOn(r.gb, g, S.sizeOpts.headroom)).sort((a, b) => b.avgBits - a.avgBits)[0] }));
  return target.filter((t) => t.best).map((t) => `${t.g} GB GPU: ${t.best.row.label} (${fmtGB(t.best.gb)}) is the highest-precision format that fits`);
}
function aiSummary() {
  const p = S.profile, r = S.result;
  const L = ['I am choosing a quantized version of an open-weight LLM. Here are measurements from a quantization planner (weights only; KV cache not included).', '', '## Model', ...profileLines(p)];
  if (p) L.push('', `## Size by format (usable GPU memory ${Math.round(S.sizeOpts.headroom * 100)}%${p.embParams && S.sizeOpts.keepEmb ? ', embeddings kept 16-bit for INT/GPTQ/AWQ/NF4' : ''})`, ...sizeLines(p));
  if (r) {
    L.push('', '## Measured precision loss', ...resultLines(), '', 'Insights:', ...insights(r, S.resultCfg).map((i) => `- ${i.text}`));
    L.push('', 'All formats on the same weights (SQNR, higher is better):', ...r.compare.map((c) => `- ${c.label}: ${fmtBits(c.bits)} bits, ${fmtDb(c.sqnr)} dB`));
  }
  if (S.scan?.rows) {
    const pl = planOf(S.scan.rows, S.scan.N);
    L.push('', '## Layer sensitivity scan (INT4 g128, sampled)', `Most sensitive: ${S.scan.rows.slice(0, 8).map((x) => `${x.name} (${fmtDb(x.int4)} dB)`).join(', ')}`,
      `Median INT4 SQNR: ${fmtDb(S.scan.rows.map((x) => x.int4).sort((a, b) => a - b)[Math.floor(S.scan.rows.length / 2)])} dB`,
      `Plan: keep ${S.scan.N} tensors in INT8, rest INT4 g128 → ${fmtGB(pl.gb)} (all INT4 ${fmtGB(pl.gb4)}, FP16 ${fmtGB(pl.gb16)})`);
  }
  const rec = recommendation();
  if (rec?.length) L.push('', '## What fits', ...rec.map((x) => `- ${x}`));
  L.push('', 'Note: weight error (SQNR) is a strong hint, not a measure of answer quality. Real GPTQ/AWQ use calibration and usually beat plain rounding.', 'Question: which version should I run for my GPU and use case, and what should I test?');
  return L.join('\n');
}
function reportMarkdown() {
  const p = S.profile, r = S.result;
  const L = ['# Quantization report', '', `_Generated ${new Date().toISOString().slice(0, 16).replace('T', ' ')} by the Model Quantization Planner (runs in the browser)._`, '', '## Model', ...profileLines(p).map((x) => `- ${x}`)];
  if (p) {
    L.push('', '## Size & GPU fit', '', `Usable GPU memory: ${Math.round(S.sizeOpts.headroom * 100)}%. Weights only — KV cache and activations need extra memory.`, '',
      `| Format | Bits/weight | Size | ${Q.GPU_SIZES.map((g) => `${g} GB`).join(' | ')} |`, `|---|---:|---:|${Q.GPU_SIZES.map(() => ':-:').join('|')}|`,
      ...sizeRows(p).map((x) => `| ${x.row.label} | ${bitsText(x)} | ${fmtGB(x.gb)} | ${Q.GPU_SIZES.map((g) => (Q.fitsOn(x.gb, g, S.sizeOpts.headroom) ? '✓' : `×${Q.gpusNeeded(x.gb, g, S.sizeOpts.headroom)}`)).join(' | ')} |`));
  }
  if (r) {
    L.push('', '## Precision loss', ...resultLines().map((x) => `- ${x}`), '', '### Insights', ...insights(r, S.resultCfg).map((i) => `- ${i.text}`),
      '', '### All formats on the same weights', '', '| Format | Bits/weight | SQNR (dB) |', '|---|---:|---:|', ...r.compare.map((c) => `| ${c.label} | ${fmtBits(c.bits)} | ${fmtDb(c.sqnr)} |`));
  }
  if (S.scan?.rows) {
    const pl = planOf(S.scan.rows, S.scan.N);
    L.push('', '## Layer sensitivity scan', '', `Sample: ${fmtInt(S.scan.perTensor)} weights per tensor, ${fmtBytes(S.scan.bytes)} read.`, '',
      `**Plan:** keep ${S.scan.N} tensors in INT8, rest INT4 g128 → **${fmtGB(pl.gb)}** (all INT4 ${fmtGB(pl.gb4)}, FP16 ${fmtGB(pl.gb16)}).`, '',
      '| # | Tensor | INT4 g128 dB | FP8 dB | INT8 dB | Outlier ratio |', '|---:|---|---:|---:|---:|---:|',
      ...S.scan.rows.slice(0, 60).map((x, i) => `| ${i + 1} | \`${x.name}\` | ${fmtDb(x.int4)} | ${fmtDb(x.fp8)} | ${fmtDb(x.int8)} | ${x.outlierRatio.toFixed(2)} |`));
  }
  L.push('', '---', 'Weight error is a strong hint, not a measure of answer quality. Verify a quantized model on your own tasks.');
  return L.join('\n');
}
function exportReportPNG() {
  if (!S.result) return;
  const lines = [...profileLines(S.profile), ...resultLines(), ...insights(S.result, S.resultCfg).map((i) => '• ' + i.text)];
  const charts = ['hist', 'stair', 'err', 'clip', 'cmp', 'heat'].map((k) => ({ canvas: $(`[data-chart=${k}] canvas`), title: CHART_TITLES[k] }));
  reportPNG('Quantization report', lines, charts).toBlob((b) => downloadBlob('quantization-report.png', b));
}
function metricsCSV() {
  const r = S.result, m = r.metrics, cfg = S.resultCfg;
  const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const L = ['section,name,value'];
  const add = (sec, k, v) => L.push([sec, k, v].map(q).join(','));
  if (S.profile) for (const [k, v] of [['model', S.profile.name], ['parameters', S.profile.paramsTotal], ['parameter source', S.profile.paramSource], ['original dtype', S.profile.originalDtype]]) add('model', k, v);
  for (const [k, v] of [['setting', cfgLabel(cfg)], ['weights', weightsLabel()], ['bits per weight', r.bits], ['SQNR dB', m.sqnr], ['MSE', m.mse], ['RMSE', m.rmse], ['max abs error', m.maxErr], ['cosine', m.cosine], ['clipped %', m.clippedPct], ['zeroed %', m.zeroedPct], ['levels used', m.levelsUsed ?? ''], ['levels available', m.levelsAvail], ['top-3 share %', m.top3Share ?? ''], ['sigma', r.stats.std], ['absmax', r.stats.amax], ['outlier ratio', r.stats.outlierRatio], ['kurtosis', r.stats.kurtosis]]) add('metrics', k, v);
  L.push('', 'format,bits per weight,SQNR dB');
  for (const c of r.compare) L.push([c.label, c.bits, c.sqnr].map(q).join(','));
  if (S.scan?.rows) {
    L.push('', 'rank,tensor,shape,weights,INT4 g128 SQNR dB,FP8 SQNR dB,INT8 SQNR dB,outlier ratio,kurtosis');
    S.scan.rows.forEach((x, i) => L.push([i + 1, x.name, x.shape.join('x'), x.numel, x.int4, x.fp8, x.int8, x.outlierRatio, x.kurtosis].map(q).join(',')));
  }
  return L.join('\n');
}

// Shareable link: settings + Hugging Face model ID in the URL hash (never tokens or local files)
function stateToHash() {
  const h = new URLSearchParams();
  const p = S.profile;
  if (p?.source === 'hf') h.set('m', `hf:${p.hf.id}${p.hf.rev !== 'main' ? '@' + p.hf.rev : ''}`);
  else if (p?.source === 'manual') h.set('m', `manual:${p.paramsTotal}:${p.originalDtype}`);
  if (S.wsrc === 'real' && S.real && p?.source === 'hf') h.set('t', S.real.tensor);
  const g = S.gen;
  h.set('g', [g.dist, g.sigma, g.count, g.outlierPct, g.outlierMul, g.seed].join(','));
  const c = S.cfg;
  h.set('q', [c.format, c.sym ? 1 : 0, c.granularity === 'group' ? 'g' + c.groupSize : c.granularity, c.scaleDtype, c.clip, c.clipPct, c.fullRange ? 1 : 0].join(','));
  h.set('hr', Math.round(S.sizeOpts.headroom * 100));
  return '#s=' + h.toString();
}
function copyLink() {
  const url = location.href.split('#')[0] + stateToHash();
  history.replaceState(null, '', url);
  copyText(url, 'Link copied — it reopens this setup (no token, no local files)');
}
function loadHash() {
  if (!location.hash.startsWith('#s=')) return null;
  let h;
  try { h = new URLSearchParams(location.hash.slice(3)); } catch { return null; }
  const g = (h.get('g') || '').split(',');
  if (g.length === 6) {
    const [dist, sigma, count, op, om, seed] = g;
    if (['laplace', 'normal', 'student'].includes(dist)) S.gen.dist = dist;
    const num = (v, lo, hi, d) => (Number.isFinite(+v) ? Math.min(hi, Math.max(lo, +v)) : d);
    S.gen.sigma = num(sigma, 0.002, 0.1, 0.02); S.gen.count = 2 ** Math.round(Math.log2(num(count, 4096, 1048576, 65536)));
    S.gen.outlierPct = num(op, 0, 2, 0); S.gen.outlierMul = num(om, 2, 100, 15); S.gen.seed = Math.round(num(seed, 1, 1e9, 1));
  }
  const q = (h.get('q') || '').split(',');
  if (q.length === 7 && Q.FORMATS[q[0]]) {
    const gr = q[2];
    S.cfg = { format: q[0], sym: q[1] === '1', granularity: gr.startsWith('g') ? 'group' : ['tensor', 'channel', 'none'].includes(gr) ? gr : 'group',
      groupSize: gr.startsWith('g') && [32, 64, 128, 256].includes(+gr.slice(1)) ? +gr.slice(1) : 128, scaleDtype: q[3] === 'fp32' ? 'fp32' : 'fp16',
      clip: ['none', 'percentile', 'auto'].includes(q[4]) ? q[4] : 'none', clipPct: Math.min(99.99, Math.max(99, +q[5] || 99.9)), fullRange: q[6] === '1' };
  }
  const hr = +h.get('hr');
  if (hr >= 50 && hr <= 100) S.sizeOpts.headroom = hr / 100;
  return { model: h.get('m'), tensor: h.get('t') };
}

/* =====================================================================
 * Modal & toast
 * ===================================================================== */
const modal = $('#modal');
function openModal(title, html) { $('#modal-title').textContent = title; $('#modal-body').innerHTML = html; modal.hidden = false; }
function closeModal() { modal.hidden = true; $('#modal-body').innerHTML = ''; modalResolve?.(false); modalResolve = null; }
let modalResolve = null;
function confirmModal(title, html, okLabel) {
  openModal(title, `<p style="margin:0">${html}</p><div class="modal-actions"><button class="primary" id="m-ok">${esc(okLabel)}</button><button id="m-cancel">Cancel</button></div>`);
  return new Promise((res) => {
    modalResolve = res;
    $('#m-ok').onclick = () => { modalResolve = null; res(true); closeModal(); };
    $('#m-cancel').onclick = closeModal;
  });
}
$('#modal-close').onclick = closeModal;
modal.addEventListener('pointerdown', (e) => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.hidden) closeModal(); });
let toastTimer = 0;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}

/* =====================================================================
 * Docs: glossary and bit lab
 * ===================================================================== */
const GLOSSARY = [
  ['Quantization', 'Storing each weight with fewer bits (e.g. 4 instead of 16) by rounding it to one of a small set of values. Smaller and often faster, at the cost of some precision.'],
  ['Scale', 'The number that converts stored integers back into real weights: weight ≈ integer × scale. One scale is shared by a whole tensor, a row, or a group of weights.'],
  ['Zero-point', 'In asymmetric quantization, the integer that stands for 0. It lets the range be lopsided (e.g. −0.1 … +0.3) instead of centred on zero.'],
  ['Absmax', 'The largest absolute value in a group. Symmetric quantization usually sets scale = absmax ÷ the largest integer, so the biggest weight maps exactly to the top level.'],
  ['Group size', 'How many weights share one scale. Smaller groups (32, 64) follow the data more closely but store more scales; 128 is the common sweet spot.'],
  ['Per-channel', 'One scale per output channel (row of the weight matrix).'],
  ['Clipping', 'Deliberately cutting values above a threshold so the scale shrinks and the remaining weights get finer steps.'],
  ['SQNR', 'Signal-to-quantization-noise ratio, in decibels: 10 · log10(Σw² ÷ Σerror²). Higher is better; +1 bit ≈ +6 dB.'],
  ['Subnormal', 'Very small float values below the format\'s normal range, stored with less precision. FP8 E4M3\'s smallest is 2⁻⁹ ≈ 0.00195; anything smaller rounds to zero unless a scale is used.'],
  ['Outlier', 'A weight much larger than the rest. With one scale for many weights, a single outlier makes the steps coarse for everyone.'],
  ['Calibration', 'Running some sample text through the model to measure activations, so a method (GPTQ, AWQ) can choose rounding / scales that hurt the outputs least. This tool does not calibrate.'],
  ['GPTQ', 'A 4-bit (or 3/8-bit) method that rounds weights one column at a time and corrects the remaining ones to cancel the error, using calibration data. Runs on GPUs (vLLM, Transformers).'],
  ['AWQ', 'Activation-aware weight quantization: protects the weights that matter most for typical activations by scaling them before 4-bit rounding. Runs on GPUs.'],
  ['NF4', '4-bit NormalFloat from bitsandbytes / QLoRA: 16 fixed levels placed to suit bell-shaped weights, with one absmax per 64 weights.'],
  ['FP8', '8-bit floating point. E4M3 (4 exponent, 3 mantissa bits, max 448) for weights and activations; E5M2 has more range, less precision. Native on H100 / RTX 40-series and newer.'],
  ['BF16', '“Brain float” 16-bit: same range as FP32 with 7 mantissa bits. The usual format of new model releases.'],
  ['GGUF', 'The file format of llama.cpp (also used by Ollama and LM Studio). Holds the model plus metadata; its “K-quants” like Q4_K_M mix bit widths across tensors.'],
  ['safetensors', 'The standard file format for model weights on Hugging Face: a small JSON header listing every tensor, followed by the raw numbers. This tool reads just the header and small slices.'],
  ['MoE', 'Mixture of experts: many parallel MLP “experts” per layer, of which only a few run for each token. Memory depends on total parameters, speed on active ones.'],
  ['KV cache', 'Memory that stores the attention keys and values of every token in the context. Grows with context length and batch size, on top of the weights.'],
];
$('#glossary-list').innerHTML = GLOSSARY.map(([t, d]) => `<dt>${esc(t)}</dt><dd>${esc(d)}</dd>`).join('');

function renderBitLab() {
  $('#bit-lab').innerHTML = `
    <div>
      <label class="field"${tipText('Type any number to see how a format stores it.')}><span>Value${qi}</span><input id="bl-v" type="number" step="any" value="0.0137" /></label>
      <label class="field"${tipText('The number format.')}><span>Format${qi}</span><select id="bl-f">
        <option value="fp32">FP32</option><option value="fp16">FP16</option><option value="bf16" selected>BF16</option><option value="e4m3">FP8 E4M3</option><option value="e5m2">FP8 E5M2</option><option value="e2m1">FP4 E2M1</option><option value="int8">INT8</option><option value="int4">INT4</option></select></label>
      <label class="field"${tipText('For INT, FP8 and FP4: the scale the value is divided by before rounding (in real models it comes from the group\'s absmax). Use 1 for plain floats.')}><span>Scale${qi}</span><input id="bl-s" type="number" step="any" value="1" /></label>
    </div>
    <div id="bl-out" class="bit-inspector" style="margin:0"></div>`;
  const upd = () => {
    const v = +$('#bl-v').value, f = $('#bl-f').value, s = +$('#bl-s').value || 1;
    let html;
    if (f.startsWith('int')) {
      const b = f === 'int8' ? 8 : 4, qmax = 2 ** (b - 1) - 1;
      const q = Math.max(-qmax, Math.min(qmax, Q.roundHalfEven(v / s)));
      const bits = ((q + 2 ** b) % 2 ** b).toString(2).padStart(b, '0');
      html = `round(${fmtNum(v, 6)} ÷ ${fmtNum(s, 6)}) = ${Q.roundHalfEven(v / s)}${Math.abs(Q.roundHalfEven(v / s)) > qmax ? ` → clamped to ${q}` : ''}<br>Stored: ${bitBoxes(bits, 'q')} <span class="muted">(${b}-bit two's complement)</span><br>Read back: ${q} × ${fmtNum(s, 6)} = <b>${fmtNum(q * s, 6)}</b> · error ${fmtNum(v - q * s, 3)}`;
    } else {
      const fmt = Q.FLOATS[f], sc = f === 'fp32' || f === 'fp16' || f === 'bf16' ? 1 : s;
      const stored = Q.roundToFloat(v / sc, fmt);
      html = `${sc !== 1 ? `${fmtNum(v, 6)} ÷ scale ${fmtNum(sc, 6)} = ${fmtNum(v / sc, 6)}<br>` : ''}Stored: ${floatBitsHtml(stored, fmt)}${sc !== 1 ? `Read back: ${fmtNum(stored, 6)} × ${fmtNum(sc, 6)} = <b>${fmtNum(stored * sc, 6)}</b>` : `Read back: <b>${fmtNum(stored, 8)}</b>`} · error ${fmtNum(v - stored * sc, 3)}${Math.abs(v / sc) > fmt.max ? '<br><span style="color:var(--danger)">Above the format\'s maximum — saturated.</span>' : ''}`;
    }
    $('#bl-out').innerHTML = html;
  };
  $('#bit-lab').addEventListener('input', upd);
  upd();
}
renderBitLab();

/* =====================================================================
 * Init
 * ===================================================================== */
// A share link pasted into this same tab only changes the hash: start over from the link
addEventListener('hashchange', () => { if (location.hash.startsWith('#s=')) location.reload(); });

(function init() {
  showSource('hf');
  const fromHash = loadHash();
  writeControls();
  updateLabels();
  updateRealAvailability();
  renderScan();
  renderResultsPanel();
  analyzeNow();
  if (fromHash?.model?.startsWith('hf:')) {
    $('#hf-id').value = fromHash.model.slice(3).replace('@', '/tree/');
    showSource('hf');
    doImport().then(() => {
      if (fromHash.tensor && S.profile?.weightAccess !== 'none') {
        setWsrc('real');
        prepareTensorList().then(() => { $('#t-search').value = ''; tensorOptions(); $('#t-select').value = fromHash.tensor; tensorInfo(); toast('Tensor from the link selected — press “Load sample” to read it.'); });
      }
    });
  } else if (fromHash?.model?.startsWith('manual:')) {
    const [, n, dt] = fromHash.model.split(':');
    $('#man-params').value = n; if (DTYPE_BITS[dt]) $('#man-dtype').value = dt;
    showSource('manual');
    doImport();
  }
})();
