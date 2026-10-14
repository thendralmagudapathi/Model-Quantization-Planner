/* Import: every source (Hugging Face, local files, direct links, local servers, config.json, manual)
 * is normalised into one "model profile". Only headers and small slices are ever read. */
import {
  parseSafetensorsHeader, parseIndexJson, countParams, detectPackedQuant, parseGGUF, ggufSummary,
  NeedMoreBytes, READABLE_DTYPES, DTYPE_BYTES, decode,
} from './parsers.js';
import { paramsFromConfig, readQuantConfig } from './params.js';
import { parseCount } from './quant.js';

/* =====================================================================
 * Errors with a plain-English reason and a next step
 * ===================================================================== */
export class UserError extends Error {
  constructor(message, next = '', level = 'error', code = '') { super(message); this.next = next; this.level = level; this.code = code; }
}
export const isAbort = (e) => e?.name === 'AbortError';
const abortError = () => Object.assign(new Error('Cancelled'), { name: 'AbortError' });

/* =====================================================================
 * Secrets: the Hugging Face token lives only in this variable (memory, this tab)
 * ===================================================================== */
let hfToken = '';
export const setToken = (t) => { hfToken = String(t || '').trim(); };
const isHfHost = (url) => /(^|\.)huggingface\.co$|(^|\.)hf\.co$/.test(new URL(url).hostname);

/* =====================================================================
 * Network accounting — every request is counted and shown on the page
 * ===================================================================== */
export const net = { requests: 0, bytes: 0, hosts: new Map(), onChange: null };
function noteNet(url, bytes = 0, newRequest = true) {
  const host = new URL(url).host;
  if (newRequest) net.requests++;
  net.bytes += bytes;
  net.hosts.set(host, (net.hosts.get(host) || 0) + (newRequest ? 1 : 0));
  net.onChange?.();
}

/* =====================================================================
 * Limits: at most 4 parallel requests, byte budgets for scans
 * ===================================================================== */
export const MAX_PARALLEL = 4;
let running = 0;
const waiting = [];
async function slot(fn) {
  if (running >= MAX_PARALLEL) await new Promise((r) => waiting.push(r));
  running++;
  try { return await fn(); }
  finally { running--; waiting.shift()?.(); }
}

export class Budget {
  constructor(limit) { this.limit = limit; this.used = 0; }
  take(n) {
    if (this.used + n > this.limit) throw new UserError(`Stopped: this would read more than the ${fmtBytes(this.limit)} limit.`, 'Use a smaller sample size.', 'warn', 'budget');
    this.used += n;
  }
}

export function fmtBytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(0) + ' kB';
  return b + ' B';
}

/* =====================================================================
 * HTTP helpers
 * ===================================================================== */
function networkError(url, signal, purpose) {
  if (signal?.aborted) return abortError();
  const u = new URL(url);
  if (/^(localhost|127\.0\.0\.1|\[::1\])$/.test(u.hostname)) {
    return new UserError(`Couldn't connect to ${u.host}.`, 'Is the server running? If it is, it is probably blocking browser access (CORS) — see the setup line below the server address.', 'error', 'local-cors');
  }
  if (isHfHost(url)) {
    return purpose === 'weights'
      ? new UserError('Couldn\'t read the weight file. Hugging Face sends weight files from its download servers (*.hf.co); a firewall, company proxy or ad-blocker may be blocking them.', 'The size table and generated-weight charts still work. For real-layer analysis, download one .safetensors file and use “File on my computer”.', 'warn', 'cdn')
      : new UserError('Couldn\'t reach huggingface.co.', 'Check your internet connection. A company firewall or proxy may block Hugging Face — if so, paste the model\'s config.json or enter the size manually.', 'error', 'offline');
  }
  return new UserError(`Couldn't load ${u.host}${u.pathname.length > 40 ? u.pathname.slice(0, 40) + '…' : u.pathname}.`, 'The server may not allow browser access (CORS), or it is offline. Download the file and use “File on my computer” instead.', 'error', 'cors');
}

function httpError(res, url, what = 'file') {
  const code = res.headers.get('x-error-code') || '';
  const hf = isHfHost(url);
  if (hf && code === 'GatedRepo') {
    return hfToken
      ? new UserError('This model is gated and your token does not have access to it yet.', 'Open the model page on huggingface.co, accept the licence and wait for approval — or paste its config.json instead.', 'error', 'gated')
      : new UserError('This model is gated.', 'Add your Hugging Face token (kept only in this browser tab), or paste its config.json instead.', 'error', 'gated');
  }
  if (hf && (res.status === 401 || code === 'RepoNotFound')) {
    return new UserError('Model not found, or it is private.', hfToken ? 'Check the ID (organisation/name). Your token may not have access to this private model.' : 'Check the ID (organisation/name, e.g. Qwen/Qwen2.5-7B-Instruct). For private models add your token.', 'error', 'notfound');
  }
  if (res.status === 404) return new UserError(`The ${what} was not found (404).`, 'Check the link or the branch name.', 'error', 'notfound');
  if (res.status === 403) return new UserError(`Access to the ${what} was refused (403).`, 'The file may be private, or the link may have expired.', 'error', 'forbidden');
  if (res.status === 429) return new UserError('Too many requests: the server is rate-limiting.', 'Wait a minute and try again (a Hugging Face token gives higher limits).', 'warn', 'ratelimit');
  return new UserError(`The server answered with an error (${res.status}).`, 'Try again in a moment.', 'error', 'http');
}

async function fetchChecked(url, { signal, auth = false, purpose, method = 'GET', body, headers = {} } = {}) {
  const h = { ...headers };
  if (auth && hfToken && isHfHost(url)) h.Authorization = `Bearer ${hfToken}`;
  let res;
  try {
    res = await fetch(url, { method, body, headers: h, signal, credentials: 'omit', referrerPolicy: 'no-referrer', cache: 'no-store' });
  } catch {
    noteNet(url);
    throw networkError(url, signal, purpose);
  }
  noteNet(url);
  return res;
}

export async function fetchJSON(url, opts = {}) {
  const res = await fetchChecked(url, opts);
  if (!res.ok) throw httpError(res, url, opts.what);
  const text = await res.text();
  noteNet(url, text.length, false);
  try { return JSON.parse(text); }
  catch { throw new UserError(`${opts.what || 'The file'} is not valid JSON.`, 'Check that the link points to the raw file, not a web page.'); }
}

/* =====================================================================
 * Byte sources: { kind: 'url', url, auth } or { kind: 'file', file }
 * ===================================================================== */
export async function readBytes(src, start, end, { signal, budget } = {}) {
  if (end <= start) return new Uint8Array(0);
  if (signal?.aborted) throw abortError();
  if (src.kind === 'file') {
    const buf = await src.file.slice(start, end).arrayBuffer();
    if (signal?.aborted) throw abortError();
    return new Uint8Array(buf);
  }
  budget?.take(end - start);
  return slot(async () => {
    if (signal?.aborted) throw abortError();
    const ctrl = new AbortController();
    const onAbort = () => ctrl.abort();
    signal?.addEventListener('abort', onAbort);
    try {
      const res = await fetchChecked(src.url, { signal: ctrl.signal, auth: src.auth, purpose: 'weights', headers: { Range: `bytes=${start}-${end - 1}` } });
      if (res.status === 200) {
        // The server ignored Range and is sending the whole file: stop immediately.
        ctrl.abort();
        throw new UserError('This server ignored the partial-download (Range) request and started sending the whole file, so the tool stopped it.', 'Download the file and use “File on my computer”, or use a host that supports Range requests (Hugging Face does).', 'error', 'norange');
      }
      if (res.status === 416) return new Uint8Array(0);
      if (res.status !== 206) {
        const finalHost = res.url ? new URL(res.url).host : '';
        const html = /text\/html/i.test(res.headers.get('content-type') || '');
        if (res.redirected && finalHost && finalHost !== new URL(src.url).host && (res.status === 403 || html)) {
          ctrl.abort();
          throw new UserError(`The download server ${finalHost} refused the request (${res.status}${html ? ', it sent a web page instead of file data' : ''}). A firewall, company proxy or ad-blocker is probably blocking it.`, 'The size table and generated-weight charts still work. For real-layer analysis, try another network, or download one .safetensors file and use “File on my computer”.', 'warn', 'cdn');
        }
        throw httpError(res, src.url);
      }
      const cr = res.headers.get('content-range');
      const total = cr && /\/(\d+)\s*$/.exec(cr);
      if (total) src.size = Number(total[1]);
      const buf = new Uint8Array(await res.arrayBuffer());
      if (buf.byteLength > end - start) throw new UserError('The server sent more data than asked for.', 'Use “File on my computer” instead.', 'error', 'norange');
      noteNet(src.url, buf.byteLength, false);
      return buf;
    } catch (e) {
      if (signal?.aborted) throw abortError();
      if (e instanceof UserError) throw e;
      throw networkError(src.url, signal, 'weights');
    } finally { signal?.removeEventListener('abort', onAbort); }
  });
}

const concat = (a, b) => { const o = new Uint8Array(a.byteLength + b.byteLength); o.set(a); o.set(b, a.byteLength); return o; };

export async function readSafetensorsHeader(src, opts = {}) {
  let bytes = await readBytes(src, 0, 256 * 1024, opts);
  let r = parseSafetensorsHeader(bytes);
  if (r.need) {
    if (bytes.byteLength < 8 || (src.kind === 'file' && src.file.size < r.need)) throw new UserError('The .safetensors file is cut short (truncated).', 'Download it again.');
    bytes = concat(bytes, await readBytes(src, bytes.byteLength, r.need, opts));
    r = parseSafetensorsHeader(bytes);
    if (r.need) throw new UserError('The .safetensors header is cut short.', 'The file may be damaged or still downloading.');
  }
  return r;
}

const GGUF_HEADER_CAP = 192 * 1024 * 1024;
export async function readGGUFHeader(src, opts = {}) {
  let bytes = await readBytes(src, 0, 1024 * 1024, opts);
  for (;;) {
    try { return parseGGUF(bytes); }
    catch (e) {
      if (!(e instanceof NeedMoreBytes)) throw new UserError(e.message, 'Check that this is a .gguf model file.');
      const want = Math.min(GGUF_HEADER_CAP, Math.max(e.need, bytes.byteLength * 2));
      if (want <= bytes.byteLength) throw new UserError('The GGUF header is larger than the 192 MB safety limit.', 'This file is unusual; try another one.');
      const more = await readBytes(src, bytes.byteLength, want, opts);
      if (!more.byteLength) throw new UserError('The GGUF file ended in the middle of its header.', 'The file may be damaged or still downloading.');
      bytes = concat(bytes, more);
    }
  }
}

/* =====================================================================
 * Model profile
 * ===================================================================== */
const DTYPE_NAMES = { bfloat16: 'BF16', float16: 'FP16', float32: 'FP32', half: 'FP16', float: 'FP32', BF16: 'BF16', F16: 'FP16', F32: 'FP32', F8_E4M3: 'FP8', F8_E5M2: 'FP8', F64: 'FP64' };
const dtypeName = (d) => DTYPE_NAMES[d] || (d ? String(d).toUpperCase() : null);
export const DTYPE_BITS = { FP32: 32, FP16: 16, BF16: 16, FP8: 8, FP64: 64 };

export function makeProfile(o = {}) {
  return {
    name: '', source: '', sourceDetail: '', paramsTotal: 0, paramsActive: null, paramSource: '', paramExact: false,
    layers: null, hiddenSize: null, intermediateSize: null, vocabSize: null, contextLength: null, architecture: null,
    heads: null, kvHeads: null, experts: null, expertsUsed: null,
    originalDtype: null, quantization: null, tensors: null, weightAccess: 'none', weightNote: '',
    embParams: 0, assumptions: [], warnings: [], files: [], fileSources: {}, weightFiles: [], hf: null, gguf: null,
    ...o,
  };
}

function applyConfig(p, cfg) {
  p.config = cfg;
  p.architecture = cfg.architectures?.[0] || cfg.model_type || p.architecture;
  p.originalDtype ||= dtypeName(cfg.torch_dtype || cfg.dtype || cfg.text_config?.torch_dtype);
  const q = readQuantConfig(cfg.quantization_config || cfg.text_config?.quantization_config);
  if (q) p.quantization = q;
  try {
    const d = paramsFromConfig(cfg);
    p.derived = d;
    Object.assign(p, {
      layers: d.layers, hiddenSize: d.hiddenSize, intermediateSize: d.intermediateSize, vocabSize: d.vocabSize,
      contextLength: d.contextLength ?? p.contextLength, heads: d.heads, kvHeads: d.kvHeads,
      experts: d.experts ?? null, expertsUsed: d.expertsUsed ?? null, embParams: d.embedding,
    });
    p.assumptions = d.assumptions;
    if (d.experts) p.paramsActive = d.active;
    return d;
  } catch (e) {
    p.warnings.push({ level: 'warn', text: e.message });
    return null;
  }
}

function setParams(p, total, source, exact) {
  p.paramsTotal = total; p.paramSource = source; p.paramExact = exact;
}

// Weight tensors are analysable when stored as plain floats (not packed / already quantized).
function decideWeightAccess(p, kind) {
  if (!p.tensors?.length) return;
  const packed = detectPackedQuant(p.tensors);
  const readable = p.tensors.filter((t) => READABLE_DTYPES.has(t.dtype) && t.shape.length >= 2);
  if (packed || (p.quantization && !['fp8'].includes(p.quantization.method) && p.quantization.bits && p.quantization.bits < 16)) {
    p.weightAccess = 'none';
    const base = p.hf?.baseModel;
    p.weightNote = `This checkpoint is already quantized (${(packed || p.quantization?.method || '').toUpperCase()}): its weights are packed integers, not normal weights, so they can't be re-analysed.${base ? ` Analyse the original model (${base}) instead.` : ''}`;
    p.analyseBase = base || null;
    return;
  }
  if (!readable.length) {
    p.weightAccess = 'none';
    p.weightNote = 'None of the weight matrices are stored as plain F32 / F16 / BF16 numbers (they are already quantized), so real-weight analysis is not available.';
    return;
  }
  p.weightAccess = kind;
}

/* =====================================================================
 * Hugging Face
 * ===================================================================== */
export function parseHfInput(s) {
  let t = String(s || '').trim();
  if (!t) throw new UserError('Type a model ID, e.g. Qwen/Qwen2.5-7B-Instruct.', '', 'warn');
  const m = t.match(/^https?:\/\/(?:www\.)?(?:huggingface\.co|hf\.co)\/(.*)$/i);
  if (/^https?:\/\//i.test(t) && !m) throw new UserError('That link is not a huggingface.co link.', 'Use “Direct file link” for files on other websites.');
  let path = (m ? m[1] : t).replace(/[?#].*$/, '').replace(/\/+$/, '');
  let parts = path.split('/').filter(Boolean);
  if (parts[0] === 'api' && parts[1] === 'models') parts = parts.slice(2);
  if (parts[0] === 'models') parts = parts.slice(1);
  if (['datasets', 'spaces', 'docs', 'blog', 'collections', 'papers'].includes(parts[0])) throw new UserError(`That is a Hugging Face ${parts[0].replace(/s$/, '')} link, not a model.`, 'Open the model page and copy its link, or type organisation/name.');
  let id, rest;
  if (parts.length >= 2 && !['tree', 'blob', 'resolve'].includes(parts[1])) { id = `${parts[0]}/${parts[1]}`; rest = parts.slice(2); }
  else { id = parts[0]; rest = parts.slice(1); }
  if (!id || !/^[\w.-]+(\/[\w.-]+)?$/.test(id)) throw new UserError(`"${t}" doesn't look like a model ID.`, 'Use the form organisation/name, e.g. meta-llama/Llama-3.1-8B.');
  let rev = 'main', file = null;
  if (['tree', 'blob', 'resolve'].includes(rest[0]) && rest[1]) {
    rev = decodeURIComponent(rest[1]);
    if (rest[0] !== 'tree' && rest.length > 2) file = rest.slice(2).map(decodeURIComponent).join('/');
  }
  return { id, rev, file };
}

export const hfFileUrl = (id, rev, path) => `https://huggingface.co/${id}/resolve/${encodeURIComponent(rev)}/${path.split('/').map(encodeURIComponent).join('/')}`;
const hfApi = (id, rev) => `https://huggingface.co/api/models/${id}${rev && rev !== 'main' ? `/revision/${encodeURIComponent(rev)}` : ''}`;

export async function importHF(input, { signal, progress = () => {} } = {}) {
  const { id, rev } = parseHfInput(input);
  progress('Asking huggingface.co about the model…');
  const info = await fetchJSON(`${hfApi(id, rev)}?blobs=true`, { signal, auth: true, what: 'Model information' });
  const files = (info.siblings || []).map((s) => ({ name: s.rfilename, size: s.size ?? null }));
  const has = (n) => files.some((f) => f.name === n);
  const card = info.cardData || {};
  const baseModel = Array.isArray(card.base_model) ? card.base_model[0] : card.base_model || null;
  const p = makeProfile({
    name: info.id || id, source: 'hf', sourceDetail: `huggingface.co/${info.id || id}${rev !== 'main' ? ` @ ${rev}` : ''}`, files,
    hf: { id: info.id || id, rev, sha: info.sha, gated: info.gated, baseModel, tags: info.tags || [], pipeline: info.pipeline_tag,
      library: info.library_name, downloads: info.downloads, likes: info.likes, license: card.license, lastModified: info.lastModified },
  });

  // config.json → architecture, details, quantization_config
  let derived = null;
  if (has('config.json')) {
    progress('Reading config.json…');
    try {
      const cfg = await fetchJSON(hfFileUrl(p.hf.id, rev, 'config.json'), { signal, auth: true, what: 'config.json' });
      derived = applyConfig(p, cfg);
    } catch (e) {
      if (isAbort(e)) throw e;
      if (e.code === 'gated') p.gatedBlocked = true;
      else p.warnings.push({ level: 'warn', text: `config.json could not be read: ${e.message}`, next: e.next });
    }
  }
  if (!p.quantization && info.config?.quantization_config) p.quantization = readQuantConfig(info.config.quantization_config);
  if (!p.architecture) p.architecture = info.config?.architectures?.[0] || info.config?.model_type || null;
  p.quantizedCheckpoint = !!p.quantization || /gguf/.test((info.tags || []).join(' '));

  // Parameter count, best source first
  const st = info.safetensors;
  if (st?.total) {
    const dtypes = Object.entries(st.parameters || {}).sort((a, b) => b[1] - a[1]);
    const mainFloat = dtypes.find(([d]) => /^(BF16|F16|F32|F8)/.test(d));
    if (!p.originalDtype && mainFloat && !p.quantization) p.originalDtype = dtypeName(mainFloat[0]);
    if (p.quantization && derived) {
      setParams(p, derived.total, 'Computed from config.json (this repo stores packed quantized weights, so file counts are less reliable)', false);
    } else if (derived?.tied && st.total - derived.total === derived.vocabSize * derived.hiddenSize) {
      setParams(p, derived.total, 'config.json (the checkpoint stores a duplicate copy of the tied embedding matrix; unique parameters shown)', true);
      p.assumptions.push(`Hugging Face reports ${st.total.toLocaleString('en-US')} because the file stores the shared embedding twice.`);
    } else {
      setParams(p, st.total, 'Hugging Face safetensors count (exact)', true);
    }
    p.dtypeBreakdown = st.parameters;
  }

  // Weight files
  const index = files.find((f) => f.name === 'model.safetensors.index.json');
  const single = files.filter((f) => f.name.endsWith('.safetensors') && !f.name.includes('/'));
  if (index) {
    progress('Reading model.safetensors.index.json…');
    try {
      const idx = parseIndexJson(await fetchJSON(hfFileUrl(p.hf.id, rev, index.name), { signal, auth: true, what: 'The index file' }));
      p.weightFiles = idx.files;
      p.weightMap = idx.weightMap;
      if (!p.paramsTotal && idx.totalSize) {
        const bytesPer = (DTYPE_BITS[p.originalDtype] || 16) / 8;
        setParams(p, Math.round(idx.totalSize / bytesPer), `model.safetensors.index.json total size ÷ ${bytesPer} bytes (approximate)`, false);
      }
    } catch (e) {
      if (isAbort(e)) throw e;
      if (e.code === 'gated') p.gatedBlocked = true;
      else p.warnings.push({ level: 'warn', text: `The weight index could not be read: ${e.message}` });
    }
  } else if (single.length) {
    p.weightFiles = single.map((f) => f.name);
  }
  for (const f of p.weightFiles) p.fileSources[f] = { kind: 'url', url: hfFileUrl(p.hf.id, rev, f), auth: true, size: files.find((x) => x.name === f)?.size };

  // GGUF-only repo
  const ggufFiles = files.filter((f) => f.name.toLowerCase().endsWith('.gguf'));
  if (!p.weightFiles.length && ggufFiles.length) {
    let g = info.gguf;
    if (!g) {
      try { g = (await fetchJSON(`${hfApi(p.hf.id, rev)}?expand[]=gguf`, { signal, auth: true })).gguf; } catch (e) { if (isAbort(e)) throw e; }
    }
    if (g) {
      if (!p.paramsTotal && g.total) setParams(p, g.total, 'GGUF header (as reported by Hugging Face)', true);
      p.architecture ||= g.architecture; p.contextLength ||= g.context_length;
    }
    p.quantization ||= { method: 'gguf', label: 'GGUF (several levels in this repo)', bits: null };
    p.quantizedCheckpoint = true;
    const plain = ggufFiles.find((f) => /(^|[-_.])(f16|bf16|fp16|f32)([-_.]|\.gguf$)/i.test(f.name) && !/-0000[2-9]-of-/.test(f.name));
    if (plain) {
      p.weightFiles = [plain.name];
      p.fileSources[plain.name] = { kind: 'url', url: hfFileUrl(p.hf.id, rev, plain.name), auth: true, size: plain.size, gguf: true };
    } else {
      p.weightNote = 'This repo only has already-quantized GGUF files, so their weights can\'t be re-analysed.' + (baseModel ? ` Analyse the original model (${baseModel}) instead.` : '');
      p.analyseBase = baseModel;
    }
  }

  if (!p.paramsTotal && derived) setParams(p, derived.total, 'Estimated from config.json', false);

  // Try the first weight-file header now (proves whether weight access works from this browser)
  if (p.weightFiles.length && !p.weightNote) {
    progress('Checking access to the weight files…');
    try {
      await ensureTensors(p, { signal, progress, limitFiles: 1 });
      if (!p.paramsTotal && p.tensors?.length && p.weightFiles.length === 1) {
        const c = countParams(p.tensors, p.quantization);
        setParams(p, c.total, 'safetensors header (exact)', true);
      }
      decideWeightAccess(p, 'http-range');
    } catch (e) {
      if (isAbort(e)) throw e;
      p.weightAccess = 'none';
      p.weightNote = `${e.message} ${e.next || ''}`.trim();
      p.weightError = e.code;
    }
  } else if (!p.weightNote && !p.weightFiles.length) {
    p.weightNote = p.gatedBlocked ? 'The weight files of this gated model need your Hugging Face token.' : 'No .safetensors weight files were found in this repo.';
  }
  if (p.gatedBlocked) {
    p.warnings.unshift({ level: 'warn', text: hfToken ? 'This model is gated and your token does not have access to it yet, so config.json and the weights could not be read. The parameter count above comes from Hugging Face and is exact.' : 'This model is gated, so config.json and the weights could not be read. The parameter count above comes from Hugging Face and is exact.',
      next: hfToken ? 'Accept the licence on the model page and wait for approval, then import again.' : 'Add your Hugging Face token (kept only in this browser tab) and import again — or paste its config.json for layer details.' });
  }
  if (p.weightNote && p.analyseBase === undefined && p.quantizedCheckpoint) p.analyseBase = baseModel;

  if (!p.paramsTotal) {
    throw new UserError('Couldn\'t find a parameter count for this model.', p.gatedBlocked ? 'This model is gated — add your token, or paste its config.json.' : 'Paste its config.json, or enter the size manually.');
  }
  if (!p.originalDtype) p.originalDtype = p.quantization ? 'BF16' : 'BF16';
  return p;
}

/** Load safetensors / GGUF headers of the profile's weight files (all of them, or the first `limitFiles`). */
export async function ensureTensors(p, { signal, progress = () => {}, limitFiles = Infinity } = {}) {
  p.loadedFiles ||= new Set();
  const todo = p.weightFiles.filter((f) => !p.loadedFiles.has(f)).slice(0, Math.max(0, limitFiles - p.loadedFiles.size));
  let done = 0;
  await Promise.all(todo.map(async (f) => {
    const src = p.fileSources[f];
    let tensors;
    if (src.gguf || f.toLowerCase().endsWith('.gguf')) {
      const g = await readGGUFHeader(src, { signal });
      p.gguf = ggufSummary(g);
      tensors = g.tensors;
    } else {
      tensors = (await readSafetensorsHeader(src, { signal })).tensors;
    }
    for (const t of tensors) t.file = f;
    p.tensors = (p.tensors || []).concat(tensors);
    p.loadedFiles.add(f);
    progress(`Read ${++done} of ${todo.length} weight-file header${todo.length === 1 ? '' : 's'}…`, done / todo.length);
  }));
  if (p.tensors) p.tensors.sort((a, b) => naturalCmp(a.name, b.name));
  p.allTensorsLoaded = p.loadedFiles.size === p.weightFiles.length;
  return p.tensors;
}
const naturalCmp = (a, b) => a.localeCompare(b, 'en', { numeric: true });

/* ---------- Ready-made quantized versions ---------- */
export function formatOfRepo(m) {
  const tags = (m.tags || []).map((t) => String(t).toLowerCase());
  const qm = String(m.config?.quantization_config?.quant_method || '').toLowerCase();
  const id = m.id.toLowerCase();
  if (tags.includes('gguf') || m.gguf) return { key: 'gguf', label: 'GGUF' };
  if (qm === 'gptq' || tags.includes('gptq') || /gptq/.test(id)) return { key: 'gptq', label: 'GPTQ' };
  if (qm === 'awq' || tags.includes('awq') || /awq/.test(id)) return { key: 'awq', label: 'AWQ' };
  if (qm === 'bitsandbytes' || tags.includes('bitsandbytes') || /bnb/.test(id)) return { key: 'bnb', label: 'bitsandbytes' };
  if (qm === 'fp8' || /fp8/.test(id)) return { key: 'fp8', label: 'FP8' };
  if (qm === 'compressed-tensors') return { key: 'fp8', label: 'compressed-tensors' };
  if (tags.includes('mlx') || m.library_name === 'mlx' || /mlx/.test(id)) return { key: 'other', label: 'MLX' };
  if (/exl2/.test(id) || tags.includes('exl2')) return { key: 'other', label: 'EXL2' };
  if (tags.includes('onnx') || /onnx/.test(id)) return { key: 'other', label: 'ONNX' };
  if (/mlc/.test(id) || m.library_name === 'mlc-llm') return { key: 'other', label: 'MLC' };
  return { key: 'other', label: m.library_name || 'other' };
}

export async function listQuantized(baseId, { signal } = {}) {
  const url = `https://huggingface.co/api/models?filter=${encodeURIComponent(`base_model:quantized:${baseId}`)}&sort=downloads&direction=-1&limit=40&expand[]=tags&expand[]=downloads&expand[]=library_name&expand[]=config&expand[]=gguf`;
  const list = await fetchJSON(url, { signal, auth: true, what: 'The list of quantized versions' });
  return list.map((m) => ({ id: m.id, downloads: m.downloads || 0, format: formatOfRepo(m), quant: readQuantConfig(m.config?.quantization_config), ggufTotal: m.gguf?.total || null }));
}

const GGUF_LEVEL = /(?:^|[-_.])((?:I?Q\d(?:_[0-9A-Z]+)*)|BF16|F16|FP16|F32)(?=[-_.]|$)/i;
/** File sizes of a repo: GGUF files grouped by level, otherwise total weight-file size. */
export async function repoSizes(id, { signal } = {}) {
  const info = await fetchJSON(`${hfApi(id)}?blobs=true`, { signal, auth: true, what: 'Repo file list' });
  const sib = info.siblings || [];
  const gg = sib.filter((s) => s.rfilename.toLowerCase().endsWith('.gguf'));
  if (gg.length) {
    const levels = new Map();
    for (const s of gg) {
      if (/mmproj/i.test(s.rfilename)) continue;
      const base = s.rfilename.split('/').pop().replace(/-\d{5}-of-\d{5}/, '').replace(/\.gguf$/i, '');
      const lv = (GGUF_LEVEL.exec(base)?.[1] || base).toUpperCase();
      levels.set(lv, (levels.get(lv) || 0) + (s.size || 0));
    }
    return { gguf: [...levels.entries()].sort((a, b) => a[1] - b[1]) };
  }
  const w = sib.filter((s) => /\.(safetensors|bin|pt)$/.test(s.rfilename));
  return { total: w.reduce((a, s) => a + (s.size || 0), 0), files: w.length };
}

/* =====================================================================
 * Direct link
 * ===================================================================== */
export async function importUrl(url, { signal, progress = () => {} } = {}) {
  let u;
  try { u = new URL(String(url).trim()); } catch { throw new UserError('That is not a valid link.', 'It should start with https://'); }
  if (!/^https?:$/.test(u.protocol)) throw new UserError('Only http:// and https:// links are supported.');
  if (globalThis.location?.protocol === 'https:' && u.protocol === 'http:' && !/^(localhost|127\.0\.0\.1|\[::1\])$/.test(u.hostname)) {
    throw new UserError('This page is served over https, so the browser blocks plain http:// links.', 'Use an https:// link, or download the file and use “File on my computer”.');
  }
  if (isHfHost(u.href) && /\/blob\//.test(u.pathname)) u.pathname = u.pathname.replace('/blob/', '/resolve/');
  const auth = isHfHost(u.href);
  const name = decodeURIComponent(u.pathname.split('/').pop() || 'file');
  const lower = name.toLowerCase();
  const p = makeProfile({ name, source: 'url', sourceDetail: u.host + u.pathname });

  if (lower.endsWith('.json')) {
    progress(`Reading ${name}…`);
    const j = await fetchJSON(u.href, { signal, auth, what: name });
    if (j.weight_map) {
      const idx = parseIndexJson(j);
      p.weightFiles = idx.files;
      for (const f of idx.files) p.fileSources[f] = { kind: 'url', url: new URL(f, u.href).href, auth };
      try { applyConfig(p, await fetchJSON(new URL('config.json', u.href).href, { signal, auth, what: 'config.json' })); } catch (e) { if (isAbort(e)) throw e; }
      p.name = guessName(u, p.config) || p.name;
      progress('Reading the weight-file headers…');
      await ensureTensors(p, { signal, progress });
      const c = countParams(p.tensors, p.quantization);
      setParams(p, c.total, `safetensors headers of ${idx.files.length} file${idx.files.length === 1 ? '' : 's'} (exact${c.unpacked ? ', packed tensors unpacked' : ''})`, true);
      decideWeightAccess(p, 'http-range');
      return finish(p);
    }
    const d = applyConfig(p, j);
    if (!d) throw new UserError('This JSON file is neither a config.json nor a model.safetensors.index.json.', 'Link to the model\'s config.json or index file.');
    p.name = guessName(u, p.config) || 'Model from config.json';
    setParams(p, d.total, 'Estimated from config.json', false);
    return finish(p);
  }
  const src = { kind: 'url', url: u.href, auth };
  if (lower.endsWith('.gguf')) {
    progress('Reading the GGUF header…');
    p.weightFiles = [name]; p.fileSources[name] = { ...src, gguf: true };
    await ensureTensors(p, { signal });
    applyGGUF(p);
    return finish(p);
  }
  progress('Reading the safetensors header…');
  p.weightFiles = [name]; p.fileSources[name] = src;
  await ensureTensors(p, { signal });
  try { applyConfig(p, await fetchJSON(new URL('config.json', u.href).href, { signal, auth, what: 'config.json' })); } catch (e) { if (isAbort(e)) throw e; }
  const c = countParams(p.tensors, p.quantization);
  setParams(p, c.total, `safetensors header (exact${c.unpacked ? ', packed tensors unpacked' : ''})`, true);
  if (/-0000\d-of-\d{5}/.test(name)) p.warnings.push({ level: 'warn', text: 'This looks like one shard of a sharded model, so the parameter count covers this file only.', next: 'Link to model.safetensors.index.json to load every shard.' });
  p.originalDtype ||= mainDtype(p.tensors);
  p.name = guessName(u, p.config) || name;
  decideWeightAccess(p, 'http-range');
  return finish(p);
}

function guessName(u, cfg) {
  const m = u.href.match(/huggingface\.co\/([\w.-]+\/[\w.-]+)\/resolve\//);
  if (m) return m[1];
  if (cfg?._name_or_path) return cfg._name_or_path;
  const dir = u.pathname.split('/').filter(Boolean).slice(-2, -1)[0];
  return dir ? decodeURIComponent(dir) : '';
}

function mainDtype(tensors) {
  const c = {};
  for (const t of tensors || []) c[t.dtype] = (c[t.dtype] || 0) + t.numel;
  const top = Object.entries(c).sort((a, b) => b[1] - a[1])[0];
  return top ? dtypeName(top[0]) : null;
}

function applyGGUF(p) {
  const g = p.gguf;
  Object.assign(p, {
    name: g.name || p.name, architecture: g.architecture, layers: g.layers, hiddenSize: g.hiddenSize,
    intermediateSize: g.intermediateSize, contextLength: g.contextLength, vocabSize: g.vocabSize, heads: g.heads, kvHeads: g.kvHeads,
    experts: g.experts || null, expertsUsed: g.expertsUsed || null,
  });
  setParams(p, g.paramsTotal, 'GGUF header (exact)', true);
  if (g.vocabSize && g.hiddenSize) p.embParams = g.vocabSize * g.hiddenSize * (p.tensors.some((t) => t.name === 'output.weight') ? 2 : 1);
  const plain = ['F32', 'F16', 'BF16'].includes(g.mainType);
  p.originalDtype = plain ? dtypeName(g.mainType) : 'BF16';
  if (!plain) {
    p.quantization = { method: 'gguf', label: `GGUF ${g.fileType} (${g.bitsPerWeight.toFixed(2)} bits/weight in this file)`, bits: g.bitsPerWeight };
    p.quantizedCheckpoint = true;
  }
  p.assumptions.push(`GGUF file: ${g.tensorBytes ? fmtBytes(g.tensorBytes) : '?'} of tensor data, ${g.bitsPerWeight.toFixed(2)} bits per weight on average.`);
  decideWeightAccess(p, p.fileSources[p.weightFiles[0]].kind === 'file' ? 'local-file' : 'http-range');
  if (p.weightAccess === 'none' && !plain) p.weightNote = `This GGUF file is already quantized (${g.fileType}), so its weights can't be re-analysed. Open an F16 / BF16 GGUF or the original .safetensors instead.`;
}

function finish(p) {
  if (!p.paramsTotal) throw new UserError('Couldn\'t work out the parameter count from this source.', 'Try “Paste config.json” or “Enter details manually”.');
  p.originalDtype ||= 'BF16';
  return p;
}

/* =====================================================================
 * Files on this computer (read with File.slice — nothing leaves the device)
 * ===================================================================== */
export async function importFiles(fileList, { signal, progress = () => {} } = {}) {
  const files = [...fileList];
  if (!files.length) throw new UserError('Choose at least one file.', 'Drop a .safetensors, .gguf or config.json file on the box.', 'warn');
  const p = makeProfile({ source: 'file', files: files.map((f) => ({ name: f.name, size: f.size })) });
  const ignored = [];
  let cfg = null, idx = null;
  for (const f of files) {
    const n = f.name.toLowerCase();
    if (n.endsWith('.json')) {
      if (f.size > 50 * 1024 * 1024) { ignored.push(f.name); continue; }
      let j;
      try { j = JSON.parse(await f.text()); } catch { throw new UserError(`${f.name} is not valid JSON.`); }
      if (j.weight_map) idx = parseIndexJson(j);
      else if (j.hidden_size || j.n_embd || j.d_model || j.text_config || j.model_type) cfg = j;
      else ignored.push(f.name);
    }
  }
  if (cfg) applyConfig(p, cfg);
  const st = files.filter((f) => f.name.toLowerCase().endsWith('.safetensors'));
  const gg = files.filter((f) => f.name.toLowerCase().endsWith('.gguf'));
  for (const f of files) if (!/\.(safetensors|gguf|json)$/i.test(f.name)) ignored.push(f.name);

  if (st.length) {
    p.weightFiles = st.map((f) => f.name);
    for (const f of st) p.fileSources[f.name] = { kind: 'file', file: f, size: f.size };
    progress(`Reading ${st.length} safetensors header${st.length === 1 ? '' : 's'}…`);
    await ensureTensors(p, { signal, progress });
    const c = countParams(p.tensors, p.quantization);
    const missing = idx ? idx.files.filter((f) => !st.some((s) => s.name === f)) : [];
    if (missing.length && p.derived) {
      setParams(p, p.derived.total, `Estimated from config.json (${missing.length} of ${idx.files.length} shards were not opened)`, false);
      p.warnings.push({ level: 'info', text: `Only ${st.length} of ${idx.files.length} shards were opened, so the tensor list is partial.`, next: 'Open all shards together to analyse every layer.' });
    } else {
      setParams(p, c.total, `safetensors header${st.length === 1 ? '' : 's'} (exact${c.unpacked ? ', packed tensors unpacked' : ''})`, true);
      if (missing.length) p.warnings.push({ level: 'warn', text: `Only ${st.length} of ${idx.files.length} shards were opened, so the parameter count covers those files only.`, next: 'Open all shards (and config.json) together.' });
    }
    p.originalDtype ||= mainDtype(p.tensors);
    p.name = cfg?._name_or_path || st[0].name.replace(/(-\d{5}-of-\d{5})?\.safetensors$/i, '');
    p.sourceDetail = `${st.length} local file${st.length === 1 ? '' : 's'} (${fmtBytes(st.reduce((a, f) => a + f.size, 0))})`;
    decideWeightAccess(p, 'local-file');
  } else if (gg.length) {
    const f = gg[0];
    p.weightFiles = [f.name]; p.fileSources[f.name] = { kind: 'file', file: f, size: f.size, gguf: true };
    progress('Reading the GGUF header…');
    await ensureTensors(p, { signal });
    applyGGUF(p);
    p.sourceDetail = `${f.name} (${fmtBytes(f.size)}, local file)`;
    if (gg.length > 1) p.warnings.push({ level: 'info', text: `Only the first GGUF file (${f.name}) was read.` });
  } else if (cfg) {
    setParams(p, p.derived?.total || 0, 'Estimated from config.json', false);
    p.name = cfg._name_or_path || 'Model from config.json';
    p.sourceDetail = 'config.json (local file)';
  } else if (idx) {
    throw new UserError('An index file alone has no weights or sizes in it.', 'Also open the .safetensors shards (and config.json) listed in it.');
  } else {
    throw new UserError('None of these files is a .safetensors, .gguf or config.json file.', 'Pick model files, e.g. model.safetensors or config.json.');
  }
  if (ignored.length) p.warnings.push({ level: 'info', text: `Ignored: ${ignored.join(', ')}.` });
  return finish(p);
}

/* =====================================================================
 * Paste config.json / manual details
 * ===================================================================== */
export function importConfigText(text) {
  let cfg;
  try { cfg = JSON.parse(text); } catch (e) { throw new UserError('That is not valid JSON.', `Check the pasted text (${e.message.replace(/^JSON\.parse: /, '')}).`); }
  const p = makeProfile({ source: 'config', sourceDetail: 'Pasted config.json', name: cfg._name_or_path || cfg.architectures?.[0] || 'Model from config.json' });
  const d = applyConfig(p, cfg);
  if (!d) throw new UserError(p.warnings[0]?.text || 'Could not read this config.json.', 'Check that you pasted the whole file.');
  setParams(p, d.total, 'Estimated from config.json', false);
  p.warnings = [];
  return finish(p);
}

export function importManual({ params, dtype, layers, hidden }) {
  const n = parseCount(params);
  if (!(n > 0)) throw new UserError('Type the number of parameters, e.g. 8B, 70B or 1.5e9.', '', 'warn');
  if (n > 1e14) throw new UserError('That is more than 100 trillion parameters — please check the number.', '', 'warn');
  return makeProfile({
    name: `${fmtCount(n)} model`, source: 'manual', sourceDetail: 'Entered by hand', paramsTotal: n, paramSource: 'Entered by hand',
    originalDtype: dtype || 'BF16', layers: Number(layers) || null, hiddenSize: Number(hidden) || null,
  });
}
export const fmtCount = (n) => (n >= 1e12 ? +(n / 1e12).toFixed(2) + 'T' : n >= 1e9 ? +(n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? +(n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? +(n / 1e3).toFixed(1) + 'k' : String(n));

/* =====================================================================
 * Local model servers: Ollama / LM Studio / vLLM
 * ===================================================================== */
const trimUrl = (s) => String(s || '').trim().replace(/\/+$/, '');

export async function listLocalModels(base, { signal } = {}) {
  base = trimUrl(base);
  try { new URL(base); } catch { throw new UserError('That server address is not valid.', 'Example: http://localhost:11434'); }
  try {
    const r = await fetchJSON(`${base}/api/tags`, { signal, what: 'The model list' });
    if (Array.isArray(r.models)) return { kind: 'ollama', models: r.models.map((m) => ({ id: m.name, label: `${m.name} · ${m.details?.parameter_size || '?'} · ${m.details?.quantization_level || ''} · ${fmtBytes(m.size || 0)}`, raw: m })) };
  } catch (e) { if (isAbort(e)) throw e; if (e.code === 'local-cors' || e.code === 'cors') throw e; }
  const r = await fetchJSON(`${base}/v1/models`, { signal, what: 'The model list' });
  const data = r.data || [];
  if (!data.length) throw new UserError('The server has no models loaded.', 'Load or download a model in the server first.', 'warn');
  return { kind: 'openai', models: data.map((m) => ({ id: m.id, label: m.id, raw: m })) };
}

export async function importLocalServer(base, modelId, listing, { signal } = {}) {
  base = trimUrl(base);
  const entry = listing?.models.find((m) => m.id === modelId);
  if (listing?.kind === 'ollama') {
    const r = await fetchJSON(`${base}/api/show`, { signal, method: 'POST', body: JSON.stringify({ model: modelId }), headers: { 'Content-Type': 'application/json' }, what: 'Model details' });
    const mi = r.model_info || {}, d = r.details || {};
    const arch = mi['general.architecture'] || d.family;
    const k = (s) => mi[`${arch}.${s}`];
    const p = makeProfile({
      name: modelId, source: 'local', sourceDetail: `Ollama at ${new URL(base).host}`, architecture: arch,
      layers: k('block_count') ?? null, hiddenSize: k('embedding_length') ?? null, intermediateSize: k('feed_forward_length') ?? null,
      contextLength: k('context_length') ?? null, heads: k('attention.head_count') ?? null, kvHeads: k('attention.head_count_kv') ?? null,
      experts: k('expert_count') ?? null, expertsUsed: k('expert_used_count') ?? null, originalDtype: 'BF16',
    });
    const n = mi['general.parameter_count'] || parseCount(d.parameter_size);
    setParams(p, n, mi['general.parameter_count'] ? 'Ollama model info (exact, from the GGUF file)' : 'Ollama "parameter_size" (rounded)', !!mi['general.parameter_count']);
    const ql = d.quantization_level;
    if (ql) {
      const plain = /^(F16|BF16|F32)$/i.test(ql);
      p.quantization = plain ? null : { method: 'gguf', label: `GGUF ${ql}`, bits: null };
      if (plain) p.originalDtype = dtypeName(ql.toUpperCase());
    }
    const size = entry?.raw?.size;
    if (size) p.assumptions.push(`Ollama stores this model in ${fmtBytes(size)} (${((size * 8) / n).toFixed(2)} bits per weight).`);
    if (Array.isArray(r.tensors)) p.serverTensors = r.tensors.length;
    p.assumptions.push('Local servers only share summary information, so real-weight analysis is not available for this source.');
    return finish(p);
  }
  // OpenAI-compatible (LM Studio, vLLM): ask LM Studio's richer endpoint if it exists
  const p = makeProfile({ name: modelId, source: 'local', sourceDetail: `OpenAI-compatible server at ${new URL(base).host}` });
  try {
    const m = await fetchJSON(`${base}/api/v0/models/${encodeURIComponent(modelId)}`, { signal, what: 'Model details' });
    p.architecture = m.arch || null; p.contextLength = m.max_context_length || null;
    if (m.quantization) p.quantization = { method: 'lmstudio', label: m.quantization, bits: null };
    p.sourceDetail = `LM Studio at ${new URL(base).host}`;
  } catch (e) { if (isAbort(e)) throw e; }
  const raw = entry?.raw || {};
  if (raw.max_model_len) p.contextLength = raw.max_model_len;
  const hfLike = raw.root && /^[\w.-]+\/[\w.-]+$/.test(raw.root) ? raw.root : /^[\w.-]+\/[\w.-]+$/.test(modelId) ? modelId : null;
  if (hfLike) p.suggestHf = hfLike;
  const g = /(\d+(?:\.\d+)?)\s*[bB](?![a-z])/.exec(modelId.replace(/[-_]/g, ' '));
  if (g) {
    setParams(p, parseFloat(g[1]) * 1e9, 'Guessed from the model name — please check', false);
    p.warnings.push({ level: 'warn', text: `This server only shares the model name, so ${fmtCount(p.paramsTotal)} parameters is a guess from the name.`, next: hfLike ? `Import ${hfLike} from Hugging Face for exact details.` : 'Use “Enter details manually” if it is wrong.' });
  } else {
    throw new UserError('This server only shares the model name, and the size can\'t be read from it.', hfLike ? `Import ${hfLike} from Hugging Face instead.` : 'Use “Enter details manually”.', 'warn');
  }
  p.originalDtype = 'BF16';
  return finish(p);
}

/* =====================================================================
 * Real weights: read a capped sample of one tensor
 * ===================================================================== */
export const MAX_SAMPLE = 1048576;

export function tensorLayout(t) {
  const shape = t.shape.length ? t.shape : [1];
  const cols = shape[shape.length - 1];
  const rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
  return { rows, cols };
}

/** Plan which byte ranges to read: `blocks` evenly spaced blocks of whole rows, about `target` values in total. */
export function planSample(t, target, blocks = 8) {
  target = Math.min(MAX_SAMPLE, target);
  const { rows, cols } = tensorLayout(t);
  const per = DTYPE_BYTES[t.dtype];
  if (cols > target) {
    // Very long rows: take `target` values from the start of one row
    return { cols: target, rows: 1, ranges: [{ start: t.offset, end: t.offset + target * per, rows: 1 }], partialRows: true };
  }
  const rowsWanted = Math.min(rows, Math.max(1, Math.floor(target / cols)));
  const nb = Math.min(blocks, rowsWanted);
  const rpb = Math.ceil(rowsWanted / nb);
  const nBlocks = Math.ceil(rowsWanted / rpb);
  const contiguous = rowsWanted * 1.25 >= rows; // spaced blocks would overlap: read consecutive chunks instead
  const ranges = [];
  for (let b = 0; b < nBlocks; b++) {
    const r = Math.min(rpb, rowsWanted - b * rpb);
    const startRow = contiguous ? b * rpb
      : nBlocks === 1 ? Math.floor((rows - r) / 2)
      : Math.floor((b * (rows - rpb)) / (nBlocks - 1));
    const s = t.offset + startRow * cols * per;
    ranges.push({ start: s, end: s + r * cols * per, rows: r, startRow });
  }
  return { cols, rows: rowsWanted, ranges, partialRows: false };
}

export async function readTensorSample(p, t, target, { signal, budget, onProgress, blocks = 8 } = {}) {
  if (!READABLE_DTYPES.has(t.dtype)) throw new UserError(`${t.name} is stored as ${t.dtype}, which is already quantized or not a float.`, 'Pick a tensor stored as BF16, F16 or F32.', 'warn');
  const src = p.fileSources[t.file];
  if (!src) throw new UserError('The file holding this tensor is not available.', 'Load the model again.');
  const plan = planSample(t, target, blocks);
  let done = 0;
  const parts = await Promise.all(plan.ranges.map(async (r) => {
    const b = await readBytes(src, r.start, r.end, { signal, budget });
    onProgress?.(++done / plan.ranges.length);
    if (b.byteLength !== r.end - r.start) throw new UserError('The file ended early while reading this tensor.', 'The file may be damaged or still downloading.');
    return decode(b, t.dtype);
  }));
  const n = parts.reduce((a, x) => a + x.length, 0);
  const w = new Float32Array(n);
  let o = 0;
  for (const x of parts) { w.set(x, o); o += x.length; }
  const bytes = plan.ranges.reduce((a, r) => a + r.end - r.start, 0);
  return { weights: w, cols: plan.cols, rows: plan.rows, blocks: plan.ranges.length, bytes, partialRows: plan.partialRows };
}

/** Tensors worth scanning: matrices (2-D or more) stored as plain floats. */
export function scanCandidates(p) {
  return (p.tensors || []).filter((t) => READABLE_DTYPES.has(t.dtype) && t.shape.length >= 2 && t.numel >= 4096);
}

export function scanEstimate(p, perTensor, blocks = 4) {
  const ts = scanCandidates(p);
  let bytes = 0, requests = 0;
  for (const t of ts) {
    const pl = planSample(t, perTensor, blocks);
    bytes += pl.ranges.reduce((a, r) => a + r.end - r.start, 0);
    requests += pl.ranges.length;
  }
  return { tensors: ts, bytes, requests };
}
