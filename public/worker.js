/* Background worker: runs the quantization maths so the page never freezes. */
import {
  quantize, metrics, effectiveBits, weightStats, generateWeights, quickSqnr,
  FORMATS, NF4_LEVELS, floatValues,
} from './quant.js';

const GEN_COLS = 1024;          // generated weights are laid out as rows of 1,024 inputs
const SUBSET = 65536;           // sweeps, comparisons and insight alternatives use at most this many weights
let gen = { key: '', w: null, stats: null };
const real = new Map();         // key → { w, cols, stats }
let cmpCache = { key: '', list: null };

function getWeights(src) {
  if (src.kind === 'gen') {
    const key = JSON.stringify(src.params);
    if (gen.key !== key) gen = { key, w: generateWeights(src.params), stats: null };
    if (!gen.stats) gen.stats = weightStats(gen.w);
    return { w: gen.w, cols: GEN_COLS, stats: gen.stats, key };
  }
  const r = real.get(src.key);
  if (!r) throw new Error('The real-layer sample is no longer loaded — load it again.');
  if (!r.stats) r.stats = weightStats(r.w);
  return { ...r, key: src.key };
}

// A row-aligned prefix of at most SUBSET weights (keeps whole rows so groups stay intact).
function subset(w, cols, max = SUBSET) {
  if (w.length <= max) return w;
  return w.subarray(0, Math.max(cols, Math.floor(max / cols) * cols));
}

function histogram(arr, lo, hi, bins) {
  const counts = new Uint32Array(bins);
  const span = hi - lo || 1;
  for (let i = 0; i < arr.length; i++) {
    let b = Math.floor(((arr[i] - lo) / span) * bins);
    if (b < 0) b = 0; else if (b >= bins) b = bins - 1;
    counts[b]++;
  }
  return counts;
}

const segIndexOf = (i, cols, cfg) => {
  if (cfg.granularity === 'tensor' || cfg.granularity === 'none') return 0;
  const row = Math.floor(i / cols), col = i % cols;
  if (cfg.granularity === 'channel') return row;
  const gpr = Math.ceil(cols / cfg.groupSize);
  return row * gpr + Math.floor(col / cfg.groupSize);
};

// Stored values of one group, in weight units (for the histogram overlay).
function levelsOf(cfg, scale, zp) {
  const f = FORMATS[cfg.format];
  if (f.kind === 'cast' || !(scale > 0)) return [];
  if (f.kind === 'int') {
    if (!cfg.sym) return Array.from({ length: 2 ** f.bits }, (_, q) => (q - zp) * scale);
    const half = 2 ** (f.bits - 1), qmin = cfg.fullRange ? -half : -(half - 1);
    return Array.from({ length: half - 1 - qmin + 1 }, (_, k) => (qmin + k) * scale);
  }
  if (f.kind === 'nf4') return NF4_LEVELS.map((v) => v * scale);
  const pos = floatValues(f.fmt);
  return [...pos.slice(1).map((v) => -v * scale).reverse(), ...pos.map((v) => v * scale)];
}

const COMPARE = [
  ['FP16', { format: 'fp16' }],
  ['BF16', { format: 'bf16' }],
  ['INT8 per channel', { format: 'int8', granularity: 'channel' }],
  ['FP8 E4M3', { format: 'fp8_e4m3', granularity: 'tensor' }],
  ['FP8 E5M2', { format: 'fp8_e5m2', granularity: 'tensor' }],
  ['INT6 g128', { format: 'int6', granularity: 'group', groupSize: 128 }],
  ['INT4 per tensor', { format: 'int4', granularity: 'tensor' }],
  ['INT4 per channel', { format: 'int4', granularity: 'channel' }],
  ['INT4 g128', { format: 'int4', granularity: 'group', groupSize: 128 }],
  ['INT4 g128 asym', { format: 'int4', granularity: 'group', groupSize: 128, sym: false }],
  ['INT4 g32', { format: 'int4', granularity: 'group', groupSize: 32 }],
  ['NF4 g64', { format: 'nf4', granularity: 'group', groupSize: 64 }],
  ['FP4 E2M1 g32', { format: 'fp4_e2m1', granularity: 'group', groupSize: 32 }],
  ['INT3 g128', { format: 'int3', granularity: 'group', groupSize: 128 }],
  ['INT2 g64', { format: 'int2', granularity: 'group', groupSize: 64 }],
];
const BASE = { sym: true, fullRange: false, granularity: 'tensor', groupSize: 128, scaleDtype: 'fp16', clip: 'none', clipPct: 99.9 };

function compareAll(w, cols, key) {
  if (cmpCache.key === key) return cmpCache.list;
  const s = subset(w, cols, 32768);       // SQNR on 32k weights is accurate to about ±0.1 dB
  const list = COMPARE.map(([label, over]) => {
    const cfg = { ...BASE, ...over };
    return { label, cfg, bits: effectiveBits(cfg, cols), sqnr: quickSqnr(s, cols, cfg).sqnr };
  });
  list.subset = s.length < w.length ? s.length : 0;
  cmpCache = { key, list };
  return list;
}

function analyze(m) {
  const t0 = performance.now();
  const { w, cols, stats, key } = getWeights(m.source);
  const cfg = m.cfg;
  const n = w.length;
  const q = quantize(w, cols, cfg);
  const met = metrics(w, q, cfg);
  let topCodes = null;
  if (met.codeCounts) {
    topCodes = [...met.codeCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
    delete met.codeCounts;
  }

  // A. histogram + levels of a typical group (median scale)
  const bins = 140;
  const hist = histogram(w, stats.min, stats.max, bins);
  let rep = 0;
  if (q.scales.length > 1) {
    const order = Array.from(q.scales.keys()).sort((a, b) => q.scales[a] - q.scales[b]);
    rep = order[Math.floor(order.length / 2)];
  }
  const levels = levelsOf(cfg, q.scales[rep], q.zps[rep]);
  // Shade only what is really clipped: the whole-tensor range, or the global percentile threshold
  let shade = null;
  const fk = FORMATS[cfg.format].kind;
  if (fk !== 'cast' && cfg.granularity !== 'none') {
    if (cfg.granularity === 'tensor') shade = [q.clipLo[0], q.clipHi[0]];
    else if (cfg.clip === 'percentile') {
      let lo = Infinity, hi = -Infinity;
      for (let s = 0; s < q.segs.length; s++) { if (q.clipLo[s] < lo) lo = q.clipLo[s]; if (q.clipHi[s] > hi) hi = q.clipHi[s]; }
      shade = [lo, hi];
    }
  }

  // B. staircase sample (≤ 5,000 points, always including the largest weight)
  const stride = Math.max(1, Math.ceil(n / 5000));
  const sx = [], sy = [];
  for (let i = 0; i < n; i += stride) { sx.push(w[i]); sy.push(q.out[i]); }
  let iMax = 0;
  for (let i = 0; i < n; i++) if (Math.abs(w[i]) > Math.abs(w[iMax])) iMax = i;
  sx.push(w[iMax]); sy.push(q.out[iMax]);

  // C. error histogram
  const err = new Float32Array(n);
  for (let i = 0; i < n; i++) err[i] = w[i] - q.out[i];
  const eMax = met.maxErr || 1e-12;
  const errHist = histogram(err, -eMax, eMax, 100);

  // E. clipping sweep: same clip ratio for every group, 100% → 20% of the max
  let sweep = null;
  const f = FORMATS[cfg.format];
  if (f.kind !== 'cast' && cfg.granularity !== 'none') {
    const s = subset(w, cols);
    const ratios = [], mse = [], sq = [];
    for (let k = 0; k < 17; k++) {
      const r = 1 - k * 0.05;            // 100%, 95%, … 20% of the max
      const v = quickSqnr(s, cols, cfg, { ratio: r });
      ratios.push(r); mse.push(v.mse); sq.push(v.sqnr);
    }
    let best = 0;
    for (let i = 1; i < mse.length; i++) if (mse[i] < mse[best]) best = i;
    sweep = { ratios, rmse: mse.map(Math.sqrt), sqnr: sq, best, subset: s.length < n ? s.length : 0 };
  }

  // D. every format on the same weights
  const compare = compareAll(w, cols, key);

  // Alternatives for the automatic insights
  const alt = {};
  if (f.kind !== 'cast') {
    const s = subset(w, cols);
    // Only the alternatives an insight can use: per-tensor (when grouping) or group 128 (when not)
    const want = cfg.granularity === 'group' ? [['tensor', { granularity: 'tensor' }]]
      : cfg.granularity === 'none' ? [] : [['g128', { granularity: 'group', groupSize: 128 }]];
    for (const [k, over] of want) {
      const c2 = { ...cfg, ...over, clip: 'none' };   // compared with the current setting without clipping (sweep at 100%)
      alt[k] = { sqnr: quickSqnr(s, cols, c2).sqnr, bits: effectiveBits(c2, cols) };
    }
    if (cfg.format !== 'int8') {
      const c8 = { ...cfg, format: 'int8', clip: 'none', granularity: cfg.granularity === 'none' ? 'tensor' : cfg.granularity };
      alt.int8 = { sqnr: quickSqnr(s, cols, c8).sqnr, bits: effectiveBits(c8, cols) };
    }
  }

  // F. 32 × 32 block around the largest weight
  const rows = Math.floor(n / cols);
  const nr = Math.min(32, rows), nc = Math.min(32, cols);
  const rStar = Math.floor(iMax / cols), cStar = iMax % cols;
  const r0 = Math.max(0, Math.min(rows - nr, rStar - 16)), c0 = Math.max(0, Math.min(cols - nc, cStar - 16));
  const hw = new Float32Array(nr * nc), hq = new Float32Array(nr * nc), hc = new Float32Array(nr * nc), hs = new Int32Array(nr * nc);
  for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
    const i = (r0 + r) * cols + (c0 + c), k = r * nc + c;
    hw[k] = w[i]; hq[k] = q.out[i]; hc[k] = q.codes ? q.codes[i] : 0; hs[k] = segIndexOf(i, cols, cfg);
  }
  const segIds = [...new Set(hs)];
  const segInfo = Object.fromEntries(segIds.map((s) => [s, { scale: q.scales[s], zp: q.zps[s], lo: q.clipLo[s], hi: q.clipHi[s] }]));

  return {
    type: 'result', id: m.id, n, cols, rows, stats, metrics: met, topCodes,
    bits: effectiveBits(cfg, cols), groups: q.segs.length,
    hist: { lo: stats.min, hi: stats.max, counts: hist, levels, repScale: q.scales[rep], repZp: q.zps[rep], clip: shade },
    stair: { x: Float32Array.from(sx), y: Float32Array.from(sy) },
    errHist: { lo: -eMax, hi: eMax, counts: errHist },
    sweep, compare, compareSubset: compare.subset || 0, alt,
    heat: { r0, c0, nr, nc, w: hw, q: hq, codes: hc, seg: hs, segInfo, rStar, cStar },
    ms: performance.now() - t0,
  };
}

// Layer scan: quick sensitivity check of one tensor sample.
function scanOne(m) {
  const { weights: w, cols } = m;
  const st = weightStats(w);
  const c = (over) => ({ ...BASE, ...over });
  const int4 = quickSqnr(w, cols, c({ format: 'int4', granularity: 'group', groupSize: 128 })).sqnr;
  const fp8 = quickSqnr(w, cols, c({ format: 'fp8_e4m3', granularity: 'tensor' })).sqnr;
  const int8 = quickSqnr(w, cols, c({ format: 'int8', granularity: 'channel' })).sqnr;
  return { type: 'scan', id: m.id, int4, fp8, int8, stats: { std: st.std, amax: st.amax, p999: st.p999, outlierRatio: st.outlierRatio, kurtosis: st.kurtosis, n: st.n } };
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'setReal') { real.clear(); real.set(m.key, { w: m.weights, cols: m.cols, stats: null }); cmpCache = { key: '', list: null }; self.postMessage({ type: 'ok', id: m.id }); }
    else if (m.type === 'analyze') self.postMessage(analyze(m));
    else if (m.type === 'scan') self.postMessage(scanOne(m));
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: err.message || String(err) });
  }
};
