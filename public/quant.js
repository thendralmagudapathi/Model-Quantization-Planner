/* Quantization maths — pure functions, no DOM. Used by the worker, the page and the Node tests. */

/* =====================================================================
 * Number formats
 * ===================================================================== */
// Minifloat formats: exponent bits, mantissa bits, exponent bias, largest finite value.
export const FLOATS = {
  fp32: { exp: 8, man: 23, bias: 127, max: 3.4028234663852886e38, bits: 32 },
  fp16: { exp: 5, man: 10, bias: 15, max: 65504, bits: 16 },
  bf16: { exp: 8, man: 7, bias: 127, max: 3.3895313892515355e38, bits: 16 },
  e4m3: { exp: 4, man: 3, bias: 7, max: 448, bits: 8 },     // OCP FP8 E4M3 "fn": no infinity, max 448
  e5m2: { exp: 5, man: 2, bias: 15, max: 57344, bits: 8 },
  e2m1: { exp: 2, man: 1, bias: 1, max: 6, bits: 4 },       // FP4: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
};

// The 16 NF4 levels, copied from bitsandbytes (csrc/kernels.cu, nf4_dequantization_lut).
export const NF4_LEVELS = [
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
];

// Formats the precision-loss explorer can simulate.
// kind: 'int' = integer + scale, 'float' = minifloat + scale, 'nf4' = lookup table + absmax, 'cast' = plain cast, no scale.
export const FORMATS = {
  int8:     { label: 'INT8',          kind: 'int',   bits: 8 },
  int6:     { label: 'INT6',          kind: 'int',   bits: 6 },
  int4:     { label: 'INT4',          kind: 'int',   bits: 4 },
  int3:     { label: 'INT3',          kind: 'int',   bits: 3 },
  int2:     { label: 'INT2',          kind: 'int',   bits: 2 },
  fp8_e4m3: { label: 'FP8 (E4M3)',    kind: 'float', bits: 8, fmt: FLOATS.e4m3 },
  fp8_e5m2: { label: 'FP8 (E5M2)',    kind: 'float', bits: 8, fmt: FLOATS.e5m2 },
  fp4_e2m1: { label: 'FP4 (E2M1)',    kind: 'float', bits: 4, fmt: FLOATS.e2m1 },
  nf4:      { label: 'NF4',           kind: 'nf4',   bits: 4 },
  fp16:     { label: 'FP16',          kind: 'cast',  bits: 16, fmt: FLOATS.fp16 },
  bf16:     { label: 'BF16',          kind: 'cast',  bits: 16, fmt: FLOATS.bf16 },
};

export const DEFAULT_CFG = {
  format: 'int4', sym: true, fullRange: false,
  granularity: 'group',      // 'none' (float formats only) | 'tensor' | 'channel' | 'group'
  groupSize: 128, scaleDtype: 'fp16',
  clip: 'none', clipPct: 99.9, // clip: 'none' | 'percentile' | 'auto'
};

/* =====================================================================
 * Rounding
 * ===================================================================== */
export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// floor(log2(a)) for a > 0, read exactly from the IEEE-754 exponent bits (no Math.log2 rounding issues).
const F64 = new Float64Array(1), U32 = new Uint32Array(F64.buffer);
const HI = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 1 : 0; // little-endian: high word is [1]
export function ilog2(a) {
  F64[0] = a;
  const e = (U32[HI] >>> 20) & 0x7ff;
  if (e) return e - 1023;
  let k = Math.floor(Math.log2(a));            // subnormal doubles (< 2^-1022): never happens for real weights
  if (2 ** k > a) k--; else if (2 ** (k + 1) <= a) k++;
  return k;
}
const POW2 = Float64Array.from({ length: 2400 }, (_, i) => 2 ** (i - 1200));
const pow2 = (k) => (k >= -1200 && k < 1200 ? POW2[k + 1200] : 2 ** k);

// Round x to the nearest value of a minifloat format (round-half-to-even, saturating at ±max).
export function roundToFloat(x, fmt) {
  if (x === 0) return 0;
  if (x !== x) return NaN;
  const a = x < 0 ? -x : x;
  if (a === Infinity) return x < 0 ? -fmt.max : fmt.max;
  const emin = 1 - fmt.bias;
  let e = ilog2(a);
  if (e < emin) e = emin;
  const step = pow2(e - fmt.man);
  const r = roundHalfEven(a / step) * step;
  return x < 0 ? -(r < fmt.max ? r : fmt.max) : r < fmt.max ? r : fmt.max;
}

// Every non-negative finite value a minifloat format can store, ascending.
export function floatValues(fmt) {
  const out = [0];
  const emin = 1 - fmt.bias;
  for (let m = 1; m < 2 ** fmt.man; m++) out.push(m * 2 ** (emin - fmt.man));          // subnormals
  for (let e = emin; ; e++) {
    for (let m = 0; m < 2 ** fmt.man; m++) {
      const v = (1 + m / 2 ** fmt.man) * 2 ** e;
      if (v > fmt.max) return out;
      out.push(v);
    }
  }
}
// Number of distinct values (±, one zero).
export const floatLevelCount = (fmt) => floatValues(fmt).length * 2 - 1;

// Split a stored minifloat value into its sign / exponent / mantissa bit fields.
export function floatBits(v, fmt) {
  const sign = v < 0 || Object.is(v, -0) ? 1 : 0;
  const a = Math.abs(v), emin = 1 - fmt.bias;
  let expField, man;
  if (a === 0) { expField = 0; man = 0; }
  else if (a < 2 ** emin) { expField = 0; man = Math.round(a / 2 ** (emin - fmt.man)); }
  else {
    const e = ilog2(a);
    expField = e + fmt.bias;
    man = Math.round((a / 2 ** e - 1) * 2 ** fmt.man);
  }
  return {
    sign, expField, man,
    expBits: expField.toString(2).padStart(fmt.exp, '0'),
    manBits: man.toString(2).padStart(fmt.man, '0'),
    subnormal: a !== 0 && expField === 0,
  };
}

/* =====================================================================
 * Groups: which weights share one scale
 * Weights are a row-major [rows × cols] matrix (rows = output channels).
 * Groups run along a row, like GPTQ / AWQ / llama.cpp blocks.
 * ===================================================================== */
export function segments(n, cols, cfg) {
  const g = cfg.granularity;
  if (g === 'tensor' || g === 'none') return [[0, n]];
  const out = [];
  const step = g === 'channel' ? cols : Math.max(1, cfg.groupSize | 0);
  for (let r0 = 0; r0 < n; r0 += cols) {
    const r1 = Math.min(n, r0 + cols);
    for (let s = r0; s < r1; s += step) out.push([s, Math.min(r1, s + step)]);
  }
  return out;
}

// Bits used per weight, including scales (and zero-points) shared by a group.
export function effectiveBits(cfg, cols = 4096) {
  const f = FORMATS[cfg.format];
  if (f.kind === 'cast' || cfg.granularity === 'none' || cfg.granularity === 'tensor') return f.bits;
  const per = cfg.granularity === 'channel' ? cols : cfg.groupSize;
  const scaleBits = cfg.scaleDtype === 'fp32' ? 32 : 16;
  const zpBits = f.kind === 'int' && !cfg.sym ? f.bits : 0;
  return f.bits + (scaleBits + zpBits) / per;
}

export const sizeGB = (params, bits) => (params * bits) / 8 / 1e9;

/* =====================================================================
 * Quantize one group (segment) — returns nothing, writes into out arrays
 * ===================================================================== */
const storeScale = (s, cfg) => (cfg.scaleDtype === 'fp32' ? Math.fround(s) : roundToFloat(s, FLOATS.fp16));

// Quantize w[a..b) with clip range [lo, hi] (hi = clipMax for symmetric formats).
// Writes dequantized values to out[], integer/level codes to codes[], returns { scale, zp }.
function quantSegment(w, a, b, cfg, lo, hi, out, codes) {
  const f = FORMATS[cfg.format];
  if (f.kind === 'cast') {
    for (let i = a; i < b; i++) { out[i] = roundToFloat(w[i], f.fmt); if (codes) codes[i] = 0; }
    return { scale: 1, zp: 0 };
  }
  if (f.kind === 'int' && !cfg.sym) {
    const qn = 2 ** f.bits - 1;
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    const scale = storeScale((hi - lo) / qn, cfg);
    if (!(scale > 0)) { for (let i = a; i < b; i++) { out[i] = 0; if (codes) codes[i] = 0; } return { scale: 0, zp: 0 }; }
    const zp = Math.min(qn, Math.max(0, Math.round(-lo / scale)));
    for (let i = a; i < b; i++) {
      const q = Math.min(qn, Math.max(0, roundHalfEven(w[i] / scale) + zp));
      out[i] = (q - zp) * scale;
      if (codes) codes[i] = q;
    }
    return { scale, zp };
  }
  const amax = hi;
  if (f.kind === 'int') {
    const half = 2 ** (f.bits - 1);
    const qmax = half - 1, qmin = cfg.fullRange ? -half : -qmax;
    const scale = storeScale(amax / (cfg.fullRange ? half : qmax), cfg);
    if (!(scale > 0)) { for (let i = a; i < b; i++) { out[i] = 0; if (codes) codes[i] = 0; } return { scale: 0, zp: 0 }; }
    for (let i = a; i < b; i++) {
      const q = Math.min(qmax, Math.max(qmin, roundHalfEven(w[i] / scale)));
      out[i] = q * scale;
      if (codes) codes[i] = q;
    }
    return { scale, zp: 0 };
  }
  if (f.kind === 'float') {
    const scale = cfg.granularity === 'none' ? 1 : storeScale(amax / f.fmt.max, cfg);
    if (!(scale > 0)) { for (let i = a; i < b; i++) { out[i] = 0; if (codes) codes[i] = 0; } return { scale: 0, zp: 0 }; }
    for (let i = a; i < b; i++) {
      const v = roundToFloat(w[i] / scale, f.fmt);
      out[i] = v * scale;
      if (codes) codes[i] = v; // codes hold the stored (scaled) float value
    }
    return { scale, zp: 0 };
  }
  // NF4: normalise by absmax, snap to the nearest of 16 levels
  const scale = storeScale(amax, cfg);
  if (!(scale > 0)) { for (let i = a; i < b; i++) { out[i] = 0; if (codes) codes[i] = 7; } return { scale: 0, zp: 0 }; }
  for (let i = a; i < b; i++) {
    const k = nearestNF4(w[i] / scale);
    out[i] = NF4_LEVELS[k] * scale;
    if (codes) codes[i] = k;
  }
  return { scale, zp: 0 };
}

const NF4_MIDS = NF4_LEVELS.slice(1).map((v, i) => (v + NF4_LEVELS[i]) / 2);
export function nearestNF4(x) {
  let lo = 0, hi = 15;                         // smallest k with x <= NF4_MIDS[k] (ties go down, like bitsandbytes' ">" tree)
  while (lo < hi) { const m = (lo + hi) >> 1; if (x > NF4_MIDS[m]) lo = m + 1; else hi = m; }
  return lo;
}

/* =====================================================================
 * Whole-tensor quantization
 * ===================================================================== */
function absPercentile(w, pct) {
  const n = w.length;
  const stride = Math.max(1, Math.floor(n / 200000)); // sample big tensors for speed
  const a = new Float32Array(Math.ceil(n / stride));
  for (let i = 0, j = 0; i < n; i += stride, j++) a[j] = Math.abs(w[i]);
  a.sort();
  return a[Math.min(a.length - 1, Math.floor((pct / 100) * (a.length - 1)))];
}

const AUTO_RATIOS = Array.from({ length: 12 }, (_, i) => 1 - i * 0.05); // 1.00 … 0.45

// Mean squared error of one segment for a given clip ratio (scratch avoids allocation).
function segmentError(w, a, b, cfg, lo, hi, scratch) {
  quantSegment(w, a, b, cfg, lo, hi, scratch, null);
  let e = 0;
  for (let i = a; i < b; i++) { const d = w[i] - scratch[i]; e += d * d; }
  return e;
}

/**
 * Quantize a [rows × cols] weight matrix.
 * Returns { out (dequantized), codes, segs, scales, zps, clipLo, clipHi, ratio }.
 * opts.ratio forces one clip ratio for every group (used by the clipping chart).
 */
export function quantize(w, cols, cfg, opts = {}) {
  const n = w.length;
  const f = FORMATS[cfg.format];
  const out = new Float32Array(n);
  const codes = opts.codes === false ? null : f.kind === 'float' ? new Float32Array(n) : new Int16Array(n);
  const segs = segments(n, cols, cfg);
  const scales = new Float32Array(segs.length), zps = new Int16Array(segs.length);
  const clipLo = new Float32Array(segs.length), clipHi = new Float32Array(segs.length);
  const T = cfg.clip === 'percentile' && opts.ratio === undefined ? absPercentile(w, cfg.clipPct) : Infinity;
  const scratch = cfg.clip === 'auto' && opts.ratio === undefined ? new Float32Array(n) : null;
  const asym = f.kind === 'int' && !cfg.sym;

  segs.forEach(([a, b], s) => {
    let mn = Infinity, mx = -Infinity;
    for (let i = a; i < b; i++) { const v = w[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const amax = Math.max(Math.abs(mn), Math.abs(mx));
    let lo, hi;
    if (opts.ratio !== undefined) { lo = mn * opts.ratio; hi = (asym ? mx : amax) * opts.ratio; }
    else if (cfg.clip === 'percentile') { lo = Math.max(mn, -T); hi = asym ? Math.min(mx, T) : Math.min(amax, T); }
    else if (cfg.clip === 'auto' && f.kind !== 'cast' && cfg.granularity !== 'none') {
      let best = Infinity, bestR = 1;
      for (const r of AUTO_RATIOS) {
        const e = segmentError(w, a, b, cfg, mn * r, (asym ? mx : amax) * r, scratch);
        if (e < best) { best = e; bestR = r; }
      }
      lo = mn * bestR; hi = (asym ? mx : amax) * bestR;
    } else { lo = mn; hi = asym ? mx : amax; }
    const { scale, zp } = quantSegment(w, a, b, cfg, lo, hi, out, codes);
    scales[s] = scale; zps[s] = zp; clipLo[s] = asym ? lo : -hi; clipHi[s] = hi;
  });
  return { out, codes, segs, scales, zps, clipLo, clipHi };
}

/* =====================================================================
 * Metrics
 * ===================================================================== */
export function sqnrDb(sumW2, sumE2) {
  if (sumE2 === 0) return Infinity;
  if (sumW2 === 0) return -Infinity;
  return 10 * Math.log10(sumW2 / sumE2);
}

// Levels a format offers (distinct stored values per group).
export function levelsAvailable(cfg) {
  const f = FORMATS[cfg.format];
  if (f.kind === 'int') return cfg.sym && !cfg.fullRange ? 2 ** f.bits - 1 : 2 ** f.bits;
  if (f.kind === 'nf4') return 16;
  return floatLevelCount(f.fmt);
}

export function metrics(w, q, cfg) {
  const { out, codes, segs, clipLo, clipHi } = q;
  const n = w.length;
  let sw2 = 0, se2 = 0, sq2 = 0, dot = 0, maxErr = 0, clipped = 0, zeroed = 0, nonzero = 0;
  for (let s = 0; s < segs.length; s++) {
    const lo = clipLo[s], hi = clipHi[s];
    for (let i = segs[s][0]; i < segs[s][1]; i++) {
      const a = w[i], b = out[i], d = a - b;
      sw2 += a * a; se2 += d * d; sq2 += b * b; dot += a * b;
      const ad = Math.abs(d);
      if (ad > maxErr) maxErr = ad;
      if (a > hi || a < lo) clipped++;
      if (a !== 0) { nonzero++; if (b === 0) zeroed++; }
    }
  }
  const f = FORMATS[cfg.format];
  const res = {
    n, mse: se2 / n, rmse: Math.sqrt(se2 / n), maxErr, sqnr: sqnrDb(sw2, se2),
    cosine: sw2 && sq2 ? dot / Math.sqrt(sw2 * sq2) : sw2 === sq2 ? 1 : 0,
    clippedPct: (100 * clipped) / n, zeroedPct: nonzero ? (100 * zeroed) / nonzero : 0,
    levelsAvail: levelsAvailable(cfg), levelsUsed: null, top3Share: null, rmsW: Math.sqrt(sw2 / n),
  };
  if (codes && f.kind !== 'cast') {
    const counts = new Map();
    for (let i = 0; i < n; i++) counts.set(codes[i], (counts.get(codes[i]) || 0) + 1);
    res.levelsUsed = counts.size;
    const top = [...counts.values()].sort((x, y) => y - x);
    res.top3Share = (100 * (top[0] + (top[1] || 0) + (top[2] || 0))) / n;
    res.codeCounts = counts;
  }
  return res;
}

// Quick SQNR only (used by sweeps and scans): no codes, no extra stats.
export function quickSqnr(w, cols, cfg, opts = {}) {
  const q = quantize(w, cols, cfg, { ...opts, codes: false });
  let sw2 = 0, se2 = 0;
  for (let i = 0; i < w.length; i++) { const d = w[i] - q.out[i]; sw2 += w[i] * w[i]; se2 += d * d; }
  return { sqnr: sqnrDb(sw2, se2), mse: se2 / w.length };
}

/* =====================================================================
 * Basic statistics of a weight sample
 * ===================================================================== */
export function weightStats(w) {
  const n = w.length;
  let mn = Infinity, mx = -Infinity, s = 0, s2 = 0, zeros = 0;
  for (let i = 0; i < n; i++) { const v = w[i]; if (v < mn) mn = v; if (v > mx) mx = v; s += v; s2 += v * v; if (v === 0) zeros++; }
  const mean = s / n, std = Math.sqrt(Math.max(0, s2 / n - mean * mean));
  let m4 = 0;
  for (let i = 0; i < n; i++) { const d = w[i] - mean; m4 += d * d * d * d; }
  const kurtosis = std ? m4 / n / std ** 4 : 0;
  const amax = Math.max(Math.abs(mn), Math.abs(mx));
  const p999 = absPercentile(w, 99.9);
  return { n, min: mn, max: mx, mean, std, rms: Math.sqrt(s2 / n), amax, p999, kurtosis, zeros,
    outlierRatio: p999 ? amax / p999 : 0 };
}

/* =====================================================================
 * Generated weights: seeded PRNG (mulberry32) + Box–Muller
 * ===================================================================== */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateWeights({ dist = 'laplace', sigma = 0.02, count = 65536, outlierPct = 0, outlierMul = 10, seed = 1, dof = 5 }) {
  const rnd = mulberry32(seed);
  let spare = null;
  const normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0; while (u === 0) u = rnd();
    const v = rnd(), r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  const w = new Float32Array(count);
  const b = sigma / Math.SQRT2;                        // Laplace scale so that std = sigma
  const tScale = sigma * Math.sqrt((dof - 2) / dof);   // Student-t scale so that std = sigma
  for (let i = 0; i < count; i++) {
    if (dist === 'normal') w[i] = sigma * normal();
    else if (dist === 'student') {
      let chi = 0; for (let k = 0; k < dof; k++) { const z = normal(); chi += z * z; }
      w[i] = (tScale * normal()) / Math.sqrt(chi / dof);
    } else {
      let u = rnd() - 0.5; while (u === -0.5) u = rnd() - 0.5;
      w[i] = -b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
    }
  }
  // Outliers: a few weights replaced by values around "outlierMul × sigma"
  const nOut = Math.round((count * outlierPct) / 100);
  for (let k = 0; k < nOut; k++) {
    const i = Math.floor(rnd() * count);
    w[i] = (rnd() < 0.5 ? -1 : 1) * outlierMul * sigma * (0.6 + 0.8 * rnd());
  }
  return w;
}

/* =====================================================================
 * Model sizes (size table)
 * ===================================================================== */
// GGUF bits/weight from llama.cpp tools/quantize/README.md (measured on Llama-3.1-8B, whole file).
export const SIZE_ROWS = [
  { id: 'fp32',  label: 'FP32',            bits: 32,     family: 'float', note: 'Full 32-bit floats. Used for training, almost never for running a model.' },
  { id: 'fp16',  label: 'FP16',            bits: 16,     family: 'float', note: '16-bit floats (5 exponent, 10 mantissa bits). The usual "unquantized" size.' },
  { id: 'bf16',  label: 'BF16',            bits: 16,     family: 'float', note: '16-bit "brain float" (8 exponent, 7 mantissa bits). Same size as FP16, wider range, less precision.' },
  { id: 'fp8',   label: 'FP8 (E4M3)',      bits: 8,      family: 'quant', note: '8-bit float with one scale per tensor. Runs natively on H100 / RTX 40-series and newer.' },
  { id: 'int8',  label: 'INT8',            bits: 'int8', family: 'quant', note: '8-bit integers with one FP16 scale per output channel (e.g. LLM.int8, W8A16).' },
  { id: 'int4',  label: 'INT4 (group 128)', bits: 4.125, family: 'quant', note: '4-bit integers, one FP16 scale per 128 weights: 4 + 16/128 = 4.125 bits.' },
  { id: 'nf4',   label: 'NF4',             bits: 4.5,    family: 'quant', note: 'bitsandbytes 4-bit NormalFloat, one FP32 absmax per 64 weights: 4 + 32/64 = 4.5 bits.\nWith "double quantization" (QLoRA) it is about 4.127 bits.' },
  { id: 'gptq',  label: 'GPTQ 4-bit g128', bits: 4.15625, family: 'quant', note: 'GPTQ stores 4-bit weights, an FP16 scale and a 4-bit zero-point per 128 weights: 4 + (16 + 4)/128 ≈ 4.156 bits.' },
  { id: 'awq',   label: 'AWQ 4-bit g128',  bits: 4.15625, family: 'quant', note: 'AWQ stores 4-bit weights, an FP16 scale and a 4-bit zero-point per 128 weights: 4 + (16 + 4)/128 ≈ 4.156 bits.' },
  { id: 'q8_0',  label: 'GGUF Q8_0',       bits: 8.5008, family: 'gguf', note: 'llama.cpp 8-bit: blocks of 32 weights with an FP16 scale (8.5 bits).' },
  { id: 'q6_k',  label: 'GGUF Q6_K',       bits: 6.5633, family: 'gguf', note: 'llama.cpp 6-bit "K-quant" (super-blocks of 256 weights).' },
  { id: 'q5_k_m', label: 'GGUF Q5_K_M',    bits: 5.7036, family: 'gguf', note: 'llama.cpp 5-bit K-quant, "medium" mix (some tensors kept at 6 bits).' },
  { id: 'q4_k_m', label: 'GGUF Q4_K_M',    bits: 4.8944, family: 'gguf', note: 'llama.cpp 4-bit K-quant, "medium" mix — the most popular choice for local use.' },
  { id: 'q3_k_m', label: 'GGUF Q3_K_M',    bits: 3.996,  family: 'gguf', note: 'llama.cpp 3-bit K-quant, "medium" mix. Noticeably lower quality.' },
  { id: 'q2_k',  label: 'GGUF Q2_K',       bits: 3.1593, family: 'gguf', note: 'llama.cpp 2-bit K-quant (many tensors kept higher). Large quality loss.' },
];

export const GPU_SIZES = [8, 12, 16, 24, 48, 80];
export const GIB = 2 ** 30;

/**
 * Size of a model in a format.
 * keepEmb: keep embeddings + output head in 16-bit (what GPTQ / AWQ / bitsandbytes usually do).
 * Returns { bits, gb, quantParams }.
 */
export function rowSize(row, params, { embParams = 0, keepEmb = true, hidden = 4096 } = {}) {
  const bits = row.bits === 'int8' ? 8 + 16 / hidden : row.bits;
  if (row.family === 'quant' && keepEmb && embParams > 0 && embParams < params) {
    const bytes = ((params - embParams) * bits + embParams * 16) / 8;
    return { bits, gb: bytes / 1e9, avgBits: (bytes * 8) / params, keptEmb: true };
  }
  return { bits, gb: sizeGB(params, bits), avgBits: bits, keptEmb: false };
}

// Does `gb` of weights fit in one GPU of `vramGB` (marketing GB = GiB) with `headroom` (0–1) usable?
export const fitsOn = (gb, vramGB, headroom = 0.85) => gb * 1e9 <= vramGB * GIB * headroom;
export const gpusNeeded = (gb, vramGB, headroom = 0.85) => Math.ceil((gb * 1e9) / (vramGB * GIB * headroom));

/* =====================================================================
 * Parse "8B", "70b", "1.5e9", "7,000,000,000", "350M"
 * ===================================================================== */
export function parseCount(s) {
  if (typeof s === 'number') return s;
  const t = String(s ?? '').trim().replace(/[, _]/g, '');
  const m = t.match(/^(\d*\.?\d+(?:e[+-]?\d+)?)([kmbt])?$/i);
  if (!m) return NaN;
  const mul = { k: 1e3, m: 1e6, b: 1e9, t: 1e12 }[(m[2] || '').toLowerCase()] || 1;
  return parseFloat(m[1]) * mul;
}
