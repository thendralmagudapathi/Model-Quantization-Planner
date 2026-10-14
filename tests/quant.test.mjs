import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  FLOATS, NF4_LEVELS, roundHalfEven, roundToFloat, floatValues, floatLevelCount, floatBits,
  quantize, metrics, effectiveBits, sizeGB, generateWeights, mulberry32, parseCount, segments,
  rowSize, SIZE_ROWS, fitsOn, gpusNeeded, quickSqnr, weightStats,
} from '../public/quant.js';

const cfg = (over) => ({ format: 'int4', sym: true, fullRange: false, granularity: 'tensor', groupSize: 128, scaleDtype: 'fp32', clip: 'none', clipPct: 99.9, ...over });
const q = (w, c, cols = w.length) => Array.from(quantize(Float32Array.from(w), cols, cfg(c)).out);

test('roundHalfEven: ties go to the even neighbour', () => {
  assert.deepEqual([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, 2.6].map((x) => roundHalfEven(x) + 0), [0, 2, 2, 0, -2, -2, 2, 3]);
});

test('INT4 symmetric (hand-checked): absmax 7 → scale 1, ties to even', () => {
  assert.deepEqual(q([7, -3.5, 2.5, 0.4, -7], {}), [7, -4, 2, 0, -7]);
});

test('INT8 symmetric (hand-checked): absmax 63.5 → scale 0.5', () => {
  // 1.25/0.5 = 2.5 → 2 ; -0.75/0.5 = -1.5 → -2 ; 10.1/0.5 = 20.2 → 20
  assert.deepEqual(q([63.5, 1.25, -0.75, 10.1], { format: 'int8' }), [63.5, 1, -1, 10]);
});

test('INT4 asymmetric (hand-checked): scale = range/15, zero-point = round(-min/scale)', () => {
  assert.deepEqual(q([-5, 0, 10], { sym: false }), [-5, 0, 10]);        // scale 1, zp 5
  assert.deepEqual(q([0, 1.5, 3, 15], { sym: false }), [0, 2, 3, 15]);  // scale 1, zp 0, 1.5 → 2
  const r = quantize(Float32Array.from([-5, 0, 10]), 3, cfg({ sym: false }));
  assert.equal(r.zps[0], 5); assert.equal(r.scales[0], 1);
  assert.deepEqual(Array.from(r.codes), [0, 5, 15]);
});

test('INT4 full range uses −8…7', () => {
  // full range: scale = absmax / 8 = 1 → 8 clamps to 7, -8 is allowed
  assert.deepEqual(q([8, -8, 3], { fullRange: true }), [7, -8, 3]);
});

test('per-group quantization gives each group its own scale', () => {
  const w = [1, 0.5, 100, 50]; // two groups of 2
  const r = quantize(Float32Array.from(w), 4, cfg({ granularity: 'group', groupSize: 2 }));
  assert.equal(r.scales.length, 2);
  assert.ok(Math.abs(r.scales[0] - 1 / 7) < 1e-7 && Math.abs(r.scales[1] - 100 / 7) < 1e-5);
  assert.deepEqual(segments(10, 5, cfg({ granularity: 'group', groupSize: 2 })), [[0, 2], [2, 4], [4, 5], [5, 7], [7, 9], [9, 10]]);
  assert.deepEqual(segments(10, 5, cfg({ granularity: 'channel' })), [[0, 5], [5, 10]]);
});

test('FP8 E4M3 edge cases', () => {
  const f = FLOATS.e4m3;
  assert.equal(roundToFloat(448, f), 448);
  assert.equal(roundToFloat(1000, f), 448);             // saturates
  assert.equal(roundToFloat(-1e9, f), -448);
  assert.equal(roundToFloat(2 ** -9, f), 2 ** -9);       // smallest subnormal
  assert.equal(roundToFloat(2 ** -10, f), 0);            // exact tie between 0 and 2^-9 → even (0)
  assert.equal(roundToFloat(0.76 * 2 ** -9, f), 2 ** -9);
  assert.equal(roundToFloat(1.0625, f), 1);              // tie between 1 and 1.125 → even mantissa
  assert.equal(roundToFloat(1.1875, f), 1.25);           // tie between 1.125 and 1.25 → even mantissa
  assert.equal(floatLevelCount(f), 253);
  assert.equal(Math.max(...floatValues(f)), 448);
});

test('FP8 E5M2, FP16, BF16 and FP4 edge cases', () => {
  assert.equal(roundToFloat(57344, FLOATS.e5m2), 57344);
  assert.equal(roundToFloat(2 ** -16, FLOATS.e5m2), 2 ** -16); // E5M2 smallest subnormal
  assert.equal(roundToFloat(65504, FLOATS.fp16), 65504);
  assert.equal(roundToFloat(65519, FLOATS.fp16), 65504);
  assert.equal(roundToFloat(2 ** -24, FLOATS.fp16), 2 ** -24); // FP16 smallest subnormal
  assert.equal(roundToFloat(2 ** -25, FLOATS.fp16), 0);        // tie → 0
  assert.equal(roundToFloat(1 + 2 ** -8, FLOATS.bf16), 1);     // BF16 has 7 mantissa bits: tie → even
  assert.equal(roundToFloat(1 + 3 * 2 ** -8, FLOATS.bf16), 1 + 2 ** -6);
  assert.deepEqual(floatValues(FLOATS.e2m1), [0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  assert.deepEqual([0.25, 0.3, 1.25, 1.75, 2.5, 3.5, 5, 7, -5.1].map((x) => roundToFloat(x, FLOATS.e2m1)), [0, 0.5, 1, 2, 2, 4, 4, 6, -6]);
  assert.equal(floatLevelCount(FLOATS.e5m2), 247);
  assert.equal(floatLevelCount(FLOATS.e2m1), 15);
});

test('every representable value round-trips unchanged', () => {
  for (const k of ['e4m3', 'e5m2', 'e2m1', 'fp16']) {
    for (const v of floatValues(FLOATS[k])) {
      assert.equal(roundToFloat(v, FLOATS[k]), v, `${k} ${v}`);
      assert.equal(roundToFloat(-v, FLOATS[k]) + 0, -v + 0, `${k} ${-v}`);
    }
  }
  const lv = Float32Array.from(NF4_LEVELS);
  assert.deepEqual(Array.from(quantize(lv, 16, cfg({ format: 'nf4' })).out), Array.from(lv));
  const ints = Float32Array.from([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(Array.from(quantize(ints, 15, cfg({})).out), Array.from(ints));
});

test('float formats match PyTorch exactly (reference vectors)', () => {
  const ref = JSON.parse(readFileSync(new URL('./fixtures/torch_floats.json', import.meta.url)));
  for (const k of ['fp16', 'bf16', 'e4m3', 'e5m2']) {
    const { x, y } = ref[k];
    let bad = 0, first = null;
    for (let i = 0; i < x.length; i++) {
      const got = roundToFloat(x[i], FLOATS[k]);
      if (!Object.is(got + 0, y[i] + 0)) { bad++; first ??= `${x[i]} → ${got}, torch ${y[i]}`; }
    }
    assert.equal(bad, 0, `${k}: ${bad}/${x.length} mismatches, first: ${first}`);
    assert.ok(x.length > 6000);
  }
});

test('float bit fields', () => {
  assert.deepEqual(floatBits(448, FLOATS.e4m3), { sign: 0, expField: 15, man: 6, expBits: '1111', manBits: '110', subnormal: false });
  assert.deepEqual(floatBits(-(2 ** -9), FLOATS.e4m3), { sign: 1, expField: 0, man: 1, expBits: '0000', manBits: '001', subnormal: true });
  assert.equal(floatBits(1, FLOATS.fp16).expBits, '01111');
});

test('SQNR rises about 6 dB per extra bit on uniform data', () => {
  const rnd = mulberry32(7);
  const w = Float32Array.from({ length: 65536 }, () => rnd() * 2 - 1);
  // Named formats 3, 4, 6, 8 bits: expect ≈ 6.02 dB per bit
  const named = { 2: 'int2', 3: 'int3', 4: 'int4', 6: 'int6', 8: 'int8' };
  const v = Object.fromEntries(Object.entries(named).map(([b, f]) => [b, quickSqnr(w, w.length, cfg({ format: f })).sqnr]));
  // Symmetric INT-b has qmax = 2^(b-1) − 1, so each step gains 20·log10(qmax_b / qmax_a) dB → 6.02 dB/bit as b grows
  const qmax = (b) => 2 ** (b - 1) - 1;
  for (const [a, b] of [[2, 3], [3, 4], [4, 6], [6, 8]]) {
    const want = 20 * Math.log10(qmax(b) / qmax(a));
    assert.ok(Math.abs(v[b] - v[a] - want) < 0.2, `int${a}→int${b}: +${(v[b] - v[a]).toFixed(2)} dB, expected ${want.toFixed(2)}`);
  }
  assert.ok(Math.abs((v[8] - v[6]) / 2 - 6.02) < 0.15, `6→8 bits: ${((v[8] - v[6]) / 2).toFixed(2)} dB/bit`);
});

test('effective bits and size', () => {
  assert.equal(effectiveBits(cfg({ granularity: 'group', groupSize: 128, scaleDtype: 'fp16' })), 4.125);
  assert.equal(effectiveBits(cfg({ granularity: 'group', groupSize: 128, scaleDtype: 'fp16', sym: false })), 4 + 20 / 128);
  assert.equal(effectiveBits(cfg({ granularity: 'group', groupSize: 32, scaleDtype: 'fp32' })), 5);
  assert.equal(effectiveBits(cfg({ format: 'bf16' })), 16);
  assert.ok(Math.abs(sizeGB(7.62e9, 4.125) - 3.929) < 0.001);
  assert.equal(sizeGB(70e9, 16), 140);
  const int4 = SIZE_ROWS.find((r) => r.id === 'int4');
  assert.equal(rowSize(int4, 70e9, { keepEmb: false }).gb, 70e9 * 4.125 / 8e9);
  // 1B params, 0.2B embeddings kept at 16 bits: (0.8 × 4.125 + 0.2 × 16) / 8 = 0.8125 GB
  assert.ok(Math.abs(rowSize(int4, 1e9, { embParams: 0.2e9, keepEmb: true }).gb - 0.8125) < 1e-12);
  assert.ok(fitsOn(20, 24) && !fitsOn(23, 24));
  assert.equal(gpusNeeded(140, 80), 2);
});

test('metrics: SQNR, cosine, clipped, zeroed, levels used', () => {
  const w = Float32Array.from([7, -3.5, 2.5, 0.4, -7, 0]);
  const c = cfg({});
  const r = quantize(w, w.length, c);
  const m = metrics(w, r, c);
  const err2 = 0.5 ** 2 + 0.5 ** 2 + 0.4 ** 2;
  const sig = 49 + 12.25 + 6.25 + 0.16 + 49;
  assert.ok(Math.abs(m.mse - err2 / 6) < 1e-6);
  assert.ok(Math.abs(m.sqnr - 10 * Math.log10(sig / err2)) < 1e-4);
  assert.equal(m.levelsUsed, 5);   // 7, -4, 2, 0, -7
  assert.equal(m.levelsAvail, 15);
  assert.ok(Math.abs(m.zeroedPct - 20) < 1e-9); // 0.4 → 0 (1 of 5 non-zero weights)
  assert.equal(m.clippedPct, 0);
  // percentile clipping clips the top values
  const big = generateWeights({ count: 10000, seed: 3, dist: 'normal' });
  const cp = cfg({ clip: 'percentile', clipPct: 99 });
  const mm = metrics(big, quantize(big, 10000, cp), cp);
  assert.ok(mm.clippedPct > 0.5 && mm.clippedPct < 1.5, `clipped ${mm.clippedPct}`);
});

test('auto clipping never does worse than no clipping', () => {
  const w = generateWeights({ dist: 'laplace', count: 16384, seed: 5, outlierPct: 0.1, outlierMul: 30 });
  for (const g of ['tensor', 'group']) {
    const none = quickSqnr(w, 1024, cfg({ granularity: g, clip: 'none' })).sqnr;
    const auto = quickSqnr(w, 1024, cfg({ granularity: g, clip: 'auto' })).sqnr;
    assert.ok(auto >= none - 1e-9, `${g}: auto ${auto} < none ${none}`);
  }
});

test('outlier disaster: per-tensor INT4 collapses, group 128 fixes it', () => {
  const w = generateWeights({ dist: 'laplace', sigma: 0.02, count: 65536, outlierPct: 0.05, outlierMul: 40, seed: 1 });
  const t = cfg({ scaleDtype: 'fp16' }), g = cfg({ granularity: 'group', scaleDtype: 'fp16' });
  const mt = metrics(w, quantize(w, 1024, t), t), mg = metrics(w, quantize(w, 1024, g), g);
  assert.ok(mt.top3Share > 95, `top-3 share ${mt.top3Share}`);
  assert.ok(mt.sqnr < 6, `per-tensor SQNR ${mt.sqnr}`);
  assert.ok(mg.sqnr > mt.sqnr + 8, `group SQNR ${mg.sqnr} vs ${mt.sqnr}`);
});

test('generated weights: seeded, right spread', () => {
  const a = generateWeights({ seed: 42, count: 200000, dist: 'laplace' });
  const b = generateWeights({ seed: 42, count: 200000, dist: 'laplace' });
  assert.deepEqual(a.slice(0, 50), b.slice(0, 50));
  for (const dist of ['normal', 'laplace', 'student']) {
    const s = weightStats(generateWeights({ seed: 9, count: 200000, dist, sigma: 0.02 }));
    assert.ok(Math.abs(s.std - 0.02) < 0.0015, `${dist} std ${s.std}`);
    assert.ok(Math.abs(s.mean) < 0.0005);
  }
  assert.ok(weightStats(generateWeights({ seed: 9, count: 200000, dist: 'laplace' })).kurtosis > 5); // Laplace kurtosis = 6
});

test('parseCount', () => {
  assert.equal(parseCount('8B'), 8e9);
  assert.equal(parseCount('70b'), 70e9);
  assert.equal(parseCount('1.5e9'), 1.5e9);
  assert.equal(parseCount('350M'), 350e6);
  assert.equal(parseCount('7,241,732,096'), 7241732096);
  assert.ok(Number.isNaN(parseCount('lots')));
});

test('NF4 lookup picks the nearest of the 16 levels', async () => {
  const { nearestNF4 } = await import('../public/quant.js');
  const rnd = mulberry32(11);
  for (let i = 0; i < 20000; i++) {
    const x = rnd() * 2.2 - 1.1;
    let best = 0; for (let j = 1; j < 16; j++) if (Math.abs(NF4_LEVELS[j] - x) < Math.abs(NF4_LEVELS[best] - x)) best = j;
    assert.equal(nearestNF4(x), best);
  }
});
