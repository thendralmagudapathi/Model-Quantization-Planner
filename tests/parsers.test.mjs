import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseSafetensorsHeader, buildSafetensors, decode, halfToFloat, bf16ToFloat, parseIndexJson,
  countParams, detectPackedQuant, parseGGUF, ggufSummary, NeedMoreBytes,
} from '../public/parsers.js';

test('FP16 → float, including subnormals, inf and NaN', () => {
  assert.equal(halfToFloat(0x3c00), 1);
  assert.equal(halfToFloat(0xc000), -2);
  assert.equal(halfToFloat(0x7bff), 65504);
  assert.equal(halfToFloat(0x0001), 2 ** -24);          // smallest subnormal
  assert.equal(halfToFloat(0x03ff), 1023 * 2 ** -24);   // largest subnormal
  assert.equal(halfToFloat(0x0400), 2 ** -14);          // smallest normal
  assert.equal(halfToFloat(0x7c00), Infinity);
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
  assert.ok(Object.is(halfToFloat(0x8000), -0));
});

test('BF16 → float', () => {
  assert.equal(bf16ToFloat(0x3f80), 1);
  assert.equal(bf16ToFloat(0xc0a0), -5);
  assert.equal(bf16ToFloat(0x3c23), 0.00994873046875);
});

test('decode F32 / F16 / BF16 byte buffers', () => {
  const f32 = Float32Array.from([1.5, -2, 0.1]);
  assert.deepEqual(Array.from(decode(new Uint8Array(f32.buffer), 'F32')), Array.from(f32));
  const f16 = new Uint8Array(Uint16Array.from([0x3c00, 0x0001, 0xc000]).buffer);
  assert.deepEqual(Array.from(decode(f16, 'F16')), [1, 2 ** -24, -2]);
  const bf = new Uint8Array(Uint16Array.from([0x3f80, 0xc0a0]).buffer);
  assert.deepEqual(Array.from(decode(bf, 'BF16')), [1, -5]);
  // unaligned view (offset 1) must still work
  const shifted = new Uint8Array(7); shifted.set(f16, 1);
  assert.deepEqual(Array.from(decode(shifted.subarray(1), 'F16')), [1, 2 ** -24, -2]);
});

test('safetensors: hand-built buffer round-trips, offsets point at the data', () => {
  const w = Float32Array.from({ length: 12 }, (_, i) => i * 0.25 - 1);
  const b = new Uint16Array([0x3c00, 0x4000, 0xbc00, 0x0000]);
  const buf = buildSafetensors([
    { name: 'model.layers.0.mlp.down_proj.weight', dtype: 'F32', shape: [3, 4], data: w },
    { name: 'model.norm.weight', dtype: 'F16', shape: [4], data: b },
  ], { format: 'pt' });
  // too short → asks for more
  assert.deepEqual(parseSafetensorsHeader(buf.subarray(0, 4)), { need: 8 });
  const n = Number(new DataView(buf.buffer).getBigUint64(0, true));
  assert.deepEqual(parseSafetensorsHeader(buf.subarray(0, 20)), { need: 8 + n });
  const h = parseSafetensorsHeader(buf);
  assert.equal(h.metadata.format, 'pt');
  assert.equal(h.tensors.length, 2);
  const t0 = h.tensors.find((t) => t.name.endsWith('down_proj.weight'));
  assert.deepEqual(t0.shape, [3, 4]);
  assert.equal(t0.numel, 12);
  assert.deepEqual(Array.from(decode(buf.subarray(t0.offset, t0.offset + t0.length), 'F32')), Array.from(w));
  const t1 = h.tensors.find((t) => t.name === 'model.norm.weight');
  assert.deepEqual(Array.from(decode(buf.subarray(t1.offset, t1.offset + t1.length), 'F16')), [1, 2, -1, 0]);
  assert.equal(countParams(h.tensors).total, 16);
});

test('safetensors: rejects garbage', () => {
  const junk = new Uint8Array(16); junk.fill(0xff);
  assert.throws(() => parseSafetensorsHeader(junk), /does not look like/);
});

test('index.json', () => {
  const idx = parseIndexJson({ metadata: { total_size: 1000 }, weight_map: { a: 'm-1.safetensors', b: 'm-2.safetensors', c: 'm-1.safetensors' } });
  assert.equal(idx.totalSize, 1000);
  assert.deepEqual(idx.files, ['m-1.safetensors', 'm-2.safetensors']);
  assert.throws(() => parseIndexJson({}), /weight_map/);
});

test('GPTQ / AWQ packed tensors are unpacked, bookkeeping tensors skipped', () => {
  // A 4096×4096 4-bit GPTQ layer: qweight int32 [4096/8, 4096], scales [32, 4096], qzeros [32, 512], g_idx [4096]
  const T = (name, dtype, shape) => ({ name, dtype, shape, numel: shape.reduce((a, b) => a * b, 1) });
  const gptq = [
    T('model.layers.0.self_attn.q_proj.qweight', 'I32', [512, 4096]),
    T('model.layers.0.self_attn.q_proj.scales', 'F16', [32, 4096]),
    T('model.layers.0.self_attn.q_proj.qzeros', 'I32', [32, 512]),
    T('model.layers.0.self_attn.q_proj.g_idx', 'I32', [4096]),
    T('model.embed_tokens.weight', 'F16', [1000, 4096]),
  ];
  assert.equal(detectPackedQuant(gptq), 'gptq');
  const naive = gptq.reduce((a, t) => a + t.numel, 0);
  const r = countParams(gptq, { bits: 4 });
  assert.equal(r.total, 4096 * 4096 + 1000 * 4096);
  assert.ok(r.unpacked && r.total > naive * 2);
  assert.equal(detectPackedQuant(gptq.filter((t) => !t.name.endsWith('g_idx'))), 'awq');
  assert.equal(detectPackedQuant([T('x.weight', 'F16', [2, 2])]), null);
});

/* ---------- GGUF: build a small file by hand ---------- */
function buildGGUF({ kv, tensors, alignment = 32 }) {
  const parts = [];
  const u32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); parts.push(b); };
  const u64 = (v) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); parts.push(b); };
  const str = (s) => { const e = new TextEncoder().encode(s); u64(e.length); parts.push(e); };
  const f32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); parts.push(b); };
  parts.push(new TextEncoder().encode('GGUF'));
  u32(3); u64(tensors.length); u64(kv.length);
  for (const [k, type, v] of kv) {
    str(k); u32(type);
    if (type === 8) str(v); else if (type === 4) u32(v); else if (type === 6) f32(v);
    else if (type === 9) { u32(8); u64(v.length); v.forEach(str); }
  }
  let off = 0;
  for (const t of tensors) {
    str(t.name); u32(t.ne.length); t.ne.forEach(u64); u32(t.type); u64(off);
    off += t.bytes;
  }
  let len = parts.reduce((a, b) => a + b.length, 0);
  const pad = (alignment - (len % alignment)) % alignment;
  parts.push(new Uint8Array(pad + off));
  const out = new Uint8Array(len + pad + off);
  let p = 0; for (const b of parts) { out.set(b, p); p += b.length; }
  return out;
}

test('GGUF header: metadata, tensor infos, sizes, data offsets', () => {
  const vocab = Array.from({ length: 100 }, (_, i) => `tok${i}`);
  const buf = buildGGUF({
    kv: [
      ['general.architecture', 8, 'llama'], ['general.name', 8, 'Tiny'], ['general.file_type', 4, 15],
      ['llama.block_count', 4, 2], ['llama.embedding_length', 4, 64], ['llama.context_length', 4, 4096],
      ['llama.rope.freq_base', 6, 10000], ['tokenizer.ggml.tokens', 9, vocab],
    ],
    tensors: [
      { name: 'token_embd.weight', ne: [64, 100], type: 12, bytes: (6400 / 256) * 144 },  // Q4_K
      { name: 'blk.0.attn_q.weight', ne: [64, 64], type: 1, bytes: 64 * 64 * 2 },          // F16
      { name: 'output_norm.weight', ne: [64], type: 0, bytes: 256 },                        // F32
    ],
  });
  const g = parseGGUF(buf);
  assert.equal(g.version, 3);
  assert.equal(g.metadata['general.name'], 'Tiny');
  assert.deepEqual(g.metadata['tokenizer.ggml.tokens'], { array: 8, count: 100 }); // big arrays are summarised
  assert.equal(g.dataStart % 32, 0);
  const q = g.tensors[1];
  assert.deepEqual(q.shape, [64, 64]);
  assert.equal(q.dtype, 'F16');
  assert.equal(g.tensors[0].length, 3600);
  assert.equal(q.offset, g.dataStart + 3600);
  const s = ggufSummary(g);
  assert.equal(s.paramsTotal, 6400 + 4096 + 64);
  assert.equal(s.layers, 2);
  assert.equal(s.vocabSize, 100);
  assert.equal(s.fileType, 'Q4_K_M');
  // Truncated input asks for more bytes instead of failing
  assert.throws(() => parseGGUF(buf.subarray(0, 60)), (e) => e instanceof NeedMoreBytes && e.need > 60);
  assert.throws(() => parseGGUF(new TextEncoder().encode('NOPE12345678')), /GGUF/);
});
