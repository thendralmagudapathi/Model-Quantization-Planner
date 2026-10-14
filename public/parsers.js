/* File-format parsers: safetensors headers, sharded index.json, GGUF headers, dtype decoding.
 * Pure functions on bytes — no network, no DOM. */

/* =====================================================================
 * Dtypes
 * ===================================================================== */
export const DTYPE_BYTES = {
  F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
  F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, F4: 0.5,
};
// Dtypes whose raw values we can decode and analyse as ordinary weights.
export const READABLE_DTYPES = new Set(['F32', 'F16', 'BF16']);

// IEEE half → float, including subnormals, infinities and NaN.
export function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;                  // zero / subnormal: m × 2^-14 × 2^-10
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const HALF_LUT = (() => { const t = new Float32Array(65536); for (let i = 0; i < 65536; i++) t[i] = halfToFloat(i); return t; })();

// BF16 is the top half of an FP32: shift left by 16.
export function bf16ToFloat(u) {
  const b = new DataView(new ArrayBuffer(4));
  b.setUint32(0, (u & 0xffff) << 16, true);
  return b.getFloat32(0, true);
}

/** Decode little-endian raw bytes of a dtype into a Float32Array. */
export function decode(bytes, dtype) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const per = DTYPE_BYTES[dtype];
  const n = Math.floor(u8.byteLength / per);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = new Float32Array(n);
  if (dtype === 'F32') for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
  else if (dtype === 'F16') for (let i = 0; i < n; i++) out[i] = HALF_LUT[dv.getUint16(i * 2, true)];
  else if (dtype === 'BF16') {
    const u32 = new Uint32Array(n);
    for (let i = 0; i < n; i++) u32[i] = dv.getUint16(i * 2, true) << 16;
    return new Float32Array(u32.buffer);
  } else if (dtype === 'F64') for (let i = 0; i < n; i++) out[i] = dv.getFloat64(i * 8, true);
  else throw new Error(`Cannot decode ${dtype} values`);
  return out;
}

const numel = (shape) => shape.reduce((a, b) => a * b, 1);

/* =====================================================================
 * safetensors
 * [u64 little-endian header length N][N bytes JSON header][tensor data]
 * ===================================================================== */
export const SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024;

/**
 * Parse a safetensors header from the first bytes of a file.
 * Returns { need: n } if more bytes are required, otherwise the parsed header.
 */
export function parseSafetensorsHeader(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8.byteLength < 8) return { need: 8 };
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = Number(dv.getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n < 2 || n > SAFETENSORS_MAX_HEADER) throw new Error('This does not look like a .safetensors file (bad header length).');
  if (u8.byteLength < 8 + n) return { need: 8 + n };
  let json;
  try { json = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n))); }
  catch { throw new Error('The .safetensors header is not valid JSON — the file may be damaged.'); }
  const metadata = json.__metadata__ || {};
  const dataStart = 8 + n;
  const tensors = Object.entries(json).filter(([k]) => k !== '__metadata__').map(([name, t]) => {
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.data_offsets)) throw new Error(`Tensor "${name}" has a malformed header entry.`);
    const [start, end] = t.data_offsets;
    return { name, dtype: t.dtype, shape: t.shape, numel: numel(t.shape), offset: dataStart + start, length: end - start };
  });
  return { headerSize: n, dataStart, metadata, tensors };
}

/** Build a safetensors file in memory (used by tests and the in-page demo). */
export function buildSafetensors(entries, metadata) {
  const header = {};
  let off = 0;
  const parts = [];
  for (const { name, dtype, shape, data } of entries) {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    header[name] = { dtype, shape, data_offsets: [off, off + bytes.byteLength] };
    parts.push(bytes);
    off += bytes.byteLength;
  }
  if (metadata) header.__metadata__ = metadata;
  let hj = new TextEncoder().encode(JSON.stringify(header));
  const pad = (8 - (hj.byteLength % 8)) % 8;
  if (pad) hj = new TextEncoder().encode(JSON.stringify(header) + ' '.repeat(pad));
  const out = new Uint8Array(8 + hj.byteLength + off);
  new DataView(out.buffer).setBigUint64(0, BigInt(hj.byteLength), true);
  out.set(hj, 8);
  let p = 8 + hj.byteLength;
  for (const b of parts) { out.set(b, p); p += b.byteLength; }
  return out;
}

/** model.safetensors.index.json → { totalSize, weightMap, files } */
export function parseIndexJson(obj) {
  const weightMap = obj?.weight_map;
  if (!weightMap || typeof weightMap !== 'object') throw new Error('This index file has no "weight_map".');
  return { totalSize: Number(obj.metadata?.total_size) || 0, weightMap, files: [...new Set(Object.values(weightMap))] };
}

/* =====================================================================
 * Already-quantized checkpoints (GPTQ / AWQ / bitsandbytes / FP8)
 * Their tensors are packed, so counting shape products gives the wrong parameter count.
 * ===================================================================== */
export function detectPackedQuant(tensors) {
  const names = tensors.map((t) => t.name);
  const has = (re) => names.some((n) => re.test(n));
  if (has(/\.qweight$/)) return has(/\.g_idx$/) ? 'gptq' : 'awq';
  if (has(/\.weight_packed$/)) return 'compressed-tensors';
  if (has(/\.(absmax|quant_map|quant_state\.bitsandbytes__(nf4|fp4))$/)) return 'bitsandbytes-4bit';
  if (has(/\.SCB$/)) return 'bitsandbytes-8bit';
  if (has(/\.weight_scale_inv$|\.weight_scale$/) && tensors.some((t) => /^F8/.test(t.dtype))) return 'fp8';
  return null;
}

// Tensors that hold quantization bookkeeping, not model parameters.
const QUANT_AUX = /\.(qzeros|scales|g_idx|absmax|quant_map|nested_absmax|nested_quant_map|quant_state\..*|SCB|weight_scale_inv|weight_scale|input_scale|weight_zero_point|weight_shape|weight_g_idx)$/;

/**
 * Count parameters from tensor headers.
 * For packed formats, unpack: GPTQ/AWQ qweight is int32 holding 32/bits values; bnb 4-bit U8 holds 2.
 */
export function countParams(tensors, quantConfig) {
  const method = detectPackedQuant(tensors);
  const bits = Number(quantConfig?.bits || quantConfig?.w_bit || quantConfig?.config_groups?.group_0?.weights?.num_bits) || 4;
  let total = 0, unpacked = false;
  for (const t of tensors) {
    if (method && QUANT_AUX.test(t.name)) continue;
    let n = t.numel;
    if (/\.qweight$/.test(t.name) && /I32|U32/.test(t.dtype)) { n *= 32 / bits; unpacked = true; }
    else if (/\.weight_packed$/.test(t.name) && /I32|U32/.test(t.dtype)) { n *= 32 / bits; unpacked = true; }
    else if (method === 'bitsandbytes-4bit' && t.dtype === 'U8' && /\.weight$/.test(t.name)) { n *= 2; unpacked = true; }
    total += n;
  }
  return { total, method, unpacked };
}

/* =====================================================================
 * GGUF (llama.cpp)
 * "GGUF" magic, u32 version, u64 tensor count, u64 metadata count, metadata KVs, tensor infos, aligned data.
 * ===================================================================== */
// ggml tensor types: [name, block size (values), bytes per block]
export const GGML_TYPES = {
  0: ['F32', 1, 4], 1: ['F16', 1, 2], 2: ['Q4_0', 32, 18], 3: ['Q4_1', 32, 20], 6: ['Q5_0', 32, 22], 7: ['Q5_1', 32, 24],
  8: ['Q8_0', 32, 34], 9: ['Q8_1', 32, 36], 10: ['Q2_K', 256, 84], 11: ['Q3_K', 256, 110], 12: ['Q4_K', 256, 144],
  13: ['Q5_K', 256, 176], 14: ['Q6_K', 256, 210], 15: ['Q8_K', 256, 292], 16: ['IQ2_XXS', 256, 66], 17: ['IQ2_XS', 256, 74],
  18: ['IQ3_XXS', 256, 98], 19: ['IQ1_S', 256, 50], 20: ['IQ4_NL', 32, 18], 21: ['IQ3_S', 256, 110], 22: ['IQ2_S', 256, 82],
  23: ['IQ4_XS', 256, 136], 24: ['I8', 1, 1], 25: ['I16', 1, 2], 26: ['I32', 1, 4], 27: ['I64', 1, 8], 28: ['F64', 1, 8],
  29: ['IQ1_M', 256, 56], 30: ['BF16', 1, 2], 34: ['TQ1_0', 256, 54], 35: ['TQ2_0', 256, 66], 39: ['MXFP4', 32, 17],
};
// general.file_type → the name people use for the whole file
export const GGUF_FILE_TYPES = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M',
  13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K', 19: 'IQ2_XXS', 20: 'IQ2_XS',
  21: 'Q2_K_S', 22: 'IQ3_XS', 23: 'IQ3_XXS', 24: 'IQ1_S', 25: 'IQ4_NL', 26: 'IQ3_S', 27: 'IQ3_M', 28: 'IQ2_S',
  29: 'IQ2_M', 30: 'IQ4_XS', 31: 'IQ1_M', 32: 'BF16', 36: 'TQ1_0', 37: 'TQ2_0', 38: 'MXFP4_MOE',
};

export class NeedMoreBytes extends Error {
  constructor(n) { super(`Need at least ${n} bytes`); this.need = n; }
}

/**
 * Parse a GGUF header. Throws NeedMoreBytes if `bytes` is too short — call again with more.
 * Large arrays (tokenizer vocab) are skipped and summarised as { array: type, count }.
 */
export function parseGGUF(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const td = new TextDecoder();
  let p = 0;
  const need = (n) => { if (p + n > u8.byteLength) throw new NeedMoreBytes(Math.max(p + n, u8.byteLength * 2)); };
  const u32 = () => { need(4); const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { need(8); const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); need(n); const s = td.decode(u8.subarray(p, p + n)); p += n; return s; };
  const SIZES = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  const scalar = (t) => {
    const n = SIZES[t];
    if (n === undefined) throw new Error(`Unknown GGUF value type ${t}`);
    need(n);
    let v;
    switch (t) {
      case 0: v = dv.getUint8(p); break; case 1: v = dv.getInt8(p); break;
      case 2: v = dv.getUint16(p, true); break; case 3: v = dv.getInt16(p, true); break;
      case 4: v = dv.getUint32(p, true); break; case 5: v = dv.getInt32(p, true); break;
      case 6: v = dv.getFloat32(p, true); break; case 7: v = dv.getUint8(p) !== 0; break;
      case 10: v = Number(dv.getBigUint64(p, true)); break; case 11: v = Number(dv.getBigInt64(p, true)); break;
      case 12: v = dv.getFloat64(p, true); break;
    }
    p += n;
    return v;
  };
  const value = (t) => {
    if (t === 8) return str();
    if (t === 9) {
      const at = u32(), count = u64();
      if (count <= 64) { const a = []; for (let i = 0; i < count; i++) a.push(value(at)); return a; }
      if (at === 8) for (let i = 0; i < count; i++) { const n = u64(); need(n); p += n; }  // skip strings
      else if (at === 9) throw new Error('Nested GGUF arrays are not supported');
      else { need(SIZES[at] * count); p += SIZES[at] * count; }
      return { array: at, count };
    }
    return scalar(t);
  };

  need(4);
  if (td.decode(u8.subarray(0, 4)) !== 'GGUF') throw new Error('This does not look like a GGUF file (missing "GGUF" magic).');
  p = 4;
  const version = u32();
  if (version < 2) throw new Error(`GGUF version ${version} is too old to read.`);
  const nTensors = u64(), nKV = u64();
  const metadata = {};
  for (let i = 0; i < nKV; i++) { const k = str(); metadata[k] = value(u32()); }
  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = str(), nd = u32(), ne = [];
    for (let d = 0; d < nd; d++) ne.push(u64());
    const type = u32(), offset = u64();
    const [typeName, block, bytesPer] = GGML_TYPES[type] || [`type ${type}`, 1, 0];
    const n = numel(ne);
    tensors.push({ name, shape: ne.slice().reverse(), dtype: typeName, ggmlType: type, numel: n, relOffset: offset, length: bytesPer ? (n / block) * bytesPer : 0 });
  }
  const alignment = Number(metadata['general.alignment']) || 32;
  const dataStart = Math.ceil(p / alignment) * alignment;
  for (const t of tensors) t.offset = dataStart + t.relOffset;
  return { version, metadata, tensors, dataStart, alignment, headerBytes: p };
}

/** Summarise GGUF metadata into model-profile fields. */
export function ggufSummary(g) {
  const m = g.metadata, arch = m['general.architecture'] || 'unknown';
  const k = (s) => m[`${arch}.${s}`];
  const total = g.tensors.reduce((a, t) => a + t.numel, 0);
  const bytes = g.tensors.reduce((a, t) => a + t.length, 0);
  const tok = m['tokenizer.ggml.tokens'];
  const counts = {};
  for (const t of g.tensors) counts[t.dtype] = (counts[t.dtype] || 0) + t.numel;
  const main = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];
  return {
    architecture: arch, name: m['general.name'] || m['general.basename'] || '',
    paramsTotal: total, layers: k('block_count'), hiddenSize: k('embedding_length'), intermediateSize: k('feed_forward_length'),
    contextLength: k('context_length'), heads: k('attention.head_count'), kvHeads: k('attention.head_count_kv'),
    experts: k('expert_count'), expertsUsed: k('expert_used_count'),
    vocabSize: k('vocab_size') || (Array.isArray(tok) ? tok.length : tok?.count),
    fileType: GGUF_FILE_TYPES[m['general.file_type']] || main, mainType: main, typeCounts: counts,
    bitsPerWeight: total ? (bytes * 8) / total : 0, tensorBytes: bytes,
  };
}
