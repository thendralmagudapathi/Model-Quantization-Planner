import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {
  parseHfInput, planSample, importFiles, importConfigText, importManual, readBytes, Budget, UserError,
  listLocalModels, importLocalServer, importUrl, readTensorSample, scanEstimate, net,
} from '../public/import.js';
import { buildSafetensors } from '../public/parsers.js';

/* ---------- a tiny Llama-shaped model in memory ---------- */
const H = 64, I = 128, V = 100, L = 2;
const CONFIG = { model_type: 'llama', architectures: ['LlamaForCausalLM'], hidden_size: H, intermediate_size: I, num_hidden_layers: L, num_attention_heads: 4, num_key_value_heads: 4, vocab_size: V, tie_word_embeddings: false, torch_dtype: 'float32', max_position_embeddings: 512 };
const f32 = (n, seed = 1) => { let s = seed; return Float32Array.from({ length: n }, () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 0.04); };
function tinyModel() {
  const e = [{ name: 'model.embed_tokens.weight', shape: [V, H] }, { name: 'lm_head.weight', shape: [V, H] }, { name: 'model.norm.weight', shape: [H] }];
  for (let l = 0; l < L; l++) {
    for (const p of ['q', 'k', 'v', 'o']) e.push({ name: `model.layers.${l}.self_attn.${p}_proj.weight`, shape: [H, H] });
    e.push({ name: `model.layers.${l}.mlp.gate_proj.weight`, shape: [I, H] }, { name: `model.layers.${l}.mlp.up_proj.weight`, shape: [I, H] }, { name: `model.layers.${l}.mlp.down_proj.weight`, shape: [H, I] });
    e.push({ name: `model.layers.${l}.input_layernorm.weight`, shape: [H] }, { name: `model.layers.${l}.post_attention_layernorm.weight`, shape: [H] });
  }
  return buildSafetensors(e.map((t, i) => ({ ...t, dtype: 'F32', data: f32(t.shape.reduce((a, b) => a * b, 1), i + 1) })));
}
const tinyParams = 2 * V * H + H + L * (4 * H * H + 3 * H * I + 2 * H);

test('parseHfInput accepts IDs, links, branches and rejects non-model links', () => {
  assert.deepEqual(parseHfInput('Qwen/Qwen2.5-7B-Instruct'), { id: 'Qwen/Qwen2.5-7B-Instruct', rev: 'main', file: null });
  assert.deepEqual(parseHfInput('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'), { id: 'Qwen/Qwen2.5-7B-Instruct', rev: 'main', file: null });
  assert.deepEqual(parseHfInput('https://huggingface.co/org/m/tree/v1.0'), { id: 'org/m', rev: 'v1.0', file: null });
  assert.deepEqual(parseHfInput('https://huggingface.co/org/m/blob/main/config.json'), { id: 'org/m', rev: 'main', file: 'config.json' });
  assert.equal(parseHfInput('gpt2').id, 'gpt2');
  assert.throws(() => parseHfInput('https://huggingface.co/datasets/x/y'), /dataset/);
  assert.throws(() => parseHfInput('https://example.com/a/b'), /not a huggingface/);
  assert.throws(() => parseHfInput('not a model!'), /doesn't look like/);
});

test('planSample: evenly spaced whole-row blocks, capped at 1,048,576 values, no overlap', () => {
  const t = { dtype: 'BF16', shape: [4096, 12288], offset: 1000 };
  const p = planSample(t, 65536);
  assert.equal(p.cols, 12288);
  assert.equal(p.rows, 5);                                   // floor(65536 / 12288)
  assert.deepEqual(p.ranges.map((r) => r.startRow), [0, 1023, 2047, 3071, 4095]);
  assert.equal(p.ranges[0].start, 1000);
  const big = planSample(t, 10_000_000);
  assert.ok(big.rows * big.cols <= 1048576);
  const small = planSample({ dtype: 'F16', shape: [10, 100], offset: 0 }, 65536); // all rows wanted → contiguous
  assert.deepEqual(small.ranges.map((r) => [r.startRow, r.rows]), [[0, 2], [2, 2], [4, 2], [6, 2], [8, 2]]);
  const wide = planSample({ dtype: 'F16', shape: [2, 5_000_000], offset: 0 }, 65536);
  assert.equal(wide.partialRows, true);
  assert.equal(wide.ranges[0].end, 65536 * 2);
  // every block of every plan stays inside the tensor and blocks never overlap
  for (const [shape, target] of [[[100, 100], 9000], [[4096, 4096], 262144], [[33, 7], 200]]) {
    const pl = planSample({ dtype: 'F32', shape, offset: 0 }, target);
    const spans = pl.ranges.map((r) => [r.startRow, r.startRow + r.rows]).sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < spans.length; i++) {
      assert.ok(spans[i][0] >= 0 && spans[i][1] <= shape[0], `${shape}: block inside`);
      if (i) assert.ok(spans[i][0] >= spans[i - 1][1], `${shape}: no overlap`);
    }
  }
});

test('local files: exact count from the header, only small slices are read', async () => {
  const bytes = tinyModel();
  const reads = [];
  const file = new File([bytes], 'model.safetensors');
  const spy = { name: file.name, size: file.size, slice: (a, b) => { reads.push([a, b]); return file.slice(a, b); }, text: () => file.text() };
  const cfg = new File([JSON.stringify(CONFIG)], 'config.json');
  const p = await importFiles([spy, cfg]);
  assert.equal(p.paramsTotal, tinyParams);
  assert.equal(p.paramExact, true);
  assert.equal(p.derived.total, tinyParams);            // config formula agrees with the real file
  assert.equal(p.layers, 2);
  assert.equal(p.weightAccess, 'local-file');
  assert.equal(p.originalDtype, 'FP32');
  // a sample of one tensor, read with File.slice
  const t = p.tensors.find((x) => x.name === 'model.layers.1.mlp.down_proj.weight');
  const s = await readTensorSample(p, t, 1000);
  assert.equal(s.cols, I);
  assert.equal(s.weights.length, s.rows * I);
  assert.ok(reads.length >= 2);
  assert.ok(scanEstimate(p, 65536).tensors.length === 2 + L * 7);
});

test('local files: clear errors for unusable input', async () => {
  await assert.rejects(importFiles([new File(['hello'], 'notes.txt')]), /None of these files/);
  await assert.rejects(importFiles([new File([JSON.stringify({ weight_map: { a: 'x.safetensors' } })], 'model.safetensors.index.json')]), /index file alone/);
  await assert.rejects(importFiles([]), (e) => e instanceof UserError);
});

test('packed GPTQ checkpoint: unpacked parameter count and analysis disabled', async () => {
  const q = buildSafetensors([
    { name: 'model.layers.0.self_attn.q_proj.qweight', dtype: 'I32', shape: [H / 8, H], data: new Int32Array(H * H / 8) },
    { name: 'model.layers.0.self_attn.q_proj.scales', dtype: 'F16', shape: [1, H], data: new Uint16Array(H) },
    { name: 'model.layers.0.self_attn.q_proj.qzeros', dtype: 'I32', shape: [1, H / 8], data: new Int32Array(H / 8) },
    { name: 'model.layers.0.self_attn.q_proj.g_idx', dtype: 'I32', shape: [H], data: new Int32Array(H) },
  ]);
  const p = await importFiles([new File([q], 'gptq.safetensors'), new File([JSON.stringify({ ...CONFIG, quantization_config: { quant_method: 'gptq', bits: 4, group_size: 128, sym: true } })], 'config.json')]);
  assert.equal(p.paramsTotal, H * H);
  assert.equal(p.weightAccess, 'none');
  assert.match(p.weightNote, /already quantized/);
  assert.equal(p.quantization.label, 'GPTQ 4-bit g128');
});

test('paste config.json and manual entry', () => {
  const p = importConfigText(JSON.stringify(CONFIG));
  assert.equal(p.paramsTotal, tinyParams);
  assert.equal(p.paramExact, false);
  assert.throws(() => importConfigText('{ nope'), /not valid JSON/);
  assert.equal(importManual({ params: '70B', dtype: 'BF16' }).paramsTotal, 70e9);
  assert.throws(() => importManual({ params: 'lots' }), /Type the number/);
});

/* ---------- HTTP: a file host with / without Range support, a mock Ollama and an OpenAI-style server ---------- */
let server, base, served = 0;
const BIG = tinyModel();
before(async () => {
  server = http.createServer((req, res) => {
    const h = { 'Access-Control-Allow-Origin': '*' };
    const u = new URL(req.url, 'http://x');
    if (u.pathname === '/api/tags' && req.headers.host.startsWith('127')) {
      return res.writeHead(200, h).end(JSON.stringify({ models: [{ name: 'qwen3:8b-q8_0', size: 8851089538, details: { parameter_size: '8.2B', quantization_level: 'Q8_0' } }] }));
    }
    if (u.pathname === '/api/show') {
      return res.writeHead(200, h).end(JSON.stringify({ details: { family: 'qwen3', quantization_level: 'Q8_0', parameter_size: '8.2B' }, model_info: { 'general.architecture': 'qwen3', 'general.parameter_count': 8190735360, 'qwen3.block_count': 36, 'qwen3.context_length': 40960, 'qwen3.embedding_length': 4096 }, tensors: [] }));
    }
    if (u.pathname === '/v1/models') return res.writeHead(200, h).end(JSON.stringify({ data: [{ id: 'Qwen/Qwen2.5-7B-Instruct', root: 'Qwen/Qwen2.5-7B-Instruct', max_model_len: 32768 }] }));
    if (u.pathname.endsWith('/config.json')) return res.writeHead(200, h).end(JSON.stringify(CONFIG));
    if (u.pathname === '/range/model.safetensors' || u.pathname === '/norange/model.safetensors') {
      const m = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range || '');
      if (m && u.pathname.startsWith('/range')) {
        const s = +m[1], e = Math.min(+m[2], BIG.length - 1);
        served += e - s + 1;
        return res.writeHead(206, { ...h, 'Content-Range': `bytes ${s}-${e}/${BIG.length}` }).end(Buffer.from(BIG.subarray(s, e + 1)));
      }
      // ignores Range: streams the whole file slowly so the client has to stop it
      res.writeHead(200, h);
      let i = 0;
      const iv = setInterval(() => { if (i >= BIG.length || res.destroyed) { clearInterval(iv); return res.end(); } served += 4096; res.write(Buffer.from(BIG.subarray(i, i += 4096))); }, 5);
      res.on('close', () => clearInterval(iv));
      return;
    }
    res.writeHead(404, h).end('{}');
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => server.close());

test('Range reads: 206 is accepted, a server that ignores Range is stopped early', async () => {
  served = 0;
  const b = await readBytes({ kind: 'url', url: `${base}/range/model.safetensors` }, 0, 8);
  assert.equal(b.byteLength, 8);
  served = 0;
  await assert.rejects(readBytes({ kind: 'url', url: `${base}/norange/model.safetensors` }, 0, 8), (e) => e.code === 'norange');
  await new Promise((r) => setTimeout(r, 50));
  assert.ok(served < BIG.length / 2, `stopped after ${served} of ${BIG.length} bytes`);
});

test('byte budget is enforced before anything is fetched', async () => {
  const budget = new Budget(100);
  await assert.rejects(readBytes({ kind: 'url', url: `${base}/range/model.safetensors` }, 0, 1000, { budget }), (e) => e.code === 'budget');
  assert.equal(budget.used, 0);
});

test('direct link to a .safetensors file (plus its config.json next to it)', async () => {
  const before = net.requests;
  const p = await importUrl(`${base}/range/model.safetensors`);
  assert.equal(p.paramsTotal, tinyParams);
  assert.equal(p.weightAccess, 'http-range');
  assert.equal(p.layers, 2);
  assert.ok(net.requests - before <= 4, 'only a few small requests');
});

test('local servers: Ollama details, and an OpenAI-compatible server (vLLM / LM Studio)', async () => {
  const list = await listLocalModels(base);
  assert.equal(list.kind, 'ollama');
  const p = await importLocalServer(base, 'qwen3:8b-q8_0', list);
  assert.equal(p.paramsTotal, 8190735360);
  assert.equal(p.layers, 36);
  assert.equal(p.quantization.label, 'GGUF Q8_0');
  assert.equal(p.weightAccess, 'none');
  // OpenAI-compatible: same server, but pretend /api/tags does not exist by using "localhost" (the mock only answers 127.*)
  const oa = `http://localhost:${new URL(base).port}`;
  const l2 = await listLocalModels(oa);
  assert.equal(l2.kind, 'openai');
  const p2 = await importLocalServer(oa, 'Qwen/Qwen2.5-7B-Instruct', l2);
  assert.equal(p2.paramsTotal, 7e9);
  assert.equal(p2.suggestHf, 'Qwen/Qwen2.5-7B-Instruct');
  assert.equal(p2.contextLength, 32768);
  assert.match(p2.paramSource, /Guessed/);
});

test('unreachable local server gives the CORS / setup hint', async () => {
  await assert.rejects(listLocalModels('http://127.0.0.1:9'), (e) => e.code === 'local-cors');
});
