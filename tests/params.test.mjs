import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { paramsFromConfig, readQuantConfig } from '../public/params.js';

// config.json files and Hugging Face's exact safetensors parameter counts, fetched from the Hub.
const FIX = JSON.parse(readFileSync(new URL('./fixtures/configs.json', import.meta.url)));

// HF's count includes non-parameter buffers for GPT-2 / Pythia (causal-mask "attn.bias" tensors), and
// Qwen3-0.6B's checkpoint stores a duplicate lm_head although embeddings are tied — so compare with the
// published counts instead (Qwen3-0.6B: "0.6B total, 0.44B non-embedding" = 440,467,456 + 151,936 × 1,024).
const PUBLISHED = {
  'openai-community/gpt2': 124439808,
  'EleutherAI/pythia-160m': 162322944,
  'Qwen/Qwen3-0.6B': 440467456 + 151936 * 1024,
};

for (const [id, { config, safetensors }] of Object.entries(FIX)) {
  test(`parameter count from config.json: ${id}`, () => {
    const r = paramsFromConfig(config);
    if (config.text_config) {
      // Multimodal: we count the language model only, so we must land below the full total (vision ≈ 10%).
      assert.ok(r.total < safetensors.total && r.total > 0.85 * safetensors.total, `${r.total} vs ${safetensors.total}`);
      assert.ok(r.assumptions.some((a) => /vision/.test(a)));
      return;
    }
    const want = PUBLISHED[id] ?? safetensors.total;
    const err = Math.abs(r.total - want) / want;
    assert.ok(err < 0.01, `${id}: got ${r.total.toLocaleString()}, expected ${want.toLocaleString()} (${(err * 100).toFixed(2)}% off)`);
  });
}

test('exact matches where the formula is complete', () => {
  assert.equal(paramsFromConfig(FIX['Qwen/Qwen2.5-0.5B-Instruct'].config).total, 494032768);
  assert.equal(paramsFromConfig(FIX['unsloth/Meta-Llama-3.1-8B'].config).total, 8030261248);
  assert.equal(paramsFromConfig(FIX['openai-community/gpt2'].config).total, 124439808);
  assert.equal(paramsFromConfig(FIX['EleutherAI/pythia-160m'].config).total, 162322944);
});

test('MoE models report total and active parameters', () => {
  const mix = paramsFromConfig(FIX['mistralai/Mixtral-8x7B-v0.1'].config);
  assert.ok(Math.abs(mix.active - 12.9e9) / 12.9e9 < 0.02, `Mixtral active ${mix.active}`);   // published ≈ 12.9B active
  const q3 = paramsFromConfig(FIX['Qwen/Qwen3-30B-A3B'].config);
  assert.ok(Math.abs(q3.active - 3.3e9) / 3.3e9 < 0.03, `Qwen3-30B-A3B active ${q3.active}`); // published 3.3B active
  const q15 = paramsFromConfig(FIX['Qwen/Qwen1.5-MoE-A2.7B'].config);
  assert.ok(Math.abs(q15.active - 2.7e9) / 2.7e9 < 0.04, `Qwen1.5-MoE active ${q15.active}`);
  assert.equal(mix.experts, 8); assert.equal(mix.expertsUsed, 2);
});

test('missing fields give a clear error', () => {
  assert.throws(() => paramsFromConfig({ hidden_size: 10 }), /missing num_hidden_layers, num_attention_heads, vocab_size/);
});

test('quantization_config normalisation', () => {
  assert.deepEqual(readQuantConfig({ quant_method: 'gptq', bits: 4, group_size: 128, sym: true }),
    { method: 'gptq', label: 'GPTQ 4-bit g128', bits: 4, groupSize: 128, sym: true });
  assert.equal(readQuantConfig({ quant_method: 'awq', bits: 4, group_size: 128, zero_point: true }).sym, false);
  assert.equal(readQuantConfig({ load_in_4bit: true, bnb_4bit_quant_type: 'nf4', quant_method: 'bitsandbytes' }).label, 'bitsandbytes NF4');
  assert.equal(readQuantConfig(null), null);
});
