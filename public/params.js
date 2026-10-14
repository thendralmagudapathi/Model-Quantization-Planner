/* Parameter count from a Hugging Face config.json (dense and Mixture-of-Experts).
 * Llama-style formula; other families handled where they differ. Every assumption is returned
 * so the page can show it. Exact counts from weight files are always preferred over this estimate. */

// Families with LayerNorm (weight + bias), a plain (non-gated) MLP and biases on linear layers.
const GPT2_LIKE = new Set(['gpt2', 'gpt_bigcode', 'starcoder']);
const NEOX_LIKE = new Set(['gpt_neox', 'gptj', 'phi', 'opt', 'bloom', 'starcoder2', 'falcon', 'mpt', 'gpt_neo']);
// Families whose config usually omits tie_word_embeddings but ties them.
const TIED_BY_DEFAULT = /^(gemma|gpt2|gpt_bigcode|gpt_neo|opt|bloom|mpt|starcoder2|falcon|smollm)/;

const first = (c, ...keys) => { for (const k of keys) if (c[k] !== undefined && c[k] !== null) return Number(c[k]); return undefined; };

/** Normalise quantization_config from GPTQ / AWQ / bitsandbytes / FP8 / compressed-tensors. */
export function readQuantConfig(qc) {
  if (!qc || typeof qc !== 'object') return null;
  const method = String(qc.quant_method || (qc.load_in_4bit ? 'bitsandbytes' : qc.load_in_8bit ? 'bitsandbytes' : 'unknown')).toLowerCase();
  if (method === 'bitsandbytes') {
    return qc.load_in_8bit && !qc.load_in_4bit
      ? { method: 'bitsandbytes', label: 'bitsandbytes INT8', bits: 8, groupSize: null, sym: true }
      : { method: 'bitsandbytes', label: `bitsandbytes ${(qc.bnb_4bit_quant_type || 'nf4').toUpperCase()}`, bits: 4, groupSize: 64, sym: true };
  }
  if (method === 'compressed-tensors') {
    const w = qc.config_groups && Object.values(qc.config_groups)[0]?.weights;
    return { method, label: `compressed-tensors ${w?.type || ''}${w?.num_bits || ''}`.trim(), bits: w?.num_bits ?? null, groupSize: w?.group_size ?? null, sym: w?.symmetric ?? null };
  }
  if (method === 'fp8') return { method, label: 'FP8', bits: 8, groupSize: qc.weight_block_size ? qc.weight_block_size.join('×') : null, sym: true };
  const bits = qc.bits ?? qc.w_bit ?? qc.num_bits ?? null;
  const groupSize = qc.group_size ?? qc.q_group_size ?? null;
  const sym = qc.sym ?? (qc.zero_point !== undefined ? !qc.zero_point : null);
  const name = method === 'unknown' ? '' : method.toUpperCase();
  return { method, label: `${name}${bits ? ` ${bits}-bit` : ''}${groupSize > 0 ? ` g${groupSize}` : ''}`.trim() || 'quantized', bits, groupSize, sym };
}

/**
 * Count parameters from a config.json object.
 * Returns { total, active, embedding, perLayer, layers, hiddenSize, …, assumptions[] } or throws.
 */
export function paramsFromConfig(raw) {
  if (!raw || typeof raw !== 'object') throw new Error('config.json must be a JSON object.');
  const assumptions = [];
  let c = raw;
  if (raw.text_config && typeof raw.text_config === 'object') {
    c = { ...raw.text_config, model_type: raw.text_config.model_type || raw.model_type };
    assumptions.push('Multimodal model: only the language model (text_config) is counted — the vision/audio encoder is not included.');
  } else if (raw.llm_config && typeof raw.llm_config === 'object') {
    c = raw.llm_config;
    assumptions.push('Only the language model (llm_config) is counted.');
  }
  const mt = String(c.model_type || raw.model_type || '').toLowerCase();

  const H = first(c, 'hidden_size', 'n_embd', 'd_model', 'dim', 'model_dim');
  const L = first(c, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers');
  const nH = first(c, 'num_attention_heads', 'n_head', 'num_heads', 'n_heads');
  const V = first(c, 'vocab_size', 'padded_vocab_size', 'n_vocab');
  if (!H || !L || !nH || !V) {
    const miss = [['hidden_size', H], ['num_hidden_layers', L], ['num_attention_heads', nH], ['vocab_size', V]].filter(([, v]) => !v).map(([k]) => k);
    throw new Error(`config.json is missing ${miss.join(', ')} — can't work out the parameter count. Try "Enter details manually".`);
  }
  let kv = first(c, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads', 'multi_query_group_num');
  if (kv === undefined) kv = c.multi_query ? 1 : nH;
  const hd = first(c, 'head_dim') ?? H / nH;
  let I = first(c, 'intermediate_size', 'n_inner', 'ffn_dim', 'ffn_hidden_size');
  if (!I) { I = 4 * H; assumptions.push(`No intermediate_size: assumed 4 × hidden = ${I}.`); }
  const ctx = first(c, 'max_position_embeddings', 'n_positions', 'seq_length', 'max_seq_len', 'max_sequence_length') ?? first(raw, 'max_position_embeddings');
  const tied = c.tie_word_embeddings ?? raw.tie_word_embeddings ?? TIED_BY_DEFAULT.test(mt);
  if (c.tie_word_embeddings === undefined && raw.tie_word_embeddings === undefined) assumptions.push(`tie_word_embeddings not set: assumed ${tied ? 'tied (one shared matrix)' : 'separate output layer'} for "${mt || 'unknown'}".`);

  const common = { modelType: mt, layers: L, hiddenSize: H, intermediateSize: I, vocabSize: V, contextLength: ctx, heads: nH, kvHeads: kv, headDim: hd, tied };
  const embedding = V * H * (tied ? 1 : 2);

  /* ---------- GPT-2 family ---------- */
  if (GPT2_LIKE.has(mt)) {
    const pos = first(c, 'n_positions', 'max_position_embeddings') || 1024;
    const inner = first(c, 'n_inner') || 4 * H;
    const kvDim = c.multi_query ? 2 * hd : 2 * H;
    const layer = H * (H + kvDim) + (H + kvDim) + H * H + H + 2 * H * inner + inner + H + 4 * H;
    const total = L * layer + V * H + pos * H + 2 * H + (tied ? 0 : V * H);
    assumptions.push('GPT-2 style: LayerNorm with bias, plain 2-matrix MLP, biases on every layer, learned position embeddings.');
    return { ...common, total, active: total, embedding, perLayer: layer, family: 'gpt2', assumptions };
  }

  /* ---------- GPT-NeoX / Pythia / Phi-2 / OPT / Falcon-like ---------- */
  if (NEOX_LIKE.has(mt)) {
    const bias = c.bias !== false && mt !== 'falcon' && mt !== 'mpt' ? 1 : 0;
    const qkv = H * (nH * hd + 2 * kv * hd) + bias * (nH * hd + 2 * kv * hd);
    const o = nH * hd * H + bias * H;
    const mlp = 2 * H * I + bias * (I + H);
    const norms = c.parallel_attn || mt === 'phi' || mt === 'gptj' ? 2 * H : 4 * H;
    const layer = qkv + o + mlp + norms;
    const total = L * layer + V * H + 2 * H + (tied ? 0 : V * H) + (mt === 'opt' ? (ctx + 2) * H : 0);
    assumptions.push(`${mt}: LayerNorm (weight + bias), plain 2-matrix MLP${bias ? ', biases on linear layers' : ''}.`);
    return { ...common, total, active: total, embedding, perLayer: layer, family: 'neox', assumptions };
  }

  /* ---------- Llama-style (Llama, Mistral, Qwen, Gemma, Phi-3, OLMo, Granite, DeepSeek…) ---------- */
  // Attention
  let attn;
  const mla = c.kv_lora_rank != null;
  if (mla) {
    const nope = first(c, 'qk_nope_head_dim'), rope = first(c, 'qk_rope_head_dim'), vh = first(c, 'v_head_dim');
    const qr = first(c, 'q_lora_rank'), kr = first(c, 'kv_lora_rank');
    const q = qr ? H * qr + qr + qr * nH * (nope + rope) : H * nH * (nope + rope);
    attn = q + H * (kr + rope) + kr + kr * nH * (nope + vh) + nH * vh * H;
    assumptions.push('Multi-head latent attention (DeepSeek MLA): low-rank q / kv projections.');
  } else {
    attn = H * nH * hd + 2 * H * kv * hd + nH * hd * H;
    if (mt.startsWith('qwen2')) { attn += nH * hd + 2 * kv * hd; assumptions.push('Qwen2: biases on the q, k and v projections.'); }
    else if (c.attention_bias || c.qkv_bias) { attn += c.attention_bias ? nH * hd + 2 * kv * hd + H : nH * hd + 2 * kv * hd; assumptions.push('attention_bias: biases on the attention projections.'); }
    if (/^qwen3|^gemma3/.test(mt)) { attn += 2 * hd; assumptions.push('Per-head q/k normalisation weights (q_norm, k_norm).'); }
    if (/^olmo2/.test(mt)) attn += nH * hd + kv * hd;
  }
  if (kv !== nH && !mla) assumptions.push(`Grouped-query attention: ${kv} key/value heads for ${nH} query heads.`);

  // MLP: gated (gate, up, down) unless the activation says otherwise
  const gated = !/^(gelu|relu|gelu_new)$/.test(String(c.hidden_act || c.activation_function || 'silu')) || /^gemma/.test(mt);
  const mlpMats = gated ? 3 : 2;
  const mlpBias = c.mlp_bias ? (gated ? 2 * I : I) + H : 0;
  const dense = mlpMats * H * I + mlpBias;
  assumptions.push(gated ? 'Gated MLP: 3 matrices per layer (gate, up, down).' : 'Plain MLP: 2 matrices per layer.');

  // Norms
  const normsPerLayer = /^gemma[23]/.test(mt) ? 4 : 2;
  const norms = normsPerLayer * H;

  // Mixture of experts
  const E = first(c, 'num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts');
  const k = first(c, 'num_experts_per_tok', 'num_selected_experts', 'moe_top_k', 'top_k') ?? 2;
  let total, active, perLayer;
  if (E && E > 1) {
    const Im = first(c, 'moe_intermediate_size', 'expert_intermediate_size') ?? I;
    const expert = mlpMats * H * Im;
    const sharedN = first(c, 'n_shared_experts') ?? 0;
    const sharedI = first(c, 'shared_expert_intermediate_size') ?? sharedN * Im;
    const shared = sharedI ? mlpMats * H * sharedI + (c.shared_expert_intermediate_size ? H : 0) : 0; // + Qwen shared_expert_gate
    const router = H * E + (c.topk_method === 'noaux_tc' ? E : 0);
    const firstDense = first(c, 'first_k_dense_replace') ?? 0;
    const step = first(c, 'decoder_sparse_step', 'moe_layer_freq') ?? 1;
    const mlpOnly = new Set(c.mlp_only_layers || []);
    let moeLayers = 0;
    for (let i = 0; i < L; i++) if (i >= firstDense && !mlpOnly.has(i) && (i + 1) % step === 0) moeLayers++;
    const denseLayers = L - moeLayers;
    const moeLayerTotal = attn + norms + E * expert + shared + router;
    const moeLayerActive = attn + norms + k * expert + shared + router;
    const denseLayer = attn + norms + dense;
    const outer = V * H + (tied ? 0 : V * H) + H;
    total = moeLayers * moeLayerTotal + denseLayers * denseLayer + outer;
    active = moeLayers * moeLayerActive + denseLayers * denseLayer + outer;
    perLayer = moeLayerTotal;
    assumptions.push(`Mixture of experts: ${E} experts per MoE layer, ${k} used per token${sharedI ? ', plus shared expert(s)' : ''}; ${moeLayers} MoE layer${moeLayers === 1 ? '' : 's'}${denseLayers ? ` and ${denseLayers} dense` : ''}.`);
    if (mt.startsWith('deepseek_v3')) assumptions.push('DeepSeek-V3: the extra multi-token-prediction layer in the checkpoint is not counted.');
    return { ...common, total, active, embedding, perLayer, family: 'moe', experts: E, expertsUsed: k, assumptions };
  }

  perLayer = attn + dense + norms;
  total = L * perLayer + V * H + (tied ? 0 : V * H) + H;
  return { ...common, total, active: total, embedding, perLayer, family: 'llama', assumptions };
}
