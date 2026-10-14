# Model Quantization Planner

A browser-only tool that answers: **"Which compressed (quantized) version of this open-weight LLM should I run, how big is
it, does it fit my GPU, how much precision do I lose — and why?"**

> Don't guess — measure your model's quantization trade-offs in seconds, privately in your browser.

- **Load a model** from Hugging Face (ID or link), local files (`.safetensors`, `.gguf`, `config.json`, index), a direct
  link, a local Ollama / LM Studio / vLLM server, a pasted config.json, or just "70B".
- **Size & GPU fit** for FP32/FP16/BF16, FP8, INT8, INT4 g128, NF4, GPTQ, AWQ and GGUF Q8_0…Q2_K, with adjustable usable memory.
- **Ready-made quantized versions** already on Hugging Face, with formats and file sizes.
- **Precision-loss explorer** on generated or *real* weights: SQNR, MSE, cosine, clipped / zeroed share, levels used,
  six charts (histogram + levels, staircase, error histogram, all formats compared, clipping trade-off, 32×32 heatmap),
  automatic plain-English insights and a bit inspector.
- **Layer sensitivity scan** of every weight matrix, plus a mixed-precision plan (INT8 for the most sensitive tensors).
- **Export & share**: Markdown / PNG report, chart PNGs, CSV, shareable link, "Copy summary for an AI assistant".

## Privacy and safety

100% static: no backend, no uploads, no build step, no dependencies. Files you open are read with `File.slice()` and never
leave the device. A Hugging Face token lives only in a JavaScript variable in the tab (never stored, logged, put in a URL
or exported). Only headers and small slices of weight files are read, with HTTP Range requests: max 1,048,576 values per
tensor, max 4 parallel requests, confirmation before any scan above 100 MB, everything cancellable. A server that ignores
Range is stopped immediately. All maths runs in a Web Worker.

## Run locally

Requires Node.js 18 or newer (only for the dev server and tests).

```bash
node server.js
```

Open <http://localhost:4180>. To deploy, copy the `public/` folder to any static host. Run the tests with:

```bash
node --test
```

## Files

| Path | Purpose |
| --- | --- |
| `server.js` | Tiny static dev server (supports Range requests), bound to 127.0.0.1 |
| `public/index.html` | Page layout and the docs sections (what it's for, how to use it, the maths, glossary) |
| `public/styles.css` | Styles (same design system as the AI System Architecture Designer) |
| `public/app.js` | UI, import flow, tooltips (`data-tip` + `TIPS`), insights, exports, URL state |
| `public/import.js` | All import sources → one model profile; Range / `File.slice` reads with safety limits |
| `public/parsers.js` | safetensors, index.json, GGUF headers, FP16/BF16 decoding, packed GPTQ/AWQ detection |
| `public/params.js` | Parameter count from config.json (dense, GQA, MoE, DeepSeek MLA, GPT-2 / NeoX families) |
| `public/quant.js` | Pure maths: INT sym/asym, minifloats (FP16/BF16/FP8/FP4), NF4, clipping, metrics, sizes |
| `public/worker.js` | Runs the maths and the layer scan off the main thread |
| `public/charts.js` | Canvas charts with hover readouts and PNG export |
| `tests/` | `node --test` unit tests; `fixtures/` holds PyTorch reference vectors and real config.json files |

## Accuracy notes

- Float rounding matches PyTorch bit-for-bit (FP16, BF16, FP8 E4M3 / E5M2; reference vectors in `tests/fixtures`).
  FP8 values beyond the max saturate here (PyTorch gives NaN / inf).
- config.json parameter counts match Hugging Face's exact counts for Llama-3.1, Qwen2.5/3, Mistral, Gemma-2, Phi-3,
  OLMo-2, Granite, Mixtral, Qwen MoE, Qwen3-MoE and DeepSeek-V2-Lite.
- GGUF sizes use llama.cpp's measured bits-per-weight (Llama-3.1-8B) and are approximate for other models.
- The tool simulates round-to-nearest. Real GPTQ / AWQ use calibration data and usually do somewhat better.
  Weight error is a strong hint, not a measure of answer quality.

## Limitations

- KV cache and activation memory are not included in the GPU-fit check.
- Weights of already-quantized checkpoints (GPTQ, AWQ, bitsandbytes, quantized GGUF) can't be re-analysed; the tool
  points to the base model instead.
- Hugging Face serves weight files from its CDN (`*.hf.co`); networks that block it (e.g. company proxies) still get
  sizes and charts, but real-layer analysis needs "File on my computer" or another network.
- Local servers (Ollama etc.) must allow the page's origin (`OLLAMA_ORIGINS`, LM Studio "Enable CORS", vLLM
  `--allowed-origins`); the page shows the exact command.
