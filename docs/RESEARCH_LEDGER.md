# HELIX Research Ledger

Each adopted technique records: hypothesis, source/reference, implementation
status, baseline, hardware, configuration, measured results, regressions,
limitations, and decision.

## v0.1 — Minimal CPU baseline (random init)

| Entry | Detail |
|-------|--------|
| Hypothesis | A tiny decoder-only transformer can be trained end-to-end on CPU from random init, with checkpoint/resume and eval, using only numpy+torch. |
| Reference | nanoGPT (karpathy/nanoGPT) — inspected as a reference; implementation written independently. |
| Techniques | RoPE, RMSNorm (pre-norm), SwiGLU, GQA, weight tying, AdamW + cosine LR + warmup, gradient accumulation, checkpoint save/resume. |
| Status | Implemented (helix/model.py, helix/train.py, helix/data.py, helix/tokenizer.py). |
| Hardware | 2 vCPU, ~3.8 GB RAM, CPU only. |
| Baseline | random initialization; seed corpus in data/seed_corpus.txt. |
| Config | configs/helix-tiny.json (2 layers, 4 heads, 128 embd, block 128, ~300 iters). |
| Results | Pending first training run (see out/tiny-run/log.jsonl). |
| Limitations | Tiny model/corpus — quality is a smoke test, not a capability claim. Byte-level BPE without regex pre-tokenization is slower to encode than a production tokenizer. No KV cache during generation (full recompute). |
| Decision | Ship v0.1 baseline, then iterate: eval harness, ingestion pipeline, architecture ablations. |

## v0.2 — Evaluation framework + corpus discovery/ingestion

| Entry | Detail |
|-------|--------|
| Hypothesis | A CPU-runnable, contamination-resistant evaluation suite plus a rights-aware corpus manifest can be built with executable code and tests, without network access. |
| Reference | nanoGPT eval patterns; OLMo/DataTrove-style manifests inspected as references; implementation written independently. |
| Techniques | Original eval tasks generated at eval time (arithmetic, code completion, cloze, needle-in-haystack, calibration); per-document provenance manifest (source id, URL, format, language, license, rights, content hash, retrieval timestamp, quality score, quarantine flag). |
| Status | Implemented (helix/eval.py, helix/corpus.py, tests/test_eval.py, tests/test_corpus.py); eval wired into `helix eval` CLI. |
| Hardware | 2 vCPU, ~3.8 GB RAM, CPU only. |
| Baseline | v0.1 tiny model on out/tiny-run/ckpt-best.pt. |
| Results | out/eval-report.json: 2/5 tasks passed on the tiny baseline (long-context recall + calibration pass; arithmetic/code/cloze fail — expected, model memorized seed corpus but did not generalize). |
| Limitations | Eval tasks are heuristics, not validated benchmarks. Corpus scan is local-only; network ingestion (Internet Archive, Wikipedia, Gutenberg) recorded as candidate sources but not fetched by default. No semantic dedup, no OCR, no malware scanning yet. |
| Decision | Ship v0.2 eval+corpus; next: benchmark harness (throughput/latency/RAM), then architecture ablations and scaling. |

## v0.3 — CPU benchmark harness (hardware detection + throughput/latency/RAM/scaling)

| Entry | Detail |
|-------|--------|
| Hypothesis | A CPU-first benchmark harness can measure forward/backward throughput, generation latency, peak RAM, and context-length scaling, with automatic hardware detection and conservative recommendations, using only numpy+torch. |
| Reference | Inspired by nanoGPT benchmarking patterns and MLPerf-style throughput/latency metrics; implementation written independently. No external benchmarking framework adopted. |
| Techniques | perf_counter wall-clock timing, /proc/self/status VmHWM for peak RSS, /proc/cpuinfo SIMD detection (AVX2/AVX-512/NEON), /proc/meminfo RAM, torch thread auto-detection, forward+backward and generation benchmarks across context lengths. |
| Status | Implemented (helix/bench.py, tests/test_bench.py); wired into `helix bench` CLI. |
| Hardware | 2 vCPU, 3.85 GB RAM, CPU only; AVX512F/AVX2/AVX/SSE4_2 detected. |
| Baseline | v0.1 tiny model (594K params, block 128). |
| Results | out/bench-report.json: forward+backward 1072.7 tok/s @ block64/batch4; 271.6 tok/s @ block128/batch4; peak RSS ~687-715 MB; generation 27.7 ms/token (~36 tok/s); context scaling ~10.5 ms (block64) to ~19.4 ms (block128) per iter. |
| Limitations | Generation uses full-context recompute (no KV cache yet). Microbenchmarks, not end-to-end training throughput. RAM figure is process RSS, not total system. |
| Decision | Ship v0.3 bench; KV-cache generation and quantized inference implemented and benchmarked (v0.4). Next: architecture ablations. |

## v0.4 — KV-cached generation + quantized CPU inference (measured, not assumed)

| Entry | Detail |
|-------|--------|
| Hypothesis | KV caching and int8 quantization can speed up CPU generation. Both are hypotheses to benchmark, not assumed wins. |
| Reference | nanoGPT/nanochat KV-cache patterns; PyTorch dynamic quantization inspected as references; KVCache and QuantizedLinear implemented independently. |
| Techniques | KVCache (per-layer rotated K/V buffers, GQA-aware, causal mask over cached keys), HelixTransformer.incremental_forward, generate_with_cache, dynamic per-tensor int8 QuantizedLinear, quantize_model. Generation modes compared: full-recompute vs KV-cached vs int8-cached. |
| Status | Implemented (helix/kvcache.py, helix/quantize.py, helix/model.py incremental_forward, helix/infer.py generate_with_cache/generate_quantized, helix/bench.py bench_generate_modes); tests/test_kvcache.py, tests/test_quantize.py; `helix generate --kv-cache --quantize` CLI flags. |
| Hardware | 2 vCPU, 3.85 GB RAM, AVX512F/AVX2, CPU only. |
| Results | out/bench-report.json (tiny 594K-param model, short prompt, 32 new tokens): full-recompute 1.046 ms/tok; KV-cached 1.128 ms/tok (0.93x, output matches); int8-cached 1.066 ms/tok (0.98x, output matches). Correctness = 1.0 across modes. Standalone run showed KV cache 1.65x on a slightly different measurement — variance due to tiny model/short prompt where Python overhead dominates. |
| Limitations | KV cache benefit scales with prompt length and model size; on this tiny model full recompute is competitive. int8 here dequantizes to float32 for matmul (no int8 BLAS), so it's memory-reduction not compute-speedup; true int8 speed needs oneDNN/int8 matmul. |
| Decision | Do NOT adopt KV cache or int8 as defaults on this hardware (no measured speedup). Expose them as optional flags (`--kv-cache --quantize`) for larger/longer-context runs where they are expected to win. Re-bench when scaling. Next: architecture ablations. |

## v0.5 — Architecture ablations: transformer vs linear attention vs recurrent

| Entry | Detail |
|-------|--------|
| Hypothesis | The best block type (transformer vs O(T) linear attention vs O(1)-state recurrent) depends on hardware and scale; it must be chosen by measuring val loss per token and per second at equal compute, not assumed. |
| Reference | Linear attention: Katharopoulos et al. 2020 "Transformers are RNNs"; recurrent/GRU: Cho et al. 2014. Implemented independently; baseline reuses helix.model.Block exactly. |
| Techniques | ModelConfig.block_type (transformer | linear_attn | recurrent | hybrid comma-list cycled); helix/arch_blocks.py (LinearAttention, LinearAttnBlock, RecurrentBlock, build_block); HelixTransformer builds mixed block stacks; helix/ablate.py trains N variants from random init on identical data + identical compute budget; reports val loss, tokens/sec, quality/sec. |
| Status | Implemented (helix/arch_blocks.py, helix/model.py block_type, helix/ablate.py, helix/cli.py `helix ablate`); tests/test_model.py block-type tests; tests/test_ablate.py. |
| Hardware | 2 vCPU, 3.85 GB RAM, AVX512F/AVX2, CPU only. |
| Results | out/ablation-report.json (174K-199K params, block 64, 60 iters, seed corpus): transformer val_loss=6.6639 tok/s=23956; linear_attn val_loss=6.7133 tok/s=17151; recurrent val_loss=6.7776 tok/s=6745. Transformer wins on BOTH quality and speed at this scale. |
| Limitations | Tiny model/corpus; val set only 58 tokens so absolute loss is not meaningful, but the equal-compute relative comparison is valid for this stage. Recurrent block uses a Python loop over T (slow); a vectorized/parallel scan might close the speed gap on longer contexts. |
| Decision | Adopt transformer as the default block type. Do NOT adopt linear attention or recurrent as defaults here — both lose on quality AND speed at this scale. Re-bench ablations when scaling to larger models/longer contexts where O(T^2) attention may dominate. Next: instruction tuning pipeline. |

## v0.6 — Supervised instruction tuning (SFT), strictly separate from pretraining

| Entry | Detail |
|-------|--------|
| Hypothesis | Instruction following is a separable stage: load a base checkpoint trained from random init, continue training on (instruction, response) pairs supervising ONLY response tokens, with its own checkpoint/optimizer/versioning. |
| Reference | Supervised fine-tuning pattern (instruction → response) common in instruction-tuned models; implemented independently with response-loss masking. No pretrained weights used — base checkpoint is HELIX's own random-init model. |
| Techniques | helix/sft.py: build_instruction_pairs (dialogue/Q&A/declarative extraction), tokenize_pairs (instruction-loss masking), train_sft (separate training loop, response-only masked cross-entropy), generate_response (response-only decoding), `helix sft` CLI. Separate out_dir, sft-log.jsonl, sckpt-best/final.pt, sft-metrics.json. |
| Status | Implemented (helix/sft.py, helix/cli.py `helix sft`); tests/test_sft.py; data/sft_pairs.json (10 instruction pairs). |
| Hardware | 2 vCPU, 3.85 GB RAM, CPU only. |
| Results | out/sft-run/: SFT on real baseline (out/tiny-run/ckpt-best.pt, 174K params) over 60 iters: sft_val_loss 2.56 → 1.43 → 1.18. Pipeline functions and loss decreases. Outputs remain low-quality because base model is tiny/tiny-corpus — a capacity/data limitation, honestly documented, not a capability claim. |
| Limitations | Tiny model cannot actually know facts; SFT teaches format/response-prediction on this toy data. 10 pairs is a demonstration, not a real instruction dataset. The general ingestion pipeline (helix/corpus.py) will supply larger instruction sources in later stages. |
| Decision | Ship SFT as a separate, versioned stage. Next: agent system (planning, sandboxed tools, approval gates, audit log). |

## v0.7 — Modular agent system: planning, sandboxed tools, approval gates, audit log

| Entry | Detail |
|-------|--------|
| Hypothesis | A small HELIX model can be wrapped in a rule-driven agent loop that selects and executes sandboxed tools with approval gates, resource limits, timeouts, and an auditable trace separating trusted instructions from untrusted tool output. |
| Reference | Agent/planning patterns (ReAct, tool-use) inspected as references; agent loop, Tool interface, sandboxed PythonTool, CalculatorTool, DocumentTool, ToolRegistry, approval gate implemented independently. |
| Techniques | helix/agent.py: Tool base class, CalculatorTool (AST-whitelisted arithmetic), PythonTool (subprocess sandbox: timeout, memory limit, temp cwd, result variable), DocumentTool (in-memory keyword lookup), ToolRegistry, Agent (plan->call->observe loop, max_steps/max_time stop conditions, approval gate for consequential tools, audit log), AgentRun/AgentStep structured traces. `helix agent` CLI with --goal, --max-steps, --max-time, --no-approval, --audit-log. |
| Status | Implemented (helix/agent.py, helix/cli.py `helix agent`); tests/test_agent.py. |
| Hardware | 2 vCPU, 3.85 GB RAM, CPU only. |
| Results | 8 agent tests pass (calculator, python sandbox incl. timeout, document tool, registry, agent loop with pre-approval, rejection of unapproved consequential tools, unknown-tool handling). End-to-end `helix agent` runs on the real baseline without crashing and writes an audit log; the tiny model cannot yet plan reliably (memorized-text output) — honestly a capacity limitation. |
| Limitations | Agent intelligence is bounded by the tiny base model (no real planning). PythonTool sandbox is best-effort (no network by default, temp cwd); host firewall/policy still governs real isolation. Approval gate is auto-approve in this non-interactive baseline (documented; interactive deployments should prompt the user). |
| Decision | Ship agent as an auditable, rule-driven baseline. Model-driven planning improves only with scaled training — do not overclaim. Next: web/streaming interface OR scaled pretraining run. |

## Adoption policy

Every technique is a hypothesis to benchmark. Adopt only when measured
evaluation justifies it: quality per token, quality per second, memory
efficiency, stability, reproducibility, or inference speed. Distinguish
established methods from experimental/unverified ideas.

## v0.8 — Image generation pipeline (latent diffusion from random init)

| Entry | Detail |
|-------|--------|
| Hypothesis | A small latent diffusion model (VAE + UNet + DDPM) can be trained from random init on CPU and generate 32x32 images; text conditioning via projected embeddings is feasible. |
| Reference | Rombach et al. 2022 (latent diffusion), Nichol & Dhariwal 2021 (cosine schedule), Ho et al. 2020 (DDPM) — inspected as references; VAE/UNet/DDPM implemented independently. |
| Techniques | helix/image_gen.py: VAE (encoder/decoder, reconstruction + KL), ConditionedUNet (residual blocks + time + attention + text conditioning), DiffusionSchedule (cosine/linear), LatentDiffusion (train + reverse sampler), train_image_gen / generate_images. |
| Status | Implemented; tests/test_image_gen.py (8 tests). |
| Hardware | 2 vCPU, 3.85 GB RAM, CPU only. |
| Results | Prototype 32x32 images train from random init; loss decreases over iterations. Quality is prototype-level at 32x32 — honestly documented, not photorealistic. No pretrained image weights used. |
| Limitations | 32x32 resolution is tiny; no photorealistic claims. Text conditioning present but unvalidated at this scale. Next: image understanding + integrated UI. |

## v0.9 — Image understanding (ViT encoder -> LLM embedding space)

| Entry | Detail |
|-------|--------|
| Hypothesis | A small VisionTransformer (patch embedding + transformer blocks) + linear projector can encode images into the LLM's embedding space as prefix tokens, enabling image-conditioned text generation, trained from random init. |
| Reference | ViT (Dosovitskiy et al. 2020), LLaVA-style multimodal projector — inspected as references; implemented independently. |
| Techniques | helix/image_understanding.py: PatchEmbedding, VisionTransformer, ImageProjection, MultimodalHelix (wraps HelixTransformer, prepends image prefix embeddings to text embeddings), train_image_understanding (next-token loss on image-text pairs), generate (image-conditioned). |
| Status | Implemented; tests/test_image_understanding.py (7 tests). |
| Results | Forward pass, generation, and training from random init all work on CPU. |
| Limitations | Tiny ViT (32x32, 64 dim); understanding is prototype-level, not a real vision model. No pretrained vision weights. Next: integrated web UI. |

## v0.10 — Integrated web UI (chat, images, code, research)

| Entry | Detail |
|-------|--------|
| Hypothesis | A single local Flask UI can unify all four first-class capabilities (LLM chat, image generation, image understanding, coding agent, research lab) with lazy model loading and SSE/JSON APIs. |
| Techniques | helix/web.py: Flask app, single-page tabbed UI, /api/chat (streaming LLM), /api/image/generate (latent diffusion), /api/image/analyze (ViT understanding), /api/code/execute (sandboxed PythonTool), /api/code/explain (LLM), /api/research/experiments (tracker), /out/web_images (serve generated images). Lazy ModelCache loads models on first request. |
| Status | Implemented; tests/test_web.py (5 tests). Verified chat + code execute endpoints work. |
| Results | UI starts and serves all tabs; chat and code execution return real results from the tiny model / sandbox. |
| Limitations | Backed by the tiny LLM (174K params) — chat quality is low, honestly a capacity limitation. Image gen/analysis require separate training runs (CLI) before use in UI. Next: coding agent + research lab as first-class modules. |

## v0.11 — Advanced coding & software-engineering agent

| Entry | Detail |
|-------|--------|
| Hypothesis | A coding agent combining repo inspection, AST static analysis, sandboxed execution, patch generation, and test generation can perform repository-level software tasks with real interpreter feedback rather than judging code from text alone. |
| Techniques | helix/coding.py: RepoInspector (read-only repo traversal), StaticAnalyzer (Python AST: symbols, imports, complexity, TODOs, syntax errors), PatchGenerator (unified diff make/apply), TestGenerator (unittest scaffolding), CodingAgent (orchestrator: list/read/analyze/execute/patch/tests), demo_code_fix (analysis + execution feedback). Reuses agent.PythonTool sandbox (timeout, memory limit, temp cwd). |
| Status | Implemented; tests/test_coding.py (7 tests). |
| Hardware | CPU only; execution sandboxed and bounded. |
| Results | Static analysis, patch generation, test generation, and sandboxed execution all work. Real compiler/interpreter feedback is the correctness signal. |
| Limitations | Python-focused static analysis; patch apply is a minimal unified-diff subset. LLM-assisted explain/fix goes through the tiny model (low quality). No pretrained code models. Next: research lab. |

## v0.12 — Research laboratory (math, statistics, data analysis, plotting)

| Entry | Detail |
|-------|--------|
| Hypothesis | A reproducible research-lab environment (experiment tracker + sympy symbolic math + statistics + data analysis + plotting) lets HELIX run computational experiments and distinguish tool-verified results from model-generated claims. |
| Techniques | helix/research.py: ExperimentTracker (record.json + metrics.json per run, legacy scan), SymbolicMath (sympy: differentiate/integrate/solve/simplify/limit, verified_by="sympy"), Statistics (describe/correlation/linear regression), load_csv/summarize_csv, plot_line/plot_histogram (matplotlib Agg), evaluate_claim (model claim vs tool-verified verdict), run_reproducible_experiment (tracked run wrapper). |
| Status | Implemented; tests/test_research.py (12 tests). sympy and matplotlib available as independent verification instruments. |
| Results | Symbolic math, statistics, CSV summarization, plotting, and experiment tracking all work; evaluate_claim confirms/disagrees model claims vs tools. |
| Limitations | sympy/matplotlib are external (OSI-approved) verification instruments, clearly distinguished from HELIX's own model. No theorem prover integration yet. Next: final integration + release. |

## Adoption policy

Every technique is a hypothesis to benchmark. Adopt only when measured
evaluation justifies it: quality per token, quality per second, memory
efficiency, stability, reproducibility, or inference speed. Distinguish
established methods from experimental/unverified ideas.
