"""HELIX benchmark harness.

CPU-first performance measurement:
  * forward / backward throughput (tokens / second)
  * generation latency (ms / token, tokens / second)
  * peak RSS memory
  * scaling across context lengths
  * hardware detection (CPU count, RAM, SIMD hints, torch threads)
  * conservative fallbacks / recommended next experiment

Everything runs on local CPU with numpy + torch only. Results are written to a
JSON report so every number is reproducible and auditable. No accelerator is
assumed; if one is present it is reported but not required.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, load_config, save_config
from helix.data import get_batch
from helix.model import HelixTransformer


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

@dataclass
class HardwareSpec:
    cpu_count: int
    cpu_count_logical: int
    ram_gb: float
    torch_threads: int
    torch_version: str
    simd_hints: List[str] = field(default_factory=list)
    has_cuda: bool = False
    device: str = "cpu"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareSpec:
    """Detect local hardware without assuming any accelerator exists."""
    import torch as _torch

    cpu_count = os.cpu_count() or 1
    try:
        import os as _os
        # logical vs physical on Linux
        physical = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else cpu_count
    except Exception:
        physical = cpu_count

    # RAM from /proc/meminfo
    ram_gb = 0.0
    try:
        with open("/proc/meminfo", "r") as f:
            mem = f.read()
        m = re.search(r"MemTotal:\s*(\d+)\s*kB", mem)
        if m:
            ram_gb = int(m.group(1)) / (1024.0 * 1024.0)
    except Exception:
        pass

    simd: List[str] = []
    try:
        with open("/proc/cpuinfo", "r", errors="ignore") as f:
            cpuinfo = f.read()
        for flag in ("avx512f", "avx2", "avx", "sse4_2", "neon", "asimd"):
            if flag in cpuinfo.lower():
                simd.append(flag.upper())
    except Exception:
        pass

    has_cuda = _torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    torch.set_num_threads(max(1, physical))

    notes = []
    if ram_gb < 2:
        notes.append("low-memory machine (<2 GB RAM): prefer small models and small batch")
    if ram_gb < 4:
        notes.append("constrained RAM: keep context short and use gradient accumulation")
    if "AVX2" not in simd and "AVX-512" not in simd and "NEON" not in simd:
        notes.append("no AVX2/NEON detected: expect slower matrix ops; pure fallback path")

    return HardwareSpec(
        cpu_count=physical,
        cpu_count_logical=cpu_count,
        ram_gb=round(ram_gb, 2),
        torch_threads=_torch.get_num_threads(),
        torch_version=_torch.__version__,
        simd_hints=simd,
        has_cuda=has_cuda,
        device=device,
        notes="; ".join(notes),
    )


def peak_rss_mb() -> float:
    """Current process peak resident set size in MB (Linux)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    category: str
    value: float
    unit: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchReport:
    hardware: HardwareSpec
    config: dict
    results: List[BenchResult] = field(default_factory=list)

    def add(self, result: BenchResult) -> None:
        self.results.append(result)

    def summary(self) -> dict:
        return {
            "hardware": self.hardware.to_dict(),
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        import os as _os
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def print_table(self) -> None:
        print(f"\n{'task':<34} {'value':>10} {'unit':<8}  context")
        print("-" * 70)
        for r in self.results:
            ctx = ", ".join(f"{k}={v}" for k, v in r.context.items())
            print(f"{r.name:<34} {r.value:>10.3f} {r.unit:<8}  {ctx}")


def _make_model(config: HelixConfig, device: str) -> HelixTransformer:
    torch.manual_seed(config.train.seed)
    model = HelixTransformer(config.model).to(device)
    return model


def bench_forward_backward(
    report: BenchReport,
    model: HelixTransformer,
    train_ids,
    val_ids,
    train_cfg,
    device: str,
    context_lengths: Optional[List[int]] = None,
    batch_size: int = 4,
    iters: int = 20,
    warmup: int = 5,
) -> None:
    """Measure tokens/sec for forward+backward at various context lengths."""
    if context_lengths is None:
        context_lengths = [64, 128, 256]

    model.train()
    opt = model.configure_optimizers(train_cfg)

    for block in context_lengths:
        if block > train_cfg.block_size:
            continue
        # warmup
        for _ in range(warmup):
            X, Y = get_batch("train", train_ids, val_ids, batch_size, block, device)
            _, loss = model(X, Y)
            loss.backward()
            opt.zero_grad(set_to_none=True)

        torch.cpu.synchronize() if hasattr(torch, "cpu") else None
        t0 = time.perf_counter()
        total_tokens = 0
        for _ in range(iters):
            X, Y = get_batch("train", train_ids, val_ids, batch_size, block, device)
            _, loss = model(X, Y)
            loss.backward()
            opt.zero_grad(set_to_none=True)
            total_tokens += X.numel()
        dt = time.perf_counter() - t0
        toks_per_sec = total_tokens / max(dt, 1e-9)
        report.add(BenchResult(
            name="forward_backward_toks_per_sec",
            category="throughput",
            value=round(toks_per_sec, 2),
            unit="tokens/s",
            context={"block": block, "batch": batch_size, "iters": iters, "device": device},
        ))
        report.add(BenchResult(
            name="ms_per_iter",
            category="latency",
            value=round(dt * 1000.0 / iters, 3),
            unit="ms",
            context={"block": block, "batch": batch_size, "device": device},
        ))
        report.add(BenchResult(
            name="peak_rss_mb",
            category="memory",
            value=peak_rss_mb(),
            unit="MB",
            context={"block": block, "batch": batch_size, "device": device},
        ))


def bench_generation(
    report: BenchReport,
    model: HelixTransformer,
    tok,
    device: str,
    max_new_tokens: int = 32,
    iters: int = 5,
) -> None:
    """Measure generation latency (full-context recompute, no KV cache yet)."""
    model.eval()
    prompts = ["The helix is", "def fibonacci(n):", "The capital of France is"]
    latencies: List[float] = []
    toks_generated = 0
    with torch.no_grad():
        for _ in range(iters):
            for p in prompts:
                ids = torch.tensor([tok.encode(p)], device=device)
                t0 = time.perf_counter()
                out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0)
                dt = time.perf_counter() - t0
                latencies.append(dt)
                toks_generated += out.size(1) - ids.size(1)

    avg_ms = (sum(latencies) / max(len(latencies), 1)) * 1000.0
    gen_toks_per_sec = toks_generated / max(sum(latencies), 1e-9)
    report.add(BenchResult(
        name="generation_ms_per_token",
        category="latency",
        value=round(avg_ms / max(max_new_tokens, 1), 3),
        unit="ms/token",
        context={"max_new_tokens": max_new_tokens, "device": device},
    ))
    report.add(BenchResult(
        name="generation_toks_per_sec",
        category="throughput",
        value=round(gen_toks_per_sec, 2),
        unit="tokens/s",
        context={"max_new_tokens": max_new_tokens, "device": device},
    ))


def bench_context_scaling(
    report: BenchReport,
    model: HelixTransformer,
    train_ids,
    val_ids,
    train_cfg,
    device: str,
    batch_size: int = 2,
    iters: int = 10,
) -> None:
    """Measure how forward/backward cost scales with context length."""
    model.train()
    opt = model.configure_optimizers(train_cfg)
    lengths = [64, 128, 256, 512]
    for block in lengths:
        if block > train_cfg.block_size:
            continue
        if len(train_ids) < block + 2:
            continue
        for _ in range(3):
            X, Y = get_batch("train", train_ids, val_ids, batch_size, block, device)
            _, loss = model(X, Y)
            loss.backward()
            opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        for _ in range(iters):
            X, Y = get_batch("train", train_ids, val_ids, batch_size, block, device)
            _, loss = model(X, Y)
            loss.backward()
            opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        report.add(BenchResult(
            name="context_scaling_ms_per_iter",
            category="scaling",
            value=round(dt * 1000.0 / iters, 3),
            unit="ms",
            context={"block": block, "batch": batch_size, "device": device},
        ))


def bench_generate_modes(
    report: BenchReport,
    model: HelixTransformer,
    tok,
    device: str,
    prompt: str = "The helix is a",
    max_new_tokens: int = 32,
    iters: int = 3,
) -> None:
    """Compare full-recompute vs KV-cached vs int8-quantized generation.

    Every mode is a hypothesis: measure ms/token and correctness, then adopt
    only what is measurably faster on this hardware. Outputs must match across
    modes (same sampling); speed is the only differentiator.
    """
    from helix.infer import (
        generate, generate_with_cache, generate_quantized,
    )
    from helix.quantize import quantize_model

    model.eval()

    def timed(fn, *a, **k):
        # warmup
        fn(*a, **k)
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*a, **k)
        dt = time.perf_counter() - t0
        return out, dt

    # 1. full-recompute (float32)
    out_full, dt_full = timed(generate, model, tok, prompt, max_new_tokens=max_new_tokens,
                              temperature=1.0, top_k=1, device=device)
    report.add(BenchResult(
        name="generate_full_ms_per_token", category="latency",
        value=round(dt_full * 1000.0 / iters / max_new_tokens, 3), unit="ms/token",
        context={"mode": "full_recompute", "max_new_tokens": max_new_tokens, "device": device},
    ))

    # 2. KV-cached (float32)
    out_cache, dt_cache = timed(generate_with_cache, model, tok, prompt,
                                max_new_tokens=max_new_tokens, temperature=1.0,
                                top_k=1, device=device)
    report.add(BenchResult(
        name="generate_kvcache_ms_per_token", category="latency",
        value=round(dt_cache * 1000.0 / iters / max_new_tokens, 3), unit="ms/token",
        context={"mode": "kv_cache", "max_new_tokens": max_new_tokens, "device": device},
    ))
    report.add(BenchResult(
        name="generate_kvcache_output_matches_full", category="correctness",
        value=1.0 if out_cache == out_full else 0.0, unit="bool",
        context={"mode": "kv_cache_vs_full"},
    ))
    if dt_cache > 0:
        report.add(BenchResult(
            name="generate_kvcache_speedup_vs_full", category="throughput",
            value=round(dt_full / max(dt_cache, 1e-9), 2), unit="x",
            context={"mode": "kv_cache_vs_full"},
        ))

    # 3. int8-quantized (cached)
    qmodel = quantize_model(model).eval()
    out_q, dt_q = timed(generate_quantized, qmodel, tok, prompt,
                        max_new_tokens=max_new_tokens, temperature=1.0,
                        top_k=1, device=device, use_cache=True)
    report.add(BenchResult(
        name="generate_int8_ms_per_token", category="latency",
        value=round(dt_q * 1000.0 / iters / max_new_tokens, 3), unit="ms/token",
        context={"mode": "int8_cached", "max_new_tokens": max_new_tokens, "device": device},
    ))
    report.add(BenchResult(
        name="generate_int8_output_matches_full", category="correctness",
        value=1.0 if out_q == out_full else 0.0, unit="bool",
        context={"mode": "int8_vs_full"},
    ))
    if dt_q > 0:
        report.add(BenchResult(
            name="generate_int8_speedup_vs_full", category="throughput",
            value=round(dt_full / max(dt_q, 1e-9), 2), unit="x",
            context={"mode": "int8_vs_full"},
        ))


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend(hardware: HardwareSpec, report: BenchReport) -> List[str]:
    """Conservative, hardware-aware next-experiment recommendations."""
    recs: List[str] = []
    if hardware.ram_gb < 2:
        recs.append("RAM < 2 GB: start with n_layer=1-2, n_embd<=64, block<=64, batch<=4")
    elif hardware.ram_gb < 4:
        recs.append("RAM < 4 GB: keep block<=128, batch<=8, use gradient accumulation")
    else:
        recs.append("RAM >= 4 GB: can try n_layer=4-6, n_embd=128-256, block=128-256")

    if hardware.cpu_count <= 2:
        recs.append("<= 2 CPUs: expect slow training; validate on tiny models before scaling")
    else:
        recs.append(f"{hardware.cpu_count} CPUs detected: can parallelize data loading and eval")

    if not hardware.simd_hints:
        recs.append("no AVX2/NEON detected: matrix ops will be slow; consider smaller models or nightly builds with optimized BLAS")

    # Find best forward_backward throughput
    fwd = [r for r in report.results if r.name == "forward_backward_toks_per_sec"]
    if fwd:
        best = max(fwd, key=lambda r: r.value)
        recs.append(f"best measured throughput {best.value} tokens/s at block={best.context.get('block')} batch={best.context.get('batch')}")
    return recs


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_bench(
    config_path: str,
    out_path: str = "out/bench-report.json",
    block_sizes: Optional[List[int]] = None,
) -> BenchReport:
    cfg = load_config(config_path)
    hardware = detect_hardware()
    print(f"[bench] hardware: {hardware.cpu_count} CPU(s), {hardware.ram_gb} GB RAM, "
          f"torch {hardware.torch_version}, device={hardware.device}")
    print(f"[bench] SIMD hints: {hardware.simd_hints or 'none detected'}")

    # Build or load data
    from helix.data import build_from_corpus, load_ids
    import os as _os

    train_ids = val_ids = None
    if _os.path.exists(cfg.data.train_file) and _os.path.exists(cfg.data.val_file):
        train_ids = load_ids(cfg.data.train_file)
        val_ids = load_ids(cfg.data.val_file)
    else:
        print(f"[bench] building data from {cfg.data.corpus_file}")
        _tok, train_ids, val_ids = build_from_corpus(
            cfg.data.corpus_file,
            cfg.data.vocab_file,
            cfg.data.merges_file,
            cfg.model.vocab_size,
            cfg.data.train_frac,
            cfg.data.train_file,
            cfg.data.val_file,
        )
        cfg.model.vocab_size = len(_tok)

    # Align vocab to learned tokenizer if present
    import json as _json
    if _os.path.exists(cfg.data.vocab_file):
        with open(cfg.data.vocab_file, encoding="utf-8") as f:
            cfg.model.vocab_size = len(_json.load(f)["vocab"])

    device = hardware.device
    model = _make_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] model parameters: {n_params:,}")

    report = BenchReport(hardware=hardware, config={
        "model": asdict(cfg.model),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
    })

    blocks = block_sizes or [64, 128, 256]
    blocks = [b for b in blocks if b <= cfg.model.block_size]

    bench_forward_backward(report, model, train_ids, val_ids, cfg.train,
                           device, context_lengths=blocks,
                           batch_size=max(1, cfg.train.batch_size // 2),
                           iters=15, warmup=3)

    # Need a tokenizer for generation bench
    from helix.tokenizer import ByteLevelBPETokenizer
    tok = None
    if _os.path.exists(cfg.data.vocab_file) and _os.path.exists(cfg.data.merges_file):
        tok = ByteLevelBPETokenizer.load(cfg.data.vocab_file, cfg.data.merges_file)
    if tok is not None:
        bench_generation(report, model, tok, device)
        bench_generate_modes(report, model, tok, device)

    bench_context_scaling(report, model, train_ids, val_ids, cfg.train,
                          device, batch_size=max(1, cfg.train.batch_size // 2),
                          iters=10)

    for r in recommend(hardware, report):
        print(f"[bench] recommendation: {r}")

    report.save(out_path)
    report.print_table()
    print(f"\n[bench] report saved to {out_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HELIX CPU benchmark")
    parser.add_argument("--config", default="configs/helix-tiny.json")
    parser.add_argument("--out-path", default="out/bench-report.json")
    parser.add_argument("--block-sizes", default="64,128,256",
                        help="comma-separated context lengths to benchmark")
    args = parser.parse_args()
    sizes = [int(s) for s in args.block_sizes.split(",") if s.strip()]
    run_bench(args.config, args.out_path, sizes)
