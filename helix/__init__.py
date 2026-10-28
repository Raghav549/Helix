"""HELIX — an open-source, research-grade, CPU-first AI system.

Trained from random initialization. Core components implemented independently:
tokenizer, model, data pipeline, training engine, evaluation, inference, and CLI.

Capability areas:
  * LLM chat (helix/model.py, helix/train.py, helix/infer.py, helix/sft.py)
  * Image generation (helix/image_gen.py)
  * Image understanding (helix/image_understanding.py)
  * Coding agent (helix/coding.py, helix/agent.py)
  * Research lab (helix/research.py)
  * Integrated web UI (helix/web.py)
  * Evaluation (helix/eval.py), corpus (helix/corpus.py), benchmarks (helix/bench.py),
    ablations (helix/ablate.py), KV cache (helix/kvcache.py), quantization (helix/quantize.py)
"""

from __future__ import annotations

__version__ = "0.12.0"

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, load_config, save_config

__all__ = [
    "__version__",
    "HelixConfig",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "load_config",
    "save_config",
]
