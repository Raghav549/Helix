"""HELIX — an open-source, research-grade, CPU-first AI system.

Trained from random initialization. Core components implemented independently:
tokenizer, model, data pipeline, training engine, evaluation, inference, and CLI.
"""

__version__ = "0.1.0"

from helix.config import ModelConfig, TrainConfig, DataConfig, HelixConfig, load_config, save_config

__all__ = [
    "__version__",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "HelixConfig",
    "load_config",
    "save_config",
]
