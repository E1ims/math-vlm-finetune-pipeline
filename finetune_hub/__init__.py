"""
Finetune Hub - Vision-Language Model (VLM) Finetuning Package

This package exposes the core components required to build, train, and run 
inference on VLM models (specifically PaliGemma). It is designed to be modular, 
allowing users to import main classes directly from the top-level package.

Usage:
    from finetune_hub import ModelConfig, VLMEngine, DataProcessor
"""

# -----------------------------------------------------------------------------
# Expose Core Modules
# -----------------------------------------------------------------------------
# We import these classes here so the user can type:
#    `from finetune_hub import ModelConfig`

from .config import ModelConfig
from .engine import VLMEngine
from .data import DataProcessor
from .trainer import TrainerWrapper
from .inference import InferenceEngine

# -----------------------------------------------------------------------------
# Public API Definition
# -----------------------------------------------------------------------------
# __all__ defines the list of names that will be exported when a client runs:
#    `from finetune_hub import *`
# This keeps the namespace clean and hides internal helper functions.
__all__ = [
    "ModelConfig",
    "VLMEngine",
    "DataProcessor",
    "TrainerWrapper",
    "InferenceEngine"
]