#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models module for Audio LLM inference and embedding extraction.

This module provides functions for:
- Loading various Audio LLM models (Qwen2-Audio, SALMONN, WavLM, etc.)
- Extracting layer-wise hidden states from audio
- Saving embeddings for downstream analysis
"""

from .audio_llm_inference import (
    load_audio_model,
    extract_hidden_states,
    extract_all_layers,
    save_embeddings,
    load_embeddings,
)
from .model_registry import (
    SUPPORTED_MODELS,
    get_model_config,
    list_available_models,
)

__all__ = [
    # Inference
    'load_audio_model',
    'extract_hidden_states',
    'extract_all_layers',
    'save_embeddings',
    'load_embeddings',
    # Registry
    'SUPPORTED_MODELS',
    'get_model_config',
    'list_available_models',
]

