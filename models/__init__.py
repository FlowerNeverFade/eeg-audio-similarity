#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models module for Audio LLM inference and embedding extraction.

This module provides functions for:
- Loading various Audio LLM models
- Extracting layer-wise hidden states from audio
- Saving and loading embeddings for downstream analysis

Supported Models (Verified from Extracted Embeddings):
------------------------------------------------------
| Model Key              | Model Name                    | Layers | Hidden Dim |
|------------------------|-------------------------------|--------|------------|
| audio-flamingo-3       | Audio-Flamingo-3              | 28     | 3584       |
| baichuan-audio-base    | Baichuan-Audio-Base           | 28     | 3584       |
| baichuan-audio-instruct| Baichuan-Audio-Instruct       | 28     | 3584       |
| glm-4-voice-9b         | GLM-4-Voice-9B                | 40     | 4096       |
| granite-speech-3.3-8b  | Granite-Speech-3.3-8B         | 40     | 4096       |
| llama-3.1-8b-omni      | Llama-3.1-8B-Omni             | 32     | 4096       |
| minicpm-o-2_6          | MiniCPM-o-2_6                 | 28     | 3584       |
| qwen2-audio            | Qwen2-Audio-7B                | 32     | 4096       |
| qwen2-audio-instruct   | Qwen2-Audio-7B-Instruct       | 32     | 4096       |
| speechgpt              | SpeechGPT-2.0-preview-7B      | 28     | 3584       |
| ultravox-llama3.1-8b   | Ultravox-v0.5-Llama-3.1-8B    | 32     | 4096       |
| ultravox-llama3.2-1b   | Ultravox-v0.5-Llama-3.2-1B    | 16     | 2048       |

Example:
--------
>>> from paper_code.models import load_audio_model, extract_all_layers
>>> model, processor, config = load_audio_model('qwen2-audio')
>>> embeddings = extract_all_layers(model, processor, audio, sample_rate=16000)
>>> print(f"Extracted {len(embeddings)} layers, dim={config.hidden_dim}")
"""

from .audio_llm_inference import (
    load_audio_model,
    extract_hidden_states,
    extract_all_layers,
    save_embeddings,
    load_embeddings,
    load_embeddings_metadata,
    batch_extract_embeddings,
    get_layer_dims,
)
from .model_registry import (
    ModelConfig,
    AudioEncoderConfig,
    SUPPORTED_MODELS,
    MODEL_ALIASES,
    MODEL_TO_EMBEDDINGS_FOLDER,
    DEFAULT_EMBEDDINGS_BASE_PATH,
    get_model_config,
    get_embeddings_path,
    list_available_models,
    get_model_summary,
    print_model_info,
)

__all__ = [
    # Inference functions
    'load_audio_model',
    'extract_hidden_states',
    'extract_all_layers',
    'save_embeddings',
    'load_embeddings',
    'load_embeddings_metadata',
    'batch_extract_embeddings',
    'get_layer_dims',
    # Model config classes
    'ModelConfig',
    'AudioEncoderConfig',
    # Registry
    'SUPPORTED_MODELS',
    'MODEL_ALIASES',
    'MODEL_TO_EMBEDDINGS_FOLDER',
    'DEFAULT_EMBEDDINGS_BASE_PATH',
    'get_model_config',
    'get_embeddings_path',
    'list_available_models',
    'get_model_summary',
    'print_model_info',
]
