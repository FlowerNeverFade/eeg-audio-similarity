#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-Audio Similarity Analysis Toolkit

A comprehensive toolkit for analyzing the similarity between EEG neural 
responses and audio language model representations.

Modules:
    models: Audio LLM loading and inference
    metrics: Similarity metrics (RSA, CKA, distance correlation, etc.)
    features: EEG and audio feature extraction
    visualization: RDM and topography visualization
    analysis: Prosody and TNC analysis
    utils: GPU and data loading utilities

Supported Audio Models (12 models):
    - Audio-Flamingo-3, Baichuan-Audio (Base/Instruct)
    - GLM-4-Voice-9B, Granite-Speech-3.3-8B
    - Llama-3.1-8B-Omni, MiniCPM-o-2_6
    - Qwen2-Audio-7B (Base/Instruct)
    - SpeechGPT-2.0-preview-7B
    - Ultravox-v0.5 (Llama-3.1-8B / Llama-3.2-1B)
"""

__version__ = "1.1.0"
__author__ = "Anonymous"

from . import models
from . import metrics
from . import features
from . import visualization
from . import analysis
from . import utils

# Convenience imports
from .metrics import (
    compute_rdm_vec,
    compute_rdm_full,
    rsa_between_rdms,
    compute_cka,
    compute_distance_correlation,
    compute_hsic_rbf,
    compute_rv_coefficient,
    permutation_pvalue,
)

from .features import (
    extract_eeg_multichannel_aligned,
    extract_eeg_rich_features,
    reduce_audio_dimensions,
)

from .visualization import (
    plot_rdm_pair,
    plot_topography_simple,
    plot_layer_profile,
)

from .utils import (
    get_device,
    setup_gpu,
    load_eeg_data,
    load_audio_embeddings,
)

from .models import (
    load_audio_model,
    extract_all_layers,
    save_embeddings,
    load_embeddings,
    get_model_config,
    list_available_models,
    SUPPORTED_MODELS,
)

__all__ = [
    # Submodules
    'models',
    'metrics',
    'features',
    'visualization',
    'analysis',
    'utils',
    # Core functions
    'compute_rdm_vec',
    'compute_rdm_full',
    'rsa_between_rdms',
    'compute_cka',
    'compute_distance_correlation',
    'compute_hsic_rbf',
    'compute_rv_coefficient',
    'permutation_pvalue',
    'extract_eeg_multichannel_aligned',
    'extract_eeg_rich_features',
    'reduce_audio_dimensions',
    'plot_rdm_pair',
    'plot_topography_simple',
    'plot_layer_profile',
    'get_device',
    'setup_gpu',
    'load_eeg_data',
    'load_audio_embeddings',
    # Models
    'load_audio_model',
    'extract_all_layers',
    'save_embeddings',
    'load_embeddings',
    'get_model_config',
    'list_available_models',
    'SUPPORTED_MODELS',
]

