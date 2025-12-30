#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-Audio Similarity Analysis Toolkit

A comprehensive toolkit for analyzing the similarity between EEG neural 
responses and audio language model representations.

Modules:
    metrics: Similarity metrics (RSA, CKA, distance correlation, etc.)
    features: EEG and audio feature extraction
    visualization: RDM and topography visualization
    analysis: Prosody and TNC analysis
    utils: GPU and data loading utilities
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

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

__all__ = [
    # Submodules
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
]

