#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features module for EEG and Audio feature extraction.
"""

from .eeg_features import (
    extract_eeg_raw_aligned,
    extract_eeg_multichannel_aligned,
    extract_eeg_rich_features,
)
from .audio_features import (
    reduce_audio_dimensions,
    normalize_audio_embeddings,
    load_audio_embeddings,
    get_layer_count,
)

__all__ = [
    # EEG Features
    'extract_eeg_raw_aligned',
    'extract_eeg_multichannel_aligned',
    'extract_eeg_rich_features',
    # Audio Features
    'reduce_audio_dimensions',
    'normalize_audio_embeddings',
    'load_audio_embeddings',
    'get_layer_count',
]

