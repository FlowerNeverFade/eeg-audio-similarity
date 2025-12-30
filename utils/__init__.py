#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities module for EEG-Audio similarity analysis.
"""

from .gpu_utils import (
    get_device,
    select_best_gpu,
    get_gpu_memory_info,
    clear_gpu_cache,
    setup_gpu,
)
from .data_loading import (
    load_eeg_data,
    load_audio_embeddings,
    find_eeg_files,
    find_audio_embedding_files,
    get_sentence_ids,
    get_layer_count,
    load_alice_eeg,
    load_alice_alignment,
    load_alice_dataset_info,
    segment_alice_by_sentence,
)

__all__ = [
    # GPU Utils
    'get_device',
    'select_best_gpu',
    'get_gpu_memory_info',
    'clear_gpu_cache',
    'setup_gpu',
    # Data Loading
    'load_eeg_data',
    'load_audio_embeddings',
    'find_eeg_files',
    'find_audio_embedding_files',
    'get_sentence_ids',
    'get_layer_count',
    # Alice Dataset
    'load_alice_eeg',
    'load_alice_alignment',
    'load_alice_dataset_info',
    'segment_alice_by_sentence',
]

