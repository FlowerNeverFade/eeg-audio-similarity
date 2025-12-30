#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing module for EEG and audio data.

This module provides functions for:
- EEG preprocessing (filtering, artifact removal, re-referencing)
- Sentence segmentation based on alignment tables
- Time alignment between EEG and audio embeddings
"""

from .eeg_preprocessing import (
    preprocess_eeg,
    bandpass_filter,
    notch_filter,
    remove_baseline,
    rereference_average,
    detect_bad_channels,
    interpolate_bad_channels,
)
from .sentence_segmentation import (
    load_alignment_table,
    segment_eeg_by_sentences,
    segment_audio_by_sentences,
    get_sentence_boundaries,
    generate_alignment_from_bids,
    load_bids_eeg,
    create_alignment_from_vad,
)
from .time_alignment import (
    align_eeg_to_audio,
    resample_to_target_length,
    interpolate_features,
)

__all__ = [
    # EEG Preprocessing
    'preprocess_eeg',
    'bandpass_filter',
    'notch_filter',
    'remove_baseline',
    'rereference_average',
    'detect_bad_channels',
    'interpolate_bad_channels',
    # Sentence Segmentation
    'load_alignment_table',
    'segment_eeg_by_sentences',
    'segment_audio_by_sentences',
    'get_sentence_boundaries',
    'generate_alignment_from_bids',
    'load_bids_eeg',
    'create_alignment_from_vad',
    # Time Alignment
    'align_eeg_to_audio',
    'resample_to_target_length',
    'interpolate_features',
]

