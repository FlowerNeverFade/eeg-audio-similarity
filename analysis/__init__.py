#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis module for EEG-Audio similarity analysis.
"""

from .prosody_analysis import (
    extract_prosody_features,
    estimate_f0_autocorr,
    compute_prosody_rdm,
    compute_tnc,
)
from .opensmile_analysis import (
    build_smile,
    extract_opensmile_features,
    add_sentiment_scores,
    find_audio_files,
    batch_extract_features,
)
from .time_window_analysis import (
    TimeWindow,
    ERP_WINDOWS,
    create_sliding_windows,
    extract_window_features,
    compute_window_rsa,
    compute_electrode_window_rsa,
    sliding_window_rsa_analysis,
    n400_analysis,
    erp_component_analysis,
    plot_time_window_topography,
    plot_sliding_window_topography_series,
    plot_rsa_time_course,
)

__all__ = [
    # Prosody Analysis
    'extract_prosody_features',
    'estimate_f0_autocorr',
    'compute_prosody_rdm',
    'compute_tnc',
    # OpenSMILE Analysis
    'build_smile',
    'extract_opensmile_features',
    'add_sentiment_scores',
    'find_audio_files',
    'batch_extract_features',
    # Time Window Analysis
    'TimeWindow',
    'ERP_WINDOWS',
    'create_sliding_windows',
    'extract_window_features',
    'compute_window_rsa',
    'compute_electrode_window_rsa',
    'sliding_window_rsa_analysis',
    'n400_analysis',
    'erp_component_analysis',
    'plot_time_window_topography',
    'plot_sliding_window_topography_series',
    'plot_rsa_time_course',
]
