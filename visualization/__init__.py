#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for EEG-Audio similarity analysis.
"""

from .rdm_visualization import (
    rdm_vec_to_matrix,
    plot_rdm,
    plot_rdm_pair,
    plot_layer_profile,
    plot_heatmap_1d,
)
from .topography import (
    default_electrode_positions,
    plot_topography_simple,
    plot_topography_contour,
    plot_topography_grid,
)
from .histogram import (
    plot_histogram,
    plot_rsa_distribution,
    plot_lag_histogram,
    plot_scatter_with_colorbar,
)
from .model_comparison import (
    load_model_results,
    compute_layer_averages,
    plot_layerwise_comparison,
    plot_model_grid,
    create_summary_table,
    plot_best_layer_bar,
)
from .group_comparison import (
    load_group_data,
    compute_group_statistics,
    plot_group_boxplot,
    plot_group_violin,
    plot_group_comparison_grid,
    perform_group_ttest,
)

__all__ = [
    # RDM Visualization
    'rdm_vec_to_matrix',
    'plot_rdm',
    'plot_rdm_pair',
    'plot_layer_profile',
    'plot_heatmap_1d',
    # Topography
    'default_electrode_positions',
    'plot_topography_simple',
    'plot_topography_contour',
    'plot_topography_grid',
    # Histogram
    'plot_histogram',
    'plot_rsa_distribution',
    'plot_lag_histogram',
    'plot_scatter_with_colorbar',
    # Model Comparison
    'load_model_results',
    'compute_layer_averages',
    'plot_layerwise_comparison',
    'plot_model_grid',
    'create_summary_table',
    'plot_best_layer_bar',
    # Group Comparison
    'load_group_data',
    'compute_group_statistics',
    'plot_group_boxplot',
    'plot_group_violin',
    'plot_group_comparison_grid',
    'perform_group_ttest',
]
