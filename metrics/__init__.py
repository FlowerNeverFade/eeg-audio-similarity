#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics module for EEG-Audio similarity analysis.

This module provides various metrics for comparing neural representations
with audio model representations.
"""

from .rdm import compute_rdm_vec, compute_rdm_full, rdm_vec_to_matrix
from .rsa import rsa_spearman, rsa_pearson, rsa_between_rdms, gpu_rankdata_average
from .cka import compute_cka, compute_cka_linear, compute_cka_rbf
from .distance_correlation import compute_distance_correlation
from .hsic import compute_hsic_rbf, compute_hsic_linear
from .rv_coefficient import compute_rv_coefficient
from .permutation_test import permutation_pvalue, permutation_test_rsa
from .mutual_info import compute_mutual_info_gaussian
from .kendall import compute_kendall_tau_b, compute_kendall_tau_exact, compute_kendall_tau_approx
from .batch_metrics import (
    compute_rdm_vec_batch,
    rsa_between_rdms_batch,
    permutation_pvalue_batch,
)

__all__ = [
    # RDM
    'compute_rdm_vec',
    'compute_rdm_full',
    'rdm_vec_to_matrix',
    # RSA
    'rsa_spearman',
    'rsa_pearson',
    'rsa_between_rdms',
    'gpu_rankdata_average',
    # CKA
    'compute_cka',
    'compute_cka_linear',
    'compute_cka_rbf',
    # Distance Correlation
    'compute_distance_correlation',
    # HSIC
    'compute_hsic_rbf',
    'compute_hsic_linear',
    # RV Coefficient
    'compute_rv_coefficient',
    # Permutation Test
    'permutation_pvalue',
    'permutation_test_rsa',
    # Mutual Information
    'compute_mutual_info_gaussian',
    # Kendall
    'compute_kendall_tau_b',
    'compute_kendall_tau_exact',
    'compute_kendall_tau_approx',
    # Batch Metrics
    'compute_rdm_vec_batch',
    'rsa_between_rdms_batch',
    'permutation_pvalue_batch',
]
