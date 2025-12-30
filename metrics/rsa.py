#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representational Similarity Analysis (RSA) functions.

This module provides GPU-accelerated functions for computing RSA
using Spearman and Pearson correlations between RDM vectors.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gpu_rankdata_average(x, device=None):
    """
    Compute ranks with average tie handling on GPU.
    
    Fully vectorized implementation without Python for loops.
    
    Args:
        x: torch.Tensor of shape (N,) - values to rank
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (N,) - ranks starting from 1
    """
    if device is None:
        device = DEVICE
    
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    n = x.shape[0]
    
    if n == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    
    # Sort and get indices
    vals, idx_sorted = torch.sort(x)
    
    # Base ranks 1, 2, 3, ..., n
    base_ranks = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    
    # Mark where values change (new group starts)
    neq = torch.ones(n, dtype=torch.bool, device=device)
    neq[1:] = vals[1:] != vals[:-1]
    
    # Assign group IDs using cumsum
    group_ids = torch.cumsum(neq.long(), dim=0) - 1
    
    # Count elements and sum ranks per group
    ones = torch.ones(n, dtype=torch.float32, device=device)
    group_counts = torch.zeros(n, dtype=torch.float32, device=device)
    group_counts.scatter_add_(0, group_ids, ones)
    
    group_rank_sums = torch.zeros(n, dtype=torch.float32, device=device)
    group_rank_sums.scatter_add_(0, group_ids, base_ranks)
    
    # Average rank per group
    group_avg_ranks = group_rank_sums / group_counts.clamp(min=1.0)
    
    # Assign to original positions
    avg_ranks_sorted = group_avg_ranks[group_ids]
    ranks = torch.empty(n, dtype=torch.float32, device=device)
    ranks[idx_sorted] = avg_ranks_sorted
    
    return ranks


def rsa_spearman(rdm1, rdm2, device=None):
    """
    Compute Spearman correlation between two RDM vectors.
    
    Args:
        rdm1: torch.Tensor of shape (N,) - first RDM vector
        rdm2: torch.Tensor of shape (N,) - second RDM vector
        device: torch.device, optional
    
    Returns:
        float: Spearman correlation coefficient
    
    Example:
        >>> rdm1 = torch.randn(4950)
        >>> rdm2 = torch.randn(4950)
        >>> rho = rsa_spearman(rdm1, rdm2)
    """
    if device is None:
        device = DEVICE
    
    t1 = torch.as_tensor(rdm1, dtype=torch.float32, device=device)
    t2 = torch.as_tensor(rdm2, dtype=torch.float32, device=device)
    
    if t1.numel() == 0 or t2.numel() == 0:
        return 0.0
    
    if torch.std(t1) < 1e-12 or torch.std(t2) < 1e-12:
        return 0.0
    
    # Rank both vectors
    rank1 = gpu_rankdata_average(t1, device)
    rank2 = gpu_rankdata_average(t2, device)
    
    # Compute Pearson on ranks (= Spearman)
    r1c = rank1 - rank1.mean()
    r2c = rank2 - rank2.mean()
    denom = torch.sqrt((r1c * r1c).sum() * (r2c * r2c).sum()) + 1e-12
    spearman_r = (r1c * r2c).sum() / denom
    
    return float(spearman_r.detach().cpu().item())


def rsa_pearson(rdm1, rdm2, device=None):
    """
    Compute Pearson correlation between two RDM vectors.
    
    Args:
        rdm1: torch.Tensor of shape (N,) - first RDM vector
        rdm2: torch.Tensor of shape (N,) - second RDM vector
        device: torch.device, optional
    
    Returns:
        float: Pearson correlation coefficient
    """
    if device is None:
        device = DEVICE
    
    t1 = torch.as_tensor(rdm1, dtype=torch.float32, device=device)
    t2 = torch.as_tensor(rdm2, dtype=torch.float32, device=device)
    
    if t1.numel() == 0 or t2.numel() == 0:
        return 0.0
    
    if torch.std(t1) < 1e-12 or torch.std(t2) < 1e-12:
        return 0.0
    
    t1c = t1 - t1.mean()
    t2c = t2 - t2.mean()
    denom = torch.sqrt((t1c * t1c).sum() * (t2c * t2c).sum()) + 1e-12
    pearson_r = (t1c * t2c).sum() / denom
    
    return float(pearson_r.detach().cpu().item())


def rsa_between_rdms(rdm1, rdm2, device=None):
    """
    Compute both Spearman and Pearson RSA between two RDM vectors.
    
    Args:
        rdm1: torch.Tensor of shape (N,) - first RDM vector
        rdm2: torch.Tensor of shape (N,) - second RDM vector
        device: torch.device, optional
    
    Returns:
        tuple: (spearman_r, p_value, pearson_r)
               Note: p_value is set to 1.0 (use permutation test for significance)
    
    Example:
        >>> rdm_eeg = torch.randn(4950)
        >>> rdm_audio = torch.randn(4950)
        >>> spearman, pval, pearson = rsa_between_rdms(rdm_eeg, rdm_audio)
    """
    spearman_r = rsa_spearman(rdm1, rdm2, device)
    pearson_r = rsa_pearson(rdm1, rdm2, device)
    
    return spearman_r, 1.0, pearson_r


if __name__ == "__main__":
    # Test the functions
    rdm1 = torch.randn(4950)
    rdm2 = rdm1 + 0.1 * torch.randn(4950)  # Correlated
    
    spearman, _, pearson = rsa_between_rdms(rdm1, rdm2)
    print(f"Spearman: {spearman:.4f}")
    print(f"Pearson: {pearson:.4f}")


