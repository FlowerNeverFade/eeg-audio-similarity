#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representational Dissimilarity Matrix (RDM) computation functions.

This module provides GPU-accelerated functions for computing RDMs
using correlation distance.
"""

import torch
import torch.nn.functional as F

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_rdm_vec(X, device=None):
    """
    Compute the upper triangular RDM vector using correlation distance.
    
    RDM is computed as (1 - correlation) between all pairs of rows.
    Only the upper triangular elements are returned as a flattened vector.
    
    Args:
        X: torch.Tensor of shape (T, D) - T time steps, D features
        device: torch.device, optional - computation device
    
    Returns:
        torch.Tensor of shape (T*(T-1)//2,) - upper triangular RDM values
    
    Example:
        >>> X = torch.randn(100, 50)  # 100 time steps, 50 features
        >>> rdm_vec = compute_rdm_vec(X)
        >>> print(rdm_vec.shape)  # torch.Size([4950])
    """
    if device is None:
        device = DEVICE
    
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    
    if X_t.shape[0] < 2:
        return torch.empty(0, device=device)
    
    # Handle 1D case
    if X_t.shape[1] == 1:
        x = X_t[:, 0]
        diff = torch.abs(x[:, None] - x[None, :])
        idx = torch.triu_indices(diff.shape[0], diff.shape[1], offset=1, device=device)
        return diff[idx[0], idx[1]]
    
    # Center the data
    Xc = X_t - X_t.mean(dim=1, keepdim=True)
    
    # Normalize
    norms = torch.linalg.norm(Xc, dim=1, keepdim=True) + 1e-12
    Xn = Xc / norms
    
    # Compute correlation matrix
    corr = Xn @ Xn.T
    
    # Convert to dissimilarity
    rdm = 1.0 - corr
    
    # Extract upper triangular
    idx = torch.triu_indices(rdm.shape[0], rdm.shape[0], offset=1, device=device)
    return rdm[idx[0], idx[1]]


def compute_rdm_full(X, device=None):
    """
    Compute the full (T x T) RDM matrix using correlation distance.
    
    Args:
        X: torch.Tensor of shape (T, D) - T time steps, D features
        device: torch.device, optional - computation device
    
    Returns:
        torch.Tensor of shape (T, T) - full RDM matrix
    
    Example:
        >>> X = torch.randn(100, 50)
        >>> rdm_full = compute_rdm_full(X)
        >>> print(rdm_full.shape)  # torch.Size([100, 100])
    """
    if device is None:
        device = DEVICE
    
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    
    if Xt.shape[0] < 2:
        return torch.zeros((Xt.shape[0], Xt.shape[0]), dtype=torch.float32, device=device)
    
    # Handle 1D case
    if Xt.shape[1] == 1:
        x = Xt[:, 0]
        diff = torch.abs(x[:, None] - x[None, :])
        return diff
    
    # Center and normalize
    Xc = Xt - Xt.mean(dim=1, keepdim=True)
    norms = torch.linalg.norm(Xc, dim=1, keepdim=True) + 1e-12
    Xn = Xc / norms
    
    # Compute correlation and convert to dissimilarity
    corr = Xn @ Xn.T
    rdm = 1.0 - corr
    
    return rdm


def rdm_vec_to_matrix(rdm_vec, device=None):
    """
    Convert an upper triangular RDM vector back to a full symmetric matrix.
    
    Args:
        rdm_vec: torch.Tensor of shape (n_pairs,) - upper triangular values
        device: torch.device, optional - computation device
    
    Returns:
        torch.Tensor of shape (T, T) - symmetric RDM matrix
    
    Example:
        >>> rdm_vec = torch.randn(4950)  # For T=100
        >>> rdm_mat = rdm_vec_to_matrix(rdm_vec)
        >>> print(rdm_mat.shape)  # torch.Size([100, 100])
    """
    if device is None:
        device = DEVICE
    
    rdm_vec = torch.as_tensor(rdm_vec, dtype=torch.float32, device=device)
    
    # Infer T from n_pairs = T*(T-1)//2
    n_pairs = rdm_vec.shape[0]
    T = int((1 + (1 + 8 * n_pairs) ** 0.5) / 2)
    
    # Create symmetric matrix
    rdm_mat = torch.zeros((T, T), dtype=torch.float32, device=device)
    idx = torch.triu_indices(T, T, offset=1, device=device)
    rdm_mat[idx[0], idx[1]] = rdm_vec
    rdm_mat[idx[1], idx[0]] = rdm_vec
    
    return rdm_mat


if __name__ == "__main__":
    # Test the functions
    X = torch.randn(100, 50)
    rdm_vec = compute_rdm_vec(X)
    rdm_full = compute_rdm_full(X)
    rdm_reconstructed = rdm_vec_to_matrix(rdm_vec)
    
    print(f"RDM vector shape: {rdm_vec.shape}")
    print(f"RDM full shape: {rdm_full.shape}")
    print(f"Reconstruction error: {(rdm_full - rdm_reconstructed).abs().max().item():.6f}")


