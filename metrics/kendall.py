#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kendall's Tau computation.

This module provides GPU-accelerated Kendall's Tau-b correlation.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_kendall_tau_b(x, y, device=None):
    """
    Compute Kendall's Tau-b correlation coefficient.
    
    Tau-b accounts for ties in the data.
    
    This is an approximate GPU implementation using sign concordance.
    
    Args:
        x: torch.Tensor of shape (n,) - first variable
        y: torch.Tensor of shape (n,) - second variable
        device: torch.device, optional
    
    Returns:
        tuple: (tau_b, p_value)
               Note: p_value is set to 1.0 (use permutation for significance)
    
    Example:
        >>> x = torch.randn(100)
        >>> y = torch.randn(100)
        >>> tau, pval = compute_kendall_tau_b(x, y)
    """
    if device is None:
        device = DEVICE
    
    a = torch.as_tensor(x, dtype=torch.float32, device=device).flatten()
    b = torch.as_tensor(y, dtype=torch.float32, device=device).flatten()
    
    if a.numel() == 0 or b.numel() == 0:
        return 0.0, 1.0
    
    if a.numel() != b.numel():
        raise ValueError("x and y must have the same length")
    
    n = a.numel()
    
    # Approximate Kendall tau using sign concordance
    # This is faster but less precise than the exact algorithm
    a0 = a - a.mean()
    b0 = b - b.mean()
    
    tau_approx = float(torch.sum(torch.sign(a0) * torch.sign(b0)).item() / n)
    
    return tau_approx, 1.0


def compute_kendall_tau_exact(x, y, device=None):
    """
    Compute exact Kendall's Tau-b (slower, O(n^2)).
    
    Args:
        x: torch.Tensor of shape (n,) - first variable
        y: torch.Tensor of shape (n,) - second variable
        device: torch.device, optional
    
    Returns:
        float: Kendall's Tau-b
    """
    if device is None:
        device = DEVICE
    
    a = torch.as_tensor(x, dtype=torch.float32, device=device).flatten()
    b = torch.as_tensor(y, dtype=torch.float32, device=device).flatten()
    
    n = a.numel()
    if n < 2:
        return 0.0
    
    # Compute all pairwise comparisons
    # sign(a[i] - a[j]) * sign(b[i] - b[j])
    diff_a = a.unsqueeze(1) - a.unsqueeze(0)  # (n, n)
    diff_b = b.unsqueeze(1) - b.unsqueeze(0)  # (n, n)
    
    sign_a = torch.sign(diff_a)
    sign_b = torch.sign(diff_b)
    
    # Only count upper triangular (i < j)
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
    
    concordant = ((sign_a * sign_b) > 0)[mask].sum()
    discordant = ((sign_a * sign_b) < 0)[mask].sum()
    
    # Ties
    ties_a = (sign_a == 0)[mask].sum()
    ties_b = (sign_b == 0)[mask].sum()
    
    n_pairs = n * (n - 1) // 2
    
    # Tau-b formula
    numerator = concordant - discordant
    denom = torch.sqrt((n_pairs - ties_a) * (n_pairs - ties_b)).float() + 1e-12
    
    tau_b = numerator.float() / denom
    
    return float(tau_b.item())


if __name__ == "__main__":
    # Test the functions
    x = torch.randn(100)
    y = x + 0.5 * torch.randn(100)  # Correlated
    
    tau_approx, _ = compute_kendall_tau_b(x, y)
    tau_exact = compute_kendall_tau_exact(x, y)
    
    print(f"Kendall Tau (approx): {tau_approx:.4f}")
    print(f"Kendall Tau (exact): {tau_exact:.4f}")

