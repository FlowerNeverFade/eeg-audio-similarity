#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutual Information computation.

This module provides functions for computing mutual information
between feature matrices using Gaussian approximation.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_mutual_info_gaussian(X, y, device=None):
    """
    Compute mean mutual information using Gaussian approximation.
    
    MI is approximated as: MI = -0.5 * log(1 - rho^2)
    where rho is the Pearson correlation.
    
    Args:
        X: torch.Tensor of shape (n, p) - feature matrix
        y: torch.Tensor of shape (n,) or (n, q) - target variable(s)
        device: torch.device, optional
    
    Returns:
        float: Mean mutual information across all feature-target pairs
    
    Example:
        >>> X = torch.randn(100, 50)
        >>> y = torch.randn(100, 10)
        >>> mi = compute_mutual_info_gaussian(X, y)
    """
    if device is None:
        device = DEVICE
    
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    
    if X_t.shape[0] < 3:
        return 0.0
    
    def _mi_gaussian(a, b):
        """Compute MI between two vectors using Gaussian assumption."""
        a0 = a - a.mean(dim=0, keepdim=True)
        b0 = b - b.mean(dim=0, keepdim=True)
        denom = torch.sqrt((a0 * a0).sum(dim=0) * (b0 * b0).sum(dim=0)) + 1e-12
        rho = (a0 * b0).sum(dim=0) / denom
        rho2 = torch.clamp(rho * rho, max=1 - 1e-7)
        mi = -0.5 * torch.log(1 - rho2)
        return mi
    
    if y_t.ndim == 1 or y_t.shape[1] == 1:
        y_col = y_t.reshape(-1, 1)
        mi_each = []
        for j in range(X_t.shape[1]):
            mi_each.append(_mi_gaussian(X_t[:, j:j+1], y_col))
        mi_stack = torch.stack(mi_each, dim=0)
        return float(mi_stack.mean().detach().cpu().item())
    else:
        mi_vals = []
        for j in range(y_t.shape[1]):
            y_col = y_t[:, j:j+1]
            mi_each = []
            for k in range(X_t.shape[1]):
                mi_each.append(_mi_gaussian(X_t[:, k:k+1], y_col))
            mi_vals.append(torch.stack(mi_each, dim=0).mean())
        return float(torch.stack(mi_vals, dim=0).mean().detach().cpu().item())


if __name__ == "__main__":
    # Test the function
    X = torch.randn(100, 50)
    y = X[:, :10] + 0.1 * torch.randn(100, 10)  # Correlated
    
    mi = compute_mutual_info_gaussian(X, y)
    print(f"Mutual Information: {mi:.4f}")

