#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hilbert-Schmidt Independence Criterion (HSIC) computation.

HSIC is a kernel-based measure of statistical dependence between
two random variables.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache for centering matrices
_CENTERING_CACHE = {}


def get_centering_matrix(n, device=None):
    """Get or create a centering matrix."""
    global _CENTERING_CACHE
    
    if device is None:
        device = DEVICE
    
    key = (str(device), int(n))
    if key in _CENTERING_CACHE:
        return _CENTERING_CACHE[key]
    
    if len(_CENTERING_CACHE) >= 10:
        oldest_key = next(iter(_CENTERING_CACHE))
        del _CENTERING_CACHE[oldest_key]
    
    H = torch.eye(n, dtype=torch.float32, device=device) - \
        (torch.ones((n, n), dtype=torch.float32, device=device) / float(n))
    
    _CENTERING_CACHE[key] = H
    return H


def compute_hsic_rbf(X, Y, device=None):
    """
    Compute biased HSIC with RBF kernels using median heuristic.
    
    HSIC measures the dependence between X and Y by computing the
    Hilbert-Schmidt norm of the cross-covariance operator in a
    reproducing kernel Hilbert space.
    
    Args:
        X: torch.Tensor of shape (n, p) - first feature matrix
        Y: torch.Tensor of shape (n, q) - second feature matrix
        device: torch.device, optional
    
    Returns:
        float: HSIC value (larger = more dependent)
    
    References:
        Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
        Measuring statistical dependence with Hilbert-Schmidt norms.
        International Conference on Algorithmic Learning Theory.
    
    Example:
        >>> X = torch.randn(100, 50)  # EEG features
        >>> Y = torch.randn(100, 20)  # Audio features
        >>> hsic = compute_hsic_rbf(X, Y)
    """
    if device is None:
        device = DEVICE
    
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    
    n = int(Xt.shape[0])
    if n < 2:
        return 0.0
    
    # Compute squared distance matrices
    Dx = torch.cdist(Xt, Xt, p=2) ** 2
    Dy = torch.cdist(Yt, Yt, p=2) ** 2
    
    # Median heuristic for kernel bandwidth
    posx = Dx[Dx > 0]
    posy = Dy[Dy > 0]
    
    sx = torch.sqrt(torch.median(posx)) if posx.numel() > 0 else Xt.std()
    sy = torch.sqrt(torch.median(posy)) if posy.numel() > 0 else Yt.std()
    
    sx = sx if float(sx.item()) > 1e-12 else torch.tensor(1.0, device=device)
    sy = sy if float(sy.item()) > 1e-12 else torch.tensor(1.0, device=device)
    
    # RBF kernel matrices
    K = torch.exp(-Dx / (2.0 * sx * sx))
    L = torch.exp(-Dy / (2.0 * sy * sy))
    
    # Centering
    H = get_centering_matrix(n, device)
    KH = H @ K @ H
    LH = H @ L @ H
    
    # Biased HSIC estimate
    hsic = (KH * LH).sum() / float((n - 1) * (n - 1))
    
    return float(hsic.detach().cpu().item())


def compute_hsic_linear(X, Y, device=None):
    """
    Compute HSIC with linear kernels.
    
    This is equivalent to computing the squared Frobenius norm of
    the cross-covariance matrix.
    
    Args:
        X: torch.Tensor of shape (n, p) - first feature matrix
        Y: torch.Tensor of shape (n, q) - second feature matrix
        device: torch.device, optional
    
    Returns:
        float: HSIC value with linear kernel
    """
    if device is None:
        device = DEVICE
    
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    
    n = int(Xt.shape[0])
    if n < 2:
        return 0.0
    
    # Linear kernel matrices
    K = Xt @ Xt.T
    L = Yt @ Yt.T
    
    # Centering
    H = get_centering_matrix(n, device)
    KH = H @ K @ H
    LH = H @ L @ H
    
    # HSIC estimate
    hsic = (KH * LH).sum() / float((n - 1) * (n - 1))
    
    return float(hsic.detach().cpu().item())


if __name__ == "__main__":
    # Test the functions
    X = torch.randn(100, 50)
    Y = X[:, :20] + 0.1 * torch.randn(100, 20)  # Correlated
    
    hsic_rbf = compute_hsic_rbf(X, Y)
    hsic_linear = compute_hsic_linear(X, Y)
    
    print(f"HSIC (RBF): {hsic_rbf:.6f}")
    print(f"HSIC (Linear): {hsic_linear:.6f}")
    
    # Test with independent data
    Y_indep = torch.randn(100, 20)
    hsic_indep = compute_hsic_rbf(X, Y_indep)
    print(f"HSIC (independent): {hsic_indep:.6f}")


