#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance Correlation computation.

Distance correlation measures both linear and non-linear dependencies
between two multivariate random variables.
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


def compute_distance_correlation(X, Y, device=None):
    """
    Compute distance correlation between two feature matrices.
    
    Distance correlation is defined as:
        dCor(X, Y) = dCov(X, Y) / sqrt(dVar(X) * dVar(Y))
    
    where dCov is the distance covariance computed using double-centered
    distance matrices.
    
    Args:
        X: torch.Tensor of shape (n, p) - first feature matrix
        Y: torch.Tensor of shape (n, q) - second feature matrix
        device: torch.device, optional
    
    Returns:
        float: Distance correlation value in [0, 1]
    
    References:
        Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
        Measuring and testing dependence by correlation of distances.
        The Annals of Statistics, 35(6), 2769-2794.
    
    Example:
        >>> X = torch.randn(100, 50)  # EEG features
        >>> Y = torch.randn(100, 20)  # Audio features
        >>> dcor = compute_distance_correlation(X, Y)
    """
    if device is None:
        device = DEVICE
    
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    
    n = int(Xt.shape[0])
    if n < 2:
        return 0.0
    
    # Compute Euclidean distance matrices
    Dx = torch.cdist(Xt, Xt, p=2)
    Dy = torch.cdist(Yt, Yt, p=2)
    
    # Double centering
    J = get_centering_matrix(n, device)
    Ax = J @ Dx @ J
    Ay = J @ Dy @ J
    
    # Distance covariance and variances
    dcov2 = (Ax * Ay).sum() / float(n * n)
    dvarx = (Ax * Ax).sum() / float(n * n)
    dvary = (Ay * Ay).sum() / float(n * n)
    
    # Distance correlation
    denom = torch.sqrt(dvarx * dvary) + 1e-12
    dcor = dcov2 / denom
    
    return float(dcor.detach().cpu().item())


if __name__ == "__main__":
    # Test the function
    X = torch.randn(100, 50)
    Y = X[:, :20] + 0.1 * torch.randn(100, 20)  # Correlated
    
    dcor = compute_distance_correlation(X, Y)
    print(f"Distance Correlation: {dcor:.4f}")
    
    # Test with independent data
    Y_indep = torch.randn(100, 20)
    dcor_indep = compute_distance_correlation(X, Y_indep)
    print(f"Distance Correlation (independent): {dcor_indep:.4f}")


