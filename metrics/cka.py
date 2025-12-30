#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centered Kernel Alignment (CKA) computation.

CKA measures similarity between two representations by comparing
their kernel matrices after centering.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache for centering matrices
_CENTERING_CACHE = {}
_MAX_CACHE_SIZE = 10


def get_centering_matrix(n, device=None):
    """
    Get or create a centering matrix H = I - (1/n) * 11^T.
    
    Args:
        n: int - matrix size
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (n, n) - centering matrix
    """
    global _CENTERING_CACHE
    
    if device is None:
        device = DEVICE
    
    key = (str(device), int(n))
    
    if key in _CENTERING_CACHE:
        return _CENTERING_CACHE[key]
    
    # Clear old entries if cache is too large
    if len(_CENTERING_CACHE) >= _MAX_CACHE_SIZE:
        oldest_key = next(iter(_CENTERING_CACHE))
        del _CENTERING_CACHE[oldest_key]
    
    H = torch.eye(n, dtype=torch.float32, device=device) - \
        (torch.ones((n, n), dtype=torch.float32, device=device) / float(n))
    
    _CENTERING_CACHE[key] = H
    return H


def compute_cka(X, Y, kernel='linear', device=None):
    """
    Compute Centered Kernel Alignment (CKA) between two feature matrices.
    
    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    
    where K and L are kernel matrices computed from X and Y.
    
    Args:
        X: torch.Tensor of shape (n, p) - first feature matrix
        Y: torch.Tensor of shape (n, q) - second feature matrix
        kernel: str - 'linear' or 'rbf'
        device: torch.device, optional
    
    Returns:
        float: CKA value in [0, 1]
    
    Example:
        >>> X = torch.randn(100, 50)  # EEG features
        >>> Y = torch.randn(100, 20)  # Audio features
        >>> cka_linear = compute_cka(X, Y, kernel='linear')
        >>> cka_rbf = compute_cka(X, Y, kernel='rbf')
    """
    if device is None:
        device = DEVICE
    
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    
    n = int(Xt.shape[0])
    if n < 2:
        return 0.0
    
    # Compute kernel matrices
    if kernel == 'linear':
        K = Xt @ Xt.T
        L = Yt @ Yt.T
    else:  # RBF kernel
        # Compute pairwise squared distances
        Dx = torch.cdist(Xt, Xt, p=2) ** 2
        Dy = torch.cdist(Yt, Yt, p=2) ** 2
        
        # Median heuristic for bandwidth
        posx = Dx[Dx > 0]
        posy = Dy[Dy > 0]
        
        sx = torch.sqrt(torch.median(posx)) if posx.numel() > 0 else Xt.std()
        sy = torch.sqrt(torch.median(posy)) if posy.numel() > 0 else Yt.std()
        
        sx = sx if float(sx.item()) > 1e-12 else torch.tensor(1.0, device=device)
        sy = sy if float(sy.item()) > 1e-12 else torch.tensor(1.0, device=device)
        
        K = torch.exp(-Dx / (2.0 * sx * sx))
        L = torch.exp(-Dy / (2.0 * sy * sy))
    
    # Centering matrix
    H = get_centering_matrix(n, device)
    
    # Centered kernel matrices
    KH = H @ K @ H
    LH = H @ L @ H
    
    # HSIC values (unnormalized)
    hsic_kl = (KH * LH).sum()
    hsic_kk = (KH * KH).sum()
    hsic_ll = (LH * LH).sum()
    
    # CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
    denom = torch.sqrt(hsic_kk * hsic_ll) + 1e-12
    cka = hsic_kl / denom
    
    return float(cka.detach().cpu().item())


def compute_cka_linear(X, Y, device=None):
    """Convenience function for linear CKA."""
    return compute_cka(X, Y, kernel='linear', device=device)


def compute_cka_rbf(X, Y, device=None):
    """Convenience function for RBF CKA."""
    return compute_cka(X, Y, kernel='rbf', device=device)


if __name__ == "__main__":
    # Test the functions
    X = torch.randn(100, 50)
    Y = X[:, :20] + 0.1 * torch.randn(100, 20)  # Correlated
    
    cka_linear = compute_cka_linear(X, Y)
    cka_rbf = compute_cka_rbf(X, Y)
    
    print(f"Linear CKA: {cka_linear:.4f}")
    print(f"RBF CKA: {cka_rbf:.4f}")


