#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV Coefficient computation.

The RV coefficient (Escoufier's RV) measures the similarity between
two multivariate datasets.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_rv_coefficient(X, Y, device=None):
    """
    Compute Escoufier's RV coefficient between two multivariate datasets.
    
    The RV coefficient is defined as:
        RV(X, Y) = trace(X'Y Y'X) / sqrt(trace(X'X X'X) * trace(Y'Y Y'Y))
    
    where X and Y are centered data matrices.
    
    Args:
        X: torch.Tensor of shape (n, p) - first dataset (n samples, p features)
        Y: torch.Tensor of shape (n, q) - second dataset (n samples, q features)
        device: torch.device, optional
    
    Returns:
        float: RV coefficient value in [0, 1]
    
    References:
        Escoufier, Y. (1973). Le traitement des variables vectorielles.
        Biometrics, 29(4), 751-760.
    
    Example:
        >>> X = torch.randn(100, 50)  # EEG features
        >>> Y = torch.randn(100, 20)  # Audio features
        >>> rv = compute_rv_coefficient(X, Y)
    """
    if device is None:
        device = DEVICE
    
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    
    if int(Xt.shape[0]) < 2 or int(Yt.shape[0]) < 2:
        return 0.0
    
    # Center the data
    Xc = Xt - Xt.mean(dim=0, keepdim=True)
    Yc = Yt - Yt.mean(dim=0, keepdim=True)
    
    # Compute cross-product matrices
    XtX = Xc.T @ Xc  # (p, p)
    YtY = Yc.T @ Yc  # (q, q)
    
    # Numerator: trace((X'Y)(Y'X))
    XtY = Xc.T @ Yc  # (p, q)
    YtX = Yc.T @ Xc  # (q, p)
    top = torch.trace(XtY @ YtX)
    
    # Denominator: sqrt(trace(X'X X'X) * trace(Y'Y Y'Y))
    bot = torch.sqrt(torch.trace(XtX @ XtX) * torch.trace(YtY @ YtY)) + 1e-12
    
    rv = top / bot
    
    return float(rv.detach().cpu().item())


if __name__ == "__main__":
    # Test the function
    X = torch.randn(100, 50)
    Y = X[:, :20] + 0.1 * torch.randn(100, 20)  # Correlated
    
    rv = compute_rv_coefficient(X, Y)
    print(f"RV Coefficient: {rv:.4f}")
    
    # Test with independent data
    Y_indep = torch.randn(100, 20)
    rv_indep = compute_rv_coefficient(X, Y_indep)
    print(f"RV Coefficient (independent): {rv_indep:.4f}")


