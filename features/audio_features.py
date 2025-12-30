#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Feature Extraction and Processing functions.

This module provides functions for processing audio embeddings from
speech language models for RSA analysis.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reduce_audio_dimensions(audio_emb, n_components=20, device=None):
    """
    Reduce dimensionality of audio embeddings using PCA.
    
    Uses GPU-accelerated SVD for fast PCA computation.
    
    Args:
        audio_emb: torch.Tensor of shape (T, D) - audio embeddings
        n_components: int - target number of components (default: 20)
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (T, min(n_components, D)) - reduced embeddings
    
    Example:
        >>> audio = torch.randn(100, 1024)  # 100 time steps, 1024 dim
        >>> reduced = reduce_audio_dimensions(audio, n_components=20)
        >>> print(reduced.shape)  # torch.Size([100, 20])
    """
    if device is None:
        device = DEVICE
    
    X = torch.as_tensor(audio_emb, dtype=torch.float32, device=device)
    
    # Handle batch dimension
    if X.ndim == 3 and X.shape[0] == 1:
        X = X.squeeze(0)
    
    # Handle 1D case
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # If n_components >= n_features, just normalize
    if n_components is None or n_components >= n_features:
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-9)
        return X
    
    n_comp = int(min(n_components, n_samples, n_features))
    if n_comp < 1:
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-9)
        return X
    
    # Center the data
    Xc = X - X.mean(dim=0, keepdim=True)
    
    # SVD for PCA
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    
    # Project onto top components
    V = Vh.T[:, :n_comp]
    Xr = Xc @ V
    
    # Normalize
    Xr = (Xr - Xr.mean(dim=0, keepdim=True)) / (Xr.std(dim=0, keepdim=True) + 1e-9)
    
    return Xr


def normalize_audio_embeddings(audio_emb, device=None):
    """
    Normalize audio embeddings (zero mean, unit variance per dimension).
    
    Args:
        audio_emb: torch.Tensor of shape (T, D) - audio embeddings
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (T, D) - normalized embeddings
    """
    if device is None:
        device = DEVICE
    
    X = torch.as_tensor(audio_emb, dtype=torch.float32, device=device)
    
    if X.ndim == 3 and X.shape[0] == 1:
        X = X.squeeze(0)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-9)
    
    return X


def load_audio_embeddings(npz_path, layer_idx=None, device=None):
    """
    Load audio embeddings from NPZ file.
    
    Args:
        npz_path: str - path to NPZ file
        layer_idx: int, optional - specific layer to load (default: all)
        device: torch.device, optional
    
    Returns:
        dict or torch.Tensor - layer embeddings
    
    Example:
        >>> embeddings = load_audio_embeddings("audio.npz")
        >>> layer_0 = embeddings['layer_0']
    """
    import numpy as np
    
    if device is None:
        device = DEVICE
    
    npz_data = np.load(npz_path, allow_pickle=True)
    
    if layer_idx is not None:
        key = f"layer_{layer_idx}"
        if key in npz_data:
            emb = npz_data[key].astype(np.float32)
            return torch.as_tensor(emb, dtype=torch.float32, device=device)
        else:
            raise KeyError(f"Layer {layer_idx} not found in {npz_path}")
    
    # Return all layers
    result = {}
    for key in npz_data.keys():
        if key.startswith('layer_'):
            emb = npz_data[key].astype(np.float32)
            result[key] = torch.as_tensor(emb, dtype=torch.float32, device=device)
    
    return result


def get_layer_count(npz_path):
    """
    Get the number of layers in an audio embedding file.
    
    Args:
        npz_path: str - path to NPZ file
    
    Returns:
        int - number of layers
    """
    import numpy as np
    
    npz_data = np.load(npz_path, allow_pickle=True)
    layer_keys = [k for k in npz_data.keys() if k.startswith('layer_')]
    return len(layer_keys)


if __name__ == "__main__":
    # Test the functions
    audio = torch.randn(100, 1024)  # 100 time steps, 1024 dimensions
    
    reduced = reduce_audio_dimensions(audio, n_components=20)
    normalized = normalize_audio_embeddings(audio)
    
    print(f"Original shape: {audio.shape}")
    print(f"Reduced shape: {reduced.shape}")
    print(f"Normalized shape: {normalized.shape}")

