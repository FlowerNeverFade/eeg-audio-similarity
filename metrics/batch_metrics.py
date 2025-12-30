#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing versions of metrics for improved GPU efficiency.

These functions process multiple samples in parallel for better throughput.
"""

import torch

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_rdm_vec_batch(X_batch, device=None):
    """
    Batch compute RDM vectors for multiple samples.
    
    Args:
        X_batch: torch.Tensor of shape (B, T, F) - B samples, T time steps, F features
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (B, T*(T-1)//2) - RDM vectors for each sample
    
    Example:
        >>> X = torch.randn(10, 100, 50)  # 10 samples
        >>> rdm_batch = compute_rdm_vec_batch(X)
        >>> print(rdm_batch.shape)  # torch.Size([10, 4950])
    """
    if device is None:
        device = DEVICE
    
    X = torch.as_tensor(X_batch, dtype=torch.float32, device=device)
    B, T, F = X.shape
    
    if T < 2:
        return torch.empty((B, 0), device=device, dtype=torch.float32)
    
    # Center: (B, T, F)
    Xc = X - X.mean(dim=2, keepdim=True)
    
    # Normalize: (B, T, 1)
    norms = torch.linalg.norm(Xc, dim=2, keepdim=True).clamp_min(1e-12)
    Xn = Xc / norms
    
    # Correlation matrix: (B, T, T)
    corr = torch.bmm(Xn, Xn.transpose(1, 2))
    rdm = 1.0 - corr
    
    # Extract upper triangular
    idx = torch.triu_indices(T, T, offset=1, device=device)
    return rdm[:, idx[0], idx[1]]


def rsa_between_rdms_batch(rdm_eeg_batch, rdm_audio, device=None):
    """
    Batch compute RSA between multiple EEG RDMs and one audio RDM.
    
    Args:
        rdm_eeg_batch: torch.Tensor of shape (B, N) - B EEG RDM vectors
        rdm_audio: torch.Tensor of shape (N,) - single audio RDM vector
        device: torch.device, optional
    
    Returns:
        tuple: (spearman_batch, pearson_batch) each of shape (B,)
    
    Example:
        >>> rdm_eeg = torch.randn(64, 4950)  # 64 electrodes
        >>> rdm_audio = torch.randn(4950)
        >>> spearman, pearson = rsa_between_rdms_batch(rdm_eeg, rdm_audio)
    """
    if device is None:
        device = DEVICE
    
    rdm_eeg = torch.as_tensor(rdm_eeg_batch, dtype=torch.float32, device=device)
    rdm_aud = torch.as_tensor(rdm_audio, dtype=torch.float32, device=device)
    
    B, N = rdm_eeg.shape
    
    if N == 0:
        return torch.zeros(B, device=device), torch.zeros(B, device=device)
    
    # Check for constant vectors
    std_eeg = rdm_eeg.std(dim=1)
    std_audio = rdm_aud.std()
    
    if std_audio < 1e-12:
        return torch.zeros(B, device=device), torch.zeros(B, device=device)
    
    # Pearson correlation
    rdm_aud_centered = rdm_aud - rdm_aud.mean()
    rdm_eeg_centered = rdm_eeg - rdm_eeg.mean(dim=1, keepdim=True)
    
    denom_aud = torch.sqrt((rdm_aud_centered ** 2).sum()).clamp_min(1e-12)
    denom_eeg = torch.sqrt((rdm_eeg_centered ** 2).sum(dim=1)).clamp_min(1e-12)
    
    pearson = (rdm_eeg_centered @ rdm_aud_centered) / (denom_aud * denom_eeg)
    
    # Spearman (rank-based Pearson)
    rank_audio = _rankdata_batch_1d(rdm_aud.unsqueeze(0), device)[0]
    rank_eeg = _rankdata_batch(rdm_eeg, device)
    
    rank_aud_centered = rank_audio - rank_audio.mean()
    rank_eeg_centered = rank_eeg - rank_eeg.mean(dim=1, keepdim=True)
    
    denom_rank_aud = torch.sqrt((rank_aud_centered ** 2).sum()).clamp_min(1e-12)
    denom_rank_eeg = torch.sqrt((rank_eeg_centered ** 2).sum(dim=1)).clamp_min(1e-12)
    
    spearman = (rank_eeg_centered @ rank_aud_centered) / (denom_rank_aud * denom_rank_eeg)
    
    # Mask constant vectors
    mask = std_eeg < 1e-12
    spearman[mask] = 0.0
    pearson[mask] = 0.0
    
    return spearman, pearson


def _rankdata_batch(x, device=None):
    """
    Batch rankdata - simplified version without tie handling.
    
    Args:
        x: torch.Tensor of shape (B, N)
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (B, N) - ranks (1-based)
    """
    if device is None:
        device = DEVICE
    
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    B, N = x.shape
    
    sorted_vals, sorted_idx = torch.sort(x, dim=1)
    
    ranks = torch.empty_like(x, dtype=torch.float32)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    ranks[batch_idx, sorted_idx] = torch.arange(1, N + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(B, N)
    
    return ranks


def _rankdata_batch_1d(x, device=None):
    """Helper for 1D ranking in batch format."""
    return _rankdata_batch(x, device)


def permutation_pvalue_batch(eeg_rdm_batch, audio_rank_full, observed_rsa_batch,
                             permutations, precomputed_perms, tri_indices, device=None):
    """
    Batch permutation p-value computation.
    
    Args:
        eeg_rdm_batch: torch.Tensor of shape (B, N_pairs) - EEG RDM vectors
        audio_rank_full: torch.Tensor of shape (T, T) - full ranked audio RDM
        observed_rsa_batch: torch.Tensor of shape (B,) - observed RSA values
        permutations: int - number of permutations
        precomputed_perms: torch.Tensor of shape (permutations, T) - permutation indices
        tri_indices: tuple - (tri_i, tri_j) upper triangular indices
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (B,) - p-values
    """
    if device is None:
        device = DEVICE
    
    B = eeg_rdm_batch.shape[0]
    n_pairs = eeg_rdm_batch.shape[1]
    
    if n_pairs == 0:
        return torch.ones(B, device=device)
    
    tri_i, tri_j = tri_indices
    
    # Rank the EEG RDMs
    rank_eeg_batch = _rankdata_batch(eeg_rdm_batch, device)
    
    # Center ranks
    x0_batch = rank_eeg_batch - rank_eeg_batch.mean(dim=1, keepdim=True)
    denom_x_batch = torch.sqrt((x0_batch ** 2).sum(dim=1)).clamp_min(1e-12)
    
    count_ge = torch.zeros(B, device=device, dtype=torch.int64)
    
    batch_size = min(1024, permutations)
    idx = 0
    total = precomputed_perms.shape[0]
    
    while idx < total:
        perms = precomputed_perms[idx:idx + batch_size]
        pairs0 = perms[:, tri_i]
        pairs1 = perms[:, tri_j]
        vects_rank = audio_rank_full[pairs0, pairs1]
        
        Y0 = vects_rank - vects_rank.mean(dim=1, keepdim=True)
        denom_Y = torch.sqrt((Y0 ** 2).sum(dim=1)).clamp_min(1e-12)
        
        # Correlations: (B, batch_size)
        corrs = (x0_batch @ Y0.T) / (denom_x_batch.unsqueeze(1) * denom_Y.unsqueeze(0))
        
        # Compare with observed
        ge = corrs >= observed_rsa_batch.unsqueeze(1)
        count_ge += ge.sum(dim=1)
        
        idx += batch_size
    
    pval = (count_ge.float() + 1.0) / (float(total) + 1.0)
    return pval


if __name__ == "__main__":
    # Test batch functions
    B, T, F = 10, 100, 50
    X_batch = torch.randn(B, T, F)
    
    rdm_batch = compute_rdm_vec_batch(X_batch)
    print(f"Batch RDM shape: {rdm_batch.shape}")
    
    rdm_audio = torch.randn(rdm_batch.shape[1])
    spearman, pearson = rsa_between_rdms_batch(rdm_batch, rdm_audio)
    print(f"Spearman batch shape: {spearman.shape}")
    print(f"Mean Spearman: {spearman.mean():.4f}")

