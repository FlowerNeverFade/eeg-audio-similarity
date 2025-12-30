#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Alignment between EEG and Audio Embeddings.

This module provides functions for aligning EEG and audio model embeddings
along the time axis through resampling/interpolation.

Key challenge: EEG has Ns samples, audio embeddings have Ta time steps.
We need to align them to a common time axis for RSA comparison.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


def resample_to_target_length(signal: np.ndarray,
                               target_length: int,
                               mode: str = 'linear') -> np.ndarray:
    """
    Resample signal to target length using interpolation.
    
    Args:
        signal: np.ndarray - input signal
                Shape can be (n_samples,), (n_channels, n_samples), or (n_samples, n_features)
        target_length: int - target number of samples
        mode: str - interpolation mode ('linear', 'nearest', 'cubic')
    
    Returns:
        np.ndarray - resampled signal with target_length samples
    
    Example:
        >>> eeg = np.random.randn(64, 5000)  # 64 channels, 5000 samples
        >>> eeg_resampled = resample_to_target_length(eeg, target_length=100)
        >>> print(eeg_resampled.shape)  # (64, 100)
    """
    if signal.ndim == 1:
        # (n_samples,) -> (1, 1, n_samples) -> interpolate -> (n_samples,)
        signal_t = torch.from_numpy(signal).float().view(1, 1, -1)
        resampled = F.interpolate(signal_t, size=target_length, mode=mode, 
                                   align_corners=False if mode != 'nearest' else None)
        return resampled.view(-1).numpy()
    
    elif signal.ndim == 2:
        # Check if shape is (n_channels, n_samples) or (n_samples, n_features)
        # Heuristic: if dim0 > dim1, assume (n_samples, n_features)
        if signal.shape[0] > signal.shape[1] * 2:
            # (n_samples, n_features) -> (1, n_features, n_samples) -> (n_samples, n_features)
            signal_t = torch.from_numpy(signal.T).float().unsqueeze(0)
            resampled = F.interpolate(signal_t, size=target_length, mode=mode,
                                       align_corners=False if mode != 'nearest' else None)
            return resampled.squeeze(0).T.numpy()
        else:
            # (n_channels, n_samples) -> (1, n_channels, n_samples) -> (n_channels, n_samples)
            signal_t = torch.from_numpy(signal).float().unsqueeze(0)
            resampled = F.interpolate(signal_t, size=target_length, mode=mode,
                                       align_corners=False if mode != 'nearest' else None)
            return resampled.squeeze(0).numpy()
    
    else:
        raise ValueError(f"Unsupported signal shape: {signal.shape}")


def interpolate_features(features: np.ndarray,
                          target_length: int,
                          device: str = 'cuda') -> torch.Tensor:
    """
    GPU-accelerated feature interpolation.
    
    Args:
        features: np.ndarray of shape (T, D) - T time steps, D feature dims
        target_length: int - target number of time steps
        device: str - 'cuda' or 'cpu'
    
    Returns:
        torch.Tensor of shape (target_length, D)
    
    Example:
        >>> audio_emb = np.random.randn(50, 1024)  # 50 time steps
        >>> eeg_aligned = interpolate_features(audio_emb, target_length=100)
        >>> print(eeg_aligned.shape)  # torch.Size([100, 1024])
    """
    # (T, D) -> (1, D, T) -> interpolate -> (1, D, target_length) -> (target_length, D)
    feat_t = torch.from_numpy(features).float()
    if device == 'cuda' and torch.cuda.is_available():
        feat_t = feat_t.cuda()
    
    feat_3d = feat_t.T.unsqueeze(0)  # (1, D, T)
    resampled = F.interpolate(feat_3d, size=target_length, mode='linear', 
                               align_corners=False)
    return resampled.squeeze(0).T  # (target_length, D)


def align_eeg_to_audio(eeg: np.ndarray,
                        audio_embedding: np.ndarray,
                        eeg_fs: float = 500.0,
                        audio_frame_shift: float = 0.02,
                        method: str = 'resample_eeg') -> Tuple[np.ndarray, np.ndarray]:
    """
    Align EEG and audio embeddings to common time axis.
    
    Two methods:
    1. 'resample_eeg': Resample EEG to match audio embedding time steps
    2. 'resample_audio': Resample audio embeddings to match EEG samples
    
    Args:
        eeg: np.ndarray - EEG data, shape (n_channels, n_samples) or (n_samples,)
        audio_embedding: np.ndarray - Audio embeddings, shape (T_audio, D)
        eeg_fs: float - EEG sampling frequency in Hz
        audio_frame_shift: float - Audio frame shift in seconds (typically 0.02s = 20ms)
        method: str - 'resample_eeg' or 'resample_audio'
    
    Returns:
        Tuple[np.ndarray, np.ndarray] - (aligned_eeg, aligned_audio)
        Both will have same number of time steps
    
    Example:
        >>> eeg = np.random.randn(64, 5000)  # 10 seconds at 500Hz
        >>> audio_emb = np.random.randn(500, 1024)  # 10 seconds at 50 fps
        >>> eeg_aligned, audio_aligned = align_eeg_to_audio(eeg, audio_emb)
        >>> print(eeg_aligned.shape, audio_aligned.shape)
    """
    T_audio = audio_embedding.shape[0]
    n_eeg_samples = eeg.shape[-1]
    
    # Calculate time duration
    eeg_duration = n_eeg_samples / eeg_fs
    audio_duration = T_audio * audio_frame_shift
    
    if method == 'resample_eeg':
        # Resample EEG to audio time steps
        target_length = T_audio
        
        if eeg.ndim == 1:
            eeg_resampled = resample_to_target_length(eeg, target_length)
            eeg_aligned = eeg_resampled.reshape(-1, 1)
        else:
            eeg_resampled = resample_to_target_length(eeg, target_length)
            eeg_aligned = eeg_resampled.T  # (T, n_channels)
        
        audio_aligned = audio_embedding
        
    elif method == 'resample_audio':
        # Resample audio embeddings to EEG samples
        target_length = n_eeg_samples
        audio_resampled = resample_to_target_length(audio_embedding, target_length)
        
        if eeg.ndim == 1:
            eeg_aligned = eeg.reshape(-1, 1)
        else:
            eeg_aligned = eeg.T  # (T, n_channels)
        
        audio_aligned = audio_resampled
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return eeg_aligned, audio_aligned


def compute_optimal_lag(eeg_features: np.ndarray,
                        audio_features: np.ndarray,
                        max_lag_ms: float = 500.0,
                        fs: float = 500.0,
                        metric: str = 'correlation') -> Tuple[int, float]:
    """
    Find optimal time lag between EEG and audio features.
    
    EEG response typically lags audio stimulus by 100-300ms.
    
    Args:
        eeg_features: np.ndarray of shape (T, D_eeg)
        audio_features: np.ndarray of shape (T, D_audio)
        max_lag_ms: float - maximum lag to search in milliseconds
        fs: float - sampling frequency for lag conversion
        metric: str - 'correlation' or 'mse'
    
    Returns:
        Tuple[int, float] - (best_lag_samples, best_score)
    
    Example:
        >>> eeg = np.random.randn(100, 64)
        >>> audio = np.random.randn(100, 20)
        >>> best_lag, score = compute_optimal_lag(eeg, audio, max_lag_ms=300)
        >>> print(f"Best lag: {best_lag} samples, score: {score:.4f}")
    """
    T = eeg_features.shape[0]
    max_lag_samples = int(max_lag_ms * fs / 1000)
    
    # Test range of lags (positive = EEG lags audio)
    lags = range(0, min(max_lag_samples, T // 2))
    
    best_lag = 0
    best_score = -np.inf if metric == 'correlation' else np.inf
    
    for lag in lags:
        if lag == 0:
            eeg_shifted = eeg_features
            audio_shifted = audio_features
        else:
            eeg_shifted = eeg_features[lag:]
            audio_shifted = audio_features[:-lag]
        
        # Compute average feature (first PCA component proxy)
        eeg_avg = np.mean(eeg_shifted, axis=1)
        audio_avg = np.mean(audio_shifted, axis=1)
        
        if metric == 'correlation':
            score = np.corrcoef(eeg_avg, audio_avg)[0, 1]
            if score > best_score:
                best_score = score
                best_lag = lag
        else:  # MSE
            score = np.mean((eeg_avg - audio_avg) ** 2)
            if score < best_score:
                best_score = score
                best_lag = lag
    
    return best_lag, best_score


def apply_lag(eeg_features: np.ndarray,
              audio_features: np.ndarray,
              lag_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply time lag to align EEG and audio features.
    
    Args:
        eeg_features: np.ndarray of shape (T, D_eeg)
        audio_features: np.ndarray of shape (T, D_audio)
        lag_samples: int - number of samples to lag EEG (positive = EEG lags audio)
    
    Returns:
        Tuple[np.ndarray, np.ndarray] - (aligned_eeg, aligned_audio)
    """
    if lag_samples == 0:
        return eeg_features, audio_features
    elif lag_samples > 0:
        return eeg_features[lag_samples:], audio_features[:-lag_samples]
    else:
        return eeg_features[:lag_samples], audio_features[-lag_samples:]


if __name__ == "__main__":
    # Test alignment
    np.random.seed(42)
    
    # Simulate EEG and audio
    eeg = np.random.randn(64, 5000)  # 10 seconds at 500Hz
    audio_emb = np.random.randn(500, 1024)  # 10 seconds at 50 fps (20ms frame shift)
    
    # Align
    eeg_aligned, audio_aligned = align_eeg_to_audio(eeg, audio_emb, 
                                                     eeg_fs=500, 
                                                     audio_frame_shift=0.02,
                                                     method='resample_eeg')
    
    print(f"Original EEG shape: {eeg.shape}")
    print(f"Original audio shape: {audio_emb.shape}")
    print(f"Aligned EEG shape: {eeg_aligned.shape}")
    print(f"Aligned audio shape: {audio_aligned.shape}")
    
    # Test lag finding
    eeg_feat = np.random.randn(100, 64)
    audio_feat = np.random.randn(100, 20)
    
    # Create artificial lag
    audio_feat[10:] = eeg_feat[:-10, :20] + 0.1 * np.random.randn(90, 20)
    
    best_lag, score = compute_optimal_lag(eeg_feat, audio_feat, max_lag_ms=200, fs=50)
    print(f"Detected lag: {best_lag} samples, correlation: {score:.4f}")

