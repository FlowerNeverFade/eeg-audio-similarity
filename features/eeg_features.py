#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Feature Extraction functions.

This module provides GPU-accelerated functions for extracting features
from raw EEG signals for RSA analysis.
"""

import torch
import torch.nn.functional as F

# Global device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_eeg_raw_aligned(eeg_signal, n_time_steps, device=None):
    """
    Extract raw EEG voltage values aligned to target time steps.
    
    Resamples the EEG signal to match the number of audio time steps
    and normalizes each dimension.
    
    Args:
        eeg_signal: torch.Tensor of shape (n_samples,) - raw EEG voltage
        n_time_steps: int - target number of time steps (from audio)
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (n_time_steps, 1) - aligned EEG features
    
    Example:
        >>> eeg = torch.randn(5000)  # 10 seconds at 500Hz
        >>> features = extract_eeg_raw_aligned(eeg, 100)
        >>> print(features.shape)  # torch.Size([100, 1])
    """
    if device is None:
        device = DEVICE
    
    sig = torch.as_tensor(eeg_signal, dtype=torch.float32, device=device)
    sig = sig.view(1, 1, -1)
    
    T = int(n_time_steps)
    if T <= 0:
        return torch.zeros((0, 1), device=device, dtype=torch.float32)
    
    # Resample using linear interpolation
    sig_rs = F.interpolate(sig, size=T, mode='linear', align_corners=False)
    sig_rs = sig_rs.view(T, 1)
    
    # Normalize
    sig_rs = (sig_rs - sig_rs.mean(dim=0, keepdim=True)) / (sig_rs.std(dim=0, keepdim=True) + 1e-9)
    
    return sig_rs


def extract_eeg_multichannel_aligned(eeg_signals, n_time_steps, device=None):
    """
    Extract multi-channel EEG features aligned to target time steps.
    
    Resamples all channels to match the number of audio time steps
    and normalizes each channel.
    
    Args:
        eeg_signals: torch.Tensor of shape (n_channels, n_samples) - multi-channel EEG
        n_time_steps: int - target number of time steps
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (n_time_steps, n_channels) - aligned multi-channel features
    
    Example:
        >>> eeg = torch.randn(64, 5000)  # 64 channels, 10 seconds at 500Hz
        >>> features = extract_eeg_multichannel_aligned(eeg, 100)
        >>> print(features.shape)  # torch.Size([100, 64])
    """
    if device is None:
        device = DEVICE
    
    X = torch.as_tensor(eeg_signals, dtype=torch.float32, device=device)
    T = int(n_time_steps)
    
    if T <= 0:
        return torch.zeros((0, 1), device=device, dtype=torch.float32)
    
    if X.ndim == 1:
        return extract_eeg_raw_aligned(X, n_time_steps, device)
    
    # Shape: (1, channels, samples)
    X = X.unsqueeze(0)
    
    # Resample: (1, channels, T)
    feats = F.interpolate(X, size=T, mode='linear', align_corners=False)
    
    # Transpose to (T, channels)
    feats = feats.squeeze(0).transpose(0, 1)
    
    # Normalize per channel
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-9, torch.ones_like(std), std)
    feats = (feats - mean) / (std + 1e-12)
    
    return feats


def extract_eeg_rich_features(eeg_signal, n_time_steps, device=None):
    """
    Extract rich features from single-channel EEG signal.
    
    Features include:
    - Raw voltage (resampled)
    - First and second derivatives
    - Sliding window statistics (mean, std, max) for windows [3, 5, 9]
    - RMS for windows [5, 11]
    - Local FFT band energies for windows [8, 16, 32]
    
    Args:
        eeg_signal: torch.Tensor of shape (n_samples,) - raw EEG
        n_time_steps: int - target number of time steps
        device: torch.device, optional
    
    Returns:
        torch.Tensor of shape (n_time_steps, n_features) - rich features
    
    Example:
        >>> eeg = torch.randn(5000)
        >>> features = extract_eeg_rich_features(eeg, 100)
        >>> print(features.shape)  # torch.Size([100, 23])
    """
    if device is None:
        device = DEVICE
    
    sig = torch.as_tensor(eeg_signal, dtype=torch.float32, device=device)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    
    N = int(sig.shape[0])
    T = int(n_time_steps)
    
    if T <= 0:
        return torch.zeros((0, 1), device=device, dtype=torch.float32)
    
    # Resample
    if N != T:
        sig_3d = sig.view(1, 1, -1)
        sig_rs = F.interpolate(sig_3d, size=T, mode='linear', align_corners=True)
        sig_rs = sig_rs.view(T)
    else:
        sig_rs = sig
    
    features = []
    
    # 1. Raw signal
    features.append(sig_rs.view(T, 1))
    
    # 2. First derivative
    fd = torch.zeros(T, dtype=torch.float32, device=device)
    fd[1:] = sig_rs[1:] - sig_rs[:-1]
    features.append(fd.view(T, 1))
    
    # 3. Second derivative
    sd = torch.zeros(T, dtype=torch.float32, device=device)
    if T > 2:
        fdd = fd[1:] - fd[:-1]
        sd[2:] = fdd[1:]
    features.append(sd.view(T, 1))
    
    # 4. Sliding window statistics
    sig_conv = sig_rs.view(1, 1, T)
    ones_conv = torch.ones_like(sig_conv)
    
    for w in [3, 5, 9]:
        half = w // 2
        kernel = torch.ones((1, 1, w), dtype=torch.float32, device=device)
        
        # Mean
        sum_x = F.conv1d(sig_conv, kernel, padding=half)
        count = F.conv1d(ones_conv, kernel, padding=half)
        mean_feat = (sum_x / count).view(T)
        
        # Std
        sum_x2 = F.conv1d(sig_conv ** 2, kernel, padding=half)
        ex2 = sum_x2 / count
        std_feat = torch.sqrt(torch.clamp(ex2 - (sum_x / count) ** 2, min=0.0)).view(T)
        
        # Max
        sig_pad = F.pad(sig_conv, (half, half), mode='constant', value=float('-inf'))
        maxv = F.max_pool1d(sig_pad, kernel_size=w, stride=1).view(T)
        
        features.append(mean_feat.view(T, 1))
        features.append(std_feat.view(T, 1))
        features.append(maxv.view(T, 1))
    
    # 5. RMS
    for w in [5, 11]:
        half = w // 2
        kernel = torch.ones((1, 1, w), dtype=torch.float32, device=device)
        sum_x2 = F.conv1d(sig_conv ** 2, kernel, padding=half)
        count = F.conv1d(ones_conv, kernel, padding=half)
        rms = torch.sqrt(sum_x2 / count).view(T)
        features.append(rms.view(T, 1))
    
    # 6. Local FFT band energies
    for win in [8, 16, 32]:
        half = win // 2
        pad_len = (win - 1) // 2
        
        if pad_len >= T or T < 3:
            features.append(torch.zeros((T, 1), device=device))
            features.append(torch.zeros((T, 1), device=device))
            features.append(torch.zeros((T, 1), device=device))
            continue
        
        # Reflect padding
        sig_padded = F.pad(sig_rs.unsqueeze(0), (pad_len, pad_len), mode='reflect').squeeze(0)
        
        if sig_padded.shape[0] >= win:
            windows = sig_padded.unfold(0, win, 1)
            num_windows = windows.shape[0]
            
            if num_windows > T:
                windows = windows[:T]
            elif num_windows < T:
                pad_windows = torch.zeros((T - num_windows, win), device=device)
                windows = torch.cat([windows, pad_windows], dim=0)
            
            fr = torch.abs(torch.fft.rfft(windows, dim=1))
            L = fr.shape[1]
            lo = max(1, L // 4)
            mi = max(1, L // 2)
            
            band_low = fr[:, :lo].sum(dim=1)
            band_mid = fr[:, lo:mi].sum(dim=1)
            band_high = fr[:, mi:].sum(dim=1)
        else:
            band_low = torch.zeros(T, device=device)
            band_mid = torch.zeros(T, device=device)
            band_high = torch.zeros(T, device=device)
        
        features.append(band_low.view(T, 1))
        features.append(band_mid.view(T, 1))
        features.append(band_high.view(T, 1))
    
    # Concatenate all features
    Ft = torch.cat(features, dim=1)
    
    # Normalize per feature
    Ft = (Ft - Ft.mean(dim=0, keepdim=True)) / (Ft.std(dim=0, keepdim=True) + 1e-9)
    
    return Ft


if __name__ == "__main__":
    # Test the functions
    eeg_single = torch.randn(5000)
    eeg_multi = torch.randn(64, 5000)
    
    raw_features = extract_eeg_raw_aligned(eeg_single, 100)
    multi_features = extract_eeg_multichannel_aligned(eeg_multi, 100)
    rich_features = extract_eeg_rich_features(eeg_single, 100)
    
    print(f"Raw features shape: {raw_features.shape}")
    print(f"Multi-channel features shape: {multi_features.shape}")
    print(f"Rich features shape: {rich_features.shape}")

