#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Preprocessing Pipeline.

This module provides functions for preprocessing raw EEG data:
- Bandpass filtering (0.5-40 Hz typical for speech processing)
- Notch filtering (50/60 Hz line noise removal)
- Baseline removal
- Re-referencing (average reference)
- Bad channel detection and interpolation
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional
import warnings


def bandpass_filter(eeg: np.ndarray, 
                    fs: float = 500.0,
                    low_freq: float = 0.5,
                    high_freq: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples) or (n_samples,)
        fs: float - sampling frequency in Hz
        low_freq: float - low cutoff frequency in Hz
        high_freq: float - high cutoff frequency in Hz
        order: int - filter order
    
    Returns:
        np.ndarray - filtered EEG data
    
    Example:
        >>> eeg_raw = np.random.randn(64, 5000)  # 64 channels, 10 seconds at 500Hz
        >>> eeg_filtered = bandpass_filter(eeg_raw, fs=500, low_freq=0.5, high_freq=40)
    """
    nyq = fs / 2.0
    low = low_freq / nyq
    high = high_freq / nyq
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    if eeg.ndim == 1:
        return signal.filtfilt(b, a, eeg)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in eeg])


def notch_filter(eeg: np.ndarray,
                 fs: float = 500.0,
                 notch_freq: float = 50.0,
                 quality_factor: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove line noise.
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples) or (n_samples,)
        fs: float - sampling frequency in Hz
        notch_freq: float - frequency to notch out (50 or 60 Hz)
        quality_factor: float - quality factor of the notch filter
    
    Returns:
        np.ndarray - filtered EEG data
    
    Example:
        >>> eeg_notched = notch_filter(eeg_raw, fs=500, notch_freq=50)  # Europe
        >>> eeg_notched = notch_filter(eeg_raw, fs=500, notch_freq=60)  # US
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    
    if eeg.ndim == 1:
        return signal.filtfilt(b, a, eeg)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in eeg])


def remove_baseline(eeg: np.ndarray,
                    baseline_samples: Optional[int] = None) -> np.ndarray:
    """
    Remove baseline (mean) from EEG data.
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples) or (n_samples,)
        baseline_samples: int, optional - number of samples to use for baseline
                         If None, uses entire signal mean
    
    Returns:
        np.ndarray - baseline-corrected EEG data
    """
    if eeg.ndim == 1:
        if baseline_samples is None:
            return eeg - np.mean(eeg)
        else:
            return eeg - np.mean(eeg[:baseline_samples])
    else:
        if baseline_samples is None:
            return eeg - np.mean(eeg, axis=1, keepdims=True)
        else:
            return eeg - np.mean(eeg[:, :baseline_samples], axis=1, keepdims=True)


def rereference_average(eeg: np.ndarray,
                        exclude_channels: Optional[List[int]] = None) -> np.ndarray:
    """
    Re-reference EEG to average reference.
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples)
        exclude_channels: list of int, optional - channel indices to exclude from average
    
    Returns:
        np.ndarray - re-referenced EEG data
    
    Example:
        >>> eeg_reref = rereference_average(eeg_raw, exclude_channels=[0, 1])  # Exclude EOG
    """
    if eeg.ndim == 1:
        return eeg - np.mean(eeg)
    
    if exclude_channels is None:
        ref = np.mean(eeg, axis=0)
    else:
        mask = np.ones(eeg.shape[0], dtype=bool)
        mask[exclude_channels] = False
        ref = np.mean(eeg[mask], axis=0)
    
    return eeg - ref


def detect_bad_channels(eeg: np.ndarray,
                        fs: float = 500.0,
                        std_threshold: float = 3.0,
                        correlation_threshold: float = 0.4,
                        flat_threshold: float = 1e-6) -> List[int]:
    """
    Detect bad EEG channels based on statistical criteria.
    
    Criteria:
    - Channels with abnormally high/low variance
    - Channels with low correlation to neighbors
    - Flat (constant) channels
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples)
        fs: float - sampling frequency
        std_threshold: float - z-score threshold for variance
        correlation_threshold: float - minimum correlation with average
        flat_threshold: float - minimum std for non-flat channel
    
    Returns:
        List[int] - indices of bad channels
    """
    n_channels = eeg.shape[0]
    bad_channels = []
    
    # Compute channel statistics
    channel_stds = np.std(eeg, axis=1)
    channel_means = np.mean(eeg, axis=1)
    
    # Z-score of standard deviations
    std_zscore = (channel_stds - np.median(channel_stds)) / (np.std(channel_stds) + 1e-12)
    
    # Average reference for correlation
    avg_ref = np.mean(eeg, axis=0)
    correlations = np.array([np.corrcoef(eeg[i], avg_ref)[0, 1] for i in range(n_channels)])
    
    for i in range(n_channels):
        # Flat channel
        if channel_stds[i] < flat_threshold:
            bad_channels.append(i)
            continue
        
        # Abnormal variance
        if np.abs(std_zscore[i]) > std_threshold:
            bad_channels.append(i)
            continue
        
        # Low correlation (might be disconnected)
        if np.abs(correlations[i]) < correlation_threshold:
            bad_channels.append(i)
            continue
    
    return sorted(set(bad_channels))


def interpolate_bad_channels(eeg: np.ndarray,
                              bad_channels: List[int],
                              channel_positions: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Interpolate bad channels using neighboring channels.
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples)
        bad_channels: list of int - indices of bad channels
        channel_positions: np.ndarray of shape (n_channels, 2 or 3), optional
                          If provided, uses distance-weighted interpolation
    
    Returns:
        np.ndarray - EEG with interpolated channels
    """
    if not bad_channels:
        return eeg.copy()
    
    eeg_interp = eeg.copy()
    n_channels = eeg.shape[0]
    good_channels = [i for i in range(n_channels) if i not in bad_channels]
    
    if len(good_channels) == 0:
        warnings.warn("No good channels available for interpolation")
        return eeg_interp
    
    if channel_positions is not None:
        # Distance-weighted interpolation
        for bad_idx in bad_channels:
            distances = np.linalg.norm(
                channel_positions[good_channels] - channel_positions[bad_idx], 
                axis=1
            )
            weights = 1.0 / (distances + 1e-6)
            weights /= weights.sum()
            eeg_interp[bad_idx] = np.sum(
                eeg[good_channels] * weights[:, np.newaxis], 
                axis=0
            )
    else:
        # Simple average of all good channels
        for bad_idx in bad_channels:
            eeg_interp[bad_idx] = np.mean(eeg[good_channels], axis=0)
    
    return eeg_interp


def preprocess_eeg(eeg: np.ndarray,
                   fs: float = 500.0,
                   low_freq: float = 0.5,
                   high_freq: float = 40.0,
                   notch_freq: Optional[float] = 50.0,
                   do_rereference: bool = True,
                   detect_bad: bool = True,
                   interpolate_bad: bool = True,
                   channel_positions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[int]]:
    """
    Full EEG preprocessing pipeline.
    
    Pipeline:
    1. Bandpass filter (0.5-40 Hz)
    2. Notch filter (50/60 Hz)
    3. Baseline removal
    4. Bad channel detection
    5. Bad channel interpolation
    6. Average re-reference
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples)
        fs: float - sampling frequency in Hz
        low_freq: float - bandpass low cutoff
        high_freq: float - bandpass high cutoff
        notch_freq: float, optional - notch frequency (None to skip)
        do_rereference: bool - whether to apply average reference
        detect_bad: bool - whether to detect bad channels
        interpolate_bad: bool - whether to interpolate bad channels
        channel_positions: np.ndarray, optional - for distance-weighted interpolation
    
    Returns:
        Tuple[np.ndarray, List[int]] - preprocessed EEG and list of bad channels
    
    Example:
        >>> eeg_raw = np.random.randn(64, 5000)
        >>> eeg_clean, bad_chs = preprocess_eeg(eeg_raw, fs=500)
        >>> print(f"Detected {len(bad_chs)} bad channels")
    """
    # Step 1: Bandpass filter
    eeg_filt = bandpass_filter(eeg, fs, low_freq, high_freq)
    
    # Step 2: Notch filter
    if notch_freq is not None:
        eeg_filt = notch_filter(eeg_filt, fs, notch_freq)
    
    # Step 3: Baseline removal
    eeg_filt = remove_baseline(eeg_filt)
    
    # Step 4: Bad channel detection
    bad_channels = []
    if detect_bad:
        bad_channels = detect_bad_channels(eeg_filt, fs)
    
    # Step 5: Interpolate bad channels
    if interpolate_bad and bad_channels:
        eeg_filt = interpolate_bad_channels(eeg_filt, bad_channels, channel_positions)
    
    # Step 6: Re-reference
    if do_rereference:
        eeg_filt = rereference_average(eeg_filt)
    
    return eeg_filt, bad_channels


if __name__ == "__main__":
    # Test preprocessing pipeline
    np.random.seed(42)
    
    # Simulate EEG data
    fs = 500
    duration = 10  # seconds
    n_channels = 64
    n_samples = int(fs * duration)
    
    # Create synthetic EEG with some artifacts
    eeg_raw = np.random.randn(n_channels, n_samples) * 50  # μV scale
    
    # Add 50Hz noise
    t = np.arange(n_samples) / fs
    eeg_raw += 10 * np.sin(2 * np.pi * 50 * t)
    
    # Make one channel bad (flat)
    eeg_raw[5] = 0.0
    
    # Preprocess
    eeg_clean, bad_chs = preprocess_eeg(eeg_raw, fs=fs)
    
    print(f"Input shape: {eeg_raw.shape}")
    print(f"Output shape: {eeg_clean.shape}")
    print(f"Detected bad channels: {bad_chs}")

