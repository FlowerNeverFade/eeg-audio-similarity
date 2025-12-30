#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prosody Feature Extraction and Analysis.

This module provides functions for extracting prosodic features
from audio signals and analyzing their relationship with EEG and LLM representations.
"""

import numpy as np


def extract_prosody_features(audio, sample_rate=16000, frame_size=512, hop_size=160):
    """
    Extract prosodic features from audio signal.
    
    Features include:
    - Fundamental frequency (F0)
    - Energy/intensity
    - Zero crossing rate
    - Spectral features (centroid, bandwidth, rolloff)
    
    Args:
        audio: np.ndarray - audio waveform
        sample_rate: int - sampling rate in Hz
        frame_size: int - frame size for analysis
        hop_size: int - hop size between frames
    
    Returns:
        dict: Dictionary of prosodic features, each (n_frames, 1) or (n_frames, n_features)
    
    Example:
        >>> audio = np.random.randn(16000)  # 1 second of audio
        >>> features = extract_prosody_features(audio)
        >>> print(features.keys())
    """
    n_frames = (len(audio) - frame_size) // hop_size + 1
    
    if n_frames <= 0:
        return {
            'f0': np.zeros((1, 1)),
            'energy': np.zeros((1, 1)),
            'zcr': np.zeros((1, 1)),
            'spectral_centroid': np.zeros((1, 1)),
            'spectral_bandwidth': np.zeros((1, 1)),
            'spectral_rolloff': np.zeros((1, 1)),
        }
    
    # Initialize feature arrays
    f0 = np.zeros(n_frames)
    energy = np.zeros(n_frames)
    zcr = np.zeros(n_frames)
    spectral_centroid = np.zeros(n_frames)
    spectral_bandwidth = np.zeros(n_frames)
    spectral_rolloff = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        frame = audio[start:end]
        
        # Energy
        energy[i] = np.sum(frame ** 2) / frame_size
        
        # Zero crossing rate
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
        
        # FFT-based features
        fft = np.fft.rfft(frame * np.hanning(frame_size))
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
        
        if magnitude.sum() > 1e-10:
            # Spectral centroid
            spectral_centroid[i] = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            # Spectral bandwidth
            spectral_bandwidth[i] = np.sqrt(
                np.sum(((freqs - spectral_centroid[i]) ** 2) * magnitude) / np.sum(magnitude)
            )
            
            # Spectral rolloff (85th percentile)
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            spectral_rolloff[i] = freqs[min(rolloff_idx, len(freqs) - 1)]
        
        # Simple F0 estimation using autocorrelation
        f0[i] = estimate_f0_autocorr(frame, sample_rate)
    
    return {
        'f0': f0.reshape(-1, 1),
        'energy': energy.reshape(-1, 1),
        'zcr': zcr.reshape(-1, 1),
        'spectral_centroid': spectral_centroid.reshape(-1, 1),
        'spectral_bandwidth': spectral_bandwidth.reshape(-1, 1),
        'spectral_rolloff': spectral_rolloff.reshape(-1, 1),
    }


def estimate_f0_autocorr(frame, sample_rate, f0_min=75, f0_max=500):
    """
    Estimate fundamental frequency using autocorrelation.
    
    Args:
        frame: np.ndarray - audio frame
        sample_rate: int - sampling rate
        f0_min: float - minimum F0 in Hz
        f0_max: float - maximum F0 in Hz
    
    Returns:
        float: Estimated F0 in Hz (0 if unvoiced)
    """
    # Autocorrelation
    frame = frame - np.mean(frame)
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    
    # Find lag range
    min_lag = int(sample_rate / f0_max)
    max_lag = int(sample_rate / f0_min)
    max_lag = min(max_lag, len(autocorr) - 1)
    
    if min_lag >= max_lag:
        return 0.0
    
    # Find peak in valid range
    search_range = autocorr[min_lag:max_lag + 1]
    if len(search_range) == 0 or autocorr[0] < 1e-10:
        return 0.0
    
    peak_idx = np.argmax(search_range) + min_lag
    
    # Check if voiced (peak should be significant)
    if autocorr[peak_idx] / autocorr[0] < 0.3:
        return 0.0
    
    return sample_rate / peak_idx


def compute_prosody_rdm(prosody_features, metric='correlation'):
    """
    Compute RDM from prosody features.
    
    Args:
        prosody_features: dict or np.ndarray - prosody features
        metric: str - distance metric ('correlation' or 'euclidean')
    
    Returns:
        np.ndarray - RDM matrix (n_frames, n_frames)
    """
    if isinstance(prosody_features, dict):
        # Concatenate all features
        features = np.hstack([v for v in prosody_features.values()])
    else:
        features = prosody_features
    
    n_frames = features.shape[0]
    rdm = np.zeros((n_frames, n_frames))
    
    if metric == 'correlation':
        # Center features
        features_centered = features - features.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(features_centered, axis=1, keepdims=True) + 1e-12
        features_normalized = features_centered / norms
        
        # Correlation matrix
        corr = features_normalized @ features_normalized.T
        rdm = 1 - corr
    else:  # euclidean
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                dist = np.linalg.norm(features[i] - features[j])
                rdm[i, j] = dist
                rdm[j, i] = dist
    
    return rdm


def compute_tnc(rdm_prosody, rdm_eeg, rdm_llm):
    """
    Compute Triangulated Neuro-Computational (TNC) similarity.
    
    TNC measures how well the LLM representation mediates the relationship
    between acoustic features and neural responses.
    
    Args:
        rdm_prosody: np.ndarray - prosody/acoustic RDM
        rdm_eeg: np.ndarray - EEG RDM
        rdm_llm: np.ndarray - LLM embedding RDM
    
    Returns:
        dict: TNC metrics including partial correlations
    """
    from scipy.stats import spearmanr
    
    # Extract upper triangular
    triu_idx = np.triu_indices(rdm_prosody.shape[0], k=1)
    
    prosody_vec = rdm_prosody[triu_idx]
    eeg_vec = rdm_eeg[triu_idx]
    llm_vec = rdm_llm[triu_idx]
    
    # Direct correlations
    r_prosody_eeg, _ = spearmanr(prosody_vec, eeg_vec)
    r_prosody_llm, _ = spearmanr(prosody_vec, llm_vec)
    r_llm_eeg, _ = spearmanr(llm_vec, eeg_vec)
    
    # Partial correlation: prosody-EEG controlling for LLM
    def partial_correlation(x, y, z):
        """Compute partial correlation of x and y controlling for z."""
        r_xy, _ = spearmanr(x, y)
        r_xz, _ = spearmanr(x, z)
        r_yz, _ = spearmanr(y, z)
        
        numerator = r_xy - r_xz * r_yz
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    r_prosody_eeg_partial = partial_correlation(prosody_vec, eeg_vec, llm_vec)
    
    # TNC score: how much variance in EEG-prosody relationship is explained by LLM
    if abs(r_prosody_eeg) > 1e-10:
        tnc_score = 1 - (r_prosody_eeg_partial ** 2) / (r_prosody_eeg ** 2)
    else:
        tnc_score = 0.0
    
    return {
        'r_prosody_eeg': r_prosody_eeg,
        'r_prosody_llm': r_prosody_llm,
        'r_llm_eeg': r_llm_eeg,
        'r_prosody_eeg_partial': r_prosody_eeg_partial,
        'tnc_score': tnc_score,
    }


if __name__ == "__main__":
    # Test the functions
    audio = np.random.randn(16000) * 0.1  # 1 second of noise
    
    features = extract_prosody_features(audio)
    print("Prosody features extracted:")
    for k, v in features.items():
        print(f"  {k}: shape {v.shape}")
    
    rdm = compute_prosody_rdm(features)
    print(f"\nProsody RDM shape: {rdm.shape}")
    
    # Test TNC with synthetic data
    n = 50
    rdm_prosody = np.random.rand(n, n)
    rdm_prosody = (rdm_prosody + rdm_prosody.T) / 2
    np.fill_diagonal(rdm_prosody, 0)
    
    rdm_eeg = rdm_prosody + 0.5 * np.random.rand(n, n)
    rdm_eeg = (rdm_eeg + rdm_eeg.T) / 2
    np.fill_diagonal(rdm_eeg, 0)
    
    rdm_llm = rdm_prosody + 0.3 * np.random.rand(n, n)
    rdm_llm = (rdm_llm + rdm_llm.T) / 2
    np.fill_diagonal(rdm_llm, 0)
    
    tnc_results = compute_tnc(rdm_prosody, rdm_eeg, rdm_llm)
    print("\nTNC results:")
    for k, v in tnc_results.items():
        print(f"  {k}: {v:.4f}")

