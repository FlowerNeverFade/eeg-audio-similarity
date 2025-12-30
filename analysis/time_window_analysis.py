#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-Window RSA Analysis.

This module provides functions for computing RSA in sliding/fixed time windows
and generating scalp topography maps for specific time ranges (e.g., N400: 300-500ms).

Key analysis paradigm from the paper:
1. Segment EEG into time windows (e.g., 250ms)
2. Compute RSA between EEG and audio embeddings in each window
3. Generate scalp topography showing RSA values across electrodes for each window
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class TimeWindow:
    """Time window specification."""
    start_ms: float
    end_ms: float
    name: str = ""
    
    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms
    
    def to_samples(self, fs: float) -> Tuple[int, int]:
        """Convert to sample indices."""
        start_sample = int(self.start_ms * fs / 1000)
        end_sample = int(self.end_ms * fs / 1000)
        return start_sample, end_sample


# Predefined ERP time windows
ERP_WINDOWS = {
    'N100': TimeWindow(80, 150, 'N100'),
    'P200': TimeWindow(150, 250, 'P200'),
    'N200': TimeWindow(200, 300, 'N200'),
    'N400': TimeWindow(300, 500, 'N400'),
    'P300': TimeWindow(250, 500, 'P300'),
    'P600': TimeWindow(500, 800, 'P600'),
    'LPC': TimeWindow(500, 800, 'Late Positive Complex'),
}


def create_sliding_windows(total_duration_ms: float,
                           window_size_ms: float = 250.0,
                           step_size_ms: float = 50.0,
                           start_ms: float = 0.0) -> List[TimeWindow]:
    """
    Create sliding time windows.
    
    Args:
        total_duration_ms: float - total duration in milliseconds
        window_size_ms: float - window size in ms (default: 250ms)
        step_size_ms: float - step size in ms (default: 50ms)
        start_ms: float - start time in ms
    
    Returns:
        List[TimeWindow] - list of time windows
    
    Example:
        >>> windows = create_sliding_windows(1000, window_size_ms=250, step_size_ms=50)
        >>> print(f"Created {len(windows)} windows")
    """
    windows = []
    current_start = start_ms
    
    while current_start + window_size_ms <= total_duration_ms:
        win = TimeWindow(
            start_ms=current_start,
            end_ms=current_start + window_size_ms,
            name=f"{int(current_start)}-{int(current_start + window_size_ms)}ms"
        )
        windows.append(win)
        current_start += step_size_ms
    
    return windows


def extract_window_features(eeg: np.ndarray,
                            window: TimeWindow,
                            fs: float = 500.0) -> np.ndarray:
    """
    Extract EEG features for a specific time window.
    
    Args:
        eeg: np.ndarray - EEG data, shape (n_channels, n_samples) or (n_samples, n_channels)
        window: TimeWindow - time window specification
        fs: float - sampling frequency in Hz
    
    Returns:
        np.ndarray - extracted window data
    """
    start_sample, end_sample = window.to_samples(fs)
    
    if eeg.ndim == 2:
        if eeg.shape[0] > eeg.shape[1]:
            # (n_samples, n_channels)
            return eeg[start_sample:end_sample, :]
        else:
            # (n_channels, n_samples)
            return eeg[:, start_sample:end_sample].T
    else:
        return eeg[start_sample:end_sample]


def compute_window_rsa(eeg_window: np.ndarray,
                       audio_window: np.ndarray,
                       device: str = 'cuda') -> Tuple[float, float]:
    """
    Compute RSA for a single time window.
    
    Args:
        eeg_window: np.ndarray - EEG data for window, shape (T, D_eeg)
        audio_window: np.ndarray - audio data for window, shape (T, D_audio)
        device: str - computation device
    
    Returns:
        Tuple[float, float] - (spearman_rsa, pearson_rsa)
    """
    from ..metrics import compute_rdm_vec, rsa_between_rdms
    
    if eeg_window.shape[0] < 3 or audio_window.shape[0] < 3:
        return 0.0, 0.0
    
    rdm_eeg = compute_rdm_vec(eeg_window, device=device)
    rdm_audio = compute_rdm_vec(audio_window, device=device)
    
    spearman, _, pearson = rsa_between_rdms(rdm_eeg, rdm_audio, device=device)
    
    return spearman, pearson


def compute_electrode_window_rsa(eeg: np.ndarray,
                                 audio_features: np.ndarray,
                                 window: TimeWindow,
                                 eeg_fs: float = 500.0,
                                 audio_fs: float = 50.0,
                                 device: str = 'cuda') -> np.ndarray:
    """
    Compute RSA for each electrode in a specific time window.
    
    Args:
        eeg: np.ndarray - multi-channel EEG, shape (n_channels, n_samples)
        audio_features: np.ndarray - audio features, shape (T_audio, D)
        window: TimeWindow - time window specification
        eeg_fs: float - EEG sampling frequency in Hz
        audio_fs: float - audio feature frame rate in Hz
        device: str - computation device
    
    Returns:
        np.ndarray - RSA values for each electrode, shape (n_channels,)
    
    Example:
        >>> eeg = np.random.randn(64, 5000)  # 64 channels, 10s at 500Hz
        >>> audio = np.random.randn(500, 20)  # 10s at 50Hz
        >>> window = TimeWindow(300, 500, 'N400')
        >>> electrode_rsa = compute_electrode_window_rsa(eeg, audio, window)
        >>> print(electrode_rsa.shape)  # (64,)
    """
    from ..metrics import compute_rdm_vec, rsa_between_rdms
    from ..preprocessing import resample_to_target_length
    
    n_channels = eeg.shape[0]
    
    # Extract EEG window
    start_eeg, end_eeg = window.to_samples(eeg_fs)
    eeg_window = eeg[:, start_eeg:end_eeg]  # (n_channels, window_samples)
    
    # Extract corresponding audio window
    start_audio = int(window.start_ms * audio_fs / 1000)
    end_audio = int(window.end_ms * audio_fs / 1000)
    audio_window = audio_features[start_audio:end_audio, :]  # (T, D)
    
    if audio_window.shape[0] < 3:
        return np.zeros(n_channels)
    
    # Compute audio RDM
    rdm_audio = compute_rdm_vec(audio_window, device=device)
    
    # Compute RSA for each electrode
    rsa_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        ch_signal = eeg_window[ch, :]
        
        # Resample to match audio time steps
        T_audio = audio_window.shape[0]
        ch_resampled = resample_to_target_length(ch_signal, T_audio)
        ch_features = ch_resampled.reshape(-1, 1)  # (T, 1)
        
        if ch_features.shape[0] < 3:
            continue
        
        rdm_eeg = compute_rdm_vec(ch_features, device=device)
        spearman, _, _ = rsa_between_rdms(rdm_eeg, rdm_audio, device=device)
        rsa_values[ch] = spearman
    
    return rsa_values


def sliding_window_rsa_analysis(eeg: np.ndarray,
                                audio_features: np.ndarray,
                                window_size_ms: float = 250.0,
                                step_size_ms: float = 50.0,
                                eeg_fs: float = 500.0,
                                audio_fs: float = 50.0,
                                device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    Perform sliding window RSA analysis across all electrodes.
    
    Args:
        eeg: np.ndarray - multi-channel EEG, shape (n_channels, n_samples)
        audio_features: np.ndarray - audio features, shape (T_audio, D)
        window_size_ms: float - window size in ms
        step_size_ms: float - step size in ms
        eeg_fs: float - EEG sampling frequency
        audio_fs: float - audio frame rate
        device: str - computation device
    
    Returns:
        Dict with:
            - 'rsa_matrix': np.ndarray of shape (n_windows, n_channels)
            - 'window_centers': np.ndarray of window center times in ms
            - 'windows': List[TimeWindow]
    
    Example:
        >>> eeg = np.random.randn(64, 5000)
        >>> audio = np.random.randn(500, 20)
        >>> results = sliding_window_rsa_analysis(eeg, audio, window_size_ms=250)
        >>> print(results['rsa_matrix'].shape)  # (n_windows, 64)
    """
    n_channels = eeg.shape[0]
    duration_ms = eeg.shape[1] / eeg_fs * 1000
    
    # Create windows
    windows = create_sliding_windows(duration_ms, window_size_ms, step_size_ms)
    n_windows = len(windows)
    
    # Compute RSA for each window
    rsa_matrix = np.zeros((n_windows, n_channels))
    window_centers = np.zeros(n_windows)
    
    for i, window in enumerate(windows):
        rsa_values = compute_electrode_window_rsa(
            eeg, audio_features, window, eeg_fs, audio_fs, device
        )
        rsa_matrix[i, :] = rsa_values
        window_centers[i] = (window.start_ms + window.end_ms) / 2
    
    return {
        'rsa_matrix': rsa_matrix,
        'window_centers': window_centers,
        'windows': windows,
    }


def n400_analysis(eeg: np.ndarray,
                  audio_features: np.ndarray,
                  eeg_fs: float = 500.0,
                  audio_fs: float = 50.0,
                  n400_start_ms: float = 300.0,
                  n400_end_ms: float = 500.0,
                  device: str = 'cuda') -> Dict:
    """
    Perform N400 time window analysis.
    
    The N400 is an ERP component typically observed 300-500ms after stimulus onset,
    associated with semantic processing.
    
    Args:
        eeg: np.ndarray - multi-channel EEG, shape (n_channels, n_samples)
        audio_features: np.ndarray - audio features, shape (T_audio, D)
        eeg_fs: float - EEG sampling frequency
        audio_fs: float - audio frame rate
        n400_start_ms: float - N400 window start (default: 300ms)
        n400_end_ms: float - N400 window end (default: 500ms)
        device: str - computation device
    
    Returns:
        Dict with:
            - 'electrode_rsa': np.ndarray of RSA per electrode
            - 'mean_rsa': float - mean RSA across electrodes
            - 'window': TimeWindow specification
    
    Example:
        >>> results = n400_analysis(eeg, audio_features)
        >>> print(f"N400 mean RSA: {results['mean_rsa']:.4f}")
    """
    window = TimeWindow(n400_start_ms, n400_end_ms, 'N400')
    
    electrode_rsa = compute_electrode_window_rsa(
        eeg, audio_features, window, eeg_fs, audio_fs, device
    )
    
    return {
        'electrode_rsa': electrode_rsa,
        'mean_rsa': np.mean(electrode_rsa),
        'std_rsa': np.std(electrode_rsa),
        'max_rsa': np.max(electrode_rsa),
        'max_electrode': int(np.argmax(electrode_rsa)),
        'window': window,
    }


def erp_component_analysis(eeg: np.ndarray,
                           audio_features: np.ndarray,
                           components: List[str] = None,
                           eeg_fs: float = 500.0,
                           audio_fs: float = 50.0,
                           device: str = 'cuda') -> Dict[str, Dict]:
    """
    Analyze multiple ERP components (N100, P200, N400, etc.).
    
    Args:
        eeg: np.ndarray - multi-channel EEG
        audio_features: np.ndarray - audio features
        components: List[str] - component names (default: all)
        eeg_fs: float - EEG sampling frequency
        audio_fs: float - audio frame rate
        device: str - computation device
    
    Returns:
        Dict[str, Dict] - results for each component
    
    Example:
        >>> results = erp_component_analysis(eeg, audio, components=['N100', 'N400', 'P600'])
        >>> for comp, data in results.items():
        ...     print(f"{comp}: mean RSA = {data['mean_rsa']:.4f}")
    """
    if components is None:
        components = ['N100', 'P200', 'N400', 'P600']
    
    results = {}
    
    for comp_name in components:
        if comp_name not in ERP_WINDOWS:
            print(f"Warning: Unknown component {comp_name}")
            continue
        
        window = ERP_WINDOWS[comp_name]
        
        electrode_rsa = compute_electrode_window_rsa(
            eeg, audio_features, window, eeg_fs, audio_fs, device
        )
        
        results[comp_name] = {
            'electrode_rsa': electrode_rsa,
            'mean_rsa': np.mean(electrode_rsa),
            'std_rsa': np.std(electrode_rsa),
            'window': window,
        }
    
    return results


def plot_time_window_topography(rsa_values: np.ndarray,
                                window: TimeWindow,
                                out_path: str,
                                title: str = None,
                                vmin: float = None,
                                vmax: float = None,
                                dpi: int = 300):
    """
    Plot scalp topography for a specific time window.
    
    Args:
        rsa_values: np.ndarray - RSA values per electrode
        window: TimeWindow - time window specification
        out_path: str - output path
        title: str - plot title (auto-generated if None)
        vmin, vmax: float - colorbar limits
        dpi: int - figure resolution
    """
    from ..visualization import plot_topography_simple
    
    if title is None:
        title = f"RSA Topography: {window.name} ({int(window.start_ms)}-{int(window.end_ms)}ms)"
    
    plot_topography_simple(
        rsa_values, 
        out_path=out_path, 
        title=title,
        vmin=vmin,
        vmax=vmax,
        dpi=dpi
    )


def plot_sliding_window_topography_series(results: Dict,
                                          out_dir: str,
                                          step: int = 1,
                                          vmin: float = None,
                                          vmax: float = None,
                                          dpi: int = 200):
    """
    Generate a series of topography plots for sliding windows.
    
    Args:
        results: Dict - output from sliding_window_rsa_analysis
        out_dir: str - output directory
        step: int - plot every n-th window
        vmin, vmax: float - colorbar limits (auto if None)
        dpi: int - figure resolution
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    rsa_matrix = results['rsa_matrix']
    windows = results['windows']
    
    if vmin is None:
        vmin = np.percentile(rsa_matrix, 5)
    if vmax is None:
        vmax = np.percentile(rsa_matrix, 95)
    
    for i in range(0, len(windows), step):
        window = windows[i]
        rsa_values = rsa_matrix[i, :]
        
        out_path = os.path.join(out_dir, f"topo_{i:03d}_{int(window.start_ms)}_{int(window.end_ms)}ms.png")
        
        plot_time_window_topography(
            rsa_values, window, out_path,
            vmin=vmin, vmax=vmax, dpi=dpi
        )
    
    print(f"Saved {len(windows) // step} topography plots to {out_dir}")


def plot_rsa_time_course(results: Dict,
                         electrodes: List[int] = None,
                         out_path: str = None,
                         title: str = "RSA Time Course",
                         figsize: Tuple[int, int] = (12, 6),
                         dpi: int = 300):
    """
    Plot RSA time course across sliding windows.
    
    Args:
        results: Dict - output from sliding_window_rsa_analysis
        electrodes: List[int] - electrode indices to plot (None = mean)
        out_path: str - output path (None = show)
        title: str - plot title
        figsize: tuple - figure size
        dpi: int - figure resolution
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    rsa_matrix = results['rsa_matrix']
    window_centers = results['window_centers']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if electrodes is None:
        # Plot mean ± std across electrodes
        mean_rsa = np.mean(rsa_matrix, axis=1)
        std_rsa = np.std(rsa_matrix, axis=1)
        
        ax.plot(window_centers, mean_rsa, 'b-', linewidth=2, label='Mean RSA')
        ax.fill_between(window_centers, mean_rsa - std_rsa, mean_rsa + std_rsa,
                        alpha=0.3, color='blue', label='±1 SD')
    else:
        # Plot specific electrodes
        colors = plt.cm.tab10(np.linspace(0, 1, len(electrodes)))
        for elec, color in zip(electrodes, colors):
            ax.plot(window_centers, rsa_matrix[:, elec], '-', 
                    color=color, linewidth=1.5, label=f'E{elec}')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('RSA (Spearman)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Mark ERP components
    for name, window in ERP_WINDOWS.items():
        if name in ['N100', 'N400', 'P600']:
            center = (window.start_ms + window.end_ms) / 2
            if window_centers.min() <= center <= window_centers.max():
                ax.axvspan(window.start_ms, window.end_ms, alpha=0.1, color='gray')
                ax.text(center, ax.get_ylim()[1] * 0.95, name, 
                        ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        return fig


if __name__ == "__main__":
    # Test time window analysis
    np.random.seed(42)
    
    # Simulate data
    n_channels = 64
    n_samples = 5000  # 10 seconds at 500Hz
    eeg = np.random.randn(n_channels, n_samples)
    audio = np.random.randn(500, 20)  # 10 seconds at 50Hz
    
    # Test N400 analysis
    print("Testing N400 analysis...")
    n400_results = n400_analysis(eeg, audio, eeg_fs=500, audio_fs=50, device='cpu')
    print(f"N400 mean RSA: {n400_results['mean_rsa']:.4f}")
    
    # Test sliding window
    print("\nTesting sliding window analysis...")
    windows = create_sliding_windows(1000, window_size_ms=250, step_size_ms=50)
    print(f"Created {len(windows)} windows")
    for w in windows[:3]:
        print(f"  {w.name}")

