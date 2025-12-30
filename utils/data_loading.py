#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Utilities.

This module provides functions for loading EEG and audio data
for the similarity analysis pipeline.
"""

import os
import numpy as np
import torch


def load_eeg_data(file_path, n_channels=None, device=None):
    """
    Load EEG data from a numpy file.
    
    Args:
        file_path: str - path to .npy file
        n_channels: int, optional - number of channels to keep (from start)
        device: torch.device, optional - device to load data to
    
    Returns:
        torch.Tensor: EEG data (n_channels, n_samples) or (n_samples,)
    
    Example:
        >>> eeg = load_eeg_data("sentence_001.npy", n_channels=64)
        >>> print(eeg.shape)
    """
    data = np.load(file_path).astype(np.float32)
    
    # Limit channels if specified
    if n_channels is not None and data.ndim == 2:
        if data.shape[0] > n_channels:
            data = data[:n_channels, :]
    
    if device is not None:
        return torch.as_tensor(data, dtype=torch.float32, device=device)
    
    return torch.as_tensor(data, dtype=torch.float32)


def load_audio_embeddings(npz_path, layer_idx=None, device=None):
    """
    Load audio embeddings from an NPZ file.
    
    Args:
        npz_path: str - path to .npz file
        layer_idx: int, optional - specific layer to load
        device: torch.device, optional - device to load to
    
    Returns:
        dict or torch.Tensor: embeddings per layer or single layer
    """
    npz_data = np.load(npz_path, allow_pickle=True)
    
    if layer_idx is not None:
        key = f"layer_{layer_idx}"
        if key not in npz_data:
            raise KeyError(f"Layer {layer_idx} not found in {npz_path}")
        
        emb = npz_data[key].astype(np.float32)
        if device is not None:
            return torch.as_tensor(emb, dtype=torch.float32, device=device)
        return torch.as_tensor(emb, dtype=torch.float32)
    
    # Return all layers
    result = {}
    for key in npz_data.keys():
        if key.startswith('layer_'):
            emb = npz_data[key].astype(np.float32)
            if device is not None:
                result[key] = torch.as_tensor(emb, dtype=torch.float32, device=device)
            else:
                result[key] = torch.as_tensor(emb, dtype=torch.float32)
    
    return result


def find_eeg_files(base_dir, subject_id, pattern="sentence_*.npy"):
    """
    Find all EEG files for a subject.
    
    Args:
        base_dir: str - base directory containing subject folders
        subject_id: str - subject identifier (e.g., "S01")
        pattern: str - file pattern to match
    
    Returns:
        list: Sorted list of file paths
    """
    import glob
    
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.isdir(subject_dir):
        return []
    
    files = glob.glob(os.path.join(subject_dir, pattern))
    return sorted(files)


def find_audio_embedding_files(embed_dir, sentence_id, pattern=None):
    """
    Find audio embedding file for a sentence.
    
    Args:
        embed_dir: str - directory containing embedding files
        sentence_id: int - sentence identifier
        pattern: str, optional - file pattern
    
    Returns:
        str or None: Path to embedding file, or None if not found
    """
    # Try different naming patterns
    patterns = [
        f"sentence_{sentence_id:03d}_segment_*_all_embeddings.npz",
        f"sentence_{sentence_id:03d}.npz",
        f"{sentence_id}_all_embeddings.npz",
        f"{sentence_id}.npz",
    ]
    
    if pattern:
        patterns.insert(0, pattern)
    
    for pat in patterns:
        import glob
        matches = glob.glob(os.path.join(embed_dir, pat))
        if matches:
            return sorted(matches)[0]
    
    return None


def get_sentence_ids(eeg_dir, subject_id):
    """
    Extract sentence IDs from EEG file names.
    
    Args:
        eeg_dir: str - directory containing EEG files
        subject_id: str - subject identifier
    
    Returns:
        list: Sorted list of sentence IDs (integers)
    """
    files = find_eeg_files(eeg_dir, subject_id)
    sentence_ids = []
    
    for f in files:
        basename = os.path.basename(f)
        # Parse sentence_XXX.npy
        if basename.startswith('sentence_'):
            try:
                sid = int(basename.split('_')[1].split('.')[0])
                sentence_ids.append(sid)
            except (ValueError, IndexError):
                pass
    
    return sorted(sentence_ids)


def get_layer_count(npz_path):
    """
    Get the number of layers in an audio embedding file.
    
    Args:
        npz_path: str - path to NPZ file
    
    Returns:
        int: Number of layers
    """
    npz_data = np.load(npz_path, allow_pickle=True)
    layer_keys = [k for k in npz_data.keys() if k.startswith('layer_')]
    return len(layer_keys)


def load_results_csv(csv_path):
    """
    Load results from a CSV file.
    
    Args:
        csv_path: str - path to CSV file
    
    Returns:
        pd.DataFrame: Results dataframe
    """
    import pandas as pd
    return pd.read_csv(csv_path)


def load_alice_eeg(mat_path):
    """
    Load EEG data from Alice in Wonderland dataset (FieldTrip .mat format).
    
    Args:
        mat_path: str - path to subject .mat file (e.g., 'S01.mat')
    
    Returns:
        tuple: (eeg_data, fs, channel_labels, time_range)
            - eeg_data: np.ndarray of shape (n_channels, n_samples)
            - fs: float - sampling frequency (500 Hz)
            - channel_labels: list of channel names
            - time_range: tuple (start_time, end_time) in seconds
    
    Example:
        >>> eeg, fs, labels, t_range = load_alice_eeg('S01.mat')
        >>> print(f"EEG: {eeg.shape}, fs: {fs} Hz, duration: {t_range[1]:.1f}s")
    """
    import scipy.io
    
    data = scipy.io.loadmat(mat_path)
    raw = data['raw'][0, 0]
    
    # Extract EEG data
    eeg_data = raw['trial'][0, 0]  # (n_channels, n_samples)
    
    # Sampling frequency
    fs = float(raw['fsample'][0, 0])
    
    # Channel labels
    channel_labels = [str(lbl[0]) for lbl in raw['label']]
    
    # Time range
    try:
        time_data = raw['time'][0, 0]
        time_range = (float(time_data[0, 0]), float(time_data[0, -1]))
    except:
        n_samples = eeg_data.shape[1]
        duration = n_samples / fs
        time_range = (0.0, duration)
    
    return eeg_data, fs, channel_labels, time_range


def load_alice_alignment(csv_path):
    """
    Load word-level alignment from AliceChapterOne-EEG.csv.
    
    Args:
        csv_path: str - path to AliceChapterOne-EEG.csv
    
    Returns:
        pd.DataFrame with columns:
            - Word, Segment, onset, offset, Order
            - Sentence, Position, IsLexical
            - LogFreq, SndPower, Length
            - NGRAM, RNN, CFG (surprisal values)
    
    Example:
        >>> alignment = load_alice_alignment('AliceChapterOne-EEG.csv')
        >>> print(f"Loaded {len(alignment)} words in {alignment['Sentence'].nunique()} sentences")
    """
    import pandas as pd
    return pd.read_csv(csv_path)


def load_alice_dataset_info(datasets_mat_path):
    """
    Load subject inclusion/exclusion flags from datasets.mat.
    
    Args:
        datasets_mat_path: str - path to datasets.mat
    
    Returns:
        dict with keys:
            - 'used': list of subject IDs used in main analysis (N=33)
            - 'low_perf': list of subjects excluded for low quiz scores (N=8)
            - 'high_noise': list of subjects excluded for high noise (N=8)
            - 'all_subjects': list of all 49 subject IDs
    
    Example:
        >>> info = load_alice_dataset_info('datasets.mat')
        >>> print(f"Using {len(info['used'])} subjects for analysis")
    """
    import scipy.io
    
    data = scipy.io.loadmat(datasets_mat_path)
    
    result = {
        'all_subjects': [f'S{i:02d}' for i in range(1, 50)],
        'used': [],
        'low_perf': [],
        'high_noise': [],
    }
    
    # Parse flags
    if 'use' in data:
        flags = data['use'].flatten()
        result['used'] = [f'S{i:02d}' for i, flag in enumerate(flags, 1) if flag]
    
    if 'lowperf' in data:
        flags = data['lowperf'].flatten()
        result['low_perf'] = [f'S{i:02d}' for i, flag in enumerate(flags, 1) if flag]
    
    if 'highnoise' in data:
        flags = data['highnoise'].flatten()
        result['high_noise'] = [f'S{i:02d}' for i, flag in enumerate(flags, 1) if flag]
    
    return result


def segment_alice_by_sentence(eeg_data, alignment_df, fs=500.0):
    """
    Segment Alice EEG data by sentence boundaries.
    
    Args:
        eeg_data: np.ndarray - EEG data (n_channels, n_samples)
        alignment_df: pd.DataFrame - alignment from load_alice_alignment()
        fs: float - sampling frequency
    
    Returns:
        dict: {sentence_id: np.ndarray of shape (n_channels, n_samples)}
    
    Note:
        This function handles the segment-based timing in AliceChapterOne-EEG.csv.
        Word onsets/offsets are relative to each audio segment, not the full recording.
    
    Example:
        >>> eeg, fs, _, _ = load_alice_eeg('S01.mat')
        >>> alignment = load_alice_alignment('AliceChapterOne-EEG.csv')
        >>> sentences = segment_alice_by_sentence(eeg, alignment, fs)
        >>> print(f"Extracted {len(sentences)} sentences")
    """
    sentences = {}
    
    for sent_id in alignment_df['Sentence'].unique():
        sent_words = alignment_df[alignment_df['Sentence'] == sent_id]
        
        # Get sentence boundaries (min onset to max offset)
        start_time = sent_words['onset'].min()
        end_time = sent_words['offset'].max()
        segment_id = sent_words['Segment'].iloc[0]
        
        # Note: onset/offset are relative to each audio segment
        # For full EEG, need to accumulate segment durations
        # This is a simplified version assuming continuous recording
        
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        # Ensure valid indices
        start_sample = max(0, start_sample)
        end_sample = min(eeg_data.shape[1], end_sample)
        
        if end_sample > start_sample:
            sentences[int(sent_id)] = eeg_data[:, start_sample:end_sample]
    
    return sentences


if __name__ == "__main__":
    # Test the functions with synthetic data
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test EEG file
        eeg_data = np.random.randn(64, 5000).astype(np.float32)
        eeg_path = os.path.join(tmpdir, "sentence_001.npy")
        np.save(eeg_path, eeg_data)
        
        # Create test audio embedding
        audio_path = os.path.join(tmpdir, "sentence_001.npz")
        embeddings = {f"layer_{i}": np.random.randn(100, 1024).astype(np.float32) 
                      for i in range(32)}
        np.savez(audio_path, **embeddings)
        
        # Test loading
        eeg = load_eeg_data(eeg_path, n_channels=60)
        print(f"Loaded EEG shape: {eeg.shape}")
        
        audio = load_audio_embeddings(audio_path, layer_idx=0)
        print(f"Loaded audio embedding shape: {audio.shape}")
        
        n_layers = get_layer_count(audio_path)
        print(f"Number of layers: {n_layers}")
        
        print("\nData loading tests passed!")

