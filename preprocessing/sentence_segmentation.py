#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence Segmentation for EEG and Audio.

This module provides functions for segmenting continuous EEG and audio
data into sentence-level segments based on alignment tables.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json


def load_alignment_table(alignment_path: str) -> pd.DataFrame:
    """
    Load sentence alignment table from file.
    
    Supported formats:
    - CSV with columns: sentence_id, start_time, end_time, text
    - JSON with list of {sentence_id, start, end, text}
    - TextGrid (Praat format)
    
    Args:
        alignment_path: str - path to alignment file
    
    Returns:
        pd.DataFrame with columns: sentence_id, start_time, end_time, text
    
    Example:
        >>> alignment = load_alignment_table('alignment.csv')
        >>> print(alignment.head())
    """
    path = Path(alignment_path)
    
    if path.suffix == '.csv':
        df = pd.read_csv(alignment_path)
        # Normalize column names
        col_map = {
            'id': 'sentence_id',
            'start': 'start_time',
            'end': 'end_time',
            'sentence': 'text',
            'content': 'text',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
    elif path.suffix == '.json':
        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            # Handle nested format
            sentences = data.get('sentences', data.get('segments', []))
            df = pd.DataFrame(sentences)
        
        # Normalize column names
        col_map = {
            'id': 'sentence_id',
            'start': 'start_time',
            'end': 'end_time',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
    elif path.suffix == '.TextGrid':
        df = _parse_textgrid(alignment_path)
        
    else:
        raise ValueError(f"Unsupported alignment file format: {path.suffix}")
    
    # Ensure required columns exist
    required = ['start_time', 'end_time']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add sentence_id if missing
    if 'sentence_id' not in df.columns:
        df['sentence_id'] = range(len(df))
    
    # Sort by start time
    df = df.sort_values('start_time').reset_index(drop=True)
    
    return df


def _parse_textgrid(textgrid_path: str) -> pd.DataFrame:
    """Parse Praat TextGrid file (simplified parser)."""
    rows = []
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple regex-free parsing
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if 'xmin' in line.lower() and '=' in line:
            try:
                xmin = float(line.split('=')[1].strip())
                i += 1
                xmax = float(lines[i].split('=')[1].strip())
                i += 1
                text_line = lines[i]
                if 'text' in text_line.lower():
                    text = text_line.split('=')[1].strip().strip('"')
                    if text:  # Non-empty interval
                        rows.append({
                            'start_time': xmin,
                            'end_time': xmax,
                            'text': text,
                        })
            except (ValueError, IndexError):
                pass
        i += 1
    
    df = pd.DataFrame(rows)
    df['sentence_id'] = range(len(df))
    return df


def get_sentence_boundaries(alignment: pd.DataFrame,
                            fs: float = 500.0) -> List[Tuple[int, int]]:
    """
    Convert time-based boundaries to sample indices.
    
    Args:
        alignment: pd.DataFrame with start_time, end_time columns (in seconds)
        fs: float - sampling frequency in Hz
    
    Returns:
        List of (start_sample, end_sample) tuples
    
    Example:
        >>> alignment = load_alignment_table('alignment.csv')
        >>> boundaries = get_sentence_boundaries(alignment, fs=500)
        >>> for start, end in boundaries[:3]:
        ...     print(f"Samples {start} to {end}")
    """
    boundaries = []
    
    for _, row in alignment.iterrows():
        start_sample = int(row['start_time'] * fs)
        end_sample = int(row['end_time'] * fs)
        boundaries.append((start_sample, end_sample))
    
    return boundaries


def segment_eeg_by_sentences(eeg: np.ndarray,
                              alignment: pd.DataFrame,
                              fs: float = 500.0,
                              min_duration: float = 0.1) -> Dict[int, np.ndarray]:
    """
    Segment continuous EEG into sentence-level segments.
    
    Args:
        eeg: np.ndarray of shape (n_channels, n_samples) or (n_samples,)
        alignment: pd.DataFrame with start_time, end_time, sentence_id
        fs: float - EEG sampling frequency in Hz
        min_duration: float - minimum segment duration in seconds
    
    Returns:
        Dict[int, np.ndarray] - {sentence_id: eeg_segment}
    
    Example:
        >>> eeg_raw = np.random.randn(64, 50000)  # 100 seconds at 500Hz
        >>> alignment = load_alignment_table('alignment.csv')
        >>> segments = segment_eeg_by_sentences(eeg_raw, alignment, fs=500)
        >>> print(f"Extracted {len(segments)} sentence segments")
    """
    segments = {}
    n_samples = eeg.shape[-1]
    
    for _, row in alignment.iterrows():
        sentence_id = int(row['sentence_id'])
        start_sample = int(row['start_time'] * fs)
        end_sample = int(row['end_time'] * fs)
        
        # Boundary checks
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        
        # Check minimum duration
        duration = (end_sample - start_sample) / fs
        if duration < min_duration:
            continue
        
        if eeg.ndim == 1:
            segment = eeg[start_sample:end_sample]
        else:
            segment = eeg[:, start_sample:end_sample]
        
        segments[sentence_id] = segment
    
    return segments


def segment_audio_by_sentences(audio: np.ndarray,
                                alignment: pd.DataFrame,
                                fs: float = 16000.0,
                                min_duration: float = 0.1) -> Dict[int, np.ndarray]:
    """
    Segment continuous audio into sentence-level segments.
    
    Args:
        audio: np.ndarray of shape (n_samples,) or (n_channels, n_samples)
        alignment: pd.DataFrame with start_time, end_time, sentence_id
        fs: float - audio sampling frequency in Hz
        min_duration: float - minimum segment duration in seconds
    
    Returns:
        Dict[int, np.ndarray] - {sentence_id: audio_segment}
    
    Example:
        >>> audio = np.random.randn(160000)  # 10 seconds at 16kHz
        >>> alignment = load_alignment_table('alignment.csv')
        >>> segments = segment_audio_by_sentences(audio, alignment, fs=16000)
    """
    return segment_eeg_by_sentences(audio, alignment, fs, min_duration)


def generate_alignment_from_bids(bids_root: str,
                                  subject: str,
                                  task: str = 'listening',
                                  session: str = None,
                                  output_path: str = None) -> pd.DataFrame:
    """
    Generate sentence alignment table from BIDS events.tsv file.
    
    This function parses the BIDS events file (which contains stimulus onset/offset
    times) and creates a sentence-level alignment table suitable for EEG segmentation.
    
    Args:
        bids_root: str - path to BIDS dataset root
        subject: str - subject ID (without 'sub-' prefix)
        task: str - task name (default: 'listening')
        session: str - session ID (optional)
        output_path: str - path to save alignment CSV (optional)
    
    Returns:
        pd.DataFrame with columns: sentence_id, start_time, end_time, 
                                   start_sample, end_sample, audio_file
    
    Example:
        >>> alignment = generate_alignment_from_bids(
        ...     '/data/ds004408', subject='01', task='listening'
        ... )
        >>> alignment.to_csv('alignment_sub01.csv', index=False)
    """
    from pathlib import Path
    
    bids_path = Path(bids_root)
    
    # Build path to events file
    if session:
        events_dir = bids_path / f'sub-{subject}' / f'ses-{session}' / 'eeg'
        events_pattern = f'sub-{subject}_ses-{session}_task-{task}_events.tsv'
    else:
        events_dir = bids_path / f'sub-{subject}' / 'eeg'
        events_pattern = f'sub-{subject}_task-{task}_events.tsv'
    
    events_file = events_dir / events_pattern
    
    if not events_file.exists():
        # Try finding any events file
        events_files = list(events_dir.glob('*events.tsv'))
        if events_files:
            events_file = events_files[0]
        else:
            raise FileNotFoundError(f"No events file found in {events_dir}")
    
    # Load events
    events_df = pd.read_csv(events_file, sep='\t')
    
    # BIDS events typically have: onset, duration, trial_type, stim_file, etc.
    required_cols = ['onset', 'duration']
    for col in required_cols:
        if col not in events_df.columns:
            raise ValueError(f"Missing required column '{col}' in events file")
    
    # Filter for sentence/stimulus events (adjust trial_type as needed)
    if 'trial_type' in events_df.columns:
        # Common trial types for speech stimuli
        speech_types = ['sentence', 'stimulus', 'audio', 'speech', 'word']
        mask = events_df['trial_type'].str.lower().isin(speech_types)
        if mask.sum() > 0:
            events_df = events_df[mask].copy()
    
    # Create alignment table
    alignment_rows = []
    
    for idx, row in events_df.iterrows():
        onset = float(row['onset'])
        duration = float(row['duration'])
        
        alignment_row = {
            'sentence_id': len(alignment_rows),
            'start_time': onset,
            'end_time': onset + duration,
            'duration': duration,
        }
        
        # Add audio file if available
        if 'stim_file' in row:
            alignment_row['audio_file'] = row['stim_file']
        
        # Add text if available
        if 'value' in row:
            alignment_row['text'] = row['value']
        elif 'text' in row:
            alignment_row['text'] = row['text']
        
        alignment_rows.append(alignment_row)
    
    alignment_df = pd.DataFrame(alignment_rows)
    
    # Sort by start time
    alignment_df = alignment_df.sort_values('start_time').reset_index(drop=True)
    alignment_df['sentence_id'] = range(len(alignment_df))
    
    # Save if output path provided
    if output_path:
        alignment_df.to_csv(output_path, index=False)
        print(f"Saved alignment table to {output_path}")
    
    return alignment_df


def load_bids_eeg(bids_root: str,
                  subject: str,
                  task: str = 'listening',
                  session: str = None,
                  preload: bool = True) -> np.ndarray:
    """
    Load EEG data from a BIDS dataset.
    
    Supports both MNE-BIDS (if available) and direct file loading.
    
    Args:
        bids_root: str - path to BIDS dataset root
        subject: str - subject ID (without 'sub-' prefix)
        task: str - task name
        session: str - session ID (optional)
        preload: bool - whether to load data into memory
    
    Returns:
        np.ndarray of shape (n_channels, n_samples)
        
        Also returns sampling frequency as second element if using raw loading
    
    Example:
        >>> eeg_data, fs = load_bids_eeg('/data/ds004408', subject='01')
        >>> print(f"EEG shape: {eeg_data.shape}, fs: {fs}")
    """
    from pathlib import Path
    
    bids_path = Path(bids_root)
    
    # Build path to EEG file
    if session:
        eeg_dir = bids_path / f'sub-{subject}' / f'ses-{session}' / 'eeg'
        file_pattern = f'sub-{subject}_ses-{session}_task-{task}_eeg'
    else:
        eeg_dir = bids_path / f'sub-{subject}' / 'eeg'
        file_pattern = f'sub-{subject}_task-{task}_eeg'
    
    # Try MNE-BIDS first
    try:
        import mne
        from mne_bids import BIDSPath, read_raw_bids
        
        bids_path_obj = BIDSPath(
            root=bids_root,
            subject=subject,
            task=task,
            session=session,
            datatype='eeg'
        )
        
        raw = read_raw_bids(bids_path_obj, verbose='ERROR')
        if preload:
            raw.load_data()
        
        data = raw.get_data()  # (n_channels, n_samples)
        fs = raw.info['sfreq']
        
        return data, fs
        
    except ImportError:
        pass  # Fall back to manual loading
    
    # Manual loading - try common EEG formats
    eeg_file = None
    
    for ext in ['.set', '.edf', '.bdf', '.vhdr', '.fif']:
        candidate = eeg_dir / f'{file_pattern}{ext}'
        if candidate.exists():
            eeg_file = candidate
            break
    
    if eeg_file is None:
        # Try finding any EEG file
        for ext in ['*.set', '*.edf', '*.bdf', '*.vhdr', '*.fif']:
            files = list(eeg_dir.glob(ext))
            if files:
                eeg_file = files[0]
                break
    
    if eeg_file is None:
        raise FileNotFoundError(f"No EEG file found in {eeg_dir}")
    
    # Load based on format
    try:
        import mne
        
        if eeg_file.suffix == '.set':
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=preload)
        elif eeg_file.suffix in ['.edf', '.bdf']:
            raw = mne.io.read_raw_edf(str(eeg_file), preload=preload)
        elif eeg_file.suffix == '.vhdr':
            raw = mne.io.read_raw_brainvision(str(eeg_file), preload=preload)
        elif eeg_file.suffix == '.fif':
            raw = mne.io.read_raw_fif(str(eeg_file), preload=preload)
        else:
            raise ValueError(f"Unsupported EEG format: {eeg_file.suffix}")
        
        data = raw.get_data()
        fs = raw.info['sfreq']
        
        return data, fs
        
    except ImportError:
        raise ImportError("MNE is required to load EEG files. Install with: pip install mne")


def create_alignment_from_vad(audio: np.ndarray,
                               fs: float = 16000.0,
                               energy_threshold: float = 0.01,
                               min_silence: float = 0.3,
                               min_speech: float = 0.5) -> pd.DataFrame:
    """
    Create simple alignment table based on Voice Activity Detection.
    
    This is a simple energy-based VAD for when no alignment is available.
    
    Args:
        audio: np.ndarray - audio signal
        fs: float - sampling frequency
        energy_threshold: float - relative energy threshold
        min_silence: float - minimum silence duration to split (seconds)
        min_speech: float - minimum speech duration to keep (seconds)
    
    Returns:
        pd.DataFrame with sentence boundaries
    """
    # Compute frame energy
    frame_size = int(0.025 * fs)
    hop_size = int(0.010 * fs)
    
    n_frames = (len(audio) - frame_size) // hop_size + 1
    energy = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        energy[i] = np.sum(audio[start:end] ** 2)
    
    # Normalize energy
    energy = energy / (np.max(energy) + 1e-12)
    
    # Find speech regions
    is_speech = energy > energy_threshold
    
    # Convert to time boundaries
    min_silence_frames = int(min_silence / (hop_size / fs))
    min_speech_frames = int(min_speech / (hop_size / fs))
    
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0
    
    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                speech_start = i
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    speech_end = i - silence_count
                    if speech_end - speech_start >= min_speech_frames:
                        segments.append({
                            'sentence_id': len(segments),
                            'start_time': speech_start * hop_size / fs,
                            'end_time': speech_end * hop_size / fs,
                            'text': '',
                        })
                    in_speech = False
                    silence_count = 0
    
    # Handle last segment
    if in_speech:
        speech_end = len(is_speech) - 1
        if speech_end - speech_start >= min_speech_frames:
            segments.append({
                'sentence_id': len(segments),
                'start_time': speech_start * hop_size / fs,
                'end_time': speech_end * hop_size / fs,
                'text': '',
            })
    
    return pd.DataFrame(segments)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create synthetic alignment
    alignment = pd.DataFrame({
        'sentence_id': [0, 1, 2],
        'start_time': [0.0, 2.5, 5.0],
        'end_time': [2.0, 4.5, 7.0],
        'text': ['First sentence.', 'Second sentence.', 'Third sentence.'],
    })
    
    # Synthetic EEG
    eeg = np.random.randn(64, 5000)  # 10 seconds at 500Hz
    
    # Segment
    segments = segment_eeg_by_sentences(eeg, alignment, fs=500)
    
    print(f"Created {len(segments)} segments")
    for sid, seg in segments.items():
        print(f"  Sentence {sid}: shape {seg.shape}")

