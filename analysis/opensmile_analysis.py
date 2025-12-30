#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSMILE-based acoustic feature extraction and sentiment analysis.

This module provides functions for extracting acoustic features using OpenSMILE
and estimating sentiment/arousal/valence from audio.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

# Optional opensmile import
try:
    import opensmile
    _HAS_OPENSMILE = True
except ImportError:
    _HAS_OPENSMILE = False


def build_smile(feature_set_name: str = 'eGeMAPSv02'):
    """
    Build OpenSMILE feature extractor.
    
    Args:
        feature_set_name: str - feature set name, one of:
            - 'eGeMAPSv02': Geneva Minimalistic Acoustic Parameter Set
            - 'ComParE_2016': ComParE 2016 feature set
            - 'emobase': Emotion recognition base set
    
    Returns:
        opensmile.Smile object
    
    Example:
        >>> smile = build_smile('eGeMAPSv02')
        >>> features = smile.process_file('audio.wav')
    """
    if not _HAS_OPENSMILE:
        raise ImportError("opensmile is not installed. Install with: pip install opensmile")
    
    feature_sets = {
        'eGeMAPSv02': opensmile.FeatureSet.eGeMAPSv02,
        'ComParE_2016': opensmile.FeatureSet.ComParE_2016,
        'emobase': opensmile.FeatureSet.emobase,
    }
    
    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}. Choose from {list(feature_sets.keys())}")
    
    return opensmile.Smile(
        feature_set=feature_sets[feature_set_name],
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_opensmile_features(audio_path: str, smile=None, feature_set: str = 'eGeMAPSv02') -> Optional[pd.DataFrame]:
    """
    Extract OpenSMILE features from an audio file.
    
    Args:
        audio_path: str - path to audio file
        smile: opensmile.Smile, optional - pre-built smile object
        feature_set: str - feature set name (if smile is None)
    
    Returns:
        pd.DataFrame with extracted features, or None if extraction fails
    
    Example:
        >>> features = extract_opensmile_features('audio.wav')
        >>> print(features.columns)
    """
    if not _HAS_OPENSMILE:
        raise ImportError("opensmile is not installed. Install with: pip install opensmile")
    
    if smile is None:
        smile = build_smile(feature_set)
    
    try:
        features = smile.process_file(audio_path)
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment/arousal/valence scores based on acoustic features.
    
    Uses a simple heuristic based on eGeMAPSv02 features:
    - Arousal: based on F0, loudness, speech rate
    - Valence: based on F0 variation, spectral features
    
    Args:
        df: pd.DataFrame with OpenSMILE features
    
    Returns:
        pd.DataFrame with added sentiment columns
    """
    df = df.copy()
    
    # Normalize features for scoring
    def safe_normalize(series):
        s = series.fillna(0)
        std = s.std()
        if std < 1e-12:
            return s * 0
        return (s - s.mean()) / std
    
    # Arousal estimation (higher pitch, louder, faster = higher arousal)
    arousal_features = []
    for col in ['F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 
                'F0semitoneFrom27.5Hz_sma3nz_stddevNorm']:
        if col in df.columns:
            arousal_features.append(safe_normalize(df[col]))
    
    if arousal_features:
        df['arousal_score'] = np.mean(arousal_features, axis=0)
    else:
        df['arousal_score'] = 0.0
    
    # Valence estimation (more variation, higher spectral centroid = more positive)
    valence_features = []
    for col in ['F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'spectralFlux_sma3_amean',
                'mfcc1_sma3_amean']:
        if col in df.columns:
            valence_features.append(safe_normalize(df[col]))
    
    if valence_features:
        df['valence_score'] = np.mean(valence_features, axis=0)
    else:
        df['valence_score'] = 0.0
    
    # Combined sentiment score
    df['sentiment_score'] = (df['arousal_score'] + df['valence_score']) / 2
    
    return df


def find_audio_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """
    Find all audio files in a directory.
    
    Args:
        input_dir: str - input directory path
        extensions: list of str - file extensions to search for
    
    Returns:
        List of Path objects
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    input_path = Path(input_dir)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(input_path.glob(f'**/*{ext}'))
    
    return sorted(audio_files)


def batch_extract_features(audio_dir: str, output_csv: str = None,
                           feature_set: str = 'eGeMAPSv02',
                           add_sentiment: bool = True) -> pd.DataFrame:
    """
    Extract OpenSMILE features from all audio files in a directory.
    
    Args:
        audio_dir: str - directory containing audio files
        output_csv: str, optional - path to save results
        feature_set: str - OpenSMILE feature set
        add_sentiment: bool - whether to add sentiment scores
    
    Returns:
        pd.DataFrame with features for all files
    """
    if not _HAS_OPENSMILE:
        raise ImportError("opensmile is not installed")
    
    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return pd.DataFrame()
    
    smile = build_smile(feature_set)
    
    all_features = []
    for audio_path in audio_files:
        features = extract_opensmile_features(str(audio_path), smile=smile)
        if features is not None:
            features['file'] = audio_path.name
            features['file_path'] = str(audio_path)
            all_features.append(features)
    
    if not all_features:
        return pd.DataFrame()
    
    df = pd.concat(all_features, ignore_index=True)
    
    if add_sentiment:
        df = add_sentiment_scores(df)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved features to {output_csv}")
    
    return df


if __name__ == "__main__":
    print("OpenSMILE Analysis Module")
    print(f"OpenSMILE available: {_HAS_OPENSMILE}")

