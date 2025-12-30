#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio LLM Inference and Embedding Extraction.

This module provides functions for:
- Loading audio language models
- Running inference to extract layer-wise hidden states
- Saving and loading embeddings
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

from .model_registry import get_model_config, ModelConfig


def load_audio_model(model_key: str,
                     device: str = 'cuda',
                     dtype: torch.dtype = torch.float16,
                     cache_dir: Optional[str] = None) -> Tuple[object, object, ModelConfig]:
    """
    Load an audio model and processor.
    
    Args:
        model_key: str - model identifier from registry
        device: str - 'cuda' or 'cpu'
        dtype: torch.dtype - model precision
        cache_dir: str, optional - HuggingFace cache directory
    
    Returns:
        Tuple[model, processor, config]
    
    Example:
        >>> model, processor, config = load_audio_model('wavlm-base')
        >>> print(f"Loaded {config.model_name} with {config.n_layers} layers")
    """
    try:
        from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")
    
    config = get_model_config(model_key)
    
    print(f"Loading {config.model_name} from {config.model_path}...")
    
    # Load model based on type
    model_path = config.model_path
    
    try:
        # Try loading with AutoModel (for encoder-only models)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"AutoModel failed: {e}, trying specific model class...")
        # Fallback for specific model types
        if 'whisper' in model_key.lower():
            from transformers import WhisperModel
            model = WhisperModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )
        elif 'wav2vec' in model_key.lower() or 'wavlm' in model_key.lower() or 'hubert' in model_key.lower():
            from transformers import Wav2Vec2Model
            model = Wav2Vec2Model.from_pretrained(
                model_path,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )
        else:
            raise
    
    model = model.to(device)
    model.eval()
    
    # Load processor/feature extractor
    try:
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    except Exception:
        try:
            processor = AutoFeatureExtractor.from_pretrained(model_path, cache_dir=cache_dir)
        except Exception:
            processor = None
            warnings.warn(f"Could not load processor for {model_key}")
    
    print(f"Loaded {config.model_name}: {config.n_layers} layers, {config.hidden_dim}d")
    
    return model, processor, config


def extract_hidden_states(model,
                          processor,
                          audio: np.ndarray,
                          sample_rate: int = 16000,
                          device: str = 'cuda',
                          layer_idx: Optional[int] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract hidden states from audio using the model.
    
    Args:
        model: loaded model
        processor: audio processor
        audio: np.ndarray - audio waveform (n_samples,)
        sample_rate: int - audio sample rate
        device: str - device for inference
        layer_idx: int, optional - specific layer to extract (None = all layers)
    
    Returns:
        If layer_idx specified: np.ndarray of shape (T, D)
        If layer_idx is None: Dict[int, np.ndarray] with all layers
    
    Example:
        >>> model, processor, config = load_audio_model('wavlm-base')
        >>> audio = np.random.randn(16000)  # 1 second
        >>> hidden_states = extract_hidden_states(model, processor, audio)
        >>> print(f"Extracted {len(hidden_states)} layers")
    """
    # Preprocess audio
    if processor is not None:
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values if hasattr(inputs, 'input_values') else inputs.input_features
        input_values = input_values.to(device)
    else:
        input_values = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
    
    # Extract hidden states
    if hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states
    elif hasattr(outputs, 'encoder_hidden_states'):
        hidden_states = outputs.encoder_hidden_states
    else:
        raise ValueError("Model output does not contain hidden states")
    
    # Convert to numpy
    if layer_idx is not None:
        # Return specific layer
        hs = hidden_states[layer_idx].squeeze(0).cpu().numpy()
        return hs
    else:
        # Return all layers
        result = {}
        for i, hs in enumerate(hidden_states):
            result[i] = hs.squeeze(0).cpu().numpy()
        return result


def extract_all_layers(model,
                       processor,
                       audio: np.ndarray,
                       sample_rate: int = 16000,
                       device: str = 'cuda',
                       batch_size: int = 1) -> Dict[int, np.ndarray]:
    """
    Extract hidden states from all layers.
    
    Args:
        model: loaded model
        processor: audio processor
        audio: np.ndarray - audio waveform
        sample_rate: int - audio sample rate
        device: str - device for inference
        batch_size: int - batch size for processing
    
    Returns:
        Dict[int, np.ndarray] - {layer_idx: hidden_states}
    """
    return extract_hidden_states(model, processor, audio, sample_rate, device, layer_idx=None)


def save_embeddings(embeddings: Dict[int, np.ndarray],
                    output_path: str,
                    metadata: Optional[Dict] = None):
    """
    Save layer-wise embeddings to npz file.
    
    Args:
        embeddings: Dict[int, np.ndarray] - {layer_idx: hidden_states}
        output_path: str - output file path (.npz)
        metadata: dict, optional - additional metadata to save
    
    Example:
        >>> embeddings = extract_all_layers(model, processor, audio)
        >>> save_embeddings(embeddings, 'sentence_001_embeddings.npz',
        ...                 metadata={'model': 'wavlm-base', 'duration': 2.5})
    """
    save_dict = {f'layer_{k}': v for k, v in embeddings.items()}
    
    if metadata is not None:
        # Save metadata as a JSON string in special key
        import json
        save_dict['_metadata'] = np.array([json.dumps(metadata)])
    
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved {len(embeddings)} layers to {output_path}")


def load_embeddings(npz_path: str,
                    layer_idx: Optional[int] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Load embeddings from npz file.
    
    Args:
        npz_path: str - path to .npz file
        layer_idx: int, optional - specific layer to load (None = all)
    
    Returns:
        If layer_idx specified: np.ndarray of shape (T, D)
        If layer_idx is None: Dict[int, np.ndarray]
    
    Example:
        >>> all_layers = load_embeddings('sentence_001_embeddings.npz')
        >>> layer_12 = load_embeddings('sentence_001_embeddings.npz', layer_idx=12)
    """
    data = np.load(npz_path, allow_pickle=True)
    
    if layer_idx is not None:
        key = f'layer_{layer_idx}'
        if key not in data:
            raise KeyError(f"Layer {layer_idx} not found in {npz_path}")
        return data[key]
    else:
        result = {}
        for key in data.files:
            if key.startswith('layer_'):
                idx = int(key.split('_')[1])
                result[idx] = data[key]
        return result


def batch_extract_embeddings(model,
                              processor,
                              audio_dir: str,
                              output_dir: str,
                              sample_rate: int = 16000,
                              device: str = 'cuda',
                              file_pattern: str = '*.wav'):
    """
    Extract embeddings for all audio files in a directory.
    
    Args:
        model: loaded model
        processor: audio processor
        audio_dir: str - directory containing audio files
        output_dir: str - output directory for embeddings
        sample_rate: int - target sample rate
        device: str - inference device
        file_pattern: str - glob pattern for audio files
    
    Example:
        >>> model, processor, config = load_audio_model('wavlm-base')
        >>> batch_extract_embeddings(model, processor, 
        ...                          'audio_sentences/', 
        ...                          'embeddings/')
    """
    import librosa
    from glob import glob
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = sorted(glob(os.path.join(audio_dir, file_pattern)))
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Extract embeddings
        embeddings = extract_all_layers(model, processor, audio, sample_rate, device)
        
        # Save
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_embeddings.npz")
        
        metadata = {
            'source_file': audio_path,
            'sample_rate': sample_rate,
            'duration': len(audio) / sample_rate,
        }
        
        save_embeddings(embeddings, output_path, metadata)
    
    print(f"Saved embeddings to {output_dir}")


if __name__ == "__main__":
    print("Audio LLM Inference Module")
    print("=" * 50)
    
    # List available models
    from .model_registry import print_model_info
    print_model_info()
    
    print("\nExample usage:")
    print("  model, processor, config = load_audio_model('wavlm-base')")
    print("  embeddings = extract_all_layers(model, processor, audio)")
    print("  save_embeddings(embeddings, 'output.npz')")

