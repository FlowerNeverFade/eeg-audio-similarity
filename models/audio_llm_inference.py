#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio LLM Inference and Embedding Extraction.

This module provides functions for:
- Loading various Audio LLM models (local or HuggingFace Hub)
- Running inference to extract layer-wise hidden states
- Saving and loading embeddings

Supported models:
- Audio-Flamingo-3 (Qwen2 + Whisper)
- Baichuan-Audio (Base/Instruct)
- GLM-4-Voice-9B
- Granite-Speech-3.3-8B
- Llama-3.1-8B-Omni
- MiniCPM-o-2_6
- Qwen2-Audio-7B (Base/Instruct)
- SpeechGPT-2.0-preview-7B
- Ultravox-v0.5 (Llama-3.1-8B / Llama-3.2-1B)
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

from .model_registry import get_model_config, ModelConfig, SUPPORTED_MODELS


def load_audio_model(
    model_key: str,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
    local_path: Optional[str] = None,
) -> Tuple[Any, Any, ModelConfig]:
    """
    Load an audio model and processor.
    
    Args:
        model_key: str - model identifier from registry
        device: str - 'cuda', 'cuda:0', 'cpu', etc.
        dtype: torch.dtype - model precision (auto-detected if None)
        cache_dir: str, optional - HuggingFace cache directory
        local_path: str, optional - override model path with local directory
    
    Returns:
        Tuple[model, processor, config]
    
    Example:
        >>> model, processor, config = load_audio_model('qwen2-audio')
        >>> print(f"Loaded {config.model_name} with {config.n_layers} layers")
    """
    try:
        from transformers import AutoModel, AutoProcessor, AutoConfig
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")
    
    config = get_model_config(model_key)
    model_path = local_path or config.model_path
    
    # Determine dtype from config if not specified
    if dtype is None:
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
    
    print(f"Loading {config.model_name} from {model_path}...")
    print(f"  Architecture: {config.architecture}")
    print(f"  Device: {device}, dtype: {dtype}")
    
    # Add model path to sys.path for custom model files
    if os.path.isdir(model_path):
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
    
    model = None
    processor = None
    
    # Model-specific loading logic
    try:
        model, processor = _load_model_by_architecture(
            config, model_path, device, dtype, cache_dir
        )
    except Exception as e:
        print(f"Model-specific loading failed: {e}")
        print("Attempting generic AutoModel loading...")
        model, processor = _load_generic_model(
            model_path, device, dtype, cache_dir
        )
    
    if model is not None:
        model.eval()
        print(f"✓ Loaded {config.model_name}: {config.n_layers} layers, {config.hidden_dim}d")
    
    return model, processor, config


def _load_model_by_architecture(
    config: ModelConfig,
    model_path: str,
    device: str,
    dtype: torch.dtype,
    cache_dir: Optional[str],
) -> Tuple[Any, Any]:
    """Load model based on its specific architecture."""
    from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
    
    architecture = config.architecture
    
    # Qwen2-Audio models
    if architecture == 'Qwen2AudioForConditionalGeneration':
        from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            device_map=device if device != 'cpu' else None,
            trust_remote_code=True,
        )
        processor = Qwen2AudioProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        return model, processor
    
    # Ultravox models
    elif architecture == 'UltravoxModel':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_path, 
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # ChatGLM models (GLM-4-Voice)
    elif architecture == 'ChatGLMForConditionalGeneration':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # Baichuan-Audio (OmniForCausalLM)
    elif architecture == 'OmniForCausalLM':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # MiniCPM-o
    elif architecture == 'MiniCPMO':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # Llama-3.1-8B-Omni (OmniSpeech2S)
    elif architecture == 'OmniSpeech2SLlamaForCausalLM':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # SpeechGPT (MIMO Llama)
    elif architecture == 'MIMOLlamaForCausalLM':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # Audio-Flamingo (LlavaLlama)
    elif architecture == 'LlavaLlamaModel':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return model, processor
    
    # Default: try AutoModel
    else:
        return _load_generic_model(model_path, device, dtype, cache_dir)


def _load_generic_model(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    cache_dir: Optional[str],
) -> Tuple[Any, Any]:
    """Generic model loading fallback."""
    from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        output_hidden_states=True,
        trust_remote_code=True,
    ).to(device)
    
    # Try loading processor
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, 
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception:
        try:
            processor = AutoFeatureExtractor.from_pretrained(
                model_path, 
                cache_dir=cache_dir,
            )
        except Exception:
            warnings.warn(f"Could not load processor for {model_path}")
    
    return model, processor


def extract_hidden_states(
    model,
    processor,
    audio: np.ndarray,
    sample_rate: int = 16000,
    device: str = 'cuda',
    layer_idx: Optional[int] = None,
    return_audio_encoder_states: bool = False,
) -> Union[np.ndarray, Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]]]:
    """
    Extract hidden states from audio using the model.
    
    Args:
        model: loaded model
        processor: audio processor
        audio: np.ndarray - audio waveform (n_samples,)
        sample_rate: int - audio sample rate
        device: str - device for inference
        layer_idx: int, optional - specific layer to extract (None = all layers)
        return_audio_encoder_states: bool - also return audio encoder hidden states
    
    Returns:
        If layer_idx specified: np.ndarray of shape (T, D)
        If layer_idx is None: Dict[int, np.ndarray] with all layers
        If return_audio_encoder_states: Dict with 'llm' and 'audio_encoder' keys
    
    Example:
        >>> model, processor, config = load_audio_model('qwen2-audio')
        >>> audio = np.random.randn(16000)  # 1 second
        >>> hidden_states = extract_hidden_states(model, processor, audio)
        >>> print(f"Extracted {len(hidden_states)} layers")
    """
    # Preprocess audio
    inputs = _preprocess_audio(processor, audio, sample_rate, device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract LLM hidden states
    llm_hidden_states = _extract_llm_hidden_states(outputs)
    
    # Convert to numpy
    if layer_idx is not None:
        hs = llm_hidden_states[layer_idx].squeeze(0).cpu().numpy()
        return hs
    else:
        result = {}
        for i, hs in enumerate(llm_hidden_states):
            result[i] = hs.squeeze(0).cpu().numpy()
        
        if return_audio_encoder_states:
            audio_encoder_states = _extract_audio_encoder_states(outputs)
            return {
                'llm': result,
                'audio_encoder': audio_encoder_states,
            }
        
        return result


def _preprocess_audio(
    processor, 
    audio: np.ndarray, 
    sample_rate: int, 
    device: str
) -> Dict[str, torch.Tensor]:
    """Preprocess audio for model input."""
    if processor is not None:
        try:
            # Standard processor
            inputs = processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            return inputs
        except Exception:
            # Try as text processor with audio features
            pass
    
    # Fallback: raw tensor
    input_values = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    return {'input_values': input_values}


def _extract_llm_hidden_states(outputs) -> tuple:
    """Extract LLM hidden states from model outputs."""
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        return outputs.hidden_states
    elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
        return outputs.decoder_hidden_states
    elif hasattr(outputs, 'last_hidden_state'):
        # Single hidden state output
        return (outputs.last_hidden_state,)
    else:
        raise ValueError("Model output does not contain hidden states. "
                        "Make sure to pass output_hidden_states=True")


def _extract_audio_encoder_states(outputs) -> Optional[Dict[int, np.ndarray]]:
    """Extract audio encoder hidden states if available."""
    encoder_states = None
    
    if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
        encoder_states = outputs.encoder_hidden_states
    elif hasattr(outputs, 'audio_encoder_hidden_states') and outputs.audio_encoder_hidden_states is not None:
        encoder_states = outputs.audio_encoder_hidden_states
    
    if encoder_states is not None:
        return {i: hs.squeeze(0).cpu().numpy() for i, hs in enumerate(encoder_states)}
    
    return None


def extract_all_layers(
    model,
    processor,
    audio: np.ndarray,
    sample_rate: int = 16000,
    device: str = 'cuda',
    batch_size: int = 1,
) -> Dict[int, np.ndarray]:
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
    return extract_hidden_states(
        model, processor, audio, sample_rate, device, layer_idx=None
    )


def save_embeddings(
    embeddings: Dict[int, np.ndarray],
    output_path: str,
    metadata: Optional[Dict] = None,
):
    """
    Save layer-wise embeddings to npz file.
    
    Args:
        embeddings: Dict[int, np.ndarray] - {layer_idx: hidden_states}
        output_path: str - output file path (.npz)
        metadata: dict, optional - additional metadata to save
    
    Example:
        >>> embeddings = extract_all_layers(model, processor, audio)
        >>> save_embeddings(embeddings, 'sentence_001_embeddings.npz',
        ...                 metadata={'model': 'qwen2-audio', 'duration': 2.5})
    """
    save_dict = {f'layer_{k}': v for k, v in embeddings.items()}
    
    if metadata is not None:
        import json
        save_dict['_metadata'] = np.array([json.dumps(metadata)])
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved {len(embeddings)} layers to {output_path}")


def load_embeddings(
    npz_path: str,
    layer_idx: Optional[int] = None,
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
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
            available = [int(k.split('_')[1]) for k in data.files if k.startswith('layer_')]
            raise KeyError(f"Layer {layer_idx} not found. Available layers: {available}")
        return data[key]
    else:
        result = {}
        for key in data.files:
            if key.startswith('layer_'):
                idx = int(key.split('_')[1])
                result[idx] = data[key]
        return result


def load_embeddings_metadata(npz_path: str) -> Optional[Dict]:
    """Load metadata from embeddings file."""
    import json
    data = np.load(npz_path, allow_pickle=True)
    if '_metadata' in data.files:
        return json.loads(data['_metadata'][0])
    return None


def batch_extract_embeddings(
    model,
    processor,
    audio_dir: str,
    output_dir: str,
    model_name: str = 'unknown',
    sample_rate: int = 16000,
    device: str = 'cuda',
    file_pattern: str = '*.wav',
):
    """
    Extract embeddings for all audio files in a directory.
    
    Args:
        model: loaded model
        processor: audio processor
        audio_dir: str - directory containing audio files
        output_dir: str - output directory for embeddings
        model_name: str - model name for metadata
        sample_rate: int - target sample rate
        device: str - inference device
        file_pattern: str - glob pattern for audio files
    
    Example:
        >>> model, processor, config = load_audio_model('qwen2-audio')
        >>> batch_extract_embeddings(model, processor, 
        ...                          'audio_sentences/', 
        ...                          'embeddings/',
        ...                          model_name=config.model_name)
    """
    import librosa
    from glob import glob
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = sorted(glob(os.path.join(audio_dir, file_pattern)))
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Extract embeddings
            embeddings = extract_all_layers(model, processor, audio, sample_rate, device)
            
            # Save
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(output_dir, f"{basename}_embeddings.npz")
            
            metadata = {
                'source_file': audio_path,
                'model': model_name,
                'sample_rate': sample_rate,
                'duration': len(audio) / sample_rate,
                'n_layers': len(embeddings),
            }
            
            save_embeddings(embeddings, output_path, metadata)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    print(f"Saved embeddings to {output_dir}")


def get_layer_dims(model_key: str) -> Tuple[int, int]:
    """
    Get the number of layers and hidden dimension for a model.
    
    Returns:
        Tuple[n_layers, hidden_dim]
    """
    config = get_model_config(model_key)
    return config.n_layers, config.hidden_dim


if __name__ == "__main__":
    print("Audio LLM Inference Module")
    print("=" * 60)
    
    # List available models
    from .model_registry import print_model_info
    print_model_info()
    
    print("\nExample usage:")
    print("  model, processor, config = load_audio_model('qwen2-audio')")
    print("  embeddings = extract_all_layers(model, processor, audio)")
    print("  save_embeddings(embeddings, 'output.npz')")
