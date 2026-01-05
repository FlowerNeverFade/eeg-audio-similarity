#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Model Registry.

Registry of supported Audio LLM models and their configurations.
These configurations are based on the ACTUAL EXTRACTED EMBEDDINGS from the
extraction scripts, verified against the npz/metadata files.

Note: The n_layers and hidden_dim values reflect what is actually captured
by the forward hooks in the extraction scripts, which may extract from:
- LLM decoder layers (most models)
- Audio encoder layers (some models like GLM-4-Voice-9B)
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AudioEncoderConfig:
    """Configuration for the audio encoder component."""
    encoder_type: str  # 'whisper', 'hubert', 'wav2vec2', 'custom'
    encoder_layers: int
    encoder_dim: int
    sample_rate: int = 16000
    num_mel_bins: int = 128


@dataclass
class ModelConfig:
    """Configuration for an audio model."""
    model_name: str
    model_path: str
    model_type: str  # 'huggingface', 'fairseq', 'custom'
    architecture: str  # The actual model architecture class
    n_layers: int  # Number of layers in extracted embeddings
    hidden_dim: int  # Hidden dimension of extracted embeddings
    sample_rate: int
    frame_shift_ms: float  # Output time resolution in ms
    requires_gpu: bool = True
    trust_remote_code: bool = True
    description: str = ""
    extraction_target: str = "llm"  # 'llm' or 'audio_encoder' - which component embeddings are extracted from
    # Audio encoder configuration
    audio_encoder: Optional[AudioEncoderConfig] = None
    # Additional config
    torch_dtype: str = "bfloat16"
    auto_map_config: Optional[str] = None  # For custom AutoConfig


# Default local model base path (set to your local path or use environment variable)
DEFAULT_MODEL_BASE_PATH = os.environ.get("AUDIO_LLM_MODEL_PATH", "./models")

# Default embeddings output base path  
DEFAULT_EMBEDDINGS_BASE_PATH = os.environ.get("AUDIO_LLM_EMBEDDINGS_PATH", "./embeddings")

# Registry of supported models with configurations verified from ACTUAL EXTRACTED EMBEDDINGS
# (verified from sentence_*_metadata.json files)
SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    # ============================================================
    # Audio-Flamingo-3 (Qwen2 7B backbone + Whisper audio encoder)
    # Verified: 28 layers (layer_0-27), hidden_dim=3584
    # ============================================================
    'audio-flamingo-3': ModelConfig(
        model_name='Audio-Flamingo-3',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/audio-flamingo-3',
        model_type='huggingface',
        architecture='LlavaLlamaModel',
        n_layers=28,  # Verified from embeddings: layer_0 to layer_27
        hidden_dim=3584,  # Verified from shape: [1, T, 3584]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Audio-Flamingo-3 multimodal model (Qwen2 7B + Whisper encoder)',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
    ),
    
    # ============================================================
    # Baichuan-Audio-Base (Omni architecture)
    # Verified: 28 layers (layer_0-27), hidden_dim=3584
    # ============================================================
    'baichuan-audio-base': ModelConfig(
        model_name='Baichuan-Audio-Base',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/Baichuan-Audio-Base',
        model_type='huggingface',
        architecture='OmniForCausalLM',
        n_layers=28,  # Verified from embeddings: layer_0 to layer_27
        hidden_dim=3584,  # Verified from shape: [1, T, 3584]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Baichuan Audio Base omni-modal model',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
        auto_map_config='configuration_omni.OmniConfig',
    ),
    
    # ============================================================
    # Baichuan-Audio-Instruct (Same architecture as Base)
    # Verified: 28 layers (layer_0-27), hidden_dim=3584
    # ============================================================
    'baichuan-audio-instruct': ModelConfig(
        model_name='Baichuan-Audio-Instruct',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/Baichuan-Audio-Instruct',
        model_type='huggingface',
        architecture='OmniForCausalLM',
        n_layers=28,  # Verified from embeddings: layer_0 to layer_27
        hidden_dim=3584,  # Verified from shape: [1, T, 3584]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Baichuan Audio instruction-tuned omni-modal model',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
        auto_map_config='configuration_omni.OmniConfig',
    ),
    
    # ============================================================
    # GLM-4-Voice-9B (Whisper encoder output)
    # Verified: 32 layers (layer_0-31) + final_output, hidden_dim=1280
    # NOTE: Extracts from Whisper encoder, not GLM transformer!
    # ============================================================
    'glm-4-voice-9b': ModelConfig(
        model_name='GLM-4-Voice-9B',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/glm-4-voice-9b',
        model_type='huggingface',
        architecture='ChatGLMForConditionalGeneration',
        n_layers=32,  # Verified: Whisper encoder layers (layer_0-31)
        hidden_dim=1280,  # Verified: Whisper large-v2 d_model
        sample_rate=16000,
        frame_shift_ms=20.0,  # 50 steps/sec = 20ms per step
        description='GLM-4 Voice 9B - extracts Whisper large-v2 encoder representations',
        extraction_target='audio_encoder',  # NOTE: Extracts from Whisper, not GLM
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,  # Whisper large-v2
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
        auto_map_config='configuration_chatglm.ChatGLMConfig',
    ),
    
    # ============================================================
    # Granite-Speech-3.3-8B (IBM Granite)
    # Verified: 40 layers (layer_0-39), hidden_dim=4096
    # ============================================================
    'granite-speech-3.3-8b': ModelConfig(
        model_name='Granite-Speech-3.3-8B',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/granite-speech-3.3-8b',
        model_type='huggingface',
        architecture='GraniteSpeechModel',
        n_layers=40,  # Verified from embeddings: layer_0 to layer_39
        hidden_dim=4096,  # Verified from shape: [1, T, 4096]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='IBM Granite Speech 3.3 8B model (decoder layers)',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
    ),
    
    # ============================================================
    # Llama-3.1-8B-Omni (OmniSpeech2S architecture)
    # Verified: 32 layers (layer_0-31), hidden_dim=4096
    # ============================================================
    'llama-3.1-8b-omni': ModelConfig(
        model_name='Llama-3.1-8B-Omni',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/Llama-3.1-8B-Omni',
        model_type='huggingface',
        architecture='OmniSpeech2SLlamaForCausalLM',
        n_layers=32,  # Verified from embeddings: layer_0 to layer_31
        hidden_dim=4096,  # Verified from shape: [1, T, 4096]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Llama 3.1 8B Omni speech-to-speech model with Whisper encoder',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='float16',
    ),
    
    # ============================================================
    # MiniCPM-o-2_6 (MiniCPMO architecture)
    # Verified: 28 layers (layer_0-27), hidden_dim=3584
    # ============================================================
    'minicpm-o-2_6': ModelConfig(
        model_name='MiniCPM-o-2_6',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/MiniCPM-o-2_6',
        model_type='huggingface',
        architecture='MiniCPMO',
        n_layers=28,  # Verified from embeddings: layer_0 to layer_27
        hidden_dim=3584,  # Verified from shape: [1, T, 3584]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='MiniCPM-o 2.6 omni-modal model with Whisper-medium encoder',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=24,
            encoder_dim=1024,
            num_mel_bins=80,
        ),
        torch_dtype='bfloat16',
        auto_map_config='configuration_minicpm.MiniCPMOConfig',
    ),
    
    # ============================================================
    # Qwen2-Audio-7B
    # Verified: 32 layers (layer_0-31), hidden_dim=4096
    # Note: Despite model name "7B", uses Qwen2-7B architecture (32 layers)
    # ============================================================
    'qwen2-audio': ModelConfig(
        model_name='Qwen2-Audio-7B',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/Qwen2-Audio-7B',
        model_type='huggingface',
        architecture='Qwen2AudioForConditionalGeneration',
        n_layers=32,  # Verified from embeddings: layer_0 to layer_31
        hidden_dim=4096,  # Verified from shape: [1, T, 4096]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Qwen2-Audio 7B model for audio understanding',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
    ),
    
    # ============================================================
    # Qwen2-Audio-7B-Instruct
    # Verified: 32 layers (layer_0-31), hidden_dim=4096
    # ============================================================
    'qwen2-audio-instruct': ModelConfig(
        model_name='Qwen2-Audio-7B-Instruct',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/Qwen2-Audio-7B-Instruct',
        model_type='huggingface',
        architecture='Qwen2AudioForConditionalGeneration',
        n_layers=32,  # Verified from embeddings: layer_0 to layer_31
        hidden_dim=4096,  # Verified from shape: [1, T, 4096]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Qwen2-Audio 7B instruction-tuned model',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
    ),
    
    # ============================================================
    # SpeechGPT-2.0-preview-7B (MIMO Llama architecture)
    # Verified: 28 layers (layer_0-27), hidden_dim=3584
    # ============================================================
    'speechgpt': ModelConfig(
        model_name='SpeechGPT-2.0-preview-7B',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/SpeechGPT-2.0-preview-7B',
        model_type='huggingface',
        architecture='MIMOLlamaForCausalLM',
        n_layers=28,  # Verified from embeddings: layer_0 to layer_27
        hidden_dim=3584,  # Verified from shape: [1, T, 3584]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='SpeechGPT 2.0 preview for speech-text understanding',
        extraction_target='llm',
        audio_encoder=None,  # Uses VQ tokens
        torch_dtype='float32',
    ),
    
    # ============================================================
    # Ultravox v0.5 (Llama-3.1-8B backbone)
    # Verified: 32 layers (layer_0-31), hidden_dim=4096
    # ============================================================
    'ultravox-llama3.1-8b': ModelConfig(
        model_name='Ultravox-v0.5-Llama-3.1-8B',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/ultravox-v0_5-llama-3_1-8b',
        model_type='huggingface',
        architecture='UltravoxModel',
        n_layers=32,  # Verified from embeddings: layer_0 to layer_31
        hidden_dim=4096,  # Verified from shape: [1, T, 4096]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Ultravox v0.5 with Llama-3.1-8B backbone and Whisper encoder',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
        auto_map_config='ultravox_config.UltravoxConfig',
    ),
    
    # ============================================================
    # Ultravox v0.5 (Llama-3.2-1B backbone)
    # Verified: 16 layers (layer_0-15), hidden_dim=2048
    # ============================================================
    'ultravox-llama3.2-1b': ModelConfig(
        model_name='Ultravox-v0.5-Llama-3.2-1B',
        model_path=f'{DEFAULT_MODEL_BASE_PATH}/ultravox-v0_5-llama-3_2-1b',
        model_type='huggingface',
        architecture='UltravoxModel',
        n_layers=16,  # Verified from embeddings: layer_0 to layer_15
        hidden_dim=2048,  # Verified from shape: [1, T, 2048]
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Ultravox v0.5 with Llama-3.2-1B backbone (lightweight)',
        extraction_target='llm',
        audio_encoder=AudioEncoderConfig(
            encoder_type='whisper',
            encoder_layers=32,
            encoder_dim=1280,
            num_mel_bins=128,
        ),
        torch_dtype='bfloat16',
        auto_map_config='ultravox_config.UltravoxConfig',
    ),
}

# Aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    'qwen2-audio-7b': 'qwen2-audio',
    'qwen2-audio-7b-instruct': 'qwen2-audio-instruct',
    'baichuan-audio': 'baichuan-audio-base',
    'glm4-voice': 'glm-4-voice-9b',
    'llama3.1-8b-omni': 'llama-3.1-8b-omni',
    'speechgpt-2.0': 'speechgpt',
    'ultravox-8b': 'ultravox-llama3.1-8b',
    'ultravox-1b': 'ultravox-llama3.2-1b',
    'minicpm': 'minicpm-o-2_6',
    'minicpm-o': 'minicpm-o-2_6',
    'granite': 'granite-speech-3.3-8b',
    'audio-flamingo': 'audio-flamingo-3',
    'af3': 'audio-flamingo-3',
}

# Mapping from model keys to embeddings folder names
# (folder names under DEFAULT_EMBEDDINGS_BASE_PATH)
MODEL_TO_EMBEDDINGS_FOLDER: Dict[str, str] = {
    'audio-flamingo-3': 'audio-flamingo-3',
    'baichuan-audio-base': 'Baichuan-Audio-Base',
    'baichuan-audio-instruct': 'Baichuan-Audio-Instruct',
    'glm-4-voice-9b': 'glm-4-voice-9b',
    'granite-speech-3.3-8b': 'granite-speech-3.3-8b',
    'llama-3.1-8b-omni': 'Llama-3.1-8B-Omni',
    'minicpm-o-2_6': 'MiniCPM-o-2_6',
    'qwen2-audio': 'Qwen2-Audio-7B',
    'qwen2-audio-instruct': 'Qwen2-Audio-7B-Instruct',
    'speechgpt': 'SpeechGPT-2.0-preview-7B',
    'ultravox-llama3.1-8b': 'ultravox-v0_5-llama-3_1-8b',
    'ultravox-llama3.2-1b': 'ultravox-v0_5-llama-3_2-1b',
}


def get_model_config(model_key: str) -> ModelConfig:
    """
    Get configuration for a model.
    
    Args:
        model_key: str - model identifier (case-insensitive)
    
    Returns:
        ModelConfig
    
    Raises:
        ValueError if model not found
    """
    key = model_key.lower().replace('_', '-')
    
    # Check aliases first
    if key in MODEL_ALIASES:
        key = MODEL_ALIASES[key]
    
    if key not in SUPPORTED_MODELS:
        available = ', '.join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return SUPPORTED_MODELS[key]


def get_embeddings_path(model_key: str, base_path: Optional[str] = None) -> str:
    """
    Get the path to pre-extracted embeddings for a model.
    
    Args:
        model_key: str - model identifier
        base_path: Optional base path (defaults to DEFAULT_EMBEDDINGS_BASE_PATH)
    
    Returns:
        str - path to embeddings folder
    """
    import os
    key = model_key.lower().replace('_', '-')
    if key in MODEL_ALIASES:
        key = MODEL_ALIASES[key]
    
    if key not in MODEL_TO_EMBEDDINGS_FOLDER:
        raise ValueError(f"No embeddings folder mapping for model: {model_key}")
    
    folder = MODEL_TO_EMBEDDINGS_FOLDER[key]
    base = base_path or DEFAULT_EMBEDDINGS_BASE_PATH
    return os.path.join(base, folder)


def list_available_models() -> List[str]:
    """List all available model keys."""
    return sorted(SUPPORTED_MODELS.keys())


def get_model_summary() -> Dict[str, Dict]:
    """Get a summary of all models with key specifications from extracted embeddings."""
    summary = {}
    for key, config in SUPPORTED_MODELS.items():
        summary[key] = {
            'name': config.model_name,
            'architecture': config.architecture,
            'n_layers': config.n_layers,  # Layers in extracted embeddings
            'hidden_dim': config.hidden_dim,  # Hidden dim in extracted embeddings
            'extraction_target': config.extraction_target,  # 'llm' or 'audio_encoder'
            'embeddings_folder': MODEL_TO_EMBEDDINGS_FOLDER.get(key, None),
        }
    return summary


def print_model_info(model_key: Optional[str] = None, verbose: bool = False):
    """Print information about models."""
    if model_key:
        config = get_model_config(model_key)
        print(f"Model: {config.model_name}")
        print(f"  Architecture: {config.architecture}")
        print(f"  Path: {config.model_path}")
        print(f"  Extracted Layers: {config.n_layers}")
        print(f"  Hidden Dim: {config.hidden_dim}")
        print(f"  Extraction Target: {config.extraction_target}")
        if config.audio_encoder:
            print(f"  Audio Encoder: {config.audio_encoder.encoder_type}")
            print(f"    Encoder Layers: {config.audio_encoder.encoder_layers}")
            print(f"    Encoder Dim: {config.audio_encoder.encoder_dim}")
        print(f"  Sample rate: {config.sample_rate}")
        print(f"  Frame shift: {config.frame_shift_ms}ms")
        print(f"  Description: {config.description}")
        # Show embeddings path if available
        key = model_key.lower().replace('_', '-')
        if key in MODEL_ALIASES:
            key = MODEL_ALIASES[key]
        if key in MODEL_TO_EMBEDDINGS_FOLDER:
            print(f"  Embeddings folder: {MODEL_TO_EMBEDDINGS_FOLDER[key]}")
    else:
        print("=" * 95)
        print("Supported Audio LLM Models (Verified from Extracted Embeddings)")
        print("=" * 95)
        print(f"{'Model Key':<25} {'Name':<30} {'Layers':<8} {'Dim':<8} {'Target':<15}")
        print("-" * 95)
        for key, config in sorted(SUPPORTED_MODELS.items()):
            print(f"{key:<25} {config.model_name:<30} {config.n_layers:<8} {config.hidden_dim:<8} {config.extraction_target:<15}")
        print("-" * 95)
        print(f"Total: {len(SUPPORTED_MODELS)} models")
        
        if verbose:
            print("\nAliases:")
            for alias, target in sorted(MODEL_ALIASES.items()):
                print(f"  {alias} -> {target}")
            print("\nEmbeddings folders:")
            for key, folder in sorted(MODEL_TO_EMBEDDINGS_FOLDER.items()):
                print(f"  {key} -> {folder}")


if __name__ == "__main__":
    print_model_info()
