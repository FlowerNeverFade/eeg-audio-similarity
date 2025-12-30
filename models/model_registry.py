#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Model Registry.

Registry of supported Audio LLM models and their configurations.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for an audio model."""
    model_name: str
    model_path: str
    model_type: str  # 'huggingface', 'fairseq', 'custom'
    n_layers: int
    hidden_dim: int
    sample_rate: int
    frame_shift_ms: float  # Output time resolution in ms
    requires_gpu: bool = True
    description: str = ""


# Registry of supported models
SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    'audio-flamingo-3': ModelConfig(
        model_name='Audio-Flamingo-3',
        model_path='nvidia/audio-flamingo-3',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Audio-Flamingo-3 multimodal audio model',
    ),
    'baichuan-audio-base': ModelConfig(
        model_name='Baichuan-Audio-Base',
        model_path='baichuan-inc/Baichuan-Audio-Base',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Baichuan Audio Base model',
    ),
    'baichuan-audio-instruct': ModelConfig(
        model_name='Baichuan-Audio-Instruct',
        model_path='baichuan-inc/Baichuan-Audio-Instruct',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Baichuan Audio instruction-tuned model',
    ),
    'glm-4-voice-9b': ModelConfig(
        model_name='GLM-4-Voice-9B',
        model_path='THUDM/glm-4-voice-9b',
        model_type='huggingface',
        n_layers=40,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='GLM-4 Voice 9B model for speech understanding',
    ),
    'granite-speech-3.3-8b': ModelConfig(
        model_name='Granite-Speech-3.3-8B',
        model_path='ibm-granite/granite-speech-3.3-8b',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='IBM Granite Speech 3.3 8B model',
    ),
    'llama-3.1-8b-omni': ModelConfig(
        model_name='Llama-3.1-8B-Omni',
        model_path='meta-llama/Llama-3.1-8B-Omni',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Llama 3.1 8B Omni multimodal model',
    ),
    'minicpm-o-2_6': ModelConfig(
        model_name='MiniCPM-o-2_6',
        model_path='openbmb/MiniCPM-o-2_6',
        model_type='huggingface',
        n_layers=28,
        hidden_dim=3584,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='MiniCPM-o 2.6 omni-modal model',
    ),
    'qwen2-audio': ModelConfig(
        model_name='Qwen2-Audio-7B',
        model_path='Qwen/Qwen2-Audio-7B',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Qwen2-Audio 7B model for audio understanding',
    ),
    'qwen2-audio-instruct': ModelConfig(
        model_name='Qwen2-Audio-7B-Instruct',
        model_path='Qwen/Qwen2-Audio-7B-Instruct',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Qwen2-Audio 7B instruction-tuned model',
    ),
    'speechgpt': ModelConfig(
        model_name='SpeechGPT-2.0-preview-7B',
        model_path='fnlp/SpeechGPT-2.0-preview-7B',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='SpeechGPT 2.0 preview for speech-text understanding',
    ),
    'ultravox-llama3.1-8b': ModelConfig(
        model_name='Ultravox-v0.5-Llama-3.1-8B',
        model_path='fixie-ai/ultravox-v0.5-llama-3.1-8b',
        model_type='huggingface',
        n_layers=32,
        hidden_dim=4096,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Ultravox v0.5 built on Llama-3.1-8B backbone',
    ),
    'ultravox-llama3.2-1b': ModelConfig(
        model_name='Ultravox-v0.5-Llama-3.2-1B',
        model_path='fixie-ai/ultravox-v0.5-llama-3.2-1b',
        model_type='huggingface',
        n_layers=16,
        hidden_dim=2048,
        sample_rate=16000,
        frame_shift_ms=20.0,
        description='Ultravox v0.5 built on Llama-3.2-1B backbone',
    ),
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
    if key not in SUPPORTED_MODELS:
        available = ', '.join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return SUPPORTED_MODELS[key]


def list_available_models() -> List[str]:
    """List all available model keys."""
    return list(SUPPORTED_MODELS.keys())


def print_model_info(model_key: Optional[str] = None):
    """Print information about models."""
    if model_key:
        config = get_model_config(model_key)
        print(f"Model: {config.model_name}")
        print(f"  Path: {config.model_path}")
        print(f"  Layers: {config.n_layers}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Sample rate: {config.sample_rate}")
        print(f"  Frame shift: {config.frame_shift_ms}ms")
        print(f"  Description: {config.description}")
    else:
        print("Available models:")
        for key, config in SUPPORTED_MODELS.items():
            print(f"  {key}: {config.model_name} ({config.n_layers} layers, {config.hidden_dim}d)")


if __name__ == "__main__":
    print_model_info()

