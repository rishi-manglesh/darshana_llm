"""LLM Provider abstraction for cross-model experiments."""

from .base import LLMProvider, LLMResponse
from .anthropic_provider import AnthropicProvider
from .mlx_provider import MLXProvider, MLX_MODELS


def get_provider(model_key: str) -> LLMProvider:
    """Create a provider for the given model key.

    Model keys:
        Anthropic: 'sonnet', 'haiku'
        MLX local: 'qwen3_8b', 'qwen3_32b', 'mistral_small'
    """
    if model_key in ("sonnet", "haiku"):
        return AnthropicProvider(model_key)
    elif model_key in MLX_MODELS:
        return MLXProvider(model_key)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "AnthropicProvider",
    "MLXProvider",
    "get_provider",
]
