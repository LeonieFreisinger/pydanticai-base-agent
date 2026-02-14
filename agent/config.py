"""Model configuration and provider factory.

This module demonstrates provider-agnostic model selection,
allowing you to switch between OpenAI, Anthropic, and Google
models with a single environment variable change.

Key pydantic-ai features showcased:
- Provider abstraction via model factory
- Model settings configuration (temperature, max_tokens)
- Special handling for reasoning models (o-series)
"""

import os
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel

# Type alias for supported providers
ProviderType = Literal["openai", "anthropic", "google"]


@dataclass
class ModelSettings:
    """Settings for a specific model configuration."""
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None  # For o-series models: low, medium, high

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }.items() if v is not None}


# Default settings per model family
MODEL_SETTINGS_MAPPER: dict[str, ModelSettings] = {
    # OpenAI standard models
    "gpt-4o": ModelSettings(temperature=0.1, max_tokens=4096),
    "gpt-4o-mini": ModelSettings(temperature=0.1, max_tokens=4096),
    "gpt-4.1": ModelSettings(temperature=0.1, max_tokens=4096),
    "gpt-4.1-mini": ModelSettings(temperature=0.1, max_tokens=4096),
    # OpenAI reasoning models (no temperature, use reasoning_effort)
    "o1": ModelSettings(max_tokens=8192, reasoning_effort="medium"),
    "o1-mini": ModelSettings(max_tokens=4096, reasoning_effort="low"),
    "o3": ModelSettings(max_tokens=8192, reasoning_effort="medium"),
    "o3-mini": ModelSettings(max_tokens=4096, reasoning_effort="low"),
    # Anthropic models
    "claude-3-5-sonnet-20241022": ModelSettings(temperature=0.1, max_tokens=4096),
    "claude-sonnet-4-20250514": ModelSettings(temperature=0.1, max_tokens=4096),
    "claude-3-5-haiku-20241022": ModelSettings(temperature=0.1, max_tokens=4096),
    # Google models
    "gemini-1.5-pro": ModelSettings(temperature=0.1, max_tokens=4096),
    "gemini-1.5-flash": ModelSettings(temperature=0.1, max_tokens=4096),
    "gemini-2.0-flash": ModelSettings(temperature=0.1, max_tokens=4096),
}

# Default model settings for unknown models
DEFAULT_SETTINGS = ModelSettings(temperature=0.1, max_tokens=4096)


def get_model_settings(model_name: str) -> ModelSettings:
    """Get settings for a model, with fallback to defaults."""
    return MODEL_SETTINGS_MAPPER.get(model_name, DEFAULT_SETTINGS)


def get_model(
    provider: ProviderType | None = None,
    model_name: str | None = None,
) -> Model:
    """
    Factory function to create a pydantic-ai model instance.
    
    This abstracts away provider-specific details, allowing you to
    switch models by changing environment variables.
    
    Args:
        provider: The LLM provider (openai, anthropic, google).
                  Defaults to LLM_PROVIDER env var or "openai".
        model_name: The specific model name.
                    Defaults to MODEL_NAME env var or provider's default.
    
    Returns:
        A configured pydantic-ai Model instance.
    
    Example:
        >>> model = get_model()  # Uses env vars
        >>> model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        >>> model = get_model("openai", "o3-mini")  # Reasoning model
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    
    # Provider-specific defaults
    default_models: dict[str, str] = {
        "openai": "gpt-4.1",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.0-flash",
    }
    
    model_name = model_name or os.getenv("MODEL_NAME", default_models.get(provider, "gpt-4.1"))
    
    if provider == "openai":
        return OpenAIModel(model_name)
    elif provider == "anthropic":
        return AnthropicModel(model_name)
    elif provider == "google":
        return GeminiModel(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use: openai, anthropic, google")


def is_reasoning_model(model_name: str) -> bool:
    """Check if the model is a reasoning model (o-series)."""
    return model_name.startswith(("o1", "o3"))


def get_provider_info() -> dict[str, str]:
    """Get current provider configuration info for display."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("MODEL_NAME", "gpt-4.1")
    return {
        "provider": provider,
        "model": model,
        "is_reasoning": str(is_reasoning_model(model)),
    }
