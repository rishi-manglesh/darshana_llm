"""Abstract LLM Provider — unified interface for cross-model experiments.

Each provider wraps a specific API (Anthropic, OpenAI-compatible, etc.)
and translates to a common call()/call_json() interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Unified response from any provider."""
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Implementations must handle:
    - Retry logic with exponential backoff on rate limits
    - JSON parsing with fallback for unreliable models
    - Provider-specific message format translation
    """

    @abstractmethod
    def call(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Single-turn text generation.

        Args:
            system: System prompt
            user_content: User message
            max_tokens: Max response tokens

        Returns:
            Response text, or None on failure after retries
        """
        ...

    @abstractmethod
    def call_json(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 500,
    ) -> Optional[dict]:
        """Generate and parse JSON response.

        Args:
            system: System prompt (should instruct JSON output)
            user_content: User message
            max_tokens: Max response tokens

        Returns:
            Parsed dict, or None on failure after retries
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Full model identifier (e.g., 'claude-sonnet-4-5-20250929')."""
        ...

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short name for results keys (e.g., 'sonnet', 'llama70b')."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name (e.g., 'anthropic', 'together')."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.short_name})"
