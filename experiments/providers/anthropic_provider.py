"""Anthropic provider — Claude Sonnet and Haiku.

Extracted from experiments/utils.py call_api() and call_api_json().
The original functions remain as thin wrappers for backward compatibility.
"""

import json
import time

import anthropic

from .base import LLMProvider


ANTHROPIC_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model_key: str = "sonnet"):
        if model_key not in ANTHROPIC_MODELS:
            raise ValueError(f"Unknown Anthropic model: {model_key}. "
                             f"Available: {list(ANTHROPIC_MODELS.keys())}")
        self._model_key = model_key
        self._model_id = ANTHROPIC_MODELS[model_key]
        self._client = anthropic.Anthropic()

    def call(self, system, user_content, max_tokens=1024, max_retries=3):
        """Call Claude API with retry logic."""
        for attempt in range(max_retries):
            try:
                msg = self._client.messages.create(
                    model=self._model_id,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                )
                if not msg.content:
                    print(f"  [WARN] Empty response (stop={msg.stop_reason}), retrying...")
                    time.sleep(2)
                    continue
                return msg.content[0].text.strip()
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                print(f"  [API ERROR] {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        return None

    def call_json(self, system, user_content, max_tokens=500, max_retries=3):
        """Call Claude API and parse JSON response."""
        for attempt in range(max_retries):
            try:
                msg = self._client.messages.create(
                    model=self._model_id,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                )
                text = msg.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                return json.loads(text)
            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"  [WARN] Parse failed after {max_retries} attempts: {e}")
                return None
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                print(f"  [API ERROR] {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        return None

    @property
    def client(self):
        """Expose raw client for darshana modules that need it directly."""
        return self._client

    @property
    def model_id(self):
        return self._model_id

    @property
    def short_name(self):
        return self._model_key

    @property
    def provider_name(self):
        return "anthropic"
