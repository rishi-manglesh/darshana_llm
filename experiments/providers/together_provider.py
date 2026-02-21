"""Together AI provider — open-source models via OpenAI-compatible API.

Covers: Llama 3.3 70B, Qwen 2.5 72B, Mistral Large 2.
All accessed through Together's OpenAI-compatible endpoint.
"""

import json
import os
import time

from .base import LLMProvider


TOGETHER_MODELS = {
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "mistral_large": "mistralai/Mistral-Large-Instruct-2411",
}


class TogetherProvider(LLMProvider):
    """Together AI provider using OpenAI-compatible API."""

    def __init__(self, model_key: str = "llama70b"):
        if model_key not in TOGETHER_MODELS:
            raise ValueError(f"Unknown Together model: {model_key}. "
                             f"Available: {list(TOGETHER_MODELS.keys())}")

        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TOGETHER_API_KEY not set. Get one at https://api.together.xyz/"
            )

        # Lazy import — openai package only needed for Together provider
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for Together provider. "
                "Install with: pip install openai"
            )

        self._model_key = model_key
        self._model_id = TOGETHER_MODELS[model_key]
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )

    def call(self, system, user_content, max_tokens=1024, max_retries=3):
        """Call Together API (OpenAI-compatible) with retry logic."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                text = response.choices[0].message.content
                if not text:
                    print(f"  [WARN] Empty response, retrying...")
                    time.sleep(2)
                    continue
                return text.strip()
            except Exception as e:
                error_str = str(e)
                if "rate" in error_str.lower() or "429" in error_str:
                    wait = 2 ** (attempt + 1)
                    print(f"  [RATE LIMIT] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [API ERROR] {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
        return None

    def call_json(self, system, user_content, max_tokens=500, max_retries=3):
        """Call Together API and parse JSON response.

        Open-source models are less reliable at JSON — extra parsing fallbacks.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                text = response.choices[0].message.content
                if not text:
                    print(f"  [WARN] Empty response, retrying...")
                    time.sleep(2)
                    continue

                text = text.strip()
                return self._parse_json(text)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"  [WARN] Parse failed after {max_retries} attempts: {e}")
                return None
            except Exception as e:
                error_str = str(e)
                if "rate" in error_str.lower() or "429" in error_str:
                    wait = 2 ** (attempt + 1)
                    print(f"  [RATE LIMIT] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [API ERROR] {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
        return None

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON with fallbacks for open-source model quirks.

        Handles: markdown fences, leading text before JSON, trailing text after.
        """
        # Strip markdown code fences
        if "```" in text:
            # Find content between first ``` and last ```
            parts = text.split("```")
            for part in parts[1:]:
                # Skip the language label (e.g., "json\n")
                content = part.strip()
                if content.startswith("json"):
                    content = content[4:].strip()
                elif content.startswith("JSON"):
                    content = content[4:].strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    @property
    def model_id(self):
        return self._model_id

    @property
    def short_name(self):
        return self._model_key

    @property
    def provider_name(self):
        return "together"
