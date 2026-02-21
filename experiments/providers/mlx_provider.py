"""MLX local provider — open-source models on Apple Silicon.

Runs models locally via mlx-lm. No API keys needed.
Requires: pip install mlx-lm

Models are auto-downloaded from HuggingFace on first use (~2-20GB per model).
Subsequent runs load from cache (~/.cache/huggingface/).
"""

from __future__ import annotations

import json
import time
from typing import Optional

from .base import LLMProvider


# Latest models as of Feb 2026
# Local models sized for 48GB M4 Mac
MLX_MODELS = {
    # Qwen3 family — best for structured output / JSON / instruction following
    "qwen3_8b":       "mlx-community/Qwen3-8B-4bit",
    "qwen3_32b":      "mlx-community/Qwen3-32B-4bit",

    # Mistral Small 3.2 — native tool use, strong instruction following
    "mistral_small":  "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
}

# RAM requirements (approximate, 4-bit quantized)
MLX_RAM_GB = {
    "qwen3_8b": 6,
    "qwen3_32b": 20,
    "mistral_small": 14,
}


class MLXProvider(LLMProvider):
    """Local MLX provider for Apple Silicon inference."""

    def __init__(self, model_key: str = "qwen3_8b"):
        if model_key not in MLX_MODELS:
            raise ValueError(
                f"Unknown MLX model: {model_key}. "
                f"Available: {list(MLX_MODELS.keys())}"
            )

        try:
            from mlx_lm import load, generate as mlx_generate
        except ImportError:
            raise ImportError(
                "mlx-lm package required. Install with: pip install mlx-lm"
            )

        self._model_key = model_key
        self._model_id = MLX_MODELS[model_key]
        self._mlx_generate = mlx_generate

        print(f"  Loading {model_key} ({self._model_id})...")
        t0 = time.time()
        self._model, self._tokenizer = load(self._model_id)
        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.1f}s")

    def _format_messages(self, system: str, user_content: str) -> str:
        """Format system + user messages using the model's chat template."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        # Disable thinking for Qwen3 models — saves ~50% generation time
        content = user_content
        if "qwen3" in self._model_key.lower():
            content = "/no_think\n" + user_content
        messages.append({"role": "user", "content": content})

        # Use the tokenizer's built-in chat template
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without chat template
            if system:
                prompt = f"System: {system}\n\nUser: {user_content}\n\nAssistant: "
            else:
                prompt = f"User: {user_content}\n\nAssistant: "

        return prompt

    def call(self, system, user_content, max_tokens=1024, max_retries=2):
        """Generate text locally via MLX."""
        prompt = self._format_messages(system, user_content)

        for attempt in range(max_retries):
            try:
                response = self._mlx_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                if response:
                    return self._strip_think(response.strip())
                print(f"  [WARN] Empty response, retrying...")
            except Exception as e:
                print(f"  [MLX ERROR] {e}")
                if attempt < max_retries - 1:
                    continue
                return None
        return None

    def call_json(self, system, user_content, max_tokens=500, max_retries=3):
        """Generate and parse JSON response locally.

        Local models are less reliable at JSON — extra parsing with fallbacks.
        """
        prompt = self._format_messages(system, user_content)

        for attempt in range(max_retries):
            try:
                text = self._mlx_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                if not text:
                    continue
                return self._parse_json(text.strip())
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    continue
                print(f"  [WARN] JSON parse failed after {max_retries} attempts: {e}")
                return None
            except Exception as e:
                print(f"  [MLX ERROR] {e}")
                if attempt < max_retries - 1:
                    continue
                return None
        return None

    @staticmethod
    def _strip_think(text: str) -> str:
        """Strip <think>...</think> blocks from reasoning models (Qwen3, etc.)."""
        import re
        # Strip closed think blocks
        cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        # Strip unclosed think block (model ran out of tokens mid-thought)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        return cleaned.strip() if cleaned.strip() else text

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON with fallbacks for local model quirks."""
        # Strip markdown code fences
        if "```" in text:
            parts = text.split("```")
            for part in parts[1:]:
                content = part.strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    continue

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found", text, 0)

    @property
    def model_id(self):
        return self._model_id

    @property
    def short_name(self):
        return self._model_key

    @property
    def provider_name(self):
        return "mlx"
