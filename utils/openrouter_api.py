import os
from typing import Any, List

import httpx
from safetytooling.data_models import Prompt, LLMResponse


class OpenRouterInferenceAPI:
    """Minimal async client for the OpenRouter API."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if self.api_key is None:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = base_url.rstrip("/")

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        **_: Any,
    ) -> List[LLMResponse]:
        """Send a chat completion request to OpenRouter."""
        messages = [{"role": m.role.value, "content": m.content} for m in prompt.messages]
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=None
            )
            resp.raise_for_status()
            data = resp.json()
            completion = data["choices"][0]["message"]["content"]
        return [LLMResponse(completion=completion)]
