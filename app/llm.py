from __future__ import annotations

from typing import Optional

from app.config import AppConfig


class LLMFallback:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._client = None

    def _ensure(self):
        key = (self.cfg.openai_api_key or "").strip()
        if self._client is None and key:
            from openai import OpenAI

            self._client = OpenAI(api_key=key)

    def available(self) -> bool:
        return bool((self.cfg.openai_api_key or "").strip())

    def answer(self, query: str, context: str | None = None) -> str:
        self._ensure()
        if not self._client:
            return "No fallback LLM configured."
        system = "You are a helpful assistant. Use provided context if it is relevant."
        user = query if not context else f"Question: {query}\n\nRelevant context (may be partial):\n{context}"
        resp = self._client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


