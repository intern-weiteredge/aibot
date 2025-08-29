from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from app.config import AppConfig


class EmbeddingProvider:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._provider = cfg.embeddings_provider.lower()
        self._st_model = None
        self._openai_client = None

    @property
    def provider_name(self) -> str:
        return self._provider

    def _ensure_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            # Force CPU to avoid device/meta tensor issues on some Windows setups
            self._st_model = SentenceTransformer(self.cfg.default_st_model, device="cpu")

    def _ensure_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.cfg.openai_api_key)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts_list:
            return np.zeros((0, 384), dtype=np.float32)

        if self._provider == "openai":
            self._ensure_openai_client()
            # Use text-embedding-3-small by default for cost efficiency unless user overrides ST_MODEL var for local
            model = "text-embedding-3-small"
            res = self._openai_client.embeddings.create(model=model, input=texts_list)
            vectors = [d.embedding for d in res.data]
            return np.array(vectors, dtype=np.float32)
        else:
            self._ensure_st_model()
            vectors = self._st_model.encode(texts_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype(np.float32, copy=False)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


