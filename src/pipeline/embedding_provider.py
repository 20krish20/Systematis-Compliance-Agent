"""
Embedding provider with two backends:
  - "local"  → sentence-transformers (all-MiniLM-L6-v2, 384-dim, no API key needed)
  - "openai" → OpenAI text-embedding-3-small (1536-dim, requires OPENAI_API_KEY + quota)

Default is "local" so the project works without any paid API keys.
Set EMBEDDING_PROVIDER=openai in .env for production-grade embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIM = 384
OPENAI_EMBEDDING_DIM = 1536


@lru_cache(maxsize=1)
def _load_local_model():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading local embedding model '%s' (first call may download ~90MB)", LOCAL_MODEL_NAME)
    return SentenceTransformer(LOCAL_MODEL_NAME)


class EmbeddingProvider:
    def __init__(self, provider: str = "local") -> None:
        self._provider = provider
        if provider == "openai":
            from src.config.settings import get_settings
            from openai import OpenAI
            cfg = get_settings()
            self._openai = OpenAI(api_key=cfg.openai_api_key)
            self._model_name = cfg.embedding_model
            self.dim = OPENAI_EMBEDDING_DIM
        else:
            self.dim = LOCAL_EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = _load_local_model()
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        response = self._openai.embeddings.create(model=self._model_name, input=texts)
        return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    from src.config.settings import get_settings
    cfg = get_settings()
    provider = getattr(cfg, "embedding_provider", "local")
    logger.info("Embedding provider: %s", provider)
    return EmbeddingProvider(provider=provider)
