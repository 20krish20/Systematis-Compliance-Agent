"""
Shared ChromaDB client factory.
- "local" mode: PersistentClient (no Docker required, data stored in ./chroma_data)
- "server" mode: HttpClient (requires chromadb Docker container)
"""
from __future__ import annotations

from functools import lru_cache

import chromadb

from src.config.settings import get_settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    cfg = get_settings()
    if cfg.chroma_mode == "server":
        return chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
    return chromadb.PersistentClient(path=cfg.chroma_local_path)
