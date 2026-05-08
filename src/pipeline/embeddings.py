"""
Batch embedding pipeline for CFPB complaint narratives into ChromaDB.
Supports checkpointed ingestion for large datasets (3.5M+ records).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from tqdm import tqdm

from src.config.settings import get_settings
from src.pipeline.embedding_provider import get_embedding_provider
from src.rag.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = get_chroma_client()
        self._embedder = get_embedding_provider()
        self._batch_size = cfg.embedding_batch_size
        self._collection_name = cfg.chroma_collection_complaints

    def get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def ingest_from_parquet(
        self,
        parquet_path: str,
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
    ) -> int:
        collection = self.get_or_create_collection()
        df = pd.read_parquet(parquet_path)
        df = df[df["masked_text"].notna() & (df["masked_text"] != "")]

        checkpoint_file = Path(checkpoint_path or f"{parquet_path}.checkpoint.json")
        start_idx = 0
        if resume and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                start_idx = json.load(f).get("last_index", 0)
            logger.info("Resuming from checkpoint at index %d", start_idx)

        total_ingested = start_idx
        batch_texts, batch_ids, batch_metas = [], [], []

        for idx, row in tqdm(
            df.iloc[start_idx:].iterrows(),
            total=len(df) - start_idx,
            desc="Embedding narratives",
        ):
            batch_texts.append(str(row["masked_text"])[:2000])
            batch_ids.append(str(row["id"]))
            batch_metas.append({
                "product": str(row.get("metadata", {}).get("product_raw", "")),
                "issue": str(row.get("metadata", {}).get("issue", "")),
                "state": str(row.get("state", "") or ""),
                "zip_code": str(row.get("zip_code", "") or ""),
                "cfpb_id": str(row.get("cfpb_id", "") or ""),
            })

            if len(batch_texts) >= self._batch_size:
                embeddings = self.embed_batch(batch_texts)
                collection.upsert(
                    documents=batch_texts,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_metas,
                )
                total_ingested += len(batch_texts)
                batch_texts, batch_ids, batch_metas = [], [], []

                with open(checkpoint_file, "w") as f:
                    json.dump({"last_index": total_ingested}, f)

        # Final batch
        if batch_texts:
            embeddings = self.embed_batch(batch_texts)
            collection.upsert(
                documents=batch_texts,
                embeddings=embeddings,
                ids=batch_ids,
                metadatas=batch_metas,
            )
            total_ingested += len(batch_texts)

        checkpoint_file.unlink(missing_ok=True)
        logger.info("Embedding ingestion complete: %d records", total_ingested)
        return total_ingested

    def similarity_search(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        collection = self.get_or_create_collection()
        query_embedding = self.embed_batch([query_text])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
