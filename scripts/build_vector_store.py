"""
CLI script: Build ChromaDB vector stores for complaints + regulatory corpus.
Usage: python scripts/build_vector_store.py --parquet data/cfpb_processed.parquet
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--parquet", "parquet_path", default="data/cfpb_processed.parquet", help="Processed complaints parquet")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoint if available")
@click.option("--regulatory/--no-regulatory", default=True, help="Also initialize regulatory corpus")
def main(parquet_path: str, resume: bool, regulatory: bool) -> None:
    """Embed complaint narratives and regulatory corpus into ChromaDB."""
    if regulatory:
        logger.info("Initializing regulatory knowledge base...")
        from src.rag.knowledge_base import RegulatoryKnowledgeBase
        kb = RegulatoryKnowledgeBase()
        kb.initialize()
        logger.info("Regulatory corpus ready")

    logger.info("Starting complaint embedding pipeline from %s", parquet_path)
    from src.pipeline.embeddings import EmbeddingPipeline
    pipeline = EmbeddingPipeline()
    count = pipeline.ingest_from_parquet(parquet_path=parquet_path, resume=resume)
    logger.info("Embedding complete: %d documents in ChromaDB", count)


if __name__ == "__main__":
    main()
