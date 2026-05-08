"""
CLI script: Download and ingest CFPB Consumer Complaint Database.
Usage: python scripts/ingest_cfpb.py --sample 50000 --output data/cfpb_processed.parquet
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
@click.option("--input", "input_path", default="data/cfpb_raw/complaints.csv", help="Path to raw CFPB CSV file")
@click.option("--output", "output_path", default="data/cfpb_processed.parquet", help="Output parquet path")
@click.option("--sample", default=None, type=int, help="Sample size (None = all records)")
@click.option("--chunk-size", default=10_000, help="Chunk size for streaming ingestion")
def main(input_path: str, output_path: str, sample: int | None, chunk_size: int) -> None:
    """Ingest, deduplicate, PII-mask, and normalize CFPB complaint data."""
    from src.pipeline.data_ingestion import CFPBIngestionPipeline

    logger.info("Starting CFPB ingestion pipeline")
    logger.info("Input: %s, Output: %s, Sample: %s", input_path, output_path, sample)

    pipeline = CFPBIngestionPipeline(
        raw_data_path=input_path,
        output_path=output_path,
        sample_size=sample,
        chunk_size=chunk_size,
    )

    count = pipeline.run()
    logger.info("Ingestion complete: %d records → %s", count, output_path)


if __name__ == "__main__":
    main()
