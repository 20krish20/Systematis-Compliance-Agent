"""
CFPB Consumer Complaint Database ingestion pipeline.
Handles download, deduplication, PII masking, and normalization.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from src.pipeline.pii_masker import PIIMasker
from src.schemas.models import ComplaintRecord, ProductCategory

logger = logging.getLogger(__name__)

CFPB_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
]

PRODUCT_TAXONOMY_MAP: dict[str, str] = {
    "Credit card": ProductCategory.CREDIT_CARD,
    "Credit card or prepaid card": ProductCategory.CREDIT_CARD,
    "Mortgage": ProductCategory.MORTGAGE,
    "Student loan": ProductCategory.STUDENT_LOAN,
    "Vehicle loan or lease": ProductCategory.AUTO_LOAN,
    "Checking or savings account": ProductCategory.CHECKING_SAVINGS,
    "Payday loan, title loan, or personal loan": ProductCategory.PERSONAL_LOAN,
    "Debt collection": ProductCategory.DEBT_COLLECTION,
    "Credit reporting, credit repair services, or other personal consumer reports": ProductCategory.CREDIT_REPORTING,
    "Money transfer, virtual currency, or money service": ProductCategory.MONEY_TRANSFER,
    "Payday loan": ProductCategory.PAYDAY_LOAN,
    "Prepaid card": ProductCategory.PREPAID_CARD,
}


class CFPBIngestionPipeline:
    def __init__(
        self,
        raw_data_path: str,
        output_path: str,
        sample_size: Optional[int] = None,
        chunk_size: int = 10_000,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self._masker = PIIMasker()
        self._seen_fingerprints: set[str] = set()

    def run(self) -> int:
        logger.info("Starting CFPB ingestion from %s", self.raw_data_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        total_processed = 0
        total_deduplicated = 0
        records: list[dict] = []

        for chunk in self._read_chunks():
            for _, row in chunk.iterrows():
                narrative = str(row.get("Consumer complaint narrative", "") or "")
                if not narrative or narrative.lower() in {"nan", "none", ""}:
                    continue

                fingerprint = PIIMasker.fingerprint(narrative)
                if fingerprint in self._seen_fingerprints:
                    total_deduplicated += 1
                    continue
                self._seen_fingerprints.add(fingerprint)

                mask_result = self._masker.mask(narrative)
                record = ComplaintRecord(
                    cfpb_id=str(row.get("Complaint ID", "")),
                    raw_text=narrative,
                    masked_text=mask_result.masked_text,
                    sha256_fingerprint=fingerprint,
                    metadata={
                        "product_raw": str(row.get("Product", "")),
                        "sub_product": str(row.get("Sub-product", "")),
                        "issue": str(row.get("Issue", "")),
                        "sub_issue": str(row.get("Sub-issue", "")),
                        "company": str(row.get("Company", "")),
                        "company_response": str(row.get("Company response to consumer", "")),
                        "timely_response": str(row.get("Timely response?", "")),
                        "consumer_disputed": str(row.get("Consumer disputed?", "")),
                        "submitted_via": str(row.get("Submitted via", "")),
                    },
                    state=str(row.get("State", "")) or None,
                    zip_code=str(row.get("ZIP code", "")) or None,
                    submitted_via=str(row.get("Submitted via", "")) or None,
                )
                records.append(record.model_dump(mode="json"))
                total_processed += 1

                if self.sample_size and total_processed >= self.sample_size:
                    break

            if self.sample_size and total_processed >= self.sample_size:
                break

        df = pd.DataFrame(records)
        df.to_parquet(self.output_path, index=False, compression="snappy")

        logger.info(
            "Ingestion complete: %d records processed, %d deduplicated, saved to %s",
            total_processed,
            total_deduplicated,
            self.output_path,
        )
        return total_processed

    def _read_chunks(self) -> Iterator[pd.DataFrame]:
        suffix = self.raw_data_path.suffix.lower()
        if suffix == ".csv":
            reader = pd.read_csv(
                self.raw_data_path,
                chunksize=self.chunk_size,
                dtype=str,
                on_bad_lines="skip",
            )
            yield from reader
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(self.raw_data_path)
            for i in range(0, len(df), self.chunk_size):
                yield df.iloc[i : i + self.chunk_size]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def normalize_product(raw_product: str) -> str:
        for key, normalized in PRODUCT_TAXONOMY_MAP.items():
            if key.lower() in raw_product.lower():
                return normalized
        return ProductCategory.OTHER
