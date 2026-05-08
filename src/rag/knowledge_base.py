"""
Regulatory knowledge base: ChromaDB-backed RAG over CFPB corpus.
Supports metadata-filtered retrieval by regulation type.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings
from src.pipeline.embedding_provider import get_embedding_provider
from src.rag.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)

REGULATORY_DOCUMENTS = [
    {
        "id": "cfpb_exam_manual_complaint_mgmt",
        "regulation": "CFPB_EXAM_MANUAL",
        "title": "CFPB Supervision and Examination Manual - Complaint Management",
        "text": (
            "Institutions are required to maintain a documented complaint management program that includes: "
            "policies and procedures for receiving, tracking, and resolving complaints; escalation processes "
            "for complaints alleging regulatory violations; root cause analysis for systemic issues; "
            "regular reporting to senior management and board; and retention of complaint records for "
            "examination. Failure to maintain an adequate complaint management program is itself an "
            "examination finding that may lead to supervisory action."
        ),
    },
    {
        "id": "reg_e_dispute_timeline",
        "regulation": "REG_E",
        "title": "Regulation E - Electronic Fund Transfer Dispute Obligations (12 CFR 1005.11)",
        "text": (
            "Upon receipt of a consumer's oral or written notice of error, the institution must: "
            "acknowledge within 5 business days; investigate and determine whether error occurred; "
            "report results to consumer within 10 business days of receiving notice; "
            "if institution cannot complete investigation within 10 business days, it must "
            "provisionally credit the consumer's account within 10 business days for the amount alleged, "
            "and complete investigation within 45 days. For POS transactions, the 10-day period extends to "
            "20 days and the 45-day period extends to 90 days."
        ),
    },
    {
        "id": "reg_z_billing_dispute",
        "regulation": "REG_Z",
        "title": "Regulation Z - Billing Dispute Resolution (12 CFR 1026.13)",
        "text": (
            "A card issuer that has received a billing error notice must: "
            "acknowledge receipt within 30 days; resolve the dispute within 2 complete billing cycles "
            "(but not more than 90 days) after receiving the billing error notice. "
            "During the dispute period, the issuer may not report the amount as delinquent, "
            "accelerate the debt, or restrict or close the account solely because the amount is disputed. "
            "The issuer must either correct the error and credit finance charges, or "
            "provide a written explanation of why the billing statement is correct."
        ),
    },
    {
        "id": "fcra_dispute_investigation",
        "regulation": "FCRA",
        "title": "Fair Credit Reporting Act - Dispute Investigation (15 U.S.C. 1681i)",
        "text": (
            "When a consumer disputes the completeness or accuracy of information in a consumer report, "
            "the consumer reporting agency must conduct a reinvestigation within 30 days "
            "(45 days if the consumer submits additional information). "
            "The furnisher must investigate the dispute, review all relevant information, "
            "and report results to the CRA. Inaccurate, incomplete, or unverifiable information "
            "must be deleted or modified. The consumer must be notified of results within 5 business days "
            "of completion. Frivolous or irrelevant disputes may be dismissed with written notice within 5 days."
        ),
    },
    {
        "id": "ecoa_adverse_action",
        "regulation": "ECOA",
        "title": "Equal Credit Opportunity Act - Adverse Action Notice (15 U.S.C. 1691)",
        "text": (
            "The ECOA prohibits discrimination in any aspect of a credit transaction based on race, color, "
            "religion, national origin, sex, marital status, age, or receipt of public assistance. "
            "Adverse action notices must be provided within 30 days and must state specific reasons "
            "for the adverse action or inform applicants of their right to request specific reasons. "
            "Creditors must retain records for 25 months (12 months for business credit). "
            "Violations may result in actual damages, punitive damages up to $10,000, "
            "and class action damages up to the lesser of $500,000 or 1% of net worth."
        ),
    },
    {
        "id": "udaap_circular_2022",
        "regulation": "UDAAP",
        "title": "CFPB Circular 2022-06 - Unfair, Deceptive, Abusive Acts or Practices",
        "text": (
            "An act or practice is unfair if it causes or is likely to cause substantial injury to consumers, "
            "the injury is not reasonably avoidable by consumers, and the injury is not outweighed by "
            "countervailing benefits. An act or practice is deceptive if it involves a material representation, "
            "omission, act, or practice that is likely to mislead a reasonable consumer. "
            "An act or practice is abusive if it materially interferes with consumers' ability to understand "
            "a term or condition of a product or service, or takes unreasonable advantage of a consumer's "
            "lack of understanding, inability to protect their interests, or reasonable reliance. "
            "UDAAP violations may result in civil money penalties, restitution, and supervisory action."
        ),
    },
]


class RegulatoryKnowledgeBase:
    def __init__(self) -> None:
        cfg = get_settings()
        self._chroma = get_chroma_client()
        self._embedder = get_embedding_provider()
        self._collection_name = cfg.chroma_collection_regulatory

    def initialize(self) -> None:
        collection = self._chroma.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        existing_ids = set(collection.get()["ids"])
        new_docs = [d for d in REGULATORY_DOCUMENTS if d["id"] not in existing_ids]

        if not new_docs:
            logger.info("Regulatory corpus already initialized (%d docs)", len(existing_ids))
            return

        texts = [d["text"] for d in new_docs]
        embeddings = self._embedder.embed(texts)

        collection.upsert(
            ids=[d["id"] for d in new_docs],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"regulation": d["regulation"], "title": d["title"]} for d in new_docs],
        )
        logger.info("Regulatory corpus initialized with %d documents", len(new_docs))

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        regulation_filter: Optional[str] = None,
    ) -> list[dict]:
        collection = self._chroma.get_or_create_collection(name=self._collection_name)
        query_embedding = self._embedder.embed([query])[0]

        where = {"regulation": regulation_filter} if regulation_filter else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "text": doc,
                "regulation": meta.get("regulation"),
                "title": meta.get("title"),
                "relevance_score": 1.0 - dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def format_context(self, results: list[dict]) -> str:
        sections = []
        for r in results:
            sections.append(f"[{r['regulation']}] {r['title']}\n{r['text']}")
        return "\n\n---\n\n".join(sections)
