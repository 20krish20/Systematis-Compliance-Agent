"""
PII masking pipeline using spaCy NER + regex patterns.
All complaint narratives must pass through this before any LLM ingestion.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import spacy

# ─── Regex patterns for financial PII ────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")),
    ("ACCOUNT_NUM", re.compile(r"\b(?:account|acct|acct#|acc)\s*[#:]?\s*\d{4,17}\b", re.IGNORECASE)),
    ("CARD_NUM", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("ROUTING_NUM", re.compile(r"\b\d{9}\b")),
    ("PHONE", re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("DOB", re.compile(r"\b(?:0?[1-9]|1[0-2])[\/\-](?:0?[1-9]|[12]\d|3[01])[\/\-](?:19|20)\d{2}\b")),
    ("DOLLAR_AMOUNT_LARGE", re.compile(r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")),
]

_SPACY_PII_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC"}


@dataclass
class MaskingResult:
    masked_text: str
    original_length: int
    masked_entities: list[dict[str, str]] = field(default_factory=list)
    pii_detected: bool = False


@lru_cache(maxsize=1)
def _load_spacy_model() -> spacy.Language:
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        return spacy.load("en_core_web_sm")


class PIIMasker:
    def __init__(self, mask_dollar_amounts: bool = False) -> None:
        self._mask_dollars = mask_dollar_amounts
        self._nlp = _load_spacy_model()

    def mask(self, text: str) -> MaskingResult:
        if not text or not text.strip():
            return MaskingResult(masked_text="", original_length=0)

        masked = text
        entities: list[dict[str, str]] = []

        # Run spaCy NER on the ORIGINAL text first, before any regex substitution.
        # This prevents spaCy from re-tagging placeholder tokens like "[SSN]" as entities.
        doc = self._nlp(text)
        spacy_replacements: list[tuple[int, int, str]] = []
        for ent in doc.ents:
            if ent.label_ in _SPACY_PII_LABELS:
                spacy_replacements.append((ent.start_char, ent.end_char, f"[{ent.label_}]"))
                entities.append({"type": ent.label_, "value": ent.text})

        # Apply spaCy replacements in reverse to preserve character offsets
        for start, end, placeholder in sorted(spacy_replacements, key=lambda x: x[0], reverse=True):
            masked = masked[:start] + placeholder + masked[end:]

        # Regex-based masking over the (possibly spaCy-modified) text
        for label, pattern in _PATTERNS:
            if label == "DOLLAR_AMOUNT_LARGE" and not self._mask_dollars:
                continue
            for match in pattern.finditer(masked):
                entities.append({"type": label, "value": match.group()})
            masked = pattern.sub(f"[{label}]", masked)

        return MaskingResult(
            masked_text=masked,
            original_length=len(text),
            masked_entities=entities,
            pii_detected=len(entities) > 0,
        )

    @staticmethod
    def fingerprint(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def mask_batch(self, texts: list[str]) -> list[MaskingResult]:
        return [self.mask(t) for t in texts]
