"""Unit tests for PII masking pipeline."""
from __future__ import annotations

import pytest

from src.pipeline.pii_masker import PIIMasker


@pytest.fixture
def masker() -> PIIMasker:
    return PIIMasker()


def test_ssn_masked(masker, pii_narrative):
    result = masker.mask(pii_narrative)
    assert "123-45-6789" not in result.masked_text
    assert "[SSN]" in result.masked_text


def test_email_masked(masker, pii_narrative):
    result = masker.mask(pii_narrative)
    assert "john.smith@example.com" not in result.masked_text
    assert "[EMAIL]" in result.masked_text


def test_phone_masked(masker, pii_narrative):
    result = masker.mask(pii_narrative)
    assert "555-867-5309" not in result.masked_text


def test_card_number_masked(masker, pii_narrative):
    result = masker.mask(pii_narrative)
    assert "4111111111111111" not in result.masked_text
    assert "[CARD_NUM]" in result.masked_text


def test_pii_detected_flag(masker, pii_narrative):
    result = masker.mask(pii_narrative)
    assert result.pii_detected is True
    assert len(result.masked_entities) > 0


def test_clean_narrative_no_pii(masker):
    text = "The bank took too long to respond to my billing dispute last month."
    result = masker.mask(text)
    assert result.masked_text == text or result.pii_detected is False


def test_fingerprint_deterministic():
    text = "Same complaint text twice"
    assert PIIMasker.fingerprint(text) == PIIMasker.fingerprint(text)


def test_fingerprint_normalized():
    text1 = "Same complaint text"
    text2 = "  Same   complaint   text  "
    assert PIIMasker.fingerprint(text1) == PIIMasker.fingerprint(text2)


def test_empty_text(masker):
    result = masker.mask("")
    assert result.masked_text == ""
    assert result.original_length == 0


def test_batch_masking(masker):
    texts = ["Call 555-123-4567 for info", "No PII here at all"]
    results = masker.mask_batch(texts)
    assert len(results) == 2
    assert "555-123-4567" not in results[0].masked_text
