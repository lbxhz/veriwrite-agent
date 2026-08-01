"""LLM semantic extraction constrained by deterministic PDF grounding."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.evidence import (
    DocumentPage,
    EvidenceCard,
    EvidenceCardProposal,
    EvidencePassageSelectionBatch,
    EvidenceQuote,
)
from veriwrite_agent.services.evidence_grounding import EvidenceGroundingValidator


class EvidenceCardExtractionError(ValueError):
    """Raised when LLM evidence cannot be tied back to supplied PDF pages."""


@dataclass(frozen=True)
class _SourcePassage:
    passage_id: str
    page_number: int
    exact_text: str


class LLMEvidenceCardExtractor:
    """Extract semantic cards while code retains identity and scope authority."""

    def __init__(
        self,
        client: LLMClient,
        *,
        page_batch_size: int = 3,
        max_chars_per_page: int = 8000,
        max_cards_per_batch: int = 4,
        max_quote_chars: int = 400,
    ) -> None:
        if not 1 <= page_batch_size <= 12:
            raise ValueError("page_batch_size must be between 1 and 12")
        if not 1 <= max_cards_per_batch <= 12:
            raise ValueError("max_cards_per_batch must be between 1 and 12")
        if not 80 <= max_quote_chars <= 1500:
            raise ValueError("max_quote_chars must be between 80 and 1500")
        self._client = client
        self._page_batch_size = page_batch_size
        self._max_chars_per_page = max_chars_per_page
        self._max_cards_per_batch = max_cards_per_batch
        self._max_quote_chars = max_quote_chars

    def extract(
        self,
        *,
        doi: str,
        title: str,
        theme_id: str,
        section_purpose: str,
        pages: list[DocumentPage],
    ) -> list[EvidenceCard]:
        if not pages:
            raise ValueError("evidence extraction requires PDF pages")
        hashes = {page.document_sha256 for page in pages}
        dois = {page.doi for page in pages}
        if len(hashes) != 1 or dois != {doi}:
            raise ValueError("all pages must belong to one DOI and one PDF hash")

        proposals = []
        ordered_pages = sorted(pages, key=lambda page: page.page_number)
        for start in range(0, len(ordered_pages), self._page_batch_size):
            batch_pages = ordered_pages[start : start + self._page_batch_size]
            proposals.extend(
                self._extract_batch(
                    title=title,
                    theme_id=theme_id,
                    section_purpose=section_purpose,
                    pages=batch_pages,
                )
            )

        cards: list[EvidenceCard] = []
        seen_claims: set[str] = set()
        for proposal in proposals:
            fingerprint = " ".join(proposal.normalized_claim.casefold().split())
            if fingerprint in seen_claims:
                continue
            seen_claims.add(fingerprint)
            digest = hashlib.sha1(
                (doi + theme_id + proposal.evidence_type + fingerprint).encode("utf-8")
            ).hexdigest()[:10]
            cards.append(
                EvidenceCard(
                    evidence_id=f"ev_{theme_id}_{digest}",
                    doi=doi,
                    theme_id=theme_id,
                    evidence_type=proposal.evidence_type,
                    normalized_claim=proposal.normalized_claim,
                    supporting_quotes=proposal.supporting_quotes,
                    source_document_sha256=next(iter(hashes)),
                    support_strength=proposal.support_strength,
                )
            )

        grounding = EvidenceGroundingValidator().validate(pages, cards)
        if not grounding.valid:
            details = "; ".join(issue.detail for issue in grounding.issues[:5])
            raise EvidenceCardExtractionError(
                f"LLM evidence quotes failed PDF grounding: {details}"
            )
        return cards

    def _extract_batch(
        self,
        *,
        title: str,
        theme_id: str,
        section_purpose: str,
        pages: list[DocumentPage],
    ) -> list:
        page_numbers = {page.page_number for page in pages}
        passages = _build_passages(
            pages,
            max_chars_per_page=self._max_chars_per_page,
            max_quote_chars=self._max_quote_chars,
        )
        passage_by_id = {passage.passage_id: passage for passage in passages}
        payload = {
            "paper_title": title,
            "theme_id": theme_id,
            "section_purpose": section_purpose,
            "passages": [
                {
                    "passage_id": passage.passage_id,
                    "page_number": passage.page_number,
                    "exact_text": passage.exact_text,
                }
                for passage in passages
            ],
        }
        schema = json.dumps(
            EvidencePassageSelectionBatch.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw = self._client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You select evidence passages from a PDF and return JSON only. "
                        "Use only the supplied passage_ids; never quote, rewrite, merge, "
                        "or invent source text. Code, not you, binds selected IDs back "
                        "to exact PDF text. Classify each selection as background, "
                        "research_object, data, method, result, limitation, or "
                        "future_work. normalized_claim may summarize only what the "
                        "selected passages support. Return no selection when evidence "
                        "is unclear. "
                        f"Return at most {self._max_cards_per_batch} selections. "
                        f"The output must satisfy this JSON Schema: {schema}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            parsed = EvidencePassageSelectionBatch.model_validate_json(raw)
        except ValidationError as exc:
            raise EvidenceCardExtractionError(
                "LLM evidence output violates the data contract: "
                f"{exc.errors(include_url=False)[:8]}"
            ) from exc
        if len(parsed.selections) > self._max_cards_per_batch:
            raise EvidenceCardExtractionError(
                "LLM evidence output exceeded the per-batch card limit"
            )
        selected_ids = {
            passage_id for selection in parsed.selections for passage_id in selection.passage_ids
        }
        unknown_ids = selected_ids - passage_by_id.keys()
        if unknown_ids:
            raise EvidenceCardExtractionError(
                f"LLM evidence output referenced unknown passage IDs: {sorted(unknown_ids)}"
            )
        proposals = []
        for selection in parsed.selections:
            quotes = [
                EvidenceQuote(
                    page_number=passage_by_id[passage_id].page_number,
                    exact_text=passage_by_id[passage_id].exact_text,
                )
                for passage_id in selection.passage_ids
            ]
            if any(quote.page_number not in page_numbers for quote in quotes):
                raise EvidenceCardExtractionError(
                    "selected passage belongs to a page outside the supplied batch"
                )
            proposals.append(
                EvidenceCardProposal(
                    evidence_type=selection.evidence_type,
                    normalized_claim=selection.normalized_claim,
                    supporting_quotes=quotes,
                    support_strength=selection.support_strength,
                )
            )
        return proposals


def _build_passages(
    pages: list[DocumentPage],
    *,
    max_chars_per_page: int,
    max_quote_chars: int,
) -> list[_SourcePassage]:
    """Create stable, code-owned excerpts that the LLM can select by ID."""

    passages: list[_SourcePassage] = []
    for page in pages:
        normalized = " ".join(page.text[:max_chars_per_page].split())
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", normalized)
        buffer = ""
        page_passages: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_quote_chars:
                if buffer:
                    page_passages.append(buffer)
                    buffer = ""
                page_passages.extend(
                    sentence[start : start + max_quote_chars]
                    for start in range(0, len(sentence), max_quote_chars)
                )
                continue
            candidate = f"{buffer} {sentence}".strip()
            if buffer and len(candidate) > max_quote_chars:
                page_passages.append(buffer)
                buffer = sentence
            else:
                buffer = candidate
        if buffer:
            page_passages.append(buffer)
        for index, exact_text in enumerate(page_passages, 1):
            if len(exact_text) < 40:
                continue
            passages.append(
                _SourcePassage(
                    passage_id=f"page_{page.page_number}_passage_{index}",
                    page_number=page.page_number,
                    exact_text=exact_text,
                )
            )
    return passages
