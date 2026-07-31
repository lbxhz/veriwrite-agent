"""LLM semantic extraction constrained by deterministic PDF grounding."""

from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.evidence import (
    DocumentPage,
    EvidenceCard,
    EvidenceCardProposalBatch,
)
from veriwrite_agent.services.evidence_grounding import EvidenceGroundingValidator


class EvidenceCardExtractionError(ValueError):
    """Raised when LLM evidence cannot be tied back to supplied PDF pages."""


class LLMEvidenceCardExtractor:
    """Extract semantic cards while code retains identity and scope authority."""

    def __init__(
        self,
        client: LLMClient,
        *,
        page_batch_size: int = 6,
        max_chars_per_page: int = 7000,
    ) -> None:
        if not 1 <= page_batch_size <= 12:
            raise ValueError("page_batch_size must be between 1 and 12")
        self._client = client
        self._page_batch_size = page_batch_size
        self._max_chars_per_page = max_chars_per_page

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
                (
                    doi
                    + theme_id
                    + proposal.evidence_type
                    + fingerprint
                ).encode("utf-8")
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
        payload = {
            "paper_title": title,
            "theme_id": theme_id,
            "section_purpose": section_purpose,
            "pages": [
                {
                    "page_number": page.page_number,
                    "text": page.text[: self._max_chars_per_page],
                }
                for page in pages
            ],
        }
        schema = json.dumps(
            EvidenceCardProposalBatch.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw = self._client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你是论文证据卡提取器，只返回JSON。"
                        "只能使用用户提供的PDF页面文本，不得补充记忆或外部知识。"
                        "每张卡归类为background、research_object、data、method、"
                        "result、limitation或future_work。"
                        "supporting_quotes必须逐字复制给定页面中的短原文，"
                        "页码必须来自给定页面。没有明确证据就不要生成卡片。"
                        f"输出必须符合JSON Schema：{schema}"
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
            parsed = EvidenceCardProposalBatch.model_validate_json(raw)
        except ValidationError as exc:
            raise EvidenceCardExtractionError(
                "LLM evidence output violates the data contract: "
                f"{exc.errors(include_url=False)[:8]}"
            ) from exc
        if any(
            quote.page_number not in page_numbers
            for proposal in parsed.proposals
            for quote in proposal.supporting_quotes
        ):
            raise EvidenceCardExtractionError(
                "LLM evidence output referenced a page outside the supplied batch"
            )
        return parsed.proposals
