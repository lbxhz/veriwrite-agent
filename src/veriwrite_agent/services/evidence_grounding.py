"""Deterministic checks that prevent V0.3 evidence cards from inventing quotes."""

from __future__ import annotations

import unicodedata
from collections import defaultdict

from veriwrite_agent.models.evidence import (
    DocumentPage,
    EvidenceCard,
    GroundingIssue,
    GroundingReport,
)


class EvidenceGroundingValidator:
    """Verify every quoted excerpt against the claimed PDF page and file hash."""

    def validate(
        self,
        pages: list[DocumentPage],
        cards: list[EvidenceCard],
    ) -> GroundingReport:
        pages_by_location: dict[tuple[str, int], list[DocumentPage]] = defaultdict(list)
        for page in pages:
            pages_by_location[(page.doi, page.page_number)].append(page)

        issues: list[GroundingIssue] = []
        for card in cards:
            for quote in card.supporting_quotes:
                location_pages = pages_by_location.get(
                    (card.doi, quote.page_number),
                    [],
                )
                if not location_pages:
                    issues.append(
                        GroundingIssue(
                            evidence_id=card.evidence_id,
                            code="document_page_missing",
                            detail=(
                                f"Page {quote.page_number} was not extracted for "
                                f"DOI {card.doi}."
                            ),
                        )
                    )
                    continue

                matching_pages = [
                    page
                    for page in location_pages
                    if page.document_sha256 == card.source_document_sha256
                ]
                if not matching_pages:
                    issues.append(
                        GroundingIssue(
                            evidence_id=card.evidence_id,
                            code="document_identity_mismatch",
                            detail=(
                                f"Page {quote.page_number} belongs to a different "
                                "PDF hash than the evidence card."
                            ),
                        )
                    )
                    continue

                normalized_quote = _normalize_text(quote.exact_text)
                if not any(
                    normalized_quote in _normalize_text(page.text)
                    for page in matching_pages
                ):
                    issues.append(
                        GroundingIssue(
                            evidence_id=card.evidence_id,
                            code="quote_not_found_on_page",
                            detail=(
                                f"The quoted text was not found on page "
                                f"{quote.page_number} of the claimed PDF."
                            ),
                        )
                    )

        return GroundingReport(valid=not issues, issues=issues)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())
