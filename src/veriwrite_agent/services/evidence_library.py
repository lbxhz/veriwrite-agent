"""Build and confirm the durable mixed-tier V0.3 literature library."""

from __future__ import annotations

from datetime import datetime, timezone

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceCard,
    EvidenceLibrary,
    LiteratureLibraryRecord,
)
from veriwrite_agent.services.literature_matrix import LiteratureMatrixBuilder
from veriwrite_agent.services.evidence_grounding import EvidenceGroundingValidator


class EvidenceLibraryBuilder:
    def build(
        self,
        *,
        records: list[LiteratureLibraryRecord],
        documents: list[DocumentAcquisition],
        pages: list[DocumentPage] | None = None,
        evidence_cards: list[EvidenceCard],
        unresolved_issues: list[str] | None = None,
    ) -> EvidenceLibrary:
        active_pages = pages or []
        issues = list(unresolved_issues or [])
        grounding = EvidenceGroundingValidator().validate(
            active_pages,
            evidence_cards,
        )
        issues.extend(
            f"{issue.code}:{issue.evidence_id}:{issue.detail}"
            for issue in grounding.issues
        )
        matrix = LiteratureMatrixBuilder().build(records, evidence_cards)
        return EvidenceLibrary(
            records=records,
            documents=documents,
            pages=active_pages,
            evidence_cards=evidence_cards,
            literature_matrix=matrix,
            unresolved_issues=issues,
        )


class EvidenceLibraryConfirmationService:
    """Apply one auditable user decision after all evidence cards are reviewed."""

    def confirm(
        self,
        library: EvidenceLibrary,
        *,
        confirmed_by: str,
    ) -> EvidenceLibrary:
        if library.unresolved_issues:
            raise ValueError("evidence library still has unresolved issues")
        confirmed_cards = [
            card.model_copy(update={"review_status": "confirmed"})
            for card in library.evidence_cards
            if card.review_status != "rejected"
        ]
        payload = library.model_dump()
        payload.update(
            {
                "status": "confirmed",
                "evidence_cards": confirmed_cards,
                "literature_matrix": LiteratureMatrixBuilder().build(
                    library.records,
                    confirmed_cards,
                ),
                "confirmed_by": confirmed_by.strip(),
                "confirmed_at": datetime.now(timezone.utc),
            }
        )
        return EvidenceLibrary.model_validate(payload)
