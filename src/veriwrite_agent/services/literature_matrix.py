"""Deterministic construction of evidence-linked literature matrix rows."""

from __future__ import annotations

from collections import defaultdict

from veriwrite_agent.models.evidence import (
    EvidenceBackedValue,
    EvidenceCard,
    LiteratureLibraryRecord,
    LiteratureMatrixRow,
)


class LiteratureMatrixBuilder:
    """Project grounded evidence cards into stable comparison dimensions."""

    def build(
        self,
        records: list[LiteratureLibraryRecord],
        cards: list[EvidenceCard],
    ) -> list[LiteratureMatrixRow]:
        cards_by_doi: dict[str, list[EvidenceCard]] = defaultdict(list)
        for card in cards:
            if card.review_status != "rejected":
                cards_by_doi[card.doi].append(card)

        rows: list[LiteratureMatrixRow] = []
        for record in records:
            dimensions: dict[str, list[EvidenceBackedValue]] = defaultdict(list)
            for card in cards_by_doi.get(record.doi, []):
                dimensions[card.evidence_type].append(
                    EvidenceBackedValue(
                        value=card.normalized_claim,
                        evidence_card_ids=[card.evidence_id],
                    )
                )
            rows.append(
                LiteratureMatrixRow(
                    doi=record.doi,
                    title=record.title,
                    theme_ids=record.theme_ids,
                    admission_status=record.admission_status,
                    centrality=record.centrality,
                    supported_claim=record.supported_claim,
                    suitable_section_id=record.suitable_section_id,
                    use_boundary=record.use_boundary,
                    research_objects=dimensions["research_object"],
                    data_sources=dimensions["data"],
                    methods=dimensions["method"],
                    key_findings=dimensions["result"],
                    limitations=dimensions["limitation"],
                    background=dimensions["background"],
                    future_work=dimensions["future_work"],
                )
            )
        return rows
