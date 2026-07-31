"""Refine the V0.2 search blueprint into a V0.4-ready writing outline."""

from __future__ import annotations

from veriwrite_agent.models.evidence import EvidenceLibrary
from veriwrite_agent.models.literature_selection import LiteratureSearchBlueprint
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.writing_handoff import (
    ConfirmedWritingOutline,
    V04WritingHandoff,
    WritingOutlineDraft,
    WritingOutlineSection,
)


class WritingOutlineBuilder:
    """Allocate words and evidence using V0.2 themes plus V0.3 coverage."""

    def build(
        self,
        blueprint: LiteratureSearchBlueprint,
        library: EvidenceLibrary,
        *,
        target_words: int,
    ) -> WritingOutlineDraft:
        if target_words < 200 * len(blueprint.themes):
            raise ValueError("target_words is too small for the confirmed themes")
        budgets = _allocate_words(
            [theme.target_count for theme in blueprint.themes],
            target_words,
        )
        cards_by_theme = {
            theme.theme_id: [
                card
                for card in library.evidence_cards
                if card.theme_id == theme.theme_id
                and card.review_status != "rejected"
            ]
            for theme in blueprint.themes
        }
        records_by_theme = {
            theme.theme_id: [
                record
                for record in library.records
                if theme.theme_id in record.theme_ids
            ]
            for theme in blueprint.themes
        }
        sections: list[WritingOutlineSection] = []
        gaps: list[str] = []
        for theme, budget in zip(blueprint.themes, budgets, strict=True):
            records = records_by_theme[theme.theme_id]
            core_dois = [
                record.doi
                for record in records
                if record.evidence_status == "full_text_verified"
            ]
            supporting_dois = [
                record.doi
                for record in records
                if record.evidence_status == "metadata_verified"
            ]
            cards = cards_by_theme[theme.theme_id]
            evidence_gap = not core_dois or not cards
            if evidence_gap:
                gaps.append(
                    f"{theme.section_title}缺少已验证全文或可追溯证据卡。"
                )
            sections.append(
                WritingOutlineSection(
                    section_id=theme.theme_id,
                    title=theme.section_title,
                    purpose=theme.section_purpose,
                    target_words=budget,
                    research_questions=theme.research_questions,
                    core_dois=core_dois,
                    supporting_dois=supporting_dois,
                    evidence_card_ids=[card.evidence_id for card in cards],
                    evidence_gap=evidence_gap,
                )
            )
        return WritingOutlineDraft(
            topic=blueprint.topic,
            writing_through_line=blueprint.writing_through_line,
            target_words=target_words,
            sections=sections,
            unresolved_gaps=gaps,
        )


class WritingHandoffService:
    def confirm_outline(
        self,
        outline: WritingOutlineDraft,
        *,
        confirmed_by: str,
        confirmation_note: str | None = None,
    ) -> ConfirmedWritingOutline:
        return ConfirmedWritingOutline(
            outline=outline,
            confirmed_by=confirmed_by,
            confirmation_note=confirmation_note,
        )

    def create(
        self,
        *,
        requirement: ConfirmedRequirementSpec,
        outline: ConfirmedWritingOutline,
        evidence_library: EvidenceLibrary,
    ) -> V04WritingHandoff:
        return V04WritingHandoff(
            requirement=requirement,
            outline=outline,
            evidence_library=evidence_library,
        )


def _allocate_words(
    weights: list[int],
    target_words: int,
) -> list[int]:
    minimum = 100
    distributable = target_words - minimum * len(weights)
    total_weight = sum(weights)
    raw = [distributable * weight / total_weight for weight in weights]
    budgets = [minimum + int(value) for value in raw]
    remainder = target_words - sum(budgets)
    order = sorted(
        range(len(weights)),
        key=lambda index: raw[index] - int(raw[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        budgets[index] += 1
    return budgets
