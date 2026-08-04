"""Deterministic language and editorial quality checks for V0.4 prose."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.writing import (
    SectionDraft,
    SectionDraftIssue,
    SectionEvidencePacket,
)
from veriwrite_agent.models.writing_plan import WritingSectionPlan
from veriwrite_agent.models.writing_quality import SectionQualityReview

OutputLanguage = Literal["Chinese", "English", "bilingual", "pending_confirmation"]


@dataclass(frozen=True)
class LanguageProfile:
    han_characters: int
    latin_words: int
    dominant_ratio: float


def language_profile(text: str, *, output_language: OutputLanguage) -> LanguageProfile:
    han_characters = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", text))
    total = han_characters + latin_words
    if not total:
        ratio = 0.0
    elif output_language == "Chinese":
        ratio = latin_words / total
    elif output_language == "English":
        ratio = han_characters / total
    else:
        ratio = 0.0
    return LanguageProfile(
        han_characters=han_characters,
        latin_words=latin_words,
        dominant_ratio=ratio,
    )


def language_mismatch_detail(
    text: str,
    *,
    output_language: OutputLanguage,
    maximum_other_language_ratio: float = 0.20,
) -> str | None:
    """Return a blocking detail when prose violates the confirmed language."""

    if output_language == "pending_confirmation":
        return "output language is still pending confirmation"
    if output_language == "bilingual":
        return None
    profile = language_profile(text, output_language=output_language)
    if profile.dominant_ratio > maximum_other_language_ratio:
        other = "English-word" if output_language == "Chinese" else "Chinese-character"
        return (
            f"{other} ratio {profile.dominant_ratio:.1%} exceeds the "
            f"{maximum_other_language_ratio:.0%} language-mixing limit"
        )
    return None


def output_language_instruction(output_language: OutputLanguage) -> str:
    if output_language == "Chinese":
        return (
            "Write the entire paragraph in natural academic Chinese. English is allowed "
            "only for necessary acronyms, model names, instrument names, and terms whose "
            "Chinese translation would be misleading. Do not write a complete English "
            "sentence and avoid literal translation syntax."
        )
    if output_language == "English":
        return (
            "Write the entire paragraph in natural academic English. Use Chinese only "
            "when quoting an indispensable proper name."
        )
    if output_language == "bilingual":
        return "Use a deliberate bilingual structure and do not switch languages mid-sentence."
    return "Do not draft prose until the output language has been confirmed."


class LLMSectionQualityReviewer:
    """Read a complete chapter and return paragraph-level editorial actions."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def review(
        self,
        section_plan: WritingSectionPlan,
        draft: SectionDraft,
        section_packet: SectionEvidencePacket,
        *,
        output_language: OutputLanguage,
    ) -> SectionQualityReview:
        if not (
            section_plan.section_id
            == draft.section_id
            == section_packet.section_id
        ):
            raise ValueError("quality review section does not match the draft")
        schema = json.dumps(
            SectionQualityReview.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = {
            "section_id": section_plan.section_id,
            "title": section_plan.title,
            "purpose": section_plan.purpose,
            "output_language": output_language,
            "planned_argument": [
                {
                    "paragraph_number": paragraph.paragraph_number,
                    "role": paragraph.role,
                    "purpose": paragraph.purpose,
                    "claim_focus": paragraph.claim_focus,
                }
                for paragraph in section_plan.paragraphs
            ],
            "paragraphs": [
                {"paragraph_number": number, "text": paragraph.text}
                for number, paragraph in enumerate(draft.paragraphs, 1)
            ],
            "locked_evidence": [
                {
                    "evidence_id": evidence.evidence_id,
                    "normalized_claim": evidence.normalized_claim,
                    "support_strength": evidence.support_strength,
                    "quotes": [
                        quote.exact_text for quote in evidence.supporting_quotes
                    ],
                }
                for evidence in section_packet.evidence_items
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Act as a strict scholarly chapter editor. Review only; do not rewrite "
                    "the chapter and do not discuss citation formatting. Identify only "
                    "actionable paragraph-level problems: topic drift, repeated argument, "
                    "broken progression, inconsistent terminology, or unnatural/generic "
                    "academic style. Also inspect factual sentences at claim level: classify "
                    "the problem as evidence_fact, reasoned_synthesis, or author_analysis; "
                    "report unsupported_claim when a factual statement is not entailed by "
                    "the locked evidence, and overstated_evidence when wording is stronger "
                    "than the evidence. Author analysis must be explicitly cautious rather "
                    "than presented as an observed fact. Judge every paragraph against the "
                    "supplied chapter purpose and planned claim focus. For evidence-related "
                    "findings, include only evidence_card_ids already assigned to that "
                    "paragraph. Give a concrete revision instruction that preserves evidence "
                    "boundaries. Do not invent a problem merely to fill the list. "
                    f"{output_language_instruction(output_language)} "
                    f"Return JSON satisfying this schema: {schema}"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            raw = self._client.complete(
                messages,
                response_format={"type": "json_object"},
            )
            try:
                review = SectionQualityReview.model_validate_json(raw)
                self._validate(review, section_plan)
                return review
            except (ValidationError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": (
                                    "Repair only the review JSON. Keep section_id unchanged, "
                                    "use only existing paragraph numbers, and remove duplicate "
                                    f"findings. Validation error: {str(exc)[:500]}"
                                ),
                            },
                        ]
                    )
        raise ValueError(f"chapter quality review violates its contract: {last_error}")

    @staticmethod
    def _validate(
        review: SectionQualityReview,
        section_plan: WritingSectionPlan,
    ) -> None:
        if review.section_id != section_plan.section_id:
            raise ValueError("quality reviewer changed section_id")
        valid_numbers = {
            paragraph.paragraph_number for paragraph in section_plan.paragraphs
        }
        identities: set[tuple[int, str]] = set()
        for finding in review.findings:
            if finding.paragraph_number not in valid_numbers:
                raise ValueError("quality reviewer used an unknown paragraph number")
            identity = (finding.paragraph_number, finding.code)
            if identity in identities:
                raise ValueError("quality reviewer duplicated a paragraph finding")
            identities.add(identity)
            paragraph = section_plan.paragraphs[finding.paragraph_number - 1]
            if not set(finding.evidence_card_ids).issubset(
                paragraph.evidence_card_ids
            ):
                raise ValueError(
                    "quality reviewer used evidence outside the paragraph plan"
                )


def apply_section_quality_review(
    draft: SectionDraft,
    review: SectionQualityReview,
) -> SectionDraft:
    """Replace prior editorial findings while preserving evidence audit results."""

    quality_codes = {
        "paragraph_repetition",
        "topic_drift",
        "coherence_gap",
        "terminology_inconsistent",
        "academic_style_problem",
        "unsupported_claim",
        "overstated_evidence",
        "quality_review_failed",
    }
    retained = [issue for issue in draft.issues if issue.code not in quality_codes]
    findings = [
        SectionDraftIssue(
            code=finding.code,
            severity=(
                "blocking"
                if finding.code in {"unsupported_claim", "overstated_evidence"}
                else "warning"
            ),
            paragraph_number=finding.paragraph_number,
            detail=(
                f"{finding.detail} Revision instruction: "
                f"{finding.revision_instruction}"
            ),
        )
        for finding in review.findings
    ]
    issues = [*retained, *findings]
    return draft.model_copy(
        update={
            "issues": issues,
            "status": (
                "needs_review"
                if any(issue.severity == "blocking" for issue in issues)
                else draft.status
            ),
            "confirmed_by": None,
            "confirmed_at": None,
        }
    )


def repeated_sentence_pairs(paragraphs: list[str]) -> list[tuple[int, int]]:
    """Find near-duplicate paragraphs from normalized token-set overlap."""

    token_sets = [_content_tokens(text) for text in paragraphs]
    repeated: list[tuple[int, int]] = []
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            if len(union) < 8:
                continue
            similarity = len(token_sets[left] & token_sets[right]) / len(union)
            if similarity >= 0.72:
                repeated.append((left + 1, right + 1))
    return repeated


def _content_tokens(text: str) -> set[str]:
    latin = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{3,}", text)
    }
    compact_han = "".join(re.findall(r"[\u3400-\u9fff]", text))
    han_bigrams = {
        compact_han[index : index + 2]
        for index in range(max(0, len(compact_han) - 1))
    }
    return latin | han_bigrams
