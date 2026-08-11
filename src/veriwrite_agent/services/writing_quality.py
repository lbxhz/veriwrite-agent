"""Deterministic language and editorial quality checks for V0.4 prose."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Literal

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.writing import (
    SectionDraft,
    SectionDraftIssue,
    SectionEvidencePacket,
    SectionQualityReviewTrace,
    V04WritingProject,
)
from veriwrite_agent.models.writing_plan import GroundedWritingPlan, WritingSectionPlan
from veriwrite_agent.models.writing_quality import (
    ManuscriptEditorialCheckpoint,
    ManuscriptQualityFinding,
    ManuscriptQualityReview,
    ParagraphQualityFinding,
    SectionQualityReview,
)

OutputLanguage = Literal["Chinese", "English", "bilingual", "pending_confirmation"]

CHAPTER_LOCAL_EDITORIAL_CODES = frozenset(
    {
        "paragraph_repetition",
        "coherence_gap",
        "terminology_inconsistent",
        "academic_style_problem",
        "oversized_paragraph",
    }
)

CHAPTER_TRUST_REPAIR_CODES = frozenset(
    {
        "topic_drift",
        "unsupported_claim",
        "overstated_evidence",
        "false_self_attribution",
    }
)

# A reviewer's vocabulary is broader than the set of findings that may be
# deferred or repaired. Keep those responsibilities separate so adding a review
# code cannot silently authorize a rewrite or weaken a trust gate.
SECTION_QUALITY_REVIEW_CODES = (
    CHAPTER_LOCAL_EDITORIAL_CODES | CHAPTER_TRUST_REPAIR_CODES
)

# Backwards-compatible public name for callers that truly mean editorial advice.
EDITORIAL_QUALITY_CODES = CHAPTER_LOCAL_EDITORIAL_CODES

HARD_CHAPTER_TRUST_CODES = frozenset(
    {"unsupported_claim", "overstated_evidence", "false_self_attribution"}
)

PROSE_REPAIRABLE_DETERMINISTIC_CODES = frozenset(
    {
        "llm_authored_citation",
        "workflow_instruction_leak",
        "language_mismatch",
        "final_audit_repair",
    }
)

PLAN_BINDING_DETERMINISTIC_CODES = frozenset(
    {
        "unknown_evidence_card",
        "unknown_source_doi",
        "evidence_source_mismatch",
        "source_permission_exceeded",
    }
)

EVIDENCE_INTEGRITY_DETERMINISTIC_CODES = frozenset({"unconfirmed_evidence"})

PROSE_REPAIRABLE_SECTION_CODES = (
    SECTION_QUALITY_REVIEW_CODES | PROSE_REPAIRABLE_DETERMINISTIC_CODES
)

# A chapter reviewer is useful for proposing edits, but its stylistic judgement is
# not equivalent to a deterministic citation/evidence gate.  These codes may consume
# a bounded local repair budget; if they persist, they are carried forward as warnings
# for the full-manuscript editor instead of stopping the whole Agent run.
DEFERABLE_CHAPTER_QUALITY_CODES = CHAPTER_LOCAL_EDITORIAL_CODES

FALSE_SELF_ATTRIBUTION_PATTERN = re.compile(
    r"(?P<subject>(?:在\s*)?(?:本文|本研究|本论文)(?:\s*中)?|我们)"
    r"\s*[，,:：]?\s*(?:首次|进一步|成功)?\s*"
    r"(?P<verb>提出(?:了)?|利用(?:了)?|采用(?:了)?|使用(?:了)?|构建(?:了)?|"
    r"设计(?:了)?|开发(?:了)?|研制(?:了)?|验证(?:了)?|获取(?:了)?|实现(?:了)?|"
    r"发现(?:了)?|结果表明|实验表明|测量(?:了)?|反演(?:了)?)",
    re.IGNORECASE,
)

ENGLISH_FALSE_SELF_ATTRIBUTION_PATTERN = re.compile(
    r"(?P<subject>\b(?:we|this\s+(?:paper|study|research)|the\s+present\s+study)\b)"
    r"\s*(?:,\s*)?(?:first|further|successfully)?\s*"
    r"(?P<verb>propos(?:e|es|ed)|develop(?:s|ed)?|design(?:s|ed)?|construct(?:s|ed)?|"
    r"build(?:s|t)?|implement(?:s|ed)?|validate(?:s|d)?|measure(?:s|d)?|retrieve(?:s|d)?|"
    r"discover(?:s|ed)?|find(?:s|ing)?|obtain(?:s|ed)?|use(?:s|d)?|utili[sz](?:e|es|ed)|"
    r"adopt(?:s|ed)?)\b",
    re.IGNORECASE,
)

REVIEW_METHOD_ALLOWLIST_PATTERN = re.compile(
    r"(?:文献综述|系统综述|叙述性综述|范围综述|文献计量|对比分析|比较分析|"
    r"内容分析|归纳分析|主题分析|已有文献|相关文献|研究进展|研究现状)"
)

ENGLISH_REVIEW_METHOD_ALLOWLIST_PATTERN = re.compile(
    r"\b(?:literature|studies|research|evidence|publications|review|bibliometric|"
    r"comparative|thematic|content)\b",
    re.IGNORECASE,
)

MATERIAL_MANUSCRIPT_REPAIR_CODES = frozenset(
    {
        "cross_section_repetition",
        "paragraph_repetition",
        "section_role_overlap",
        "academic_style_problem",
        "oversized_paragraph",
    }
)

FORMULAIC_ACADEMIC_PATTERNS = (
    re.compile(r"值得注意的是"),
    re.compile(r"(?:奠定了|奠定)\s*.{0,18}?基础"),
    re.compile(r"具有(?:十分|非常)?重要(?:的)?(?:意义|价值)"),
    re.compile(r"不容忽视"),
    re.compile(r"为.{0,18}?提供了有力支持"),
)


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


def workflow_instruction_leak_detail(text: str) -> str | None:
    """Detect internal retrieval/planning instructions that leaked into submit-ready prose."""

    patterns = (
        r"为满足.{0,20}(?:文献|参考文献|引用).{0,20}(?:数量|覆盖|配额|政策)",
        r"(?:文献|参考文献|引用)(?:覆盖|配额)政策",
        r"(?:证据卡|检索蓝图|写作交接包|锁定证据包)",
        r"(?:bibliography|reference|citation)\s+(?:coverage|quota)\s+(?:policy|requirement)",
        r"(?:locked|assigned)\s+(?:evidence|source)s?",
        r"(?:evidence|writing)\s+packet",
        r"metadata-supported\s+sources?",
        r"(?:本段|本节)\s*(?:不再|仅)\s*(?:重复|复述|保留).{0,40}(?:前文|前述|核心判断|内容)",
        r"(?:根据|按照)\s*(?:审稿|返修|修订|编辑)\s*(?:意见|要求|指令)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"internal workflow instruction leaked into prose: {match.group(0)}"
    return None


def false_self_attribution_detail(text: str) -> str | None:
    """Reject review prose that claims a cited study as the current paper's work."""

    for pattern, allowlist in (
        (FALSE_SELF_ATTRIBUTION_PATTERN, REVIEW_METHOD_ALLOWLIST_PATTERN),
        (ENGLISH_FALSE_SELF_ATTRIBUTION_PATTERN, ENGLISH_REVIEW_METHOD_ALLOWLIST_PATTERN),
    ):
        for match in pattern.finditer(text):
            verb = match.group("verb").casefold()
            trailing_context = text[match.end() : min(len(text), match.end() + 36)]
            if _is_allowed_review_operation(verb, trailing_context, allowlist=allowlist):
                continue
            excerpt_start = max(0, match.start() - 16)
            excerpt_end = min(len(text), match.end() + 42)
            excerpt = " ".join(text[excerpt_start:excerpt_end].split())
            return (
                f"综述错误地使用“{match.group(0)}”认领被引研究成果：{excerpt}。"
                "应根据锁定来源的作者元数据改写为“作者等提出/使用/发现”。"
            )
    return None


def _is_allowed_review_operation(
    verb: str,
    trailing_context: str,
    *,
    allowlist: re.Pattern[str],
) -> bool:
    """Allow only explicit review/meta-analysis operations for the current paper."""

    adoption_verbs = {
        "利用",
        "利用了",
        "采用",
        "采用了",
        "使用",
        "使用了",
        "use",
        "uses",
        "used",
        "utilize",
        "utilizes",
        "utilized",
        "utilise",
        "utilises",
        "utilised",
        "adopt",
        "adopts",
        "adopted",
    }
    return verb in adoption_verbs and allowlist.search(trailing_context) is not None


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
        evidence_by_id = {
            evidence.evidence_id: evidence
            for evidence in section_packet.evidence_items
        }
        sources_by_doi = {source.doi: source for source in section_packet.sources}
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
                    "central_question": paragraph.central_question,
                    "argument_move": paragraph.argument_move,
                    "comparison_axis": paragraph.comparison_axis,
                    "locked_evidence": [
                        {
                            "evidence_id": evidence.evidence_id,
                            "normalized_claim": evidence.normalized_claim,
                            "support_strength": evidence.support_strength,
                            "quotes": [
                                quote.exact_text
                                for quote in evidence.supporting_quotes
                            ],
                        }
                        for evidence_id in paragraph.evidence_card_ids
                        if (evidence := evidence_by_id.get(evidence_id)) is not None
                    ],
                    "admitted_sources": [
                        {
                            "doi": source.doi,
                            "centrality": source.centrality,
                            "supported_claim": source.supported_claim,
                            "use_boundary": source.use_boundary,
                        }
                        for doi in paragraph.source_dois
                        if (source := sources_by_doi.get(doi)) is not None
                    ],
                }
                for paragraph in section_plan.paragraphs
            ],
            "paragraphs": [
                {"paragraph_number": number, "text": paragraph.text}
                for number, paragraph in enumerate(draft.paragraphs, 1)
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
                    "academic style. This is a literature review, not an original empirical "
                    "study. Report false_self_attribution whenever wording such as '本文提出', "
                    "'本研究利用', or '我们设计' incorrectly claims a cited source's method, "
                    "dataset, experiment, or result as the current paper's work. Use the cited "
                    "authors as the grammatical subject instead. Phrases such as "
                    "'Naseem等提出', 'Li等构建', or 'Smith et al. found' "
                    "already attribute the work to the cited authors and MUST NOT be reported "
                    "as false_self_attribution. That code is reserved for first-person/current-"
                    "paper claims such as '本文提出', '本研究利用', or '我们设计'. Report "
                    "oversized_paragraph when a paragraph combines several independent "
                    "argument moves and needs "
                    "to be split or structurally reduced. Also inspect factual sentences at "
                    "claim level: classify "
                    "the problem as evidence_fact, reasoned_synthesis, or author_analysis; "
                    "report unsupported_claim when a factual statement is not entailed by "
                    "the locked evidence, and overstated_evidence when wording is stronger "
                    "than the evidence. Author analysis must be explicitly cautious rather "
                    "than presented as an observed fact. Judge paragraphs against the "
                    "supplied chapter purpose, planned claim focus, and that paragraph's "
                    "own locked_evidence and admitted_sources only. Treat a paragraph as "
                    "topic_drift when a source is used beyond its supported_claim or "
                    "use_boundary, or when a contextual source becomes the main subject. "
                    "Omit paragraphs that are acceptable. Return "
                    "no more than six of the chapter's highest-confidence, materially useful "
                    "findings; an empty findings list is valid. Report unsupported_claim or "
                    "overstated_evidence only when the mismatch is clear from the supplied "
                    "paragraph-level evidence, not for a minor wording preference. Include "
                    "severity=blocking only when the problem makes the paragraph materially "
                    "unreliable or outside the confirmed topic and therefore must be repaired "
                    "before use. Use severity=warning for optional improvements in style, "
                    "terminology, repetition, or transitions. Never block merely because a "
                    "different phrasing would be preferable. Include "
                    "only evidence_card_ids present in that paragraph's locked_evidence. "
                    "Give a concrete revision instruction that preserves evidence boundaries. "
                    "Do not invent a problem merely to fill the list. Write finding details "
                    "in natural academic Chinese when the output language is Chinese. "
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
                review = _normalize_section_evidence_scope(review, section_plan)
                review = _normalize_section_content_findings(review, draft)
                review = _normalize_section_gate_severity(review)
                review = _deduplicate_section_findings(review)
                _validate_section_review(review, section_plan)
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


class LLMManuscriptQualityReviewer:
    """Review the complete confirmed body across chapter boundaries.

    The reviewer can request only paragraph-level body revisions. Citation selection
    remains deterministic and every requested rewrite therefore reuses the original
    paragraph evidence packet.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def review(
        self,
        plan: GroundedWritingPlan,
        project: V04WritingProject,
    ) -> ManuscriptQualityReview:
        deterministic = _deterministic_manuscript_findings(plan, project)
        deferred = _deferred_manuscript_findings(project)
        schema = json.dumps(
            ManuscriptQualityReview.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        plan_by_id = {section.section_id: section for section in plan.sections}
        payload = {
            "paper_genre": "literature_review",
            "topic": plan.topic,
            "output_language": plan.output_language,
            "sections": [
                {
                    "section_id": state.section_id,
                    "title": plan_by_id[state.section_id].title,
                    "purpose": plan_by_id[state.section_id].purpose,
                    "paragraphs": [
                        {
                            "paragraph_number": number,
                            "planned_claim": plan_by_id[state.section_id]
                            .paragraphs[number - 1]
                            .claim_focus,
                            "text": paragraph.text,
                        }
                        for number, paragraph in enumerate(
                            state.draft.paragraphs if state.draft else [], 1
                        )
                    ],
                }
                for state in project.sections
            ],
            "deferred_chapter_findings": [
                finding.model_dump(mode="json") for finding in deferred
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Act as the independent full-manuscript structure editor for a "
                    "literature review. Review across all chapters, not one chapter at a "
                    "time. Identify only material cross-section repetition, a chapter "
                    "performing another chapter's role, false claims that the current "
                    "review conducted a cited study, paragraphs that contain several "
                    "unrelated argument moves and should be reduced, or a broken global "
                    "argument progression. Also report academic_style_problem when the "
                    "manuscript repeatedly uses generic AI-like transitions, inflated "
                    "importance claims, or template conclusions instead of a concrete "
                    "scholarly judgment. Do not report a necessary technical term merely "
                    "because it recurs. Do not review citation formatting and do not "
                    "invent evidence problems. Locate every finding at one existing body "
                    "section_id and paragraph_number. Use blocking only for clear duplicate "
                    "content, false research ownership, severe role overlap, or an unusable "
                    "oversized paragraph; otherwise use warning. Give a targeted revision "
                    "instruction that preserves the paragraph's existing evidence scope. "
                    "Set disposition=targeted_repair when the finding is high-confidence, "
                    "material to manuscript quality, and safely repairable inside the "
                    "paragraph's locked evidence scope. The repair executor replaces one "
                    "paragraph at a time: never instruct it to move, delete, merge, or split "
                    "paragraphs. Express structural repairs as an executable replacement, "
                    "for example 'reduce this paragraph to one unique bridge judgment and "
                    "omit the repeated cases' or 'retain one central argument move'. Use "
                    "disposition=report_only when a useful fix would require changing the "
                    "chapter outline, moving material between chapters, or adding evidence. "
                    "A targeted_repair finding MUST use severity=blocking. Every warning "
                    "MUST use disposition=report_only so optional editorial preferences do "
                    "not create a non-converging rewrite loop. Use report_only for a minor "
                    "or uncertain suggestion. A targeted repair will be sent "
                    "back to the writer and independently reviewed again. "
                    "The payload may contain deferred_chapter_findings from bounded local "
                    "review. Recheck each one explicitly in full-manuscript context. Use "
                    "paragraph_repetition for repetition inside one chapter, coherence_gap "
                    "for a local transition or progression break, and "
                    "terminology_inconsistent for conflicting terms. Preserve a still-useful "
                    "unresolved item in the output; omit it only when the current prose no "
                    "longer has that defect. "
                    "Return at most 12 high-confidence findings; an empty list is valid. "
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
                review = ManuscriptQualityReview.model_validate_json(raw)
                review = _validate_manuscript_review(review, plan)
                return _merge_manuscript_findings(review, [*deterministic, *deferred])
            except (ValidationError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": (
                                    "Repair only the review JSON. Use existing section IDs "
                                    "and paragraph numbers, remove duplicates, and do not "
                                    f"invent findings. Error: {str(exc)[:700]}"
                                ),
                            },
                        ]
                    )
        return ManuscriptQualityReview(
            review_status="deterministic_fallback",
            findings=[*deterministic, *deferred][:16],
            review_error=str(last_error)[:1000] if last_error else "unknown review error",
        )


class FullManuscriptEditorialService:
    """Run and persist the independent body-editing gate before final matter.

    This stage is deliberately read-only: it may report paragraph-level actions, but it
    cannot silently change prose, evidence bindings, or citations. The existing targeted
    repair executor applies accepted actions later under the original paragraph packets.
    """

    def __init__(self, reviewer: LLMManuscriptQualityReviewer) -> None:
        self._reviewer = reviewer

    def run(
        self,
        plan: GroundedWritingPlan,
        project: V04WritingProject,
    ) -> ManuscriptEditorialCheckpoint:
        if plan.status != "confirmed":
            raise ValueError("full-manuscript editing requires a confirmed writing plan")
        if project.status != "body_complete":
            raise ValueError("full-manuscript editing requires a complete body draft")

        review = self._reviewer.review(plan, project)
        blocking = [
            finding
            for finding in review.findings
            if finding.severity == "blocking"
            or finding.disposition == "targeted_repair"
        ]
        warnings = [
            finding
            for finding in review.findings
            if finding not in blocking
        ]
        return ManuscriptEditorialCheckpoint(
            body_fingerprint=manuscript_body_fingerprint(plan, project),
            status="needs_revision" if blocking else "passed",
            review=review,
            blocking_count=len(blocking),
            warning_count=len(warnings),
            completed_at=datetime.now(timezone.utc),
        )


def manuscript_body_fingerprint(
    plan: GroundedWritingPlan,
    project: V04WritingProject,
) -> str:
    """Fingerprint body prose and locked support so refreshes reuse one review."""

    payload = {
        "plan_fingerprint": plan.plan_fingerprint,
        "sections": [
            {
                "section_id": state.section_id,
                "status": state.status,
                "paragraphs": [
                    {
                        "text": paragraph.text,
                        "evidence_card_ids": paragraph.evidence_card_ids,
                        "source_dois": paragraph.source_dois,
                    }
                    for paragraph in (state.draft.paragraphs if state.draft else [])
                ],
            }
            for state in project.sections
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def refine_writing_plan_for_manuscript_review(
    plan: GroundedWritingPlan,
    review: ManuscriptQualityReview | None,
    *,
    evidence_doi_by_id: dict[str, str] | None = None,
) -> GroundedWritingPlan:
    """Align paragraph assignments with executable full-manuscript repairs.

    A paragraph writer cannot reliably obey a global editor when its locked plan still
    requires the very material the editor asked it to remove.  This refinement step makes
    the editorial decision part of the deterministic plan before prose is regenerated:
    repeated source coverage is dropped when another paragraph already carries it, the
    paragraph budget is reduced, and the claim focus becomes the global repair task.

    Required bibliography coverage is preserved: a source is removed from a target only
    when it is planned elsewhere.  Citation selection therefore remains code-owned.
    """

    if review is None:
        return plan
    findings_by_target: dict[tuple[str, int], list[ManuscriptQualityFinding]] = {}
    for finding in review.findings:
        if finding.disposition != "targeted_repair":
            continue
        findings_by_target.setdefault(
            (finding.section_id, finding.paragraph_number), []
        ).append(finding)
    if not findings_by_target:
        return plan

    source_use_count: dict[str, int] = {}
    for section in plan.sections:
        for paragraph in section.paragraphs:
            for doi in paragraph.source_dois:
                source_use_count[doi] = source_use_count.get(doi, 0) + 1
    evidence_sources = evidence_doi_by_id or {}
    refined_sections: list[WritingSectionPlan] = []
    changed = False
    for section in plan.sections:
        paragraphs = list(section.paragraphs)
        section_changed = False
        released_budget = 0
        target_numbers: set[int] = set()
        for index, paragraph in enumerate(paragraphs):
            findings = findings_by_target.get(
                (section.section_id, paragraph.paragraph_number)
            )
            if not findings:
                continue
            target_numbers.add(paragraph.paragraph_number)
            codes = {finding.code for finding in findings}
            instruction = " ".join(
                dict.fromkeys(finding.revision_instruction for finding in findings)
            )
            cap = _global_editor_target_cap(codes)
            new_target = min(paragraph.target_words, cap)
            released_budget += paragraph.target_words - new_target

            retained_sources = list(paragraph.source_dois)
            retained_cards = list(paragraph.evidence_card_ids)
            if codes & {"cross_section_repetition", "section_role_overlap"}:
                unique_sources = [
                    doi
                    for doi in paragraph.source_dois
                    if source_use_count.get(doi, 0) == 1
                ]
                if unique_sources:
                    retained_sources = unique_sources
                    retained_cards = [
                        evidence_id
                        for evidence_id in paragraph.evidence_card_ids
                        if evidence_sources.get(evidence_id) in set(unique_sources)
                    ]
                elif paragraph.source_dois:
                    retained_sources = [paragraph.source_dois[0]]
                    retained_cards = [
                        evidence_id
                        for evidence_id in paragraph.evidence_card_ids
                        if evidence_sources.get(evidence_id) == retained_sources[0]
                    ]
            role = paragraph.role
            if role == "detailed_evidence" and not retained_cards:
                role = "synthesis"
            style_only = codes == {"academic_style_problem"}
            paragraphs[index] = paragraph.model_copy(
                update={
                    "role": role,
                    "purpose": (
                        (
                            paragraph.purpose
                            + " Editorial constraint: "
                            + instruction
                        )
                        if style_only
                        else (
                            "Global manuscript repair: replace the old paragraph with one "
                            "evidence-bounded argument that performs only this task: "
                            + instruction
                        )
                    ),
                    "claim_focus": (
                        paragraph.claim_focus if style_only else instruction
                    ),
                    "central_question": (
                        paragraph.central_question
                        if style_only
                        else (
                            "How can this paragraph perform its unique chapter role without "
                            "repeating neighbouring or earlier material?"
                        )
                    ),
                    "argument_move": (
                        paragraph.argument_move
                        if style_only
                        else "author_judgment"
                    ),
                    "comparison_axis": (
                        paragraph.comparison_axis if style_only else None
                    ),
                    "target_words": new_target,
                    "evidence_card_ids": retained_cards,
                    "source_dois": retained_sources,
                }
            )
            section_changed = True
        if section_changed:
            # A prior implementation moved the complete released allowance to one
            # untouched paragraph.  Its prose was then reused under a much larger target,
            # which made the plan claim that the section still met its word budget while
            # the real manuscript became shorter.  Spread the allowance across every
            # non-target paragraph instead.  Reuse validation will rewrite only paragraphs
            # that are materially shorter than their new target.
            paragraphs = _redistribute_released_budget(
                paragraphs,
                released_budget=released_budget,
                fixed_paragraph_numbers=target_numbers,
            )
            refined_sections.append(
                section.model_copy(
                    update={
                        "paragraphs": paragraphs,
                        "target_words": sum(
                            paragraph.target_words for paragraph in paragraphs
                        ),
                    }
                )
            )
            changed = True
        else:
            refined_sections.append(section)
    if not changed:
        return plan
    refined_sections = _restore_required_plan_sources(
        plan,
        refined_sections,
        evidence_doi_by_id=evidence_sources,
    )
    fingerprint = _refined_plan_fingerprint(plan, refined_sections)
    return GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={
                "sections": refined_sections,
                "plan_fingerprint": fingerprint,
            }
        ).model_dump(mode="json")
    )


def _global_editor_target_cap(codes: set[str]) -> int:
    if "section_role_overlap" in codes:
        return 220
    if "cross_section_repetition" in codes:
        return 240
    if "oversized_paragraph" in codes:
        return 320
    if "global_coherence_gap" in codes:
        return 320
    return 360


def mark_manuscript_editor_targets(
    plan: GroundedWritingPlan,
    repair_instructions: dict[tuple[str, int], list[str]],
) -> GroundedWritingPlan:
    """Keep remapped manuscript repairs inside their existing evidence authority.

    Structural de-duplication may remove a reviewed paragraph and remap its repair to a
    surviving neighbour. The neighbour must inherit an editorial task, not keep an older
    detailed-claim or comparison task that can incorrectly trigger literature/PDF recovery.
    DOI and evidence-card bindings are preserved unchanged.
    """

    if not repair_instructions:
        return plan
    changed = False
    sections: list[WritingSectionPlan] = []
    for section in plan.sections:
        paragraphs = []
        for paragraph in section.paragraphs:
            instructions = repair_instructions.get(
                (section.section_id, paragraph.paragraph_number)
            )
            if not instructions:
                paragraphs.append(paragraph)
                continue
            instruction = " ".join(dict.fromkeys(instructions)).strip()
            if not instruction:
                instruction = (
                    "Retain only the paragraph's unique evidence-bounded judgment."
                )
            instruction = instruction[:500]
            paragraphs.append(
                paragraph.model_copy(
                    update={
                        "role": (
                            paragraph.role
                            if paragraph.evidence_card_ids
                            else "background"
                        ),
                        "purpose": "Global manuscript editorial repair: " + instruction,
                        "claim_focus": instruction,
                        "central_question": (
                            "How can this paragraph retain only its unique, cautious "
                            "judgment without adding facts or evidence?"
                        ),
                        "argument_move": "author_judgment",
                        "comparison_axis": None,
                    }
                )
            )
            changed = True
        sections.append(section.model_copy(update={"paragraphs": paragraphs}))
    if not changed:
        return plan
    return GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={
                "sections": sections,
                "plan_fingerprint": _refined_plan_fingerprint(plan, sections),
            }
        ).model_dump(mode="json")
    )


def _redistribute_released_budget(
    paragraphs: list,
    *,
    released_budget: int,
    fixed_paragraph_numbers: set[int],
) -> list:
    """Distribute a repair budget without creating one artificial mega-paragraph."""

    if released_budget <= 0:
        return paragraphs
    recipients = [
        index
        for index, paragraph in enumerate(paragraphs)
        if paragraph.paragraph_number not in fixed_paragraph_numbers
    ]
    if not recipients:
        return paragraphs
    share, remainder = divmod(released_budget, len(recipients))
    updated = list(paragraphs)
    for order, index in enumerate(recipients):
        paragraph = updated[index]
        increment = share + (1 if order < remainder else 0)
        updated[index] = paragraph.model_copy(
            update={"target_words": paragraph.target_words + increment}
        )
    return updated


def _restore_required_plan_sources(
    original: GroundedWritingPlan,
    sections: list[WritingSectionPlan],
    *,
    evidence_doi_by_id: dict[str, str],
) -> list[WritingSectionPlan]:
    """Prevent two simultaneously edited paragraphs from dropping the same source."""

    planned = {
        doi
        for section in sections
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    }
    missing = [doi for doi in original.required_source_dois if doi not in planned]
    if not missing:
        return sections
    original_owners = {
        doi: [
            (section.section_id, paragraph.paragraph_number)
            for section in original.sections
            for paragraph in section.paragraphs
            if doi in paragraph.source_dois
        ]
        for doi in missing
    }
    payloads = [section.model_dump(mode="json") for section in sections]
    by_id = {section["section_id"]: section for section in payloads}
    for doi in missing:
        candidates = []
        for section_id, number in original_owners[doi]:
            section = by_id.get(section_id)
            if section is None or number > len(section["paragraphs"]):
                continue
            paragraph = section["paragraphs"][number - 1]
            if len(paragraph["source_dois"]) < 8:
                candidates.append(paragraph)
        if not candidates:
            candidates = [
                paragraph
                for section in payloads
                for paragraph in section["paragraphs"]
                if len(paragraph["source_dois"]) < 8
            ]
        if not candidates:
            raise ValueError("required source cannot be restored to the refined plan")
        recipient = min(candidates, key=lambda paragraph: len(paragraph["source_dois"]))
        recipient["source_dois"].append(doi)
        cards = [
            evidence_id
            for section in original.sections
            for paragraph in section.paragraphs
            if doi in paragraph.source_dois
            for evidence_id in paragraph.evidence_card_ids
            if evidence_doi_by_id.get(evidence_id) == doi
        ]
        for evidence_id in cards:
            if (
                evidence_id not in recipient["evidence_card_ids"]
                and len(recipient["evidence_card_ids"]) < 5
            ):
                recipient["evidence_card_ids"].append(evidence_id)
    return [WritingSectionPlan.model_validate(section) for section in payloads]


def _refined_plan_fingerprint(
    plan: GroundedWritingPlan,
    sections: list[WritingSectionPlan],
) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v2-global-editor",
            "topic": plan.topic,
            "output_language": plan.output_language,
            "required_source_dois": plan.required_source_dois,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_section_review(
    review: SectionQualityReview,
    section_plan: WritingSectionPlan,
) -> None:
    if review.section_id != section_plan.section_id:
        raise ValueError("quality reviewer changed section_id")
    valid_numbers = {paragraph.paragraph_number for paragraph in section_plan.paragraphs}
    identities: set[tuple[int, str]] = set()
    for finding in review.findings:
        if finding.paragraph_number not in valid_numbers:
            raise ValueError("quality reviewer used an unknown paragraph number")
        identity = (finding.paragraph_number, finding.code)
        if identity in identities:
            raise ValueError("quality reviewer duplicated a paragraph finding")
        identities.add(identity)


def _deduplicate_section_findings(
    review: SectionQualityReview,
) -> SectionQualityReview:
    """Collapse mechanically duplicated reviewer findings before validation.

    A duplicate paragraph/code pair does not carry new editorial information and is
    safe to normalize deterministically. Treating it as a fatal model-contract error
    caused the Agent to discard a usable review and restart an already-written
    chapter. When duplicates differ, retain the stronger and more informative one.
    """

    positions: dict[tuple[int, str], int] = {}
    normalized = []
    for finding in review.findings:
        identity = (finding.paragraph_number, finding.code)
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(normalized)
            normalized.append(finding)
            continue
        current = normalized[position]
        if _section_finding_priority(finding) > _section_finding_priority(current):
            normalized[position] = finding
    return review.model_copy(update={"findings": normalized})


def _section_finding_priority(
    finding: ParagraphQualityFinding,
) -> tuple[int, int, int]:
    return (
        int(finding.severity == "blocking"),
        len(finding.evidence_card_ids),
        len(finding.detail) + len(finding.revision_instruction),
    )


def _normalize_section_evidence_scope(
    review: SectionQualityReview,
    section_plan: WritingSectionPlan,
) -> SectionQualityReview:
    """Drop reviewer-selected card IDs outside each paragraph's locked scope."""

    normalized = []
    for finding in review.findings:
        if finding.paragraph_number > len(section_plan.paragraphs):
            normalized.append(finding)
            continue
        allowed = set(
            section_plan.paragraphs[finding.paragraph_number - 1].evidence_card_ids
        )
        normalized.append(
            finding.model_copy(
                update={
                    "evidence_card_ids": [
                        evidence_id
                        for evidence_id in finding.evidence_card_ids
                        if evidence_id in allowed
                    ]
                }
            )
        )
    return review.model_copy(update={"findings": normalized})


def _normalize_section_content_findings(
    review: SectionQualityReview,
    draft: SectionDraft,
) -> SectionQualityReview:
    """Remove semantic-review findings contradicted by deterministic text checks.

    A cited author as grammatical subject (for example, ``Li 等提出``) is the
    required form in a literature review.  LLM reviewers occasionally mistake
    the verb ``提出`` itself for self-attribution, so only the explicit
    current-paper/first-person patterns may activate this blocking code.
    """

    normalized = []
    for finding in review.findings:
        if finding.code != "false_self_attribution":
            normalized.append(finding)
            continue
        number = finding.paragraph_number
        if number > len(draft.paragraphs):
            normalized.append(finding)
            continue
        paragraph = draft.paragraphs[number - 1]
        if false_self_attribution_detail(paragraph.text) is not None:
            normalized.append(finding)
    return review.model_copy(update={"findings": normalized})


def _normalize_section_gate_severity(
    review: SectionQualityReview,
) -> SectionQualityReview:
    """Keep semantic trust boundaries under code control, not model discretion."""

    return review.model_copy(
        update={
            "findings": [
                finding.model_copy(update={"severity": "blocking"})
                if finding.code in HARD_CHAPTER_TRUST_CODES
                else finding
                for finding in review.findings
            ]
        }
    )


def _validate_manuscript_review(
    review: ManuscriptQualityReview,
    plan: GroundedWritingPlan,
) -> ManuscriptQualityReview:
    valid = {
        section.section_id: {
            paragraph.paragraph_number for paragraph in section.paragraphs
        }
        for section in plan.sections
    }
    positions: dict[tuple[str, int, str], int] = {}
    normalized: list[ManuscriptQualityFinding] = []
    for finding in review.findings:
        if finding.paragraph_number not in valid.get(finding.section_id, set()):
            raise ValueError("manuscript reviewer used an unknown paragraph target")
        if finding.code in MATERIAL_MANUSCRIPT_REPAIR_CODES:
            # These defects are locally executable and materially affect the final
            # manuscript.  Review models often under-label them as optional warnings,
            # causing the editor loop to pass while preserving known repetition or
            # formulaic prose.  Code owns the gate and upgrades the action safely.
            finding = finding.model_copy(
                update={
                    "severity": "blocking",
                    "disposition": "targeted_repair",
                    "revision_instruction": _safe_manuscript_repair_instruction(
                        finding.code
                    ),
                }
            )
        elif finding.severity == "warning" and finding.disposition == "targeted_repair":
            finding = finding.model_copy(update={"disposition": "report_only"})
        elif finding.severity == "blocking" and finding.disposition == "report_only":
            finding = finding.model_copy(update={"disposition": "targeted_repair"})
        identity = (finding.section_id, finding.paragraph_number, finding.code)
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(normalized)
            normalized.append(finding)
            continue
        current = normalized[position]
        if _manuscript_finding_priority(finding) > _manuscript_finding_priority(current):
            normalized[position] = finding
    return review.model_copy(update={"findings": normalized})


def _manuscript_finding_priority(
    finding: ManuscriptQualityFinding,
) -> tuple[int, int, int]:
    return (
        int(finding.severity == "blocking"),
        int(finding.disposition == "targeted_repair"),
        len(finding.detail) + len(finding.revision_instruction),
    )


def normalize_manuscript_review_for_current_policy(
    review: ManuscriptQualityReview,
    plan: GroundedWritingPlan,
) -> ManuscriptQualityReview:
    """Upgrade a persisted review to the current executable repair policy."""

    return _validate_manuscript_review(review, plan)


def _safe_manuscript_repair_instruction(code: str) -> str:
    instructions = {
        "cross_section_repetition": (
            "仅重写本段：删除与其他段落重复的背景、案例和结论，只保留本段独有的"
            "中心判断及必要证据；不得移动段落或新增证据。"
        ),
        "paragraph_repetition": (
            "仅重写本段：删除与本章其他段落重复的背景、案例和判断，只保留本段"
            "独有的论证推进；不得新增证据或改写其他段落。"
        ),
        "section_role_overlap": (
            "仅重写本段：使其只完成当前章节计划规定的论证任务，删除属于其他章节"
            "的展开；不得移动段落或新增证据。"
        ),
        "academic_style_problem": (
            "仅重写本段：删除模板化过渡、空泛重要性判断和夸张措辞，用具体、克制"
            "的学术判断连接现有证据；不得新增事实或证据。"
        ),
        "oversized_paragraph": (
            "仅重写本段：围绕一个中心判断压缩并列案例、重复背景和次要细节，保留"
            "必要比较与局限；不得拆分、移动段落或新增证据。"
        ),
        "coherence_gap": (
            "仅重写本段：明确它与前一段中心判断的逻辑关系，并完成计划规定的"
            "论证动作；不得新增事实、来源或证据。"
        ),
        "terminology_inconsistent": (
            "仅重写本段：采用全文已经使用的规范术语，消除同一概念的冲突命名；"
            "不得改变技术含义或证据边界。"
        ),
    }
    return instructions.get(
        code,
        "仅在原段落与原证据范围内执行定点修订，不得新增事实或引用。",
    )


def _merge_manuscript_findings(
    review: ManuscriptQualityReview,
    deterministic: list[ManuscriptQualityFinding],
) -> ManuscriptQualityReview:
    merged = list(deterministic)
    positions = {
        (finding.section_id, finding.paragraph_number, finding.code): position
        for position, finding in enumerate(merged)
    }
    for finding in review.findings:
        identity = (finding.section_id, finding.paragraph_number, finding.code)
        if finding.code == "false_self_attribution" and identity not in positions:
            # This high-impact genre error has an explicit deterministic text
            # predicate.  Do not let an LLM reinterpret ordinary source-summary
            # prose as first-person ownership when that predicate is absent.
            continue
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(merged)
            merged.append(finding)
        elif _manuscript_finding_priority(finding) > _manuscript_finding_priority(
            merged[position]
        ):
            merged[position] = finding
    return review.model_copy(update={"findings": merged[:16]})


def _deferred_manuscript_findings(
    project: V04WritingProject,
) -> list[ManuscriptQualityFinding]:
    """Carry bounded V0.4 findings into V0.5 instead of relying on rediscovery."""

    carried: list[ManuscriptQualityFinding] = []
    for state in project.sections:
        if state.draft is None:
            continue
        if not any(
            issue.code == "quality_review_deferred" for issue in state.draft.issues
        ):
            continue
        for issue in state.draft.issues:
            if (
                issue.code not in DEFERABLE_CHAPTER_QUALITY_CODES
                or issue.severity != "warning"
                or issue.paragraph_number is None
            ):
                continue
            carried.append(
                ManuscriptQualityFinding(
                    section_id=state.section_id,
                    paragraph_number=issue.paragraph_number,
                    code=issue.code,
                    severity="warning",
                    disposition="report_only",
                    detail=f"V0.4 deferred finding: {issue.detail}"[:500],
                    revision_instruction=_safe_manuscript_repair_instruction(issue.code),
                )
            )
    return carried[:16]


def _deterministic_manuscript_findings(
    plan: GroundedWritingPlan,
    project: V04WritingProject,
) -> list[ManuscriptQualityFinding]:
    """Catch high-confidence structural defects even if the reviewer JSON fails."""

    findings: list[ManuscriptQualityFinding] = []
    paragraphs: list[tuple[str, int, str]] = []
    for state in project.sections:
        if state.draft is None:
            continue
        for number, paragraph in enumerate(state.draft.paragraphs, 1):
            paragraphs.append((state.section_id, number, paragraph.text))
            attribution = false_self_attribution_detail(paragraph.text)
            if attribution:
                findings.append(
                    ManuscriptQualityFinding(
                        section_id=state.section_id,
                        paragraph_number=number,
                        code="false_self_attribution",
                        severity="blocking",
                        disposition="targeted_repair",
                        detail=attribution,
                        revision_instruction=(
                            "保留原证据和结论边界，以锁定来源的作者为主语，删除“本文/"
                            "本研究/我们提出、利用或设计”等错误研究归属。"
                        ),
                    )
                )
            units = _prose_units(paragraph.text, output_language=plan.output_language)
            if units > 1200:
                findings.append(
                    ManuscriptQualityFinding(
                        section_id=state.section_id,
                        paragraph_number=number,
                        code="oversized_paragraph",
                        severity="blocking",
                        disposition="targeted_repair",
                        detail=f"单段达到 {units} 个统计单位，已包含过多独立论证动作。",
                        revision_instruction=(
                            "围绕计划中心问题收缩本段，只保留一个中心判断、必要比较和"
                            "局限；不要复述本章或其他章节。"
                        ),
                    )
                )
    for left in range(len(paragraphs)):
        left_section, left_number, left_text = paragraphs[left]
        for right in range(left + 1, len(paragraphs)):
            right_section, right_number, right_text = paragraphs[right]
            if left_section == right_section:
                continue
            similarity = content_similarity(left_text, right_text)
            if similarity < 0.82:
                continue
            findings.append(
                ManuscriptQualityFinding(
                    section_id=right_section,
                    paragraph_number=right_number,
                    code="cross_section_repetition",
                    severity="blocking",
                    disposition="targeted_repair",
                    detail=(
                        f"与 {left_section}:{left_number} 的跨章节内容相似度为 "
                        f"{similarity:.0%}。"
                    ),
                    revision_instruction=(
                        "删除重复背景，按照本章独有目的重新组织中心判断；不得用同义"
                        "改写继续复述前文。"
                    ),
                )
            )
    formulaic_occurrences: dict[str, list[tuple[str, int]]] = {}
    for section_id, number, text in paragraphs:
        for pattern in FORMULAIC_ACADEMIC_PATTERNS:
            if pattern.search(text):
                formulaic_occurrences.setdefault(pattern.pattern, []).append(
                    (section_id, number)
                )
    for pattern, occurrences in formulaic_occurrences.items():
        if len(occurrences) < 2:
            continue
        # Keep the first occurrence when it is contextually useful; repeated uses
        # are the high-confidence signal that the prose has become templated.
        for section_id, number in occurrences[1:]:
            findings.append(
                ManuscriptQualityFinding(
                    section_id=section_id,
                    paragraph_number=number,
                    code="academic_style_problem",
                    severity="blocking",
                    disposition="targeted_repair",
                    detail=(
                        "全文重复使用模板化学术表达，模式为 "
                        f"{pattern!r}，削弱了具体论证。"
                    ),
                    revision_instruction=_safe_manuscript_repair_instruction(
                        "academic_style_problem"
                    ),
                )
            )
    return findings[:16]


def apply_section_quality_review(
    draft: SectionDraft,
    review: SectionQualityReview,
) -> SectionDraft:
    """Replace prior editorial findings while preserving evidence audit results."""

    quality_codes = {
        *SECTION_QUALITY_REVIEW_CODES,
        "quality_review_failed",
        "quality_review_deferred",
    }
    retained = [issue for issue in draft.issues if issue.code not in quality_codes]
    findings = [
        SectionDraftIssue(
            code=finding.code,
            severity=finding.severity,
            paragraph_number=finding.paragraph_number,
            detail=(
                f"{finding.detail} Revision instruction: "
                f"{finding.revision_instruction}"
            ),
        )
        for finding in review.findings
    ]
    issues = [*retained, *findings]
    blocking_signatures = sorted(
        {
            f"{issue.paragraph_number or 0}:{issue.code}"
            for issue in issues
            if issue.severity == "blocking"
        }
    )
    review_round = draft.quality_review_rounds + 1
    history = [
        *draft.quality_review_history,
        SectionQualityReviewTrace(
            round_number=review_round,
            body_fingerprint=_section_review_body_fingerprint(draft),
            blocking_signatures=blocking_signatures,
            blocking_count=len(blocking_signatures),
        ),
    ][-6:]
    return draft.model_copy(
        update={
            "issues": issues,
            "status": (
                "needs_review"
                if any(issue.severity == "blocking" for issue in issues)
                else "draft"
            ),
            "confirmed_by": None,
            "confirmed_at": None,
            "quality_review_status": (
                "findings"
                if any(issue.severity == "blocking" for issue in findings)
                else "passed"
            ),
            "quality_review_rounds": review_round,
            "quality_reviewed_at": datetime.now(timezone.utc),
            "quality_review_history": history,
        }
    )


def _section_review_body_fingerprint(draft: SectionDraft) -> str:
    canonical = json.dumps(
        [paragraph.text for paragraph in draft.paragraphs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mark_section_quality_review_failed(
    draft: SectionDraft,
    error: Exception | str,
) -> SectionDraft:
    """Keep a generated checkpoint but make a failed independent review explicit."""

    retained = [issue for issue in draft.issues if issue.code != "quality_review_failed"]
    issues = [
        *retained,
        SectionDraftIssue(
            code="quality_review_failed",
            severity="warning",
            detail=f"Independent chapter review did not complete: {error}",
        ),
    ]
    return draft.model_copy(
        update={
            "issues": issues,
            "status": (
                "needs_review"
                if any(issue.severity == "blocking" for issue in issues)
                else "draft"
            ),
            "confirmed_by": None,
            "confirmed_at": None,
            "quality_review_status": "failed",
            "quality_review_rounds": draft.quality_review_rounds + 1,
            "quality_reviewed_at": datetime.now(timezone.utc),
        }
    )


def mark_section_quality_review_degraded(
    draft: SectionDraft,
    error: Exception | str,
) -> SectionDraft:
    """Continue after reviewer-contract failure while recording degraded assurance.

    Deterministic evidence, permission, citation, language and length gates have already
    run. A secondary LLM reviewer that cannot produce valid JSON is an availability fault,
    not proof that the chapter is invalid. The manuscript-wide editor and external scorer
    still inspect the assembled paper later.
    """

    retained = [
        issue
        for issue in draft.issues
        if issue.code not in {"quality_review_failed", "quality_review_degraded"}
    ]
    issues = [
        *retained,
        SectionDraftIssue(
            code="quality_review_degraded",
            severity="warning",
            detail=(
                "Independent chapter review returned unusable structured output after "
                "automatic repair; deterministic gates passed and processing continued: "
                f"{error}"
            ),
        ),
    ]
    return draft.model_copy(
        update={
            "issues": issues,
            "status": "draft",
            "confirmed_by": None,
            "confirmed_at": None,
            "quality_review_status": "passed",
            "quality_review_rounds": draft.quality_review_rounds + 1,
            "quality_reviewed_at": datetime.now(timezone.utc),
        }
    )


def defer_noncritical_section_findings(draft: SectionDraft) -> SectionDraft:
    """Stop a non-converging style loop without weakening trust boundaries.

    Repetition, transitions, terminology, generic style and paragraph size are
    editorial quality concerns.  They should be attempted locally, then inspected
    again by the manuscript-wide editor.  By contrast, topic drift, unsupported or
    overstated evidence and false research ownership remain blocking findings.
    """

    changed = False
    normalized: list[SectionDraftIssue] = []
    for issue in draft.issues:
        if (
            issue.severity == "blocking"
            and issue.code in DEFERABLE_CHAPTER_QUALITY_CODES
        ):
            normalized.append(issue.model_copy(update={"severity": "warning"}))
            changed = True
        else:
            normalized.append(issue)
    if not changed:
        return draft

    normalized = [
        issue
        for issue in normalized
        if issue.code not in {"quality_review_degraded", "quality_review_deferred"}
    ]
    normalized.append(
        SectionDraftIssue(
            code="quality_review_deferred",
            severity="warning",
            detail=(
                "The chapter reviewer exhausted its bounded local repair budget for "
                "non-critical editorial findings. The draft was retained and these "
                "items were deferred to the full-manuscript editor; evidence and topic "
                "trust gates were not weakened."
            ),
        )
    )
    blocking_remains = any(issue.severity == "blocking" for issue in normalized)
    return draft.model_copy(
        update={
            "issues": normalized,
            "status": "needs_review" if blocking_remains else "draft",
            "confirmed_by": None,
            "confirmed_at": None,
            "quality_review_status": "findings" if blocking_remains else "passed",
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


def content_similarity(left: str, right: str) -> float:
    """Return a conservative similarity score for prose-level duplication."""

    left_compact = _normalized_content(left)
    right_compact = _normalized_content(right)
    if not left_compact or not right_compact:
        return 0.0
    sequence = SequenceMatcher(None, left_compact, right_compact, autojunk=False).ratio()
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    shorter = min(len(left_tokens), len(right_tokens))
    containment = (
        len(left_tokens & right_tokens) / shorter if shorter >= 12 else 0.0
    )
    return max(sequence, jaccard, containment)


def _normalized_content(text: str) -> str:
    without_citations = re.sub(r"\[(?:@|\d)[^\]]*\]", "", text)
    return "".join(
        re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", without_citations.casefold())
    )


def _prose_units(text: str, *, output_language: str) -> int:
    if output_language == "English":
        return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text, re.UNICODE))
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    without_han = re.sub(r"[\u3400-\u9fff]", " ", text)
    words = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", without_han, re.UNICODE))
    return han + words


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
