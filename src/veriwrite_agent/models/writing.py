"""V0.4 contracts for evidence-constrained, section-by-section writing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.evidence import EvidenceQuote
from veriwrite_agent.models.literature_discovery import canonicalize_doi
from veriwrite_agent.models.requirements import StrictModel
from veriwrite_agent.models.writing_handoff import V04WritingHandoff

ParagraphRole = Literal[
    "detailed_evidence",
    "section_support",
    "background",
    "synthesis",
]


class SectionEvidenceItem(StrictModel):
    """One confirmed V0.3 evidence card exposed to a single section."""

    evidence_id: str = Field(pattern=r"^ev_[a-z0-9_]{3,80}$")
    doi: str
    normalized_claim: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    support_strength: Literal["direct", "partial"]
    supporting_quotes: list[EvidenceQuote] = Field(min_length=1, max_length=3)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class SectionSourceRecord(StrictModel):
    """Bibliographic context with an explicit permission boundary."""

    doi: str
    citation_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{2,79}$")
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int = Field(ge=1000, le=2100)
    journal: str | None = None
    abstract: str | None = None
    evidence_tier: Literal["A_core", "B_supporting", "C_background"]
    permitted_use: Literal[
        "detailed_claims",
        "section_support",
        "background_only",
    ]
    admission_status: Literal["admitted", "legacy_unreviewed"] = (
        "legacy_unreviewed"
    )
    centrality: Literal[
        "central", "supporting", "peripheral", "out_of_scope", "legacy_unreviewed"
    ] = "legacy_unreviewed"
    supported_claim: str | None = None
    suitable_section_id: str | None = None
    use_boundary: str | None = None

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def admitted_source_needs_writing_scope(self) -> SectionSourceRecord:
        if self.admission_status == "admitted" and (
            self.centrality not in {"central", "supporting"}
            or not self.supported_claim
            or not self.suitable_section_id
        ):
            raise ValueError("admitted section sources require an explicit writing scope")
        return self


class SectionEvidencePacket(StrictModel):
    """Deterministic section context sent to an LLM."""

    schema_version: Literal["0.4.0"] = "0.4.0"
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    target_words: int = Field(ge=100)
    counting_policy: Literal[
        "chinese_chars_and_english_words", "words"
    ] = "chinese_chars_and_english_words"
    output_language: Literal[
        "Chinese", "English", "bilingual", "pending_confirmation"
    ] = "pending_confirmation"
    research_questions: list[str] = Field(default_factory=list)
    required_source_dois: list[str] = Field(default_factory=list)
    max_sources_per_paragraph: int = Field(default=3, ge=1, le=8)
    evidence_items: list[SectionEvidenceItem] = Field(min_length=1)
    sources: list[SectionSourceRecord] = Field(min_length=1)
    ai_writing_mode: Literal["generation_allowed", "generation_blocked"] = (
        "generation_allowed"
    )
    ai_policy_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_must_resolve_to_section_sources(self) -> SectionEvidencePacket:
        evidence_ids = [item.evidence_id for item in self.evidence_items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("section evidence_id values must be unique")
        source_dois = [source.doi for source in self.sources]
        if len(source_dois) != len(set(source_dois)):
            raise ValueError("section source DOI values must be unique")
        if any(item.doi not in source_dois for item in self.evidence_items):
            raise ValueError("every section evidence item needs a source record")
        if any(doi not in source_dois for doi in self.required_source_dois):
            raise ValueError("required section sources must exist in the packet")
        if (
            self.ai_writing_mode == "generation_blocked"
            and not self.ai_policy_reasons
        ):
            raise ValueError("blocked AI writing requires policy reasons")
        return self


class DraftParagraphContent(StrictModel):
    """LLM prose and tentative support before support completeness validation."""

    role: ParagraphRole
    text: str = Field(min_length=1)
    evidence_card_ids: list[str] = Field(default_factory=list)
    source_dois: list[str] = Field(default_factory=list)

    @field_validator("evidence_card_ids", "source_dois", mode="after")
    @classmethod
    def values_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("paragraph support identifiers must be unique")
        return values

    @field_validator("source_dois")
    @classmethod
    def normalize_dois(cls, values: list[str]) -> list[str]:
        return [canonicalize_doi(value) for value in values]


class DraftParagraphProposal(DraftParagraphContent):
    """LLM prose proposal without authority to create citations."""

    @model_validator(mode="after")
    def paragraph_needs_declared_support(self) -> DraftParagraphProposal:
        if not self.evidence_card_ids and not self.source_dois:
            raise ValueError("every paragraph requires declared source support")
        if self.role == "detailed_evidence" and not self.evidence_card_ids:
            raise ValueError("detailed_evidence paragraphs require evidence cards")
        return self


class UnboundSectionDraftProposal(StrictModel):
    """Parse valid prose even when its support declarations need repair."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    paragraphs: list[DraftParagraphContent] = Field(min_length=1, max_length=20)


class ParagraphSupportBinding(StrictModel):
    """One repair-only LLM response that cannot alter paragraph prose."""

    paragraph_number: int = Field(ge=1)
    evidence_card_ids: list[str] = Field(default_factory=list)
    source_dois: list[str] = Field(default_factory=list)

    @field_validator("evidence_card_ids", "source_dois", mode="after")
    @classmethod
    def values_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("paragraph support identifiers must be unique")
        return values

    @field_validator("source_dois")
    @classmethod
    def normalize_dois(cls, values: list[str]) -> list[str]:
        return [canonicalize_doi(value) for value in values]

    @model_validator(mode="after")
    def support_cannot_be_empty(self) -> ParagraphSupportBinding:
        if not self.evidence_card_ids and not self.source_dois:
            raise ValueError("support repair cannot leave a paragraph unbound")
        return self


class SectionSupportBindingBatch(StrictModel):
    """Repair-only bindings for every paragraph in one immutable draft."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    bindings: list[ParagraphSupportBinding] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def paragraph_numbers_must_be_unique(self) -> SectionSupportBindingBatch:
        numbers = [binding.paragraph_number for binding in self.bindings]
        if len(numbers) != len(set(numbers)):
            raise ValueError("support repair paragraph numbers must be unique")
        return self


class SectionDraftProposal(StrictModel):
    """Structured LLM response later rendered and audited by code."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    paragraphs: list[DraftParagraphProposal] = Field(min_length=1, max_length=20)


class CitationBinding(StrictModel):
    """One deterministic link from rendered prose to source evidence."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    paragraph_number: int = Field(ge=1)
    citation_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{2,79}$")
    doi: str
    evidence_card_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class SectionDraftIssue(StrictModel):
    """A deterministic writing or citation gate result."""

    code: Literal[
        "unknown_evidence_card",
        "unknown_source_doi",
        "evidence_source_mismatch",
        "source_permission_exceeded",
        "unconfirmed_evidence",
        "llm_authored_citation",
        "workflow_instruction_leak",
        "partial_support",
        "word_count_low",
        "word_count_high",
        "final_audit_repair",
        "language_mismatch",
        "paragraph_repetition",
        "topic_drift",
        "coherence_gap",
        "terminology_inconsistent",
        "academic_style_problem",
        "quality_review_failed",
        "quality_review_degraded",
        "quality_review_deferred",
        "unsupported_claim",
        "overstated_evidence",
        "false_self_attribution",
        "oversized_paragraph",
    ]
    severity: Literal["warning", "blocking"]
    detail: str = Field(min_length=1)
    paragraph_number: int | None = Field(default=None, ge=1)


class SectionQualityReviewTrace(StrictModel):
    """Small persisted signal used to detect a non-converging review loop."""

    round_number: int = Field(ge=1)
    body_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocking_signatures: list[str] = Field(default_factory=list, max_length=20)
    blocking_count: int = Field(ge=0)


class SectionDraft(StrictModel):
    """Auditable section draft with code-generated citation bindings."""

    schema_version: Literal["0.4.0"] = "0.4.0"
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    title: str = Field(min_length=1)
    status: Literal["needs_review", "draft", "confirmed"]
    target_words: int = Field(ge=100)
    counted_words: int = Field(ge=0)
    paragraphs: list[DraftParagraphProposal] = Field(min_length=1)
    markdown: str = Field(min_length=1)
    citations: list[CitationBinding] = Field(default_factory=list)
    issues: list[SectionDraftIssue] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None
    quality_review_status: Literal[
        "not_run", "passed", "findings", "failed"
    ] = "not_run"
    quality_review_rounds: int = Field(default=0, ge=0)
    quality_reviewed_at: datetime | None = None
    quality_review_history: list[SectionQualityReviewTrace] = Field(
        default_factory=list,
        max_length=6,
    )

    @model_validator(mode="after")
    def status_must_match_issues_and_confirmation(self) -> SectionDraft:
        blocking = any(issue.severity == "blocking" for issue in self.issues)
        if self.status == "needs_review" and not blocking:
            raise ValueError("needs_review drafts require a blocking issue")
        if self.status == "draft" and blocking:
            raise ValueError("draft sections cannot contain blocking issues")
        if self.status == "confirmed":
            if blocking:
                raise ValueError("confirmed sections cannot contain blocking issues")
            if not self.confirmed_by or self.confirmed_at is None:
                raise ValueError("confirmed sections need confirmation audit fields")
        elif self.confirmed_by is not None or self.confirmed_at is not None:
            raise ValueError("unconfirmed sections cannot claim confirmation")
        if self.quality_review_status == "not_run":
            if self.quality_reviewed_at is not None:
                raise ValueError("the current unreviewed revision cannot have a review time")
        elif self.quality_review_rounds < 1 or self.quality_reviewed_at is None:
            raise ValueError("quality-reviewed sections need review audit fields")
        return self


class WritingSectionState(StrictModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    status: Literal["pending", "draft", "needs_review", "confirmed"] = "pending"
    draft: SectionDraft | None = None

    @model_validator(mode="after")
    def state_must_match_draft(self) -> WritingSectionState:
        if self.status == "pending" and self.draft is not None:
            raise ValueError("pending sections cannot contain a draft")
        if self.status != "pending":
            if self.draft is None or self.draft.status != self.status:
                raise ValueError("section state must match its draft status")
        return self


class V04WritingProject(StrictModel):
    """Durable state for staged body writing."""

    schema_version: Literal["0.4.0"] = "0.4.0"
    status: Literal["drafting", "body_complete"] = "drafting"
    handoff: V04WritingHandoff
    sections: list[WritingSectionState] = Field(min_length=1)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def section_states_must_match_confirmed_outline(self) -> V04WritingProject:
        expected_ids = [
            section.section_id
            for section in self.handoff.outline.outline.sections
        ]
        actual_ids = [section.section_id for section in self.sections]
        if actual_ids != expected_ids:
            raise ValueError("writing project sections must match outline order")
        all_confirmed = all(
            section.status == "confirmed" for section in self.sections
        )
        if (self.status == "body_complete") != all_confirmed:
            raise ValueError(
                "body_complete is valid exactly when every section is confirmed"
            )
        return self


class BodyDraftPackage(StrictModel):
    """Confirmed body Markdown plus its complete evidence trace."""

    schema_version: Literal["0.4.0"] = "0.4.0"
    topic: str = Field(min_length=1)
    markdown: str = Field(min_length=1)
    counted_words: int = Field(ge=1)
    citations: list[CitationBinding] = Field(min_length=1)
    source_dois: list[str] = Field(min_length=1)
