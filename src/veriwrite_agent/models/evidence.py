"""V0.3 contracts for PDF acquisition, grounded evidence, and literature matrices."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.literature_discovery import canonicalize_doi
from veriwrite_agent.models.requirements import StrictModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
EvidenceType = Literal[
    "background",
    "research_object",
    "data",
    "method",
    "result",
    "limitation",
    "future_work",
]
PdfIdentityBasis = Literal[
    "doi_text",
    "doi_metadata",
    "title_text",
    "title_metadata",
    "filename",
]


class DocumentAcquisition(StrictModel):
    """Auditable result of trying to obtain one selected paper PDF."""

    doi: str
    status: Literal["available", "upload_required", "excluded"]
    method: Literal["automatic_download", "user_upload", "none"]
    source_url: str | None = None
    local_path: str | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    media_type: Literal["application/pdf"] | None = None
    file_size_bytes: int | None = Field(default=None, ge=1)
    attempts: int = Field(default=0, ge=0, le=3)
    reason_codes: list[str] = Field(default_factory=list)
    acquired_at: datetime | None = None

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def availability_must_match_artifact(self) -> DocumentAcquisition:
        artifact_fields = (
            self.local_path,
            self.sha256,
            self.media_type,
            self.file_size_bytes,
        )
        if self.status == "available":
            if self.method == "none" or not all(artifact_fields):
                raise ValueError("available documents need a complete PDF artifact")
            if self.acquired_at is None:
                self.acquired_at = datetime.now(timezone.utc)
            if self.reason_codes:
                raise ValueError("available documents cannot contain failure reasons")
        else:
            if self.method != "none" or any(artifact_fields):
                raise ValueError("unavailable documents cannot claim a local artifact")
            if not self.reason_codes:
                raise ValueError("unavailable documents need an explicit reason")
        return self


class CorePaperExpectation(StrictModel):
    """One selected paper for which the user may need to download a PDF."""

    doi: str
    title: str = Field(min_length=1)
    source_url: str
    theme_id: str = Field(min_length=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class PdfInspectionIssue(StrictModel):
    """One deterministic problem found while checking a downloaded PDF."""

    code: Literal[
        "file_missing",
        "not_pdf",
        "pdf_unreadable",
        "pdf_encrypted",
        "empty_pdf",
        "missing_eof_marker",
        "text_not_extractable",
        "identity_not_confirmed",
    ]
    severity: Literal["warning", "blocking"]
    detail: str = Field(min_length=1)


class PdfInspectionReport(StrictModel):
    """Identity and integrity result for one user-downloaded core paper."""

    expectation: CorePaperExpectation
    status: Literal["verified", "needs_review", "invalid", "missing"]
    local_path: str | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    file_size_bytes: int | None = Field(default=None, ge=1)
    page_count: int | None = Field(default=None, ge=1)
    extractable_page_count: int = Field(default=0, ge=0)
    identity_score: float = Field(default=0, ge=0, le=1)
    identity_basis: list[PdfIdentityBasis] = Field(default_factory=list)
    issues: list[PdfInspectionIssue] = Field(default_factory=list)
    inspected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def status_must_match_inspection(self) -> PdfInspectionReport:
        blocking = any(issue.severity == "blocking" for issue in self.issues)
        if self.status == "verified":
            if blocking or self.identity_score < 0.8:
                raise ValueError(
                    "verified PDFs require confirmed identity and no blocking issues"
                )
            if not all(
                (
                    self.local_path,
                    self.sha256,
                    self.file_size_bytes,
                    self.page_count,
                )
            ):
                raise ValueError("verified PDFs require a complete local artifact")
        if self.status == "missing":
            if self.local_path is not None or self.sha256 is not None:
                raise ValueError("missing PDFs cannot claim a local artifact")
            if not any(issue.code == "file_missing" for issue in self.issues):
                raise ValueError("missing PDFs require a file_missing issue")
        return self


class PdfInspectionBatch(StrictModel):
    """Recoverable human-in-the-loop checkpoint for all core papers."""

    schema_version: Literal["0.3.0"] = "0.3.0"
    download_directory: str
    inspected_file_count: int = Field(ge=0)
    reports: list[PdfInspectionReport] = Field(default_factory=list)
    unmatched_files: list[str] = Field(default_factory=list)
    inspected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def each_expected_doi_must_appear_once(self) -> PdfInspectionBatch:
        dois = [report.expectation.doi for report in self.reports]
        if len(dois) != len(set(dois)):
            raise ValueError("a core paper can only have one inspection report")
        return self


class DocumentPage(StrictModel):
    """One page of extracted text with stable document and page identity."""

    doi: str
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    extraction_method: Literal["native_text", "ocr", "hybrid"]
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def confidence_only_applies_to_ocr(self) -> DocumentPage:
        if self.extraction_method == "native_text" and self.ocr_confidence is not None:
            raise ValueError("native text pages cannot claim OCR confidence")
        return self


class DocumentExtractionIssue(StrictModel):
    """One PDF page extraction problem kept for audit and retry."""

    code: Literal[
        "file_missing",
        "hash_mismatch",
        "page_text_missing",
        "ocr_unavailable",
        "pdf_unreadable",
    ]
    page_number: int | None = Field(default=None, ge=1)
    severity: Literal["warning", "blocking"]
    detail: str = Field(min_length=1)


class DocumentExtractionResult(StrictModel):
    """Page-preserving extraction output for one verified PDF."""

    doi: str
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal["complete", "needs_ocr", "failed"]
    page_count: int = Field(ge=0)
    pages: list[DocumentPage] = Field(default_factory=list)
    issues: list[DocumentExtractionIssue] = Field(default_factory=list)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def status_must_match_pages_and_issues(self) -> DocumentExtractionResult:
        if any(page.doi != self.doi for page in self.pages):
            raise ValueError("all extracted pages must belong to the result DOI")
        if any(
            page.document_sha256 != self.document_sha256 for page in self.pages
        ):
            raise ValueError("all extracted pages must use the result PDF hash")
        if self.status == "complete":
            if len(self.pages) != self.page_count or any(
                issue.severity == "blocking" for issue in self.issues
            ):
                raise ValueError(
                    "complete extraction requires text for every page and no blockers"
                )
        if self.status == "failed" and not any(
            issue.severity == "blocking" for issue in self.issues
        ):
            raise ValueError("failed extraction requires a blocking issue")
        return self


class EvidenceQuote(StrictModel):
    """Short source excerpt whose location can be checked deterministically."""

    page_number: int = Field(ge=1)
    exact_text: str = Field(min_length=1, max_length=1500)
    section_title: str | None = None


class EvidenceCard(StrictModel):
    """One LLM-normalized claim that remains attached to source text."""

    evidence_id: str = Field(pattern=r"^ev_[a-z0-9_]{3,80}$")
    doi: str
    theme_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    evidence_type: EvidenceType
    normalized_claim: str = Field(min_length=1)
    supporting_quotes: list[EvidenceQuote] = Field(min_length=1, max_length=3)
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    support_strength: Literal["direct", "partial"]
    review_status: Literal["needs_review", "confirmed", "rejected"] = "needs_review"

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class EvidenceCardProposal(StrictModel):
    """LLM proposal without authority to choose DOI, hash, ID, or theme."""

    evidence_type: EvidenceType
    normalized_claim: str = Field(min_length=1)
    supporting_quotes: list[EvidenceQuote] = Field(min_length=1, max_length=3)
    support_strength: Literal["direct", "partial"]


class EvidenceCardProposalBatch(StrictModel):
    proposals: list[EvidenceCardProposal] = Field(default_factory=list, max_length=12)


class EvidenceBackedValue(StrictModel):
    """One matrix value with explicit links to its supporting evidence cards."""

    value: str = Field(min_length=1)
    evidence_card_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_card_ids", mode="after")
    @classmethod
    def evidence_ids_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("matrix evidence_card_ids must be unique")
        return values


class LiteratureMatrixRow(StrictModel):
    """Evidence-linked comparison dimensions for one verified paper."""

    doi: str
    title: str = Field(min_length=1)
    theme_ids: list[str] = Field(min_length=1)
    research_objects: list[EvidenceBackedValue] = Field(default_factory=list)
    data_sources: list[EvidenceBackedValue] = Field(default_factory=list)
    methods: list[EvidenceBackedValue] = Field(default_factory=list)
    key_findings: list[EvidenceBackedValue] = Field(default_factory=list)
    limitations: list[EvidenceBackedValue] = Field(default_factory=list)
    background: list[EvidenceBackedValue] = Field(default_factory=list)
    future_work: list[EvidenceBackedValue] = Field(default_factory=list)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class GroundingIssue(StrictModel):
    """Deterministic evidence-chain failure found before user review."""

    evidence_id: str
    code: Literal[
        "document_page_missing",
        "quote_not_found_on_page",
        "document_identity_mismatch",
    ]
    detail: str = Field(min_length=1)


class GroundingReport(StrictModel):
    """Batch result of checking all quoted evidence against extracted pages."""

    valid: bool
    issues: list[GroundingIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validity_must_match_issues(self) -> GroundingReport:
        if self.valid == bool(self.issues):
            raise ValueError("valid must be true exactly when no grounding issues exist")
        return self


class LiteratureLibraryRecord(StrictModel):
    """One bibliographic record with an explicit full-text evidence tier."""

    doi: str
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int = Field(ge=1000, le=2100)
    journal: str | None = None
    abstract: str | None = None
    source_url: str
    theme_ids: list[str] = Field(min_length=1)
    evidence_tier: Literal["A_core", "B_supporting", "C_background"]
    evidence_status: Literal["full_text_verified", "metadata_verified"]
    permitted_use: Literal[
        "detailed_claims",
        "section_support",
        "background_only",
    ]

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def tier_must_match_evidence_status(self) -> LiteratureLibraryRecord:
        if self.evidence_tier == "A_core":
            if self.evidence_status != "full_text_verified":
                raise ValueError("A_core records require verified full text")
            if self.permitted_use != "detailed_claims":
                raise ValueError("A_core records must permit detailed claims")
        if self.evidence_status == "metadata_verified" and (
            self.permitted_use == "detailed_claims"
        ):
            raise ValueError("metadata-only records cannot support detailed claims")
        return self


class EvidenceLibrary(StrictModel):
    """V0.3 hand-off consumed later by outline and writing modules."""

    schema_version: Literal["0.3.0"] = "0.3.0"
    status: Literal["draft", "confirmed"] = "draft"
    records: list[LiteratureLibraryRecord] = Field(default_factory=list)
    documents: list[DocumentAcquisition] = Field(default_factory=list)
    pages: list[DocumentPage] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    literature_matrix: list[LiteratureMatrixRow] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> EvidenceLibrary:
        document_dois = {
            document.doi
            for document in self.documents
            if document.status == "available"
        }
        card_by_id = {card.evidence_id: card for card in self.evidence_cards}
        record_by_doi = {record.doi: record for record in self.records}
        if len(record_by_doi) != len(self.records):
            raise ValueError("literature library DOI values must be unique")
        if len(card_by_id) != len(self.evidence_cards):
            raise ValueError("evidence_id values must be unique")
        if any(card.doi not in document_dois for card in self.evidence_cards):
            raise ValueError("evidence cards require an available source document")
        if any(page.doi not in document_dois for page in self.pages):
            raise ValueError("extracted pages require an available source document")
        document_hashes = {
            document.doi: document.sha256
            for document in self.documents
            if document.status == "available"
        }
        if any(
            page.document_sha256 != document_hashes.get(page.doi)
            for page in self.pages
        ):
            raise ValueError("extracted pages must match the available PDF hash")
        if any(
            record.evidence_status == "full_text_verified"
            and record.doi not in document_dois
            for record in self.records
        ):
            raise ValueError("full-text records require an available document")
        if any(
            record.evidence_status == "metadata_verified"
            and record.doi in document_dois
            for record in self.records
        ):
            raise ValueError(
                "available PDFs must be represented as full-text records"
            )

        for row in self.literature_matrix:
            if row.doi not in document_dois:
                raise ValueError("matrix rows require an available source document")
            if self.records and row.doi not in record_by_doi:
                raise ValueError("matrix rows require a literature library record")
            cells = (
                row.research_objects
                + row.data_sources
                + row.methods
                + row.key_findings
                + row.limitations
                + row.background
                + row.future_work
            )
            for cell in cells:
                for evidence_id in cell.evidence_card_ids:
                    card = card_by_id.get(evidence_id)
                    if card is None:
                        raise ValueError("matrix cells reference an unknown evidence card")
                    if card.doi != row.doi:
                        raise ValueError("matrix evidence must come from the same paper")

        if self.status == "confirmed":
            if self.unresolved_issues:
                raise ValueError("confirmed libraries cannot have unresolved issues")
            if not self.confirmed_by or self.confirmed_at is None:
                raise ValueError("confirmed libraries need confirmation audit fields")
            if any(
                card.review_status != "confirmed" for card in self.evidence_cards
            ):
                raise ValueError("confirmed libraries require confirmed evidence cards")
            if self.records and any(
                not any(page.doi == card.doi for page in self.pages)
                for card in self.evidence_cards
            ):
                raise ValueError("confirmed evidence cards require extracted pages")
        elif self.confirmed_by is not None or self.confirmed_at is not None:
            raise ValueError("draft libraries cannot claim confirmation")
        return self
