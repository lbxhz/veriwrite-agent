from veriwrite_agent.literature.fake import (
    FakeAuthoritativeMetadataProvider,
    FakeDoiResolver,
)
from veriwrite_agent.models.literature_discovery import LiteratureCandidate
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
    RisBibliographicMetadata,
)
from veriwrite_agent.services.literature_identity_verification import (
    LiteratureIdentityVerificationService,
)


def candidate(**updates: object) -> LiteratureCandidate:
    base: dict[str, object] = {
        "doi": "10.1000/geoai.1",
        "title": "GeoAI: Methods & Applications",
        "authors": ["San Zhang", "Jane Q. Doe"],
        "year": 2025,
        "journal_title": "Annals of GIS",
        "publisher": "Example Publisher",
        "source_type": "journal-article",
        "source_provider": "crossref",
        "source_url": "https://doi.org/10.1000/geoai.1",
    }
    base.update(updates)
    return LiteratureCandidate.model_validate(base)


def resolved(doi: str = "10.1000/geoai.1") -> DoiResolutionEvidence:
    return DoiResolutionEvidence(
        doi=doi,
        status="resolved",
        resolver_url=f"https://doi.org/{doi}",
        final_url="https://publisher.example/article/geoai",
        http_status=200,
        attempts=1,
        reason="resolved",
    )


def authority(
    *,
    doi: str = "10.1000/geoai.1",
    metadata_doi: str | None = "10.1000/geoai.1",
    title: str | None = "geoai methods and applications",
    authors: list[str] | None = None,
    year: int | None = 2025,
    journal: str | None = "Annals of GIS",
) -> AuthoritativeMetadataEvidence:
    return AuthoritativeMetadataEvidence(
        doi=doi,
        status="available",
        source_url="https://api.crossref.org/transform",
        metadata=RisBibliographicMetadata(
            doi=metadata_doi,
            title=title,
            authors=authors if authors is not None else ["Zhang, San", "Doe, Jane Q"],
            year=year,
            journal_title=journal,
            publisher="Example Publisher",
            ris_type="JOUR",
        ),
        raw_ris="TY  - JOUR\nER  -",
        attempts=1,
        reason="available",
    )


def test_uses_complete_authority_ris_as_the_canonical_record() -> None:
    resolver = FakeDoiResolver({"10.1000/geoai.1": resolved()})
    provider = FakeAuthoritativeMetadataProvider(
        {"10.1000/geoai.1": authority()}
    )
    service = LiteratureIdentityVerificationService(resolver, provider)

    result = service.verify(candidate())

    assert result.status == "verified"
    assert result.reason_codes == []
    assert result.authority is not None
    assert result.authority.metadata is not None
    assert result.authority.metadata.title == "geoai methods and applications"
    assert resolver.calls == ["10.1000/geoai.1"]
    assert provider.calls == ["10.1000/geoai.1"]


def test_complete_ris_doi_is_then_checked_for_resolution() -> None:
    resolution = DoiResolutionEvidence(
        doi="10.1000/geoai.1",
        status="unresolvable",
        resolver_url="https://doi.org/10.1000/geoai.1",
        http_status=404,
        attempts=1,
        reason="not found",
    )
    resolver = FakeDoiResolver({"10.1000/geoai.1": resolution})
    provider = FakeAuthoritativeMetadataProvider(
        {"10.1000/geoai.1": authority()}
    )
    service = LiteratureIdentityVerificationService(resolver, provider)

    result = service.verify(candidate())

    assert result.status == "excluded"
    assert result.reason_codes == ["doi_unresolvable"]
    assert provider.calls == ["10.1000/geoai.1"]


def test_restricted_landing_page_can_verify_with_authority_ris() -> None:
    resolution = DoiResolutionEvidence(
        doi="10.1000/geoai.1",
        status="landing_unavailable",
        resolver_url="https://doi.org/10.1000/geoai.1",
        final_url="https://publisher.example/restricted",
        http_status=403,
        attempts=1,
        reason="publisher rejected automated access",
    )
    provider = FakeAuthoritativeMetadataProvider(
        {"10.1000/geoai.1": authority()}
    )
    service = LiteratureIdentityVerificationService(
        FakeDoiResolver({"10.1000/geoai.1": resolution}),
        provider,
    )

    result = service.verify(candidate())

    assert result.status == "verified"
    assert result.reason_codes == []
    assert result.warning_codes == ["landing_page_unavailable"]
    assert provider.calls == ["10.1000/geoai.1"]


def test_candidate_title_does_not_override_the_canonical_ris_title() -> None:
    service = LiteratureIdentityVerificationService(
        FakeDoiResolver({"10.1000/geoai.1": resolved()}),
        FakeAuthoritativeMetadataProvider(
            {"10.1000/geoai.1": authority(title="A Different Paper")}
        ),
    )

    result = service.verify(candidate())

    assert result.status == "verified"
    assert result.authority is not None
    assert result.authority.metadata is not None
    assert result.authority.metadata.title == "A Different Paper"


def test_authority_ris_for_another_doi_is_excluded() -> None:
    service = LiteratureIdentityVerificationService(
        FakeDoiResolver({"10.1000/geoai.1": resolved()}),
        FakeAuthoritativeMetadataProvider(
            {
                "10.1000/geoai.1": authority(
                    metadata_doi="10.1000/different-paper"
                )
            }
        ),
    )

    result = service.verify(candidate())

    assert result.status == "excluded"
    assert result.reason_codes == ["ris_doi_mismatch"]


def test_missing_authority_field_is_not_silently_accepted() -> None:
    service = LiteratureIdentityVerificationService(
        FakeDoiResolver({"10.1000/geoai.1": resolved()}),
        FakeAuthoritativeMetadataProvider(
            {"10.1000/geoai.1": authority(authors=[])}
        ),
    )

    result = service.verify(candidate())

    assert result.status == "excluded"
    assert "authority_missing_authors" in result.reason_codes


def test_batch_deduplicates_the_crossref_doi_identity_key() -> None:
    resolver = FakeDoiResolver({"10.1000/geoai.1": resolved()})
    service = LiteratureIdentityVerificationService(
        resolver,
        FakeAuthoritativeMetadataProvider(
            {"10.1000/geoai.1": authority()}
        ),
    )
    duplicated = candidate(title="Duplicate presentation")

    batch = service.verify_many([candidate(), duplicated])

    assert len(batch.results) == 1
    assert len(batch.verified_records) == 1
    assert resolver.calls == ["10.1000/geoai.1"]
