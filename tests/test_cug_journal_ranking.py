from pathlib import Path

import pytest

from veriwrite_agent.literature.cug_catalog import (
    CugJournalRankingProvider,
    JournalCatalogError,
    normalize_journal_title,
)


def test_default_catalog_has_expected_source_shape() -> None:
    provider = CugJournalRankingProvider.from_default_catalog()

    assert provider.record_count == 15875
    assert len(provider.available_disciplines) == 38
    assert "测绘科学与技术" in provider.available_disciplines


def test_matches_remote_sensing_journals_in_surveying_discipline() -> None:
    provider = CugJournalRankingProvider.from_default_catalog()

    top = provider.lookup("Remote Sensing of Environment", "测绘科学与技术")
    second = provider.lookup("REMOTE-SENSING", "测绘科学与技术")

    assert top.status == "matched"
    assert top.resolved_tier == "T1"
    assert top.records[0].source_workbook == "测绘科学与技术.xlsx"
    assert second.status == "matched"
    assert second.resolved_tier == "T2"


def test_reports_source_catalog_conflict_instead_of_guessing() -> None:
    provider = CugJournalRankingProvider.from_default_catalog()

    result = provider.lookup("Energy & Environment", "管理科学与工程")

    assert result.status == "ambiguous"
    assert {record.tier for record in result.records} == {"T3", "T4"}


def test_unknown_journal_is_not_silently_classified() -> None:
    provider = CugJournalRankingProvider.from_default_catalog()

    result = provider.lookup("Imaginary Journal of GeoAI", "测绘科学与技术")

    assert result.status == "not_found"
    assert result.resolved_tier is None


def test_title_normalization_handles_formatting_not_semantic_aliases() -> None:
    assert normalize_journal_title("Energy & Environment") == (
        normalize_journal_title("  ENERGY and ENVIRONMENT ")
    )


def test_invalid_catalog_columns_fail_early(tmp_path: Path) -> None:
    catalog = tmp_path / "invalid.csv"
    catalog.write_text("journal,level\nExample,T1\n", encoding="utf-8")

    with pytest.raises(JournalCatalogError, match="columns"):
        CugJournalRankingProvider(catalog)
