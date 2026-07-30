from pathlib import Path

import pytest

from veriwrite_agent.literature.norwegian_register import (
    NorwegianRegisterCatalogError,
    NorwegianRegisterRankingProvider,
)


@pytest.fixture(scope="module")
def provider() -> NorwegianRegisterRankingProvider:
    return NorwegianRegisterRankingProvider.from_default_catalog()


def test_default_snapshot_has_fixed_2025_source_shape(
    provider: NorwegianRegisterRankingProvider,
) -> None:
    assert provider.record_count == 36265


def test_issn_match_has_priority_over_a_noisy_title(
    provider: NorwegianRegisterRankingProvider,
) -> None:
    result = provider.lookup("A noisy title from metadata", ["0034-4257"])

    assert result.status == "matched"
    assert result.resolved_level == 2
    assert result.match_basis == "issn"
    assert result.records[0].original_title == "Remote Sensing of Environment"


def test_normalized_title_is_a_fallback_when_issn_is_missing(
    provider: NorwegianRegisterRankingProvider,
) -> None:
    result = provider.lookup("  REMOTE\u00a0 SENSING  ")

    assert result.status == "matched"
    assert result.resolved_level == 1
    assert result.match_basis == "title"


def test_level_zero_remains_visible_as_not_approved(
    provider: NorwegianRegisterRankingProvider,
) -> None:
    result = provider.lookup("# ISOJ Journal", ["2328-0700"])

    assert result.status == "matched"
    assert result.resolved_level == 0


def test_unknown_journal_is_not_silently_classified(
    provider: NorwegianRegisterRankingProvider,
) -> None:
    result = provider.lookup("Imaginary Journal of GeoAI", ["1234-5679"])

    assert result.status == "not_found"
    assert result.resolved_level is None


def test_invalid_snapshot_columns_fail_early(tmp_path: Path) -> None:
    catalog = tmp_path / "invalid.csv"
    catalog.write_text("journal,level\nExample,2\n", encoding="utf-8")

    with pytest.raises(NorwegianRegisterCatalogError, match="columns"):
        NorwegianRegisterRankingProvider(catalog)
