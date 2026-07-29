"""Provider ports and adapters for V0.2 literature discovery."""

from veriwrite_agent.literature.base import (
    AuthoritativeMetadataProvider,
    DoiResolver,
    JournalRankingProvider,
    LiteratureSearchError,
    LiteratureSearchProvider,
)
from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.crossref import CrossrefSearchProvider
from veriwrite_agent.literature.doi import DoiOrgResolver, DoiRisMetadataProvider
from veriwrite_agent.literature.fake import (
    FakeAuthoritativeMetadataProvider,
    FakeDoiResolver,
    FakeLiteratureSearchProvider,
)
from veriwrite_agent.literature.ris import RisParseError, parse_ris

__all__ = [
    "AuthoritativeMetadataProvider",
    "CrossrefSearchProvider",
    "CugJournalRankingProvider",
    "DoiOrgResolver",
    "DoiResolver",
    "DoiRisMetadataProvider",
    "FakeAuthoritativeMetadataProvider",
    "FakeDoiResolver",
    "FakeLiteratureSearchProvider",
    "JournalRankingProvider",
    "LiteratureSearchError",
    "LiteratureSearchProvider",
    "RisParseError",
    "parse_ris",
]
