"""Provider ports and adapters for V0.2 literature discovery."""

from veriwrite_agent.literature.base import (
    JournalRankingProvider,
    LiteratureSearchError,
    LiteratureSearchProvider,
)
from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.crossref import CrossrefSearchProvider
from veriwrite_agent.literature.fake import FakeLiteratureSearchProvider

__all__ = [
    "CrossrefSearchProvider",
    "CugJournalRankingProvider",
    "FakeLiteratureSearchProvider",
    "JournalRankingProvider",
    "LiteratureSearchError",
    "LiteratureSearchProvider",
]
