"""Create the audited user-confirmed hand-off for literature retrieval."""

from __future__ import annotations

from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
)


class LiteratureBlueprintConfirmationService:
    """Freeze the exact provisional blueprint a user approved for retrieval."""

    def confirm(
        self,
        blueprint: LiteratureSearchBlueprint,
        *,
        confirmed_by: str,
        note: str | None = None,
    ) -> ConfirmedLiteratureSearchBlueprint:
        return ConfirmedLiteratureSearchBlueprint(
            confirmed_by=confirmed_by,
            confirmation_note=note,
            blueprint=blueprint,
        )
