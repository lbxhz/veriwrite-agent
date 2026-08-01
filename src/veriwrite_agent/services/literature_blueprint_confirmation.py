"""Create the audited user-confirmed hand-off for literature retrieval."""

from __future__ import annotations

from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
)
from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy


class LiteratureBlueprintConfirmationService:
    """Freeze the exact provisional blueprint a user approved for retrieval."""

    def confirm(
        self,
        blueprint: LiteratureSearchBlueprint,
        *,
        confirmed_by: str,
        note: str | None = None,
        expected_policy: ExecutableRequirementPolicy | None = None,
    ) -> ConfirmedLiteratureSearchBlueprint:
        if expected_policy is not None and blueprint.requirement_policy != expected_policy:
            raise ValueError("the V0.1 executable policy is immutable during blueprint editing")
        return ConfirmedLiteratureSearchBlueprint(
            confirmed_by=confirmed_by,
            confirmation_note=note,
            blueprint=blueprint,
        )
