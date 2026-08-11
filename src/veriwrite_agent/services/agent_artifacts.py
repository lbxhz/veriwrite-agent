"""Adapters between existing stage contracts and the compact Agent runtime state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from veriwrite_agent.models.agent_runtime import (
    AgentState,
    ArtifactKind,
    ArtifactReference,
)
from veriwrite_agent.models.evidence import EvidenceLibrary
from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.final_delivery import FinalPaperPackage
from veriwrite_agent.models.literature_discovery import LiteratureDiscoveryResult
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
)
from veriwrite_agent.models.literature_verification import LiteratureVerificationBatch
from veriwrite_agent.models.paper_quality import PaperQualityScorecard
from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.models.writing import BodyDraftPackage, V04WritingProject
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.models.writing_quality import ManuscriptQualityReview

ArtifactStatus = Literal[
    "draft",
    "confirmed",
    "ready",
    "needs_revision",
    "superseded",
    "invalid",
]


@dataclass(frozen=True)
class _ArtifactAdapter:
    kind: ArtifactKind
    fallback_schema_version: str


_ADAPTERS: dict[type[BaseModel], _ArtifactAdapter] = {
    RequirementSpec: _ArtifactAdapter("requirement_spec", "0.1.2"),
    ExecutableRequirementPolicy: _ArtifactAdapter("requirement_policy", "1.0.0"),
    ConfirmedLiteratureSearchBlueprint: _ArtifactAdapter(
        "literature_blueprint", "0.2.2"
    ),
    LiteratureDiscoveryResult: _ArtifactAdapter("literature_result", "0.2.0"),
    LiteratureVerificationBatch: _ArtifactAdapter(
        "literature_verification", "0.2.1"
    ),
    EvidenceLibrary: _ArtifactAdapter("evidence_library", "0.3.1"),
    V04WritingHandoff: _ArtifactAdapter("writing_handoff", "0.4-handoff.0"),
    GroundedWritingPlan: _ArtifactAdapter("writing_plan", "0.4-plan.0"),
    V04WritingProject: _ArtifactAdapter("writing_project", "0.4.0"),
    BodyDraftPackage: _ArtifactAdapter("body_draft", "0.4.0"),
    ManuscriptQualityReview: _ArtifactAdapter(
        "manuscript_review", "0.4-quality-review.0"
    ),
    FinalPaperPackage: _ArtifactAdapter("final_package", "mvp-2.2"),
    PaperQualityScorecard: _ArtifactAdapter("quality_scorecard", "paper-quality.0"),
}


def artifact_fingerprint(artifact: BaseModel) -> str:
    """Return a stable digest of one fully validated Pydantic artifact."""

    serialized = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _artifact_status(artifact: BaseModel) -> ArtifactStatus:
    if isinstance(artifact, RequirementSpec):
        return "draft"
    if isinstance(
        artifact,
        (ExecutableRequirementPolicy, ConfirmedLiteratureSearchBlueprint),
    ):
        return "confirmed"
    if isinstance(artifact, LiteratureDiscoveryResult):
        return "ready" if artifact.target_reached else "needs_revision"
    if isinstance(artifact, LiteratureVerificationBatch):
        return "ready" if artifact.results else "draft"
    if isinstance(artifact, EvidenceLibrary):
        return artifact.status
    if isinstance(artifact, V04WritingHandoff):
        return "ready"
    if isinstance(artifact, GroundedWritingPlan):
        return artifact.status
    if isinstance(artifact, V04WritingProject):
        return "ready" if artifact.status == "body_complete" else "draft"
    if isinstance(artifact, BodyDraftPackage):
        return "ready"
    if isinstance(artifact, ManuscriptQualityReview):
        return (
            "needs_revision"
            if any(finding.severity == "blocking" for finding in artifact.findings)
            else "ready"
        )
    if isinstance(artifact, FinalPaperPackage):
        return {
            "needs_revision": "needs_revision",
            "ready_for_confirmation": "ready",
            "confirmed": "confirmed",
        }[artifact.status]
    if isinstance(artifact, PaperQualityScorecard):
        return "ready" if artifact.release_gate == "passed" else "needs_revision"
    raise TypeError(f"unsupported Agent artifact type: {type(artifact).__name__}")


def artifact_reference_from_model(
    artifact: BaseModel,
    *,
    storage_key: str,
) -> ArtifactReference:
    """Build a validated compact reference without copying artifact contents."""

    adapter = _ADAPTERS.get(type(artifact))
    if adapter is None:
        raise TypeError(f"unsupported Agent artifact type: {type(artifact).__name__}")
    fingerprint = artifact_fingerprint(artifact)
    schema_version = str(
        getattr(artifact, "schema_version", adapter.fallback_schema_version)
    )
    return ArtifactReference(
        artifact_id=f"{adapter.kind}_{fingerprint[:16]}",
        kind=adapter.kind,
        schema_version=schema_version,
        fingerprint=fingerprint,
        storage_key=storage_key,
        status=_artifact_status(artifact),
    )


def register_artifact(state: AgentState, reference: ArtifactReference) -> AgentState:
    """Register one artifact and supersede only older active versions of its kind."""

    existing = next(
        (
            artifact
            for artifact in state.artifacts
            if artifact.artifact_id == reference.artifact_id
        ),
        None,
    )
    if existing is not None:
        if existing.fingerprint != reference.fingerprint or existing.kind != reference.kind:
            raise ValueError("artifact_id is already bound to different artifact content")
        return state

    updated_artifacts: list[ArtifactReference] = []
    for artifact in state.artifacts:
        if artifact.kind == reference.kind and artifact.status not in {
            "superseded",
            "invalid",
        }:
            updated_artifacts.append(artifact.model_copy(update={"status": "superseded"}))
        else:
            updated_artifacts.append(artifact)
    updated_artifacts.append(reference)
    return state.model_copy(
        update={
            "artifacts": updated_artifacts,
            "event_sequence": state.event_sequence + 1,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def active_artifact(
    state: AgentState,
    kind: ArtifactKind,
) -> ArtifactReference | None:
    """Return the current usable artifact of a kind, if one exists."""

    for artifact in reversed(state.artifacts):
        if artifact.kind == kind and artifact.status not in {"superseded", "invalid"}:
            return artifact
    return None
