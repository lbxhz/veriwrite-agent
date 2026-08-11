"""Deterministic release gate for topic-scoped literature admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from veriwrite_agent.models.evidence import EvidenceLibrary
from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy


@dataclass(frozen=True)
class TopicAdmissionAudit:
    """Explain whether a literature library is safe to expose to writers."""

    boundary_actionable: bool
    invalid_dois: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.boundary_actionable and not self.invalid_dois

    @property
    def detail(self) -> str:
        reasons: list[str] = []
        if not self.boundary_actionable:
            reasons.append("已确认的主题边界不完整")
        if self.invalid_dois:
            preview = ", ".join(self.invalid_dois[:8])
            suffix = (
                f"（另有 {len(self.invalid_dois) - 8} 篇）"
                if len(self.invalid_dois) > 8
                else ""
            )
            reasons.append(
                f"{len(self.invalid_dois)} 篇文献缺少完整的准入用途：{preview}{suffix}"
            )
        return "；".join(reasons) or "主题准入已通过"


def audit_topic_admission(
    library: EvidenceLibrary,
    policy: ExecutableRequirementPolicy,
    *,
    valid_section_ids: Iterable[str] | None = None,
) -> TopicAdmissionAudit:
    """Fail closed when a record has not passed the relevance/use-boundary gate."""

    section_ids = set(valid_section_ids) if valid_section_ids is not None else None
    invalid: list[str] = []
    for record in library.records:
        complete = (
            record.admission_status == "admitted"
            and record.centrality in {"central", "supporting"}
            and bool(record.supported_claim and record.supported_claim.strip())
            and bool(record.suitable_section_id and record.suitable_section_id.strip())
            and bool(record.use_boundary and record.use_boundary.strip())
        )
        if section_ids is not None:
            complete = complete and record.suitable_section_id in section_ids
        if not complete:
            invalid.append(record.doi)
    return TopicAdmissionAudit(
        boundary_actionable=policy.topic_boundary.is_actionable,
        invalid_dois=tuple(invalid),
    )
