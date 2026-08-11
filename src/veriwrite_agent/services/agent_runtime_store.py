"""Crash-safe local event and checkpoint storage for the bounded Agent loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from veriwrite_agent.models.agent_runtime import (
    AgentActionRequest,
    AgentCheckpoint,
    AgentState,
    ControllerDecision,
    CriticReport,
    ToolObservation,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ACTION_ID = re.compile(r"^act_[0-9a-f]{16}$")
_OBSERVATION_ID = re.compile(r"^obs_[0-9a-f]{16}$")
_CRITIC_ID = re.compile(r"^crit_[0-9a-f]{16}$")
_DECISION_ID = re.compile(r"^dec_[0-9a-f]{16}$")
_CHECKPOINT_ID = re.compile(r"^ckpt_[0-9a-f]{16}$")
_IDEMPOTENCY_KEY = re.compile(r"^[0-9a-f]{64}$")


class AgentRuntimeStore:
    """Store compact runtime events separately from full paper artifacts."""

    def __init__(self, root: Path, *, max_event_bytes: int = 4 * 1024 * 1024) -> None:
        self.root = root
        self.max_event_bytes = max_event_bytes

    def save_action(self, action: AgentActionRequest) -> None:
        self._save_model("actions", action.action_id, action)

    def load_action(self, action_id: str) -> AgentActionRequest | None:
        self._validate_id(action_id, _ACTION_ID, "action_id")
        return self._load_model("actions", action_id, AgentActionRequest)

    def save_observation(self, observation: ToolObservation) -> None:
        action = self.load_action(observation.action_id)
        if action is None:
            raise ValueError("observation requires a previously stored action")
        if action.idempotency_key != observation.idempotency_key:
            raise ValueError("observation idempotency_key does not match its action")

        index = self._load_idempotency_index()
        existing_id = index.get(observation.idempotency_key)
        if observation.status == "succeeded" and existing_id not in {
            None,
            observation.observation_id,
        }:
            raise ValueError("idempotency_key already has a successful observation")

        self._save_model(
            "observations",
            observation.observation_id,
            observation,
        )
        if observation.status == "succeeded" and existing_id is None:
            index[observation.idempotency_key] = observation.observation_id
            self._save_json(self.root / "idempotency_index.json", index)

    def load_observation(self, observation_id: str) -> ToolObservation | None:
        self._validate_id(observation_id, _OBSERVATION_ID, "observation_id")
        return self._load_model(
            "observations",
            observation_id,
            ToolObservation,
        )

    def load_success_by_idempotency(
        self,
        idempotency_key: str,
    ) -> ToolObservation | None:
        self._validate_id(idempotency_key, _IDEMPOTENCY_KEY, "idempotency_key")
        observation_id = self._load_idempotency_index().get(idempotency_key)
        if observation_id is None:
            return None
        observation = self.load_observation(observation_id)
        if observation is None or observation.status != "succeeded":
            raise ValueError("idempotency index points to a missing or unsuccessful result")
        return observation

    def save_critic_report(self, report: CriticReport) -> None:
        self._save_model("critics", report.report_id, report)

    def load_critic_report(self, report_id: str) -> CriticReport | None:
        self._validate_id(report_id, _CRITIC_ID, "report_id")
        return self._load_model("critics", report_id, CriticReport)

    def save_decision(self, decision: ControllerDecision) -> None:
        missing_observations = [
            observation_id
            for observation_id in decision.based_on_observation_ids
            if self.load_observation(observation_id) is None
        ]
        if missing_observations:
            raise ValueError(
                "controller decision references unknown observations: "
                + ", ".join(missing_observations)
            )
        missing_reports = [
            report_id
            for report_id in decision.based_on_critic_report_ids
            if self.load_critic_report(report_id) is None
        ]
        if missing_reports:
            raise ValueError(
                "controller decision references unknown critic reports: "
                + ", ".join(missing_reports)
            )
        if decision.next_action is not None:
            self.save_action(decision.next_action)
        self._save_model("decisions", decision.decision_id, decision)

    def load_decision(self, decision_id: str) -> ControllerDecision | None:
        self._validate_id(decision_id, _DECISION_ID, "decision_id")
        return self._load_model("decisions", decision_id, ControllerDecision)

    def save_checkpoint(self, checkpoint: AgentCheckpoint) -> None:
        checkpoint_dir = self.root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        matches = list(checkpoint_dir.glob(f"*_{checkpoint.checkpoint_id}.json"))
        payload = checkpoint.model_dump_json(indent=2)
        if matches:
            if len(matches) != 1 or matches[0].read_text(encoding="utf-8") != payload:
                raise ValueError("checkpoint_id is already bound to different content")
            return

        latest = self.load_latest_checkpoint()
        if latest is None:
            if checkpoint.sequence != 0 or checkpoint.parent_checkpoint_id is not None:
                raise ValueError("the first stored checkpoint must be sequence 0")
        else:
            if checkpoint.state.run_id != latest.state.run_id:
                raise ValueError("one runtime store cannot mix Agent run IDs")
            if checkpoint.sequence != latest.sequence + 1:
                raise ValueError("checkpoint sequence must advance exactly once")
            if checkpoint.parent_checkpoint_id != latest.checkpoint_id:
                raise ValueError("checkpoint parent must be the latest checkpoint")

        path = checkpoint_dir / f"{checkpoint.sequence:08d}_{checkpoint.checkpoint_id}.json"
        self._save_text(path, payload)
        self._save_json(
            self.root / "latest_checkpoint.json",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "sequence": checkpoint.sequence,
                "file": path.name,
            },
        )

    def load_latest_checkpoint(self) -> AgentCheckpoint | None:
        checkpoint_dir = self.root / "checkpoints"
        if not checkpoint_dir.is_dir():
            return None

        valid: list[AgentCheckpoint] = []
        saw_files = False
        for path in sorted(checkpoint_dir.glob("*.json")):
            saw_files = True
            try:
                valid.append(self._parse_model(path, AgentCheckpoint))
            except (OSError, UnicodeError, ValueError, ValidationError):
                continue
        if not valid:
            if saw_files:
                raise ValueError("checkpoint files exist but no valid recovery chain was found")
            return None

        pointed = self._load_pointed_checkpoint(valid)
        if pointed is not None:
            return pointed

        longest = self._load_unique_longest_chain(valid)
        if longest is not None:
            return longest

        roots = [
            checkpoint
            for checkpoint in valid
            if checkpoint.sequence == 0 and checkpoint.parent_checkpoint_id is None
        ]
        if len(roots) != 1:
            raise ValueError("checkpoint history must contain exactly one valid root")
        current = roots[0]
        while True:
            children = [
                checkpoint
                for checkpoint in valid
                if checkpoint.sequence == current.sequence + 1
                and checkpoint.parent_checkpoint_id == current.checkpoint_id
                and checkpoint.state.run_id == current.state.run_id
            ]
            if not children:
                return current
            if len(children) != 1:
                raise ValueError("checkpoint history contains an ambiguous fork")
            current = children[0]

    def _load_unique_longest_chain(
        self,
        valid: list[AgentCheckpoint],
    ) -> AgentCheckpoint | None:
        """Recover the sole furthest valid tip when the atomic pointer is stale."""

        by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in valid}
        parent_ids = {
            checkpoint.parent_checkpoint_id
            for checkpoint in valid
            if checkpoint.parent_checkpoint_id is not None
        }
        tips = [
            checkpoint
            for checkpoint in valid
            if checkpoint.checkpoint_id not in parent_ids
            and self._has_valid_checkpoint_ancestry(checkpoint, by_id)
        ]
        if not tips:
            return None
        furthest_sequence = max(checkpoint.sequence for checkpoint in tips)
        furthest = [
            checkpoint for checkpoint in tips if checkpoint.sequence == furthest_sequence
        ]
        return furthest[0] if len(furthest) == 1 else None

    @staticmethod
    def _has_valid_checkpoint_ancestry(
        checkpoint: AgentCheckpoint,
        by_id: dict[str, AgentCheckpoint],
    ) -> bool:
        current = checkpoint
        seen: set[str] = set()
        while current.parent_checkpoint_id is not None:
            if current.checkpoint_id in seen:
                return False
            seen.add(current.checkpoint_id)
            parent = by_id.get(current.parent_checkpoint_id)
            if (
                parent is None
                or parent.sequence != current.sequence - 1
                or parent.state.run_id != current.state.run_id
            ):
                return False
            current = parent
        return current.sequence == 0

    def _load_pointed_checkpoint(
        self,
        valid: list[AgentCheckpoint],
    ) -> AgentCheckpoint | None:
        """Recover the branch selected by the last atomic checkpoint write.

        Two Streamlit sessions can briefly share one run ID and create sibling
        checkpoints.  The store already atomically writes latest_checkpoint.json
        after every successful checkpoint, so that pointer is the durable branch
        selection.  Validate its complete ancestry before trusting it; corrupt or
        stale pointers fall back to the strict unique-chain scan below.
        """

        pointer_path = self.root / "latest_checkpoint.json"
        if not pointer_path.is_file():
            return None
        try:
            if pointer_path.stat().st_size > self.max_event_bytes:
                return None
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            checkpoint_id = pointer["checkpoint_id"]
            sequence = pointer["sequence"]
            self._validate_id(checkpoint_id, _CHECKPOINT_ID, "checkpoint_id")
            if not isinstance(sequence, int) or sequence < 0:
                return None
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

        by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in valid}
        current = by_id.get(checkpoint_id)
        if current is None or current.sequence != sequence:
            return None
        selected = current
        while current.parent_checkpoint_id is not None:
            parent = by_id.get(current.parent_checkpoint_id)
            if (
                parent is None
                or parent.sequence != current.sequence - 1
                or parent.state.run_id != current.state.run_id
            ):
                return None
            current = parent
        if current.sequence != 0:
            return None
        return selected

    def load_state(self) -> AgentState | None:
        checkpoint = self.load_latest_checkpoint()
        return checkpoint.state if checkpoint is not None else None

    def _save_model(self, directory: str, object_id: str, model: BaseModel) -> None:
        path = self.root / directory / f"{object_id}.json"
        payload = model.model_dump_json(indent=2)
        if path.is_file():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"{object_id} is already bound to different content")
            return
        self._save_text(path, payload)

    def _load_model(
        self,
        directory: str,
        object_id: str,
        model_type: type[_ModelT],
    ) -> _ModelT | None:
        path = self.root / directory / f"{object_id}.json"
        if not path.is_file():
            return None
        return self._parse_model(path, model_type)

    def _parse_model(self, path: Path, model_type: type[_ModelT]) -> _ModelT:
        if path.stat().st_size > self.max_event_bytes:
            raise ValueError(f"Agent runtime event exceeds {self.max_event_bytes} bytes")
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_idempotency_index(self) -> dict[str, str]:
        path = self.root / "idempotency_index.json"
        if not path.is_file():
            return {}
        if path.stat().st_size > self.max_event_bytes:
            raise ValueError("Agent idempotency index is too large")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Agent idempotency index must be a JSON object")
        index: dict[str, str] = {}
        for key, observation_id in raw.items():
            self._validate_id(key, _IDEMPOTENCY_KEY, "idempotency_key")
            self._validate_id(observation_id, _OBSERVATION_ID, "observation_id")
            index[key] = observation_id
        return index

    def _save_json(self, path: Path, value: object) -> None:
        self._save_text(
            path,
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
            allow_replace=True,
        )

    def _save_text(self, path: Path, payload: str, *, allow_replace: bool = False) -> None:
        encoded = payload.encode("utf-8")
        if len(encoded) > self.max_event_bytes:
            raise ValueError(f"Agent runtime event exceeds {self.max_event_bytes} bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        if path.exists() and not allow_replace:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"refusing to overwrite existing Agent event: {path.name}")
        temporary.replace(path)

    @staticmethod
    def _validate_id(value: str, pattern: re.Pattern[str], field: str) -> None:
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"invalid {field}")
