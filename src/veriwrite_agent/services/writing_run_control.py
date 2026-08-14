"""Cross-session pause and single-executor control for V0.4 writing."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time
from uuid import uuid4


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class WritingRunControlStore:
    """Persist pause state and grant one bounded executor lease per project."""

    def __init__(
        self,
        root: Path,
        *,
        project_key: str,
        stale_after_seconds: float = 1800.0,
    ) -> None:
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", project_key).strip("._")
        if not safe_key:
            raise ValueError("writing control requires a non-empty project key")
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self._root = root / safe_key
        self._state_path = self._root / "control.json"
        self._lease_path = self._root / "executor.lock"
        self._stale_after_seconds = stale_after_seconds

    def is_paused(self) -> bool:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(payload.get("paused"))

    def pause(self) -> None:
        self._write_state(paused=True)

    def resume(self) -> None:
        self._write_state(paused=False)

    def try_acquire(self, owner_id: str) -> WritingExecutorLease | None:
        if not owner_id.strip() or self.is_paused():
            return None
        self._root.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                descriptor = os.open(
                    self._lease_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if attempt == 0 and self._lease_is_stale():
                    try:
                        self._lease_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                return None
            payload = {
                "owner_id": owner_id,
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            return WritingExecutorLease(self, owner_id)
        return None

    def _write_state(self, *, paused: bool) -> None:
        _atomic_write(
            self._state_path,
            json.dumps(
                {
                    "paused": paused,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _lease_is_stale(self) -> bool:
        try:
            age = time() - self._lease_path.stat().st_mtime
        except FileNotFoundError:
            return True
        return age > self._stale_after_seconds

    def _lease_owned_by(self, owner_id: str) -> bool:
        try:
            payload = json.loads(self._lease_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return payload.get("owner_id") == owner_id

    def _heartbeat(self, owner_id: str) -> bool:
        if self.is_paused() or not self._lease_owned_by(owner_id):
            return False
        try:
            os.utime(self._lease_path, None)
        except FileNotFoundError:
            return False
        return True

    def _release(self, owner_id: str) -> None:
        if not self._lease_owned_by(owner_id):
            return
        try:
            self._lease_path.unlink()
        except FileNotFoundError:
            pass


@dataclass
class WritingExecutorLease:
    """One exclusive V0.4 execution lease."""

    _store: WritingRunControlStore
    owner_id: str
    _released: bool = False

    def continue_allowed(self) -> bool:
        return not self._released and self._store._heartbeat(self.owner_id)

    def release(self) -> None:
        if self._released:
            return
        self._store._release(self.owner_id)
        self._released = True

    def __enter__(self) -> WritingExecutorLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
