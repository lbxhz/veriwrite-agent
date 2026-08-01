"""Atomic local persistence for the single-user MVP project checkpoint."""

from __future__ import annotations

from pathlib import Path


class LocalProjectStore:
    """Persist validated snapshot JSON outside Streamlit's volatile session."""

    def __init__(self, path: Path, *, max_bytes: int = 20 * 1024 * 1024) -> None:
        self.path = path
        self.max_bytes = max_bytes

    def load(self) -> str | None:
        if not self.path.is_file():
            return None
        if self.path.stat().st_size > self.max_bytes:
            raise ValueError("本地项目自动存档超过20 MB，已拒绝载入")
        return self.path.read_text(encoding="utf-8")

    def save(self, snapshot_json: str) -> None:
        payload = snapshot_json.encode("utf-8")
        if len(payload) > self.max_bytes:
            raise ValueError("本地项目自动存档超过20 MB，已拒绝保存")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
        self.path.with_suffix(self.path.suffix + ".tmp").unlink(missing_ok=True)
