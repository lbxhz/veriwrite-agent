"""Fail when a candidate public commit contains private runtime data or local paths."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cs",
    ".example",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATHS = {".env"}
FORBIDDEN_PREFIXES = (".venv/", "output/", "outputs/", "runtime/")
FORBIDDEN_CONTENT = {
    "developer user directory": re.compile(
        "C:" + r"\\Users\\" + r"(?!<|user(?:name)?(?:\\|/))[^\\/]+",
        re.IGNORECASE,
    ),
    "private workspace path": re.compile(
        r"20\d{2}-\d{2}-\d{2}[/\\]" + "new-" + "chat",
        re.IGNORECASE,
    ),
    "private evidence path": re.compile(
        r"[A-Z]:\\" + "AI-" + "Agent-Projects",
        re.IGNORECASE,
    ),
    "OpenAI-compatible API token": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub personal token": re.compile(
        r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
}


def candidate_files() -> tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(
        item.replace("\\", "/")
        for item in result.stdout.decode("utf-8").split("\0")
        if item
    )


def main() -> int:
    violations: list[str] = []
    files = candidate_files()
    for relative in files:
        if relative in FORBIDDEN_PATHS or relative.startswith(FORBIDDEN_PREFIXES):
            violations.append(f"private path is a release candidate: {relative}")
            continue
        path = ROOT / relative
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN_CONTENT.items():
            if pattern.search(text):
                violations.append(f"{relative}: contains {label}")

    required = ("README.md", ".github/workflows/ci.yml", ".env.example")
    for relative in required:
        if relative not in files:
            violations.append(f"required public repository file is missing: {relative}")

    if violations:
        print("Public release check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print(f"Public release check passed for {len(files)} candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
