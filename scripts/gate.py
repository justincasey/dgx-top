"""Commit gate for dgx-top: run validation commands and write a receipt.

Exit 0 plus a fresh receipt in the git directory proves the gate passed; any
failing command stops the run and leaves no new receipt on disk. The receipt
resolves through ``git rev-parse --git-path`` so a linked worktree (where
``.git`` is a file pointing at the main checkout's git dir) writes its own
receipt instead of failing on ``NotADirectoryError``.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECEIPT_NAME = "agent-workflows-gate-receipt"


def _receipt_path() -> Path:
    """The receipt inside this checkout's git directory (per-worktree)."""
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", RECEIPT_NAME],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path = Path(out)
    return path if path.is_absolute() else ROOT / path


RECEIPT = _receipt_path()

COMMANDS = [
    ["uv", "run", "ruff", "check", "."],
    ["uv", "run", "ruff", "format", "--check", "."],
    ["uv", "run", "pytest", "-q"],
]


def main() -> int:
    lines = [f"gate run at {datetime.now(timezone.utc).isoformat()}"]
    for cmd in COMMANDS:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
        lines.append(f"$ {' '.join(cmd)} -> exit {proc.returncode}")
        lines.extend(f"  {line}" for line in tail)
        if proc.returncode != 0:
            lines.append("GATE FAILED")
            print("\n".join(lines))
            return proc.returncode or 1
    lines.append("GATE PASS")
    RECEIPT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
