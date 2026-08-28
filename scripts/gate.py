"""Commit gate for dgx-top: run validation commands and write a receipt.

Exit 0 plus a fresh receipt under .git/ proves the gate passed; any failing
command stops the run and leaves no new receipt on disk.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = ROOT / ".git" / "agent-workflows-gate-receipt"

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
