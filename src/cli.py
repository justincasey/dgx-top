"""Command-line entry point for dgx-top."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from config import ConfigError, configure, default_config_path, example_config_path, load_config
from preflight import run_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dgx-top",
        description="Terminal monitoring for one- or two-node NVIDIA DGX Spark clusters.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="configuration file (default: $DGX_TOP_CONFIG or ~/.config/dgx-top/config.toml)",
    )
    subparsers = parser.add_subparsers(dest="command")
    init_parser = subparsers.add_parser("init", help="create an editable example configuration")
    init_parser.add_argument("--force", action="store_true", help="replace an existing file")
    subparsers.add_parser("check", help="verify configuration, SSH, telemetry, and vLLM")
    return parser


def _init_config(path: Path | None, force: bool) -> int:
    destination = path.expanduser() if path else default_config_path()
    if destination.exists() and not force:
        print(f"Configuration already exists: {destination}", file=sys.stderr)
        print("Use --force to replace it.", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = example_config_path()
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    print(f"Created {destination}")
    print("Edit its SSH targets and vLLM URLs, then run: dgx-top check")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        return _init_config(args.config, args.force)
    try:
        if args.command == "check":
            settings = load_config(args.config)
            return asyncio.run(run_preflight(settings))
        configure(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    from app import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
