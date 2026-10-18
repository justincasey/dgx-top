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
        description="Terminal monitoring for 1-12 node NVIDIA DGX Spark clusters.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="configuration file (default: $DGX_TOP_CONFIG or ~/.config/dgx-top/config.toml)",
    )
    parser.add_argument(
        "--theme",
        metavar="NAME",
        help="color theme override (default: the theme from the configuration file)",
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--simulate",
        type=int,
        metavar="N",
        help="run with N synthetic Spark nodes (no SSH/HTTP); N is 1-12",
    )
    init_parser = subparsers.add_parser("init", help="create an editable example configuration")
    init_parser.add_argument("--force", action="store_true", help="replace an existing file")
    subparsers.add_parser("check", help="verify configuration, SSH, telemetry, and engine access")
    subparsers.add_parser("themes", help="list available color themes")
    discover_parser = subparsers.add_parser(
        "discover", help="list inference endpoints listening on the configured nodes (read-only)"
    )
    discover_parser.add_argument("--json", action="store_true", help="print JSON instead")
    discover_parser.add_argument(
        "--all-ports",
        action="store_true",
        help="also probe ports in the nodes' ephemeral range",
    )
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
    print("Edit its SSH targets and engine URLs, then run: dgx-top check")
    return 0


def _list_themes() -> int:
    """Print every supported theme, marking the default and light themes."""
    from themes import DEFAULT_THEME, get_theme, theme_names

    print(f"default theme: {DEFAULT_THEME}")
    for name in theme_names():
        theme = get_theme(name)
        note = " (light)" if theme is not None and not theme.dark else ""
        print(f"  {name}{note}")
    return 0


async def _collect_scans(nodes, all_ports: bool) -> list:
    from discovery import sweep

    return [scan async for scan in sweep(nodes, all_ports=all_ports)]


def _discover(json_output: bool, all_ports: bool) -> int:
    """One read-only sweep: print what listens where, change nothing."""
    from config import get_settings
    from discovery import format_table, to_json

    scans = asyncio.run(_collect_scans(get_settings().nodes, all_ports))
    print(to_json(scans) if json_output else format_table(scans))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        return _init_config(args.config, args.force)
    if args.command == "themes":
        return _list_themes()
    try:
        if args.command == "check":
            if args.simulate:
                print(
                    "`check` verifies real SSH/HTTP endpoints; it cannot run with --simulate.",
                    file=sys.stderr,
                )
                return 2
            settings = load_config(args.config, theme=args.theme)
            return asyncio.run(run_preflight(settings))
        configure(args.config, theme=args.theme, simulate=args.simulate)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "discover":
        return _discover(args.json, args.all_ports)

    from app import run

    # Effective config path as resolved by load_config: --config wins, then
    # DGX_TOP_CONFIG/default. run() logs next to it.
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
