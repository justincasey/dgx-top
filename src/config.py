"""Configuration loading and validation for dgx-top."""

from __future__ import annotations

import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from themes import DEFAULT_THEME, theme_names

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib


class ConfigError(ValueError):
    """Raised when a dgx-top configuration is missing or invalid."""


@dataclass(frozen=True)
class NodeConfig:
    label: str
    ssh_target: str
    vllm_url: str
    worker: bool = False


METER_TREATMENTS = {"gradient", "spark", "tick", "line"}


@dataclass(frozen=True)
class Settings:
    nodes: tuple[NodeConfig, ...]
    poll_interval: int = 5
    history_length: int = 40
    theme: str = DEFAULT_THEME
    meter_treatment: str = "gradient"
    quiet: bool = False


def _parse_theme(app: dict, override: str | None) -> str:
    """Resolve the configured theme, validating it against known theme names."""
    theme = app.get("theme", DEFAULT_THEME)
    if not isinstance(theme, str) or not theme.strip():
        raise ConfigError("app.theme must be a non-empty theme name")
    theme = theme.strip()
    if override is not None:
        theme = override.strip()
        if not theme:
            raise ConfigError("--theme must be a non-empty theme name")
    if theme not in theme_names():
        available = ", ".join(theme_names())
        raise ConfigError(f"unknown theme {theme!r}; available themes: {available}")
    return theme


def default_config_path() -> Path:
    override = os.environ.get("DGX_TOP_CONFIG")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "dgx-top" / "config.toml"


def example_config_path() -> Path:
    """Resolve the example configuration shipped with dgx-top."""
    source = Path(__file__).with_name("config.example.toml")
    if source.is_file():
        return source
    installed = Path(sysconfig.get_path("data")) / "share" / "dgx-top" / "config.example.toml"
    if installed.is_file():
        return installed
    raise ConfigError(
        "example configuration not found; reinstall dgx-top or run from the source tree"
    )


def _require_text(raw: dict, key: str, node_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"nodes[{node_number}].{key} must be a non-empty string")
    return value.strip()


def _parse_node(raw: object, node_number: int) -> NodeConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"nodes[{node_number}] must be a TOML table")
    label = _require_text(raw, "label", node_number)
    ssh_target = _require_text(raw, "ssh_target", node_number)
    vllm_url = _require_text(raw, "vllm_url", node_number).rstrip("/")
    parsed = urlparse(vllm_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"nodes[{node_number}].vllm_url must be an http(s) base URL")
    worker = raw.get("worker", False)
    if not isinstance(worker, bool):
        raise ConfigError(f"nodes[{node_number}].worker must be true or false")
    return NodeConfig(label, ssh_target, vllm_url, worker)


def load_config(path: str | Path | None = None, theme: str | None = None) -> Settings:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.is_file():
        raise ConfigError(
            f"configuration not found: {config_path}\n"
            "Run `dgx-top init` (or `uv run dgx-top init` from the source tree) to create it, "
            "then edit the node settings."
        )
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ConfigError("configuration must define at least one [[nodes]] table")
    if len(raw_nodes) > 2:
        raise ConfigError("this release supports one or two displayed nodes")
    nodes = tuple(_parse_node(node, index) for index, node in enumerate(raw_nodes, 1))
    labels = [node.label.casefold() for node in nodes]
    if len(set(labels)) != len(labels):
        raise ConfigError("node labels must be unique")

    app = raw.get("app", {})
    if not isinstance(app, dict):
        raise ConfigError("[app] must be a TOML table")
    poll_interval = app.get("poll_interval", 5)
    history_length = app.get("history_length", 40)
    if not isinstance(poll_interval, int) or not 1 <= poll_interval <= 60:
        raise ConfigError("app.poll_interval must be an integer from 1 to 60")
    if not isinstance(history_length, int) or not 10 <= history_length <= 1000:
        raise ConfigError("app.history_length must be an integer from 10 to 1000")
    meter_treatment = app.get("meter_treatment", "gradient")
    if not isinstance(meter_treatment, str) or meter_treatment not in METER_TREATMENTS:
        raise ConfigError(
            "app.meter_treatment must be one of: " + ", ".join(sorted(METER_TREATMENTS))
        )
    quiet = app.get("quiet", False)
    if not isinstance(quiet, bool):
        raise ConfigError("app.quiet must be true or false")
    return Settings(
        nodes,
        poll_interval,
        history_length,
        _parse_theme(app, theme),
        meter_treatment,
        quiet,
    )


_SETTINGS: Settings | None = None
SPARK_UNITS: dict[int, dict[str, object]] = {}
HISTORY_LEN = 40


def configure(path: str | Path | None = None, theme: str | None = None) -> Settings:
    """Load settings and update compatibility globals used by the collector."""
    global _SETTINGS, HISTORY_LEN
    settings = load_config(path, theme=theme)
    _SETTINGS = settings
    HISTORY_LEN = settings.history_length
    SPARK_UNITS.clear()
    SPARK_UNITS.update(
        {
            index: {
                "label": node.label,
                "ssh_target": node.ssh_target,
                "vllm_url": node.vllm_url,
                "worker": node.worker,
            }
            for index, node in enumerate(settings.nodes, 1)
        }
    )
    return settings


def get_settings() -> Settings:
    if _SETTINGS is None:
        raise ConfigError("dgx-top has not loaded a configuration")
    return _SETTINGS


NVIDIA_SMI_CMD = (
    "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,temperature.gpu,clocks.current.sm,clocks_throttle_reasons.active"
    " --format=csv,noheader,nounits"
)
