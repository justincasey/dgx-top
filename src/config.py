"""Configuration loading and validation for dgx-top."""

from __future__ import annotations

import os
import stat
import sysconfig
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

import tomlkit

from themes import DEFAULT_THEME, theme_names

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 (tomllib not in stdlib until 3.11)
    import tomli as tomllib


class ConfigError(ValueError):
    """Raised when a dgx-top configuration is missing or invalid."""


@dataclass(frozen=True)
class NodeConfig:
    label: str
    ssh_target: str
    vllm_url: str
    worker: bool = False
    engine: str | None = None
    """Expected inference engine (``"vllm"``, ``"sglang"`` or
    ``"tensorfold"``); ``None`` means infer it from the endpoint's own
    metrics. The URL is engine-agnostic, so the field selects no metric
    profile and no protocol of its own: it suppresses SGLang's load-API
    probe on a node declared as an engine without one (vLLM, TensorFold),
    and it names the engine on a load-only row (``collector.poll_unit``
    sets ``model_source`` from it)."""


ENGINES = {"vllm", "sglang", "tensorfold"}
"""Recognised values for ``NodeConfig.engine``.

Must stay equal to ``collector.ENGINE_PROFILES``: every engine the collector
can parse needs a declared value. The collector imports this module, so the
two sets cannot be derived from one another; ``tests/test_config.py`` pins
the coupling instead of leaving it to drift."""


METER_TREATMENTS = {"gradient", "spark", "tick", "line"}
CHART_MODES = {"oscilloscope", "lines"}
"""Decode-chart renderers: the oscilloscope (default) and the legacy
braille-fill line chart."""

MAX_NODES = 12
"""Largest cluster dgx-top displays (one or more Sparks)."""

CLOCK_CEILING_MIN = 1500
CLOCK_CEILING_MAX = 2400
"""Bounds (MHz) of ``app.clock_ceiling_mhz`` and of the `c` menu's ceiling: the
menu only locks a ceiling from 1500 to 2400 MHz."""


@dataclass(frozen=True)
class Settings:
    nodes: tuple[NodeConfig, ...]
    poll_interval: int = 5
    history_length: int = 40
    theme: str = DEFAULT_THEME
    meter_treatment: str = "gradient"
    chart: str = "oscilloscope"
    quiet: bool = False
    # GPU clock ceiling (MHz) the `c` menu applies: nvidia-smi -lgc 0,<ceiling>.
    clock_ceiling_mhz: int = 2200


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
    engine = raw.get("engine")
    if engine is not None:
        if not isinstance(engine, str) or engine.strip().lower() not in ENGINES:
            raise ConfigError(
                f"nodes[{node_number}].engine must be one of: " + ", ".join(sorted(ENGINES))
            )
        engine = engine.strip().lower()
    return NodeConfig(label, ssh_target, vllm_url, worker, engine)


def _synthetic_nodes(n: int) -> tuple[NodeConfig, ...]:
    """Build ``n`` offline-equivalent NodeConfigs for simulation mode."""
    return tuple(
        NodeConfig(
            label=f"spark-{i}",
            ssh_target=f"sim-{i}",
            vllm_url=f"http://sim-{i}:8000",
            worker=(i > 1),
        )
        for i in range(1, n + 1)
    )


def _parse_app(raw: dict, theme: str | None) -> tuple[int, int, str, str, str, bool, int]:
    """Validate the ``[app]`` block and return (poll_interval, history_length,
    theme, meter_treatment, chart, quiet, clock_ceiling_mhz)."""
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
    chart = app.get("chart", "oscilloscope")
    if not isinstance(chart, str) or chart not in CHART_MODES:
        raise ConfigError("app.chart must be one of: " + ", ".join(sorted(CHART_MODES)))
    quiet = app.get("quiet", False)
    if not isinstance(quiet, bool):
        raise ConfigError("app.quiet must be true or false")
    ceiling = app.get("clock_ceiling_mhz", 2200)
    # bools fall outside the range too
    if not isinstance(ceiling, int) or not CLOCK_CEILING_MIN <= ceiling <= CLOCK_CEILING_MAX:
        raise ConfigError(
            f"app.clock_ceiling_mhz must be an integer from {CLOCK_CEILING_MIN} "
            f"to {CLOCK_CEILING_MAX}"
        )
    return (
        poll_interval,
        history_length,
        _parse_theme(app, theme),
        meter_treatment,
        chart,
        quiet,
        ceiling,
    )


def load_config(
    path: str | Path | None = None,
    theme: str | None = None,
    simulate: int | None = None,
) -> Settings:
    config_path = effective_config_path(path)
    raw: dict = {}
    if config_path.is_file():
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    if simulate is not None:
        if not isinstance(simulate, int) or simulate < 1 or simulate > MAX_NODES:
            raise ConfigError(f"simulate must be an integer from 1 to {MAX_NODES}")
        app_settings = _parse_app(raw, theme)
        return Settings(_synthetic_nodes(simulate), *app_settings)

    if not config_path.is_file():
        raise ConfigError(
            f"configuration not found: {config_path}\n"
            "Run `dgx-top init` (or `uv run dgx-top init` from the source tree) to create it, "
            "then edit the node settings."
        )
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ConfigError("configuration must define at least one [[nodes]] table")
    if len(raw_nodes) > MAX_NODES:
        raise ConfigError(f"configuration supports up to {MAX_NODES} displayed nodes")
    nodes = tuple(_parse_node(node, index) for index, node in enumerate(raw_nodes, 1))
    labels = [node.label.casefold() for node in nodes]
    if len(set(labels)) != len(labels):
        raise ConfigError("node labels must be unique")

    app_settings = _parse_app(raw, theme)
    return Settings(nodes, *app_settings)


def effective_config_path(path: str | Path | None = None) -> Path:
    """The file ``load_config`` reads: an explicit path, else the default."""
    return Path(path).expanduser() if path else default_config_path()


_SETTINGS: Settings | None = None
_CONFIG_PATH: Path | None = None
SPARK_UNITS: dict[int, dict[str, object]] = {}
HISTORY_LEN = 40
SIMULATION_NODES: int | None = None
"""When set, ``collector.poll_cluster`` returns synthetic data for this many
units instead of SSH/HTTP polling. Owned here (not in the collector) so
``configure(simulate=N)`` is the single source of truth."""


def configure(
    path: str | Path | None = None, theme: str | None = None, simulate: int | None = None
) -> Settings:
    """Load settings and update compatibility globals used by the collector."""
    global _SETTINGS, _CONFIG_PATH, HISTORY_LEN, SIMULATION_NODES
    settings = load_config(path, theme=theme, simulate=simulate)
    _SETTINGS = settings
    # Simulated nodes have no file behind them: nothing may be written back.
    _CONFIG_PATH = None if simulate is not None else effective_config_path(path)
    HISTORY_LEN = settings.history_length
    SIMULATION_NODES = simulate
    SPARK_UNITS.clear()
    SPARK_UNITS.update(
        {
            index: {
                "label": node.label,
                "ssh_target": node.ssh_target,
                "vllm_url": node.vllm_url,
                "worker": node.worker,
                "engine": node.engine,
            }
            for index, node in enumerate(settings.nodes, 1)
        }
    )
    return settings


def get_settings() -> Settings:
    if _SETTINGS is None:
        raise ConfigError("dgx-top has not loaded a configuration")
    return _SETTINGS


def config_path() -> Path | None:
    """The configuration file backing the loaded settings; None when the
    settings are synthetic (``--simulate``) and nothing may be saved."""
    return _CONFIG_PATH


def set_node_endpoint(unit_id: int, url: str, engine: str | None) -> NodeConfig:
    """Point a configured node (1-based ``unit_id``) at another engine URL.

    Updates the loaded ``Settings`` and the collector's ``SPARK_UNITS`` view
    together. ``engine`` replaces the node's engine only when the node
    declares one; an undeclared engine stays detected from the endpoint's
    own metrics, exactly as ``save_node_endpoint`` leaves the file."""
    global _SETTINGS
    settings = get_settings()
    old = settings.nodes[unit_id - 1]
    node = replace(
        old,
        vllm_url=url.rstrip("/"),
        engine=engine if old.engine is not None else None,
    )
    nodes = settings.nodes[: unit_id - 1] + (node,) + settings.nodes[unit_id:]
    _SETTINGS = replace(settings, nodes=nodes)
    SPARK_UNITS[unit_id]["vllm_url"] = node.vllm_url
    SPARK_UNITS[unit_id]["engine"] = node.engine
    return node


def save_node_endpoint(
    path: str | Path, unit_id: int, label: str, url: str, engine: str | None
) -> None:
    """Write one node's new endpoint into the configuration file.

    Edits the ``unit_id``-th (1-based) ``[[nodes]]`` entry in place with
    tomlkit, so comments and every other key survive; a file tomlkit cannot
    reproduce byte for byte (``[[nodes]]`` split by another table) is refused
    instead. ``engine`` is written only when the entry already declares one;
    ``worker`` is never touched.
    The entry's label must equal ``label`` — a file edited since it was
    loaded is refused rather than written to the wrong node. The edited text
    must load through ``load_config`` before it replaces the original, which
    happens atomically (sibling temp file, ``os.replace``) with the
    original permission bits. Every refusal raises ``ConfigError`` and leaves
    the file byte-for-byte unchanged."""
    target = Path(path).expanduser().resolve()
    url = url.rstrip("/")
    try:
        original = target.read_text(encoding="utf-8")
        mode = stat.S_IMODE(target.stat().st_mode)
    except (OSError, UnicodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        raise ConfigError(f"cannot read {target}: {reason}") from exc
    try:
        doc = tomlkit.parse(original)
    except Exception as exc:
        raise ConfigError(f"invalid TOML in {target}: {exc}") from exc
    if doc.as_string() != original:
        # tomlkit regroups an array of tables split by another table
        # ([[nodes]] … [app] … [[nodes]]); writing it back would move whole
        # sections and re-attach their comments. Refuse rather than reformat.
        raise ConfigError(
            f"{target} cannot be edited in place (tables or comments would move, "
            f"e.g. [[nodes]] entries split by another table); set nodes[{unit_id}] by hand"
        )
    nodes = doc.get("nodes")
    if not isinstance(nodes, list) or not 1 <= unit_id <= len(nodes):
        raise ConfigError(f"{target} has no nodes[{unit_id}] entry")
    entry = nodes[unit_id - 1]
    if not isinstance(entry, dict) or str(entry.get("label", "")).strip() != label:
        raise ConfigError(f"nodes[{unit_id}] in {target} is no longer {label!r}")
    entry["vllm_url"] = url
    if engine is not None and "engine" in entry:
        entry["engine"] = engine
    try:
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    except OSError as exc:
        raise ConfigError(f"cannot write {target.parent}: {exc.strerror or exc}") from exc
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(tomlkit.dumps(doc))
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        # Re-validate what this edit can break. The theme rides as an
        # override: a session started with --theme may run on a file whose
        # own theme name is not valid, and that is no reason to refuse.
        saved = load_config(tmp, theme=DEFAULT_THEME).nodes[unit_id - 1]
        if saved.label != label or saved.vllm_url != url:
            raise ConfigError(f"edited nodes[{unit_id}] did not read back as {label!r} at {url}")
        os.replace(tmp, target)
    except ConfigError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        raise ConfigError(f"cannot write {target}: {reason}") from exc


NVIDIA_SMI_CMD = (
    "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,temperature.gpu,clocks.current.sm,clocks_throttle_reasons.active"
    " --format=csv,noheader,nounits"
)
