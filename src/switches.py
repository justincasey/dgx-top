"""Read-only polling of the network switches configured as ``[[switches]]``.

RouterOS v7's REST API is the only protocol today. Every request is a GET,
except RouterOS's ``/interface/ethernet/monitor`` command, which REST exposes
as a POST but which only reads (``once`` returns one sample and exits).

The parse helpers are pure and tolerant: a field that is missing or garbled
keeps its unknown sentinel (-1, "", empty) and never drops the switch. Only a
failed request for one of the three core endpoints marks the switch offline.

This module must not import ``collector``: the collector imports it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import os
import re
import ssl
import time
from urllib.parse import urlparse

import httpx

from config import SWITCH_PORT_LABEL_MAX, SwitchConfig
from stats import SWITCH_TEMP_ALERT, SwitchPort, SwitchPsu, SwitchStats

HTTP_TIMEOUT = 5.0
HTTP_BODY_LIMIT = 16 * 1024 * 1024
"""Same cap as ``collector.HTTP_BODY_LIMIT``; duplicated because importing the
collector here would be circular."""

_baselines: dict[tuple[str, str], tuple[float, int, int, int, int, int]] = {}
"""(switch label, port name) -> the previous poll's (monotonic time, rx bytes,
tx bytes, rx packets, tx packets, tx-queue-drop), for rates."""
_seen_running: dict[str, set[str]] = {}
"""switch label -> ports seen running by an earlier poll. With no configured
port list these stay listed after their link drops, so the drop is visible
(and a finding) instead of the row silently vanishing."""


async def _request_capped(
    client: httpx.AsyncClient, method: str, url: str, body: object | None = None
) -> bytes:
    """Send one request and return the body, aborting past HTTP_BODY_LIMIT.

    Raises an httpx.HTTPError subclass on transport, status or size problems."""
    async with client.stream(method, url, json=body) as resp:
        resp.raise_for_status()
        # One buffer, not one object per chunk (see collector._read_capped).
        data = bytearray()
        async for chunk in resp.aiter_bytes():
            data += chunk
            if len(data) > HTTP_BODY_LIMIT:
                raise httpx.HTTPError(f"response body exceeds {HTTP_BODY_LIMIT} bytes")
        return bytes(data)


def _float(value: object, default: float = -1.0) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: object, default: int = -1) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_resource(data: object, stats: SwitchStats) -> None:
    """Fill board, version, uptime, CPU and memory from ``/system/resource``."""
    if not isinstance(data, dict):
        return
    stats.board = _text(data.get("board-name"))
    # "7.20.1 (stable)": the release channel is noise on a tile.
    version = _text(data.get("version"))
    stats.version = version.split()[0] if version else ""
    stats.uptime = _text(data.get("uptime"))
    load = _float(data.get("cpu-load"))
    stats.cpu_pct = load if 0.0 <= load <= 100.0 else -1.0
    stats.cpu_mhz = max(_float(data.get("cpu-frequency"), 0.0), 0.0)
    stats.cpu_count = max(_int(data.get("cpu-count"), 0), 0)
    total = _int(data.get("total-memory"))
    free = _int(data.get("free-memory"))
    if total > 0 and 0 <= free <= total:
        stats.mem_total_bytes = total
        stats.mem_used_bytes = total - free


_PSU_FIELD = re.compile(r"(psu\d+)-(state|power|input-voltage)")
# "board-temperature1" -> board1; a single-sensor model's bare "temperature" -> temp.
_TEMP_FIELD = re.compile(r"(?:(.+)-)?temperature(\d*)")
_PSU_FAULTS = {"fail", "no-input"}
"""PSU states that are a finding. ``not-present`` (an empty bay) and ``idle``
(a redundant supply on standby) are normal for a switch with spare bays."""


def parse_health(rows: object, stats: SwitchStats) -> None:
    """Fill temperatures, fans and PSUs from ``/system/health`` rows."""
    if not isinstance(rows, list):
        return
    psus: dict[str, SwitchPsu] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _text(row.get("name"))
        value = row.get("value")
        if match := _TEMP_FIELD.fullmatch(name):
            temp = _float(value, math.nan)
            if not math.isnan(temp):
                stats.temps[(match[1] or "temp") + match[2]] = temp
        elif name.endswith("-speed"):
            rpm = _float(value)
            if rpm >= 0:
                stats.fans[name.removesuffix("-speed")] = rpm
        elif name == "fan-state":
            stats.fan_state = _text(value)
        elif match := _PSU_FIELD.fullmatch(name):
            psu = psus.setdefault(match[1], SwitchPsu(match[1]))
            if match[2] == "state":
                psu.state = _text(value)
            elif match[2] == "power":
                psu.watts = _float(value)
            else:
                psu.volts = _float(value)
    stats.psus = [psus[key] for key in sorted(psus, key=_natural_key)]


_RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)bps", re.IGNORECASE)
_RATE_SCALE = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}


def parse_rate(rate: object) -> float:
    """``"200Gbps"`` -> 2e11 bits/s; 0 when not a rate."""
    match = _RATE.fullmatch(_text(rate))
    if not match:
        return 0.0
    return float(match[1]) * _RATE_SCALE[match[2].lower()]


def _natural_key(name: str) -> list[tuple[int, int | str]]:
    return [(0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", name)]


_LABEL_RULES = (
    (re.compile(r"qsfp\d*-dd-(\d+)-(\d+)"), "dd"),
    (re.compile(r"qsfp\d*-(\d+)-(\d+)"), "q"),
    (re.compile(r"qsfp\d*-(\d+)"), "q"),
    (re.compile(r"sfp(?:\d+|-sfpplus|plus)?-?(\d+)"), "sfp"),
    (re.compile(r"ether(\d+)"), "eth"),
)


def port_label(name: str) -> str:
    """Short display label for a RouterOS interface name.

    qsfp56-dd-1-1 -> dd1, qsfp56-dd-1-2 -> dd1.2, qsfp28-3-1 -> q3,
    sfp28-2 / sfp-sfpplus2 -> sfp2, ether1 -> eth1; anything else is truncated."""
    for pattern, prefix in _LABEL_RULES:
        match = pattern.fullmatch(name)
        if match:
            label = prefix + match[1]
            if match.lastindex == 2 and match[2] != "1":
                label += "." + match[2]  # a breakout lane past the first
            return label[:SWITCH_PORT_LABEL_MAX]
    return name[:SWITCH_PORT_LABEL_MAX]


def _by_name(interfaces: object) -> dict[str, dict]:
    if not isinstance(interfaces, list):
        return {}
    return {
        _text(row.get("name")): row
        for row in interfaces
        if isinstance(row, dict) and _text(row.get("name"))
    }


def select_ports(cfg: SwitchConfig, interfaces: object) -> list[SwitchPort]:
    """The ports to show, as ``SwitchPort`` rows carrying ``running`` only.

    Configured ports: exactly those, in order (one absent on the switch reads
    as not running). Otherwise every enabled running ethernet port plus any
    seen running by an earlier poll, in natural name order."""
    rows = _by_name(interfaces)

    def running(name: str) -> bool:
        return _text(rows.get(name, {}).get("running")) == "true"

    if cfg.ports:
        return [SwitchPort(name, label, running=running(name)) for name, label in cfg.ports]
    seen = _seen_running.setdefault(cfg.label, set())
    for name, row in rows.items():
        if _text(row.get("disabled")) == "true":
            seen.discard(name)  # disabled on purpose: not a dropped link
        elif _text(row.get("type")) == "ether" and running(name):
            seen.add(name)
    return [
        SwitchPort(name, port_label(name), running=running(name))
        for name in sorted(seen, key=_natural_key)
    ]


def apply_counters(switch: str, port: SwitchPort, row: object, now: float) -> None:
    """Rate one port from its interface counters against the previous poll.

    The first observation only seeds. A counter that went backwards (switch
    reboot, wrap) re-seeds and reports 0, unmeasured, rather than a bogus rate."""
    key = (switch, port.name)
    if not isinstance(row, dict):
        _baselines.pop(key, None)
        return
    counters = (
        now,
        _int(row.get("rx-byte")),
        _int(row.get("tx-byte")),
        _int(row.get("rx-packet")),
        _int(row.get("tx-packet")),
        _int(row.get("tx-queue-drop"), 0),
    )
    if min(counters[1:]) < 0:
        _baselines.pop(key, None)  # unreadable counters: no rate, start over
        return
    previous = _baselines.get(key)
    _baselines[key] = counters
    if previous is None:
        return
    elapsed = now - previous[0]
    deltas = [cur - prev for cur, prev in zip(counters[1:], previous[1:])]
    if elapsed <= 0 or min(deltas) < 0:
        return
    rx_bytes, tx_bytes, rx_pkts, tx_pkts, drops = deltas
    port.rx_bps = rx_bytes * 8 / elapsed
    port.tx_bps = tx_bytes * 8 / elapsed
    port.rx_pps = rx_pkts / elapsed
    port.tx_pps = tx_pkts / elapsed
    port.queue_drops = drops
    port.measured = True


def _sensor_name(key: str) -> str:
    if key == "switch":
        return "sw"
    if key.startswith("board"):
        return "brd" + key.removeprefix("board")
    return key


def findings(stats: SwitchStats) -> list[tuple[str, str]]:
    """(level, text) findings, critical first, each group in display order."""
    crit = [
        ("crit", f"{_sensor_name(key)} {temp:.0f}°")
        for key, temp in stats.temps.items()
        if temp >= SWITCH_TEMP_ALERT
    ]
    crit += [("crit", f"{port.label} link down") for port in stats.ports if not port.running]
    warn = [("warn", f"{psu.name} {psu.state}") for psu in stats.psus if psu.state in _PSU_FAULTS]
    if stats.fan_state not in ("", "ok"):
        warn.append(("warn", f"fan {stats.fan_state}"))
    warn += [
        ("warn", f"{port.label} drops +{port.queue_drops}")
        for port in stats.ports
        if port.queue_drops > 0
    ]
    return crit + warn


def _tls_untrusted(exc: BaseException) -> bool:
    """True when certificate verification is what failed, anywhere in the
    cause chain (httpx wraps the ssl error in a ConnectError)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def failure_reason(exc: BaseException) -> str:
    """Short reason a switch request failed. Never the URL's credentials (it
    has none) nor the password (only ever in the auth header)."""
    if _tls_untrusted(exc):
        return "TLS certificate not trusted (see ca_file)"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return "authentication failed (HTTP 401)" if code == 401 else f"HTTP {code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    if isinstance(exc, httpx.ConnectError):
        return "unreachable"
    if isinstance(exc, ValueError):
        return "malformed response"
    if isinstance(exc, httpx.HTTPError) and str(exc):
        return str(exc)[:80]
    return type(exc).__name__


@functools.lru_cache(maxsize=None)
def _ca_context(ca_file: str) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=ca_file)


def tls_verify(cfg: SwitchConfig) -> ssl.SSLContext | bool:
    """httpx ``verify`` for a switch: its private CA's context (built once per
    file; httpx 0.28 deprecates a path string here), else the system store."""
    return _ca_context(cfg.ca_file) if cfg.ca_file else True


def _forget_rates(switch: str) -> None:
    """Drop a switch's rate baselines: after an outage the first sample must
    reseed, not average (or count drops over) the whole outage."""
    for key in [key for key in _baselines if key[0] == switch]:
        del _baselines[key]


def _offline(cfg: SwitchConfig, host: str, error: str) -> SwitchStats:
    _forget_rates(cfg.label)
    return SwitchStats(cfg.label, host, error=error)


async def _get_json(client: httpx.AsyncClient, url: str) -> object:
    return json.loads(await _request_capped(client, "GET", url))


async def _monitor(client: httpx.AsyncClient, base: str, names: list[str]) -> dict[str, dict]:
    """Negotiated rate per port from one ``monitor once``; {} when it fails, so
    the rest of the reading survives with the rates unknown."""
    body = {"numbers": ",".join(names), "once": "", ".proplist": "name,rate,status"}
    try:
        rows = json.loads(
            await _request_capped(client, "POST", f"{base}/rest/interface/ethernet/monitor", body)
        )
    except (httpx.HTTPError, ValueError):
        return {}
    return _by_name(rows)


async def poll_switch(index: int, cfg: SwitchConfig) -> SwitchStats:
    """Read one switch. Never raises: a failure returns it offline with a reason.

    ``index`` is the switch's position in the configuration (the collector's
    flight key); rate baselines and sticky ports are keyed by the label."""
    del index
    host = urlparse(cfg.url).hostname or ""
    password = os.environ.get(cfg.password_env)
    if password is None:
        return _offline(cfg, host, f"${cfg.password_env} is not set")
    stats = SwitchStats(cfg.label, host, online=True)
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, auth=(cfg.user, password), verify=tls_verify(cfg)
        ) as client:
            resource, health, interfaces = await asyncio.gather(
                _get_json(client, f"{cfg.url}/rest/system/resource"),
                _get_json(client, f"{cfg.url}/rest/system/health"),
                _get_json(client, f"{cfg.url}/rest/interface"),
            )
            now = time.monotonic()
            rows = _by_name(interfaces)
            stats.ports = select_ports(cfg, interfaces)
            # Only ports the switch has: one unknown name fails the whole command.
            present = [port.name for port in stats.ports if port.name in rows]
            monitor = await _monitor(client, cfg.url, present) if present else {}
    except Exception as exc:  # any failure is an offline reading, never a crash
        return _offline(cfg, host, failure_reason(exc))
    parse_resource(resource, stats)
    parse_health(health, stats)
    for port in stats.ports:
        apply_counters(cfg.label, port, rows.get(port.name), now)
        port.rate = _text(monitor.get(port.name, {}).get("rate"))
        port.rate_bps = parse_rate(port.rate)
    stats.findings = findings(stats)
    return stats
