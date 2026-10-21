"""One-shot discovery of inference-engine endpoints on the configured nodes.

A sweep lists each node's listening TCP ports over the same multiplexed SSH
connection the collector uses (``ss -Htln``, or ``/proc/net/tcp{,6}`` where
``ss`` is missing), then probes every candidate port from this machine — the
vantage point the dashboard polls from — with the collector's own engine
classifier. Nothing runs on a timer: callers ask for one sweep and get one.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator, Iterable, Optional, Sequence
from urllib.parse import urlparse

import httpx

import collector
import config
from config import NodeConfig

RANGE_MARK = "@@DGX_TOP_RANGE"
SS_MARK = "@@DGX_TOP_SS"
PROC_MARK = "@@DGX_TOP_PROC"

LISTEN_CMD = (
    f"echo {RANGE_MARK}; cat /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null; "
    "if command -v ss >/dev/null 2>&1 && ss -Htln >/dev/null 2>&1; then "
    f"echo {SS_MARK}; ss -Htln 2>/dev/null; "
    f"else echo {PROC_MARK}; cat /proc/net/tcp /proc/net/tcp6 2>/dev/null; fi; "
    "exit 0"
)
"""One POSIX ``sh`` command per node: the ephemeral port range, then the
listening sockets from whichever source the node has. Forced to exit 0 so a
missing ``tcp6`` table never reads as an SSH failure."""

SKIP_PORTS = frozenset({22})
PROBE_TIMEOUT = 2.0
PROBE_CONCURRENCY = 16

# Scope precedence when several sockets share a port: a wildcard bind is
# reachable on every address, a specific bind on one, loopback on none.
_SCOPE_RANK = {"any": 2, "host": 1, "loopback": 0}


@dataclass(frozen=True)
class Listener:
    address: str
    port: int
    scope: str  # "any" | "host" | "loopback"


@dataclass(frozen=True)
class Endpoint:
    url: str
    port: int
    engine: str
    model: str  # "" when /v1/models named nothing
    status: str  # "monitored" | "new"


@dataclass
class NodeScan:
    uid: int  # 1-based, as collector.SPARK_UNITS
    label: str
    worker: bool
    endpoints: list[Endpoint] = field(default_factory=list)
    local_only: list[int] = field(default_factory=list)
    error: Optional[str] = None  # set ⇔ the node could not be listed (ssh-failed)


# ─── parsing (total over remote text) ────────────────────────────────


def _scope(address: str) -> Optional[str]:
    if address in ("", "*", "0.0.0.0", "::"):
        return "any"
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return None
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip.is_unspecified:
        return "any"
    return "loopback" if ip.is_loopback else "host"


def _split_local(local: str) -> Optional[tuple[str, int]]:
    """``0.0.0.0:8000`` / ``[::]:8000`` / ``*:8000`` / ``127.0.0.53%lo:53`` /
    ``[fe80::1]%eth0:8006`` (ss prints a device zone outside the brackets)."""
    address, sep, port = local.rpartition(":")
    if not sep or not port.isdigit():
        return None
    address = address.split("%", 1)[0].strip("[]")
    return address, int(port)


def _parse_ss(lines: Iterable[str]) -> list[Listener]:
    found: list[Listener] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 4 or fields[0] != "LISTEN":
            continue
        split = _split_local(fields[3])
        if split is None:
            continue
        address, port = split
        scope = _scope(address)
        if scope is not None and 0 < port < 65536:
            found.append(Listener(address, port, scope))
    return found


def _proc_address(hex_addr: str) -> Optional[str]:
    try:
        raw = bytes.fromhex(hex_addr)
    except ValueError:
        return None
    if len(raw) == 4:
        return str(ipaddress.IPv4Address(raw[::-1]))
    if len(raw) == 16:
        # Four host-order (little-endian) 32-bit words.
        words = b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))
        return str(ipaddress.IPv6Address(words))
    return None


def _parse_proc(lines: Iterable[str]) -> list[Listener]:
    found: list[Listener] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 4 or fields[3] != "0A":  # 0A = TCP_LISTEN
            continue
        hex_addr, _, hex_port = fields[1].partition(":")
        address = _proc_address(hex_addr)
        try:
            port = int(hex_port, 16)
        except ValueError:
            continue
        if address is None or not 0 < port < 65536:
            continue
        scope = _scope(address)
        if scope is not None:
            found.append(Listener(address, port, scope))
    return found


def _dedupe(listeners: Iterable[Listener]) -> list[Listener]:
    """One listener per port, the most reachable bind winning."""
    best: dict[int, Listener] = {}
    for listener in listeners:
        held = best.get(listener.port)
        if held is None or _SCOPE_RANK[listener.scope] > _SCOPE_RANK[held.scope]:
            best[listener.port] = listener
    return [best[port] for port in sorted(best)]


def _parse_range(lines: list[str]) -> Optional[tuple[int, int]]:
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            low, high = int(parts[0]), int(parts[1])
            if 0 < low <= high < 65536:
                return low, high
    return None


def parse_listen_output(text: str) -> tuple[list[Listener], Optional[tuple[int, int]]]:
    """Split ``LISTEN_CMD`` output into deduplicated listeners and the
    ephemeral port range (None when the node did not state one)."""
    sections: dict[str, list[str]] = {}
    current: Optional[list[str]] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in (RANGE_MARK, SS_MARK, PROC_MARK):
            current = sections.setdefault(stripped, [])
        elif current is not None and stripped:
            current.append(stripped)
    listeners = _parse_ss(sections.get(SS_MARK, [])) + _parse_proc(sections.get(PROC_MARK, []))
    return _dedupe(listeners), _parse_range(sections.get(RANGE_MARK, []))


def candidates(
    listeners: Sequence[Listener],
    port_range: Optional[tuple[int, int]],
    all_ports: bool = False,
) -> list[Listener]:
    """Drop SSH and, unless ``all_ports``, the kernel's ephemeral range —
    where Ray/NCCL workers park dozens of listeners. An unknown range keeps
    every port: nothing is filtered on a guess."""
    kept = []
    for listener in listeners:
        if listener.port in SKIP_PORTS:
            continue
        if not all_ports and port_range and port_range[0] <= listener.port <= port_range[1]:
            continue
        kept.append(listener)
    return kept


# ─── probing ─────────────────────────────────────────────────────────


def _is_local_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _ssh_hostname(target: str) -> Optional[str]:
    """The host ``ssh`` connects to for ``target`` (``ssh -G``: local config
    evaluation, no connection is made)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-G",
            target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception:
        return None
    for line in stdout.decode(errors="replace").splitlines():
        key, _, value = line.partition(" ")
        if key == "hostname" and value.strip():
            return value.strip()
    return None


async def node_host(node: NodeConfig) -> str:
    """Host this machine reaches the node at: the configured URL's host,
    else (a tunnelled ``localhost`` URL names no node) ``ssh -G``'s, else
    the configured host as written."""
    host = urlparse(node.vllm_url).hostname or ""
    if host and not _is_local_host(host):
        return host
    return await _ssh_hostname(node.ssh_target) or host


def _url(scheme: str, host: str, port: int) -> str:
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


async def probe(client: httpx.AsyncClient, base_url: str) -> Optional[tuple[str, str]]:
    """``(engine, model)`` when ``base_url`` serves an inference engine.

    ``/metrics`` sample evidence first, then SGLang's load API (a server
    started without ``--enable-metrics``), then ``/v1/models`` for the name.
    Never raises: a foreign or dead port is simply not an engine."""
    engine: Optional[str] = None
    try:
        resp = await client.get(f"{base_url}/metrics")
        resp.raise_for_status()
        engine = collector.metrics_engine(resp.text)
    except Exception:
        engine = None
    if engine is None:
        load = await collector.fetch_engine_load(base_url, timeout=PROBE_TIMEOUT)
        if load is None:
            return None
        engine = "sglang"
    return engine, await collector._fetch_model_name(client, base_url)


def _status(node: NodeConfig, host: str, port: int) -> str:
    configured = urlparse(node.vllm_url)
    try:
        configured_port = configured.port or (443 if configured.scheme == "https" else 80)
    except ValueError:
        return "new"
    same_host = (configured.hostname or "").lower() == host.lower()
    return "monitored" if same_host and configured_port == port else "new"


async def scan_node(uid: int, node: NodeConfig, all_ports: bool = False) -> NodeScan:
    """List one node's listeners and classify each reachable one."""
    scan = NodeScan(uid=uid, label=node.label, worker=node.worker)
    try:
        output = await collector._ssh_run(node.ssh_target, LISTEN_CMD)
    except Exception as exc:
        scan.error = str(exc).strip() or type(exc).__name__
        return scan
    listeners, port_range = parse_listen_output(output)
    probe_set = candidates(listeners, port_range, all_ports)
    scan.local_only = [lst.port for lst in probe_set if lst.scope == "loopback"]
    remote = [lst for lst in probe_set if lst.scope != "loopback"]
    if not remote:
        return scan
    node_addr = await node_host(node)
    scheme = urlparse(node.vllm_url).scheme or "http"
    limit = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def one(client: httpx.AsyncClient, listener: Listener) -> Optional[Endpoint]:
        host = node_addr if listener.scope == "any" else listener.address
        url = _url(scheme, host, listener.port)
        async with limit:
            found = await probe(client, url)
        if found is None:
            return None
        engine, model = found
        return Endpoint(url, listener.port, engine, model, _status(node, host, listener.port))

    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
        results = await asyncio.gather(*(one(client, lst) for lst in remote))
    scan.endpoints = [ep for ep in results if ep is not None]
    return scan


def simulated_scan(uid: int, node: NodeConfig) -> NodeScan:
    """Made-up results for ``--simulate``: no SSH, no HTTP. Tagged against
    the node's current URL like a real scan, so a switched node reads back
    as monitored on the next sweep."""
    scan = NodeScan(uid=uid, label=node.label, worker=node.worker)
    host = urlparse(node.vllm_url).hostname or node.label
    served = {
        1: [(8000, "vllm", "Qwen3.6-27B"), (8001, "sglang", "Gemma-4-26B")],
        2: [(30000, "sglang", "DeepSeek-V4-Flash")],
    }.get(uid, [])
    scan.endpoints = [
        Endpoint(_url("http", host, port), port, engine, model, _status(node, host, port))
        for port, engine, model in served
    ]
    if uid == 1:
        scan.local_only = [8100]
    elif uid == 3:
        scan.error = f"ssh: connect to host {host} port 22: Connection timed out"
    return scan


async def sweep(nodes: Sequence[NodeConfig], all_ports: bool = False) -> AsyncIterator[NodeScan]:
    """Scan every node once, yielding each node's result as it finishes."""
    if config.SIMULATION_NODES:
        for uid, node in enumerate(nodes, 1):
            await asyncio.sleep(0.15)  # stagger so the modal visibly fills in
            yield simulated_scan(uid, node)
        return
    tasks = [
        asyncio.ensure_future(scan_node(uid, node, all_ports)) for uid, node in enumerate(nodes, 1)
    ]
    try:
        for done in asyncio.as_completed(tasks):
            yield await done
    finally:
        # A closed scan window must not leave node scans running behind it.
        for task in tasks:
            task.cancel()


# ─── reporting ───────────────────────────────────────────────────────


def nothing_found_message(scans: Sequence[NodeScan]) -> Optional[str]:
    """The empty-sweep line, or None when any endpoint was found. Nodes that
    could not be listed are not counted as scanned."""
    if any(scan.endpoints for scan in scans):
        return None
    scanned = sum(1 for scan in scans if scan.error is None)
    if not scanned:
        return "No nodes could be scanned."
    return f"No inference endpoints found on {scanned} node{'s' if scanned != 1 else ''}."


def format_table(scans: Sequence[NodeScan]) -> str:
    """Plain-text table for ``dgx-top discover``."""
    rows = [("NODE", "PORT", "ENGINE", "MODEL", "STATUS", "URL")]
    for scan in sorted(scans, key=lambda s: s.uid):
        if scan.error is not None:
            rows.append((scan.label, "-", "-", "-", "ssh-failed", scan.error))
            continue
        for ep in scan.endpoints:
            rows.append((scan.label, str(ep.port), ep.engine, ep.model or "—", ep.status, ep.url))
        for port in scan.local_only:
            rows.append((scan.label, str(port), "?", "—", "local-only", "loopback bind"))
        if not scan.endpoints and not scan.local_only:
            note = "worker" if scan.worker else ""
            rows.append((scan.label, "-", "-", "-", "none", note))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[-1]
        for row in rows
    ]
    message = nothing_found_message(scans)
    if message:
        lines += ["", message]
    return "\n".join(line.rstrip() for line in lines)


def to_json(scans: Sequence[NodeScan]) -> str:
    return json.dumps([asdict(scan) for scan in sorted(scans, key=lambda s: s.uid)], indent=2)
