"""Connectivity and prerequisite checks for configured DGX Spark nodes and
network switches."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from collector import LOAD_API_ENGINES, _read_capped, fetch_engine_load, metrics_engine
from config import NodeConfig, Settings, SwitchConfig
from switches import HTTP_TIMEOUT, _request_capped, failure_reason, tls_verify


@dataclass(frozen=True)
class CheckResult:
    node: str
    check: str
    ok: bool
    detail: str
    warning: bool = False  # passes, but deserves a look (printed WARN)


async def _check_ssh(node: NodeConfig) -> CheckResult:
    command = "command -v nvidia-smi >/dev/null && test -r /proc/stat && echo ready"
    try:
        process = await asyncio.create_subprocess_exec(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            node.ssh_target,
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    except FileNotFoundError:
        return CheckResult(node.label, "ssh", False, "ssh executable not found")
    except TimeoutError:
        return CheckResult(node.label, "ssh", False, "connection timed out")
    if process.returncode == 0 and stdout.decode().strip() == "ready":
        return CheckResult(node.label, "ssh", True, "passwordless SSH and telemetry ready")
    detail = stderr.decode().strip() or "nvidia-smi or /proc/stat is unavailable"
    return CheckResult(node.label, "ssh", False, detail.splitlines()[-1])


async def _load_probe_detail(node: NodeConfig, reason: str) -> str | None:
    """Detail when the node's load API answers, else None.

    The single place the declared-engine rule lives: a node declared as an
    engine without a load API (vLLM, TensorFold) is never probed for
    SGLang's routes, in any branch.
    """
    if node.engine is not None and node.engine not in LOAD_API_ENGINES:
        return None
    load = await fetch_engine_load(node.vllm_url)
    return f"load endpoint ready ({reason})" if load is not None else None


async def _check_engine(node: NodeConfig) -> CheckResult:
    """Validate a node's model-serving endpoint, whichever engine it runs.

    /metrics is probed first and accepted when any namespace carries sample
    evidence (vLLM, SGLang and TensorFold do not expose the same series, so
    a ``vllm:``-only test rejected every SGLang node). A server without
    ``--enable-metrics`` serves no /metrics at all but still answers SGLang's
    load API, which is the fallback probe — a metrics-disabled SGLang is
    healthy, not a hard failure. That fallback runs wherever /metrics yields
    no engine evidence, a 2xx with none included (a redirect the client did
    not follow, an empty registry, a foreign service): the collector monitors
    such a node through the same route, so the check must not fail a node the
    dashboard reports on. A node declared as an engine without a load API
    (``engine = "vllm"``, ``engine = "tensorfold"``) skips the fallback,
    exactly as the collector's poll does.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            text = (await _read_capped(client, f"{node.vllm_url}/metrics")).decode(
                "utf-8", errors="replace"
            )
    except Exception as exc:
        detail = await _load_probe_detail(node, "metrics disabled")
        if detail is not None:
            return CheckResult(node.label, "engine", True, detail)
        if node.engine is None or node.engine in LOAD_API_ENGINES:
            return CheckResult(
                node.label, "engine", False, f"no metrics and no load endpoint: {exc}"
            )
        return CheckResult(node.label, "engine", False, f"no metrics endpoint: {exc}")
    engine = metrics_engine(text)
    if engine is None:
        detail = await _load_probe_detail(node, "no engine metrics")
        if detail is not None:
            return CheckResult(node.label, "engine", True, detail)
        return CheckResult(node.label, "engine", False, "endpoint returned no engine metrics")
    return CheckResult(node.label, "engine", True, f"{engine} metrics endpoint ready")


async def check_node(node: NodeConfig) -> tuple[CheckResult, CheckResult]:
    ssh_result, engine_result = await asyncio.gather(_check_ssh(node), _check_engine(node))
    return ssh_result, engine_result


async def check_switch(switch: SwitchConfig) -> list[CheckResult]:
    """Credentials present and ``/rest/system/identity`` answering with a name.

    A plain-http url still passes, with a warning: Basic auth sends the
    password across the network in cleartext."""
    results: list[CheckResult] = []
    if urlparse(switch.url).scheme == "http":
        results.append(
            CheckResult(
                switch.label,
                "transport",
                True,
                "http sends the switch password in cleartext; prefer https (www-ssl)",
                warning=True,
            )
        )
    password = os.environ.get(switch.password_env) or None
    if password is None:
        results.append(
            CheckResult(switch.label, "rest", False, f"${switch.password_env} is not set")
        )
        return results
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, auth=(switch.user, password), verify=tls_verify(switch)
        ) as client:
            identity = json.loads(
                await _request_capped(client, "GET", f"{switch.url}/rest/system/identity")
            )
    except Exception as exc:  # reported as a failed check, never a crash
        results.append(CheckResult(switch.label, "rest", False, failure_reason(exc)))
        return results
    name = identity.get("name") if isinstance(identity, dict) else None
    if not isinstance(name, str) or not name:
        results.append(CheckResult(switch.label, "rest", False, "identity response has no name"))
    else:
        results.append(CheckResult(switch.label, "rest", True, "RouterOS REST API ready"))
    return results


async def run_preflight(settings: Settings) -> int:
    grouped, switch_groups = await asyncio.gather(
        asyncio.gather(*(check_node(node) for node in settings.nodes)),
        asyncio.gather(*(check_switch(switch) for switch in settings.switches)),
    )
    results = [result for pair in grouped for result in pair]
    results += [result for group in switch_groups for result in group]
    for result in results:
        marker = "WARN" if result.warning else "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {result.node} {result.check}: {result.detail}")
    failures = sum(not result.ok for result in results)
    if failures:
        print(f"\nPreflight failed: {failures} check(s) need attention.")
        return 1
    print("\nPreflight passed. Run `dgx-top` to start monitoring.")
    return 0
