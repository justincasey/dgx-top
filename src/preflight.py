"""Connectivity and prerequisite checks for configured DGX Spark nodes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from collector import fetch_engine_load, metrics_engine
from config import NodeConfig, Settings


@dataclass(frozen=True)
class CheckResult:
    node: str
    check: str
    ok: bool
    detail: str


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

    The single place the declared-vLLM rule lives: such a node is never
    probed for SGLang's routes, in any branch.
    """
    if node.engine == "vllm":
        return None
    load = await fetch_engine_load(node.vllm_url)
    return f"load endpoint ready ({reason})" if load is not None else None


async def _check_engine(node: NodeConfig) -> CheckResult:
    """Validate a node's model-serving endpoint, whichever engine it runs.

    /metrics is probed first and accepted when EITHER namespace carries
    sample evidence (vLLM and SGLang do not expose the same series, so a
    ``vllm:``-only test rejected every SGLang node). A server without
    ``--enable-metrics`` serves no /metrics at all but still answers SGLang's
    load API, which is the fallback probe — a metrics-disabled SGLang is
    healthy, not a hard failure. That fallback runs wherever /metrics yields
    no engine evidence, a 2xx with none included (a redirect the client did
    not follow, an empty registry, a foreign service): the collector monitors
    such a node through the same route, so the check must not fail a node the
    dashboard reports on. A node declared ``engine = "vllm"`` skips the
    fallback, exactly as the collector's poll does.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{node.vllm_url}/metrics")
            response.raise_for_status()
    except Exception as exc:
        detail = await _load_probe_detail(node, "metrics disabled")
        if detail is not None:
            return CheckResult(node.label, "engine", True, detail)
        if node.engine == "vllm":
            return CheckResult(node.label, "engine", False, f"no metrics endpoint: {exc}")
        return CheckResult(node.label, "engine", False, f"no metrics and no load endpoint: {exc}")
    engine = metrics_engine(response.text)
    if engine is None:
        detail = await _load_probe_detail(node, "no engine metrics")
        if detail is not None:
            return CheckResult(node.label, "engine", True, detail)
        return CheckResult(node.label, "engine", False, "endpoint returned no engine metrics")
    return CheckResult(node.label, "engine", True, f"{engine} metrics endpoint ready")


async def check_node(node: NodeConfig) -> tuple[CheckResult, CheckResult]:
    ssh_result, engine_result = await asyncio.gather(_check_ssh(node), _check_engine(node))
    return ssh_result, engine_result


async def run_preflight(settings: Settings) -> int:
    grouped = await asyncio.gather(*(check_node(node) for node in settings.nodes))
    results = [result for pair in grouped for result in pair]
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {result.node} {result.check}: {result.detail}")
    failures = sum(not result.ok for result in results)
    if failures:
        print(f"\nPreflight failed: {failures} check(s) need attention.")
        return 1
    print("\nPreflight passed. Run `dgx-top` to start monitoring.")
    return 0
