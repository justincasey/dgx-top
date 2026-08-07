"""Connectivity and prerequisite checks for configured DGX Spark nodes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

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


async def _check_vllm(node: NodeConfig) -> CheckResult:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{node.vllm_url}/metrics")
            response.raise_for_status()
    except Exception as exc:
        return CheckResult(node.label, "vllm", False, str(exc))
    if "vllm:" not in response.text:
        return CheckResult(node.label, "vllm", False, "endpoint returned no vLLM metrics")
    return CheckResult(node.label, "vllm", True, "metrics endpoint ready")


async def check_node(node: NodeConfig) -> tuple[CheckResult, CheckResult]:
    ssh_result, vllm_result = await asyncio.gather(_check_ssh(node), _check_vllm(node))
    return ssh_result, vllm_result


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
