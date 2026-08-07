from unittest.mock import AsyncMock, patch

import pytest

from config import NodeConfig, Settings
from preflight import CheckResult, run_preflight


@pytest.mark.asyncio
async def test_preflight_returns_nonzero_when_any_check_fails(capsys):
    settings = Settings((NodeConfig("node-a", "node-a", "http://node-a.example.com"),))
    results = (
        CheckResult("node-a", "ssh", True, "ready"),
        CheckResult("node-a", "vllm", False, "unreachable"),
    )
    with patch("preflight.check_node", AsyncMock(return_value=results)):
        status = await run_preflight(settings)

    assert status == 1
    output = capsys.readouterr().out
    assert "[PASS] node-a ssh" in output
    assert "[FAIL] node-a vllm" in output


@pytest.mark.asyncio
async def test_preflight_returns_zero_when_all_checks_pass(capsys):
    settings = Settings((NodeConfig("node-a", "node-a", "http://node-a.example.com"),))
    results = (
        CheckResult("node-a", "ssh", True, "ready"),
        CheckResult("node-a", "vllm", True, "ready"),
    )
    with patch("preflight.check_node", AsyncMock(return_value=results)):
        status = await run_preflight(settings)

    assert status == 0
    assert "Preflight passed" in capsys.readouterr().out
