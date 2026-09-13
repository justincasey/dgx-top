from unittest.mock import AsyncMock, patch

import pytest

from collector import EngineLoad
from config import NodeConfig, Settings
from preflight import CheckResult, _check_engine, run_preflight


@pytest.mark.asyncio
async def test_preflight_returns_nonzero_when_any_check_fails(capsys):
    settings = Settings((NodeConfig("node-a", "node-a", "http://node-a.example.com"),))
    results = (
        CheckResult("node-a", "ssh", True, "ready"),
        CheckResult("node-a", "engine", False, "unreachable"),
    )
    with patch("preflight.check_node", AsyncMock(return_value=results)):
        status = await run_preflight(settings)

    assert status == 1
    output = capsys.readouterr().out
    assert "[PASS] node-a ssh" in output
    assert "[FAIL] node-a engine" in output


@pytest.mark.asyncio
async def test_preflight_returns_zero_when_all_checks_pass(capsys):
    settings = Settings((NodeConfig("node-a", "node-a", "http://node-a.example.com"),))
    results = (
        CheckResult("node-a", "ssh", True, "ready"),
        CheckResult("node-a", "engine", True, "ready"),
    )
    with patch("preflight.check_node", AsyncMock(return_value=results)):
        status = await run_preflight(settings)

    assert status == 0
    assert "Preflight passed" in capsys.readouterr().out


class _Resp:
    def __init__(self, text: str = "", status_error: Exception | None = None):
        self.text = text
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error


class _Client:
    """Fake httpx client returning ``response`` for /metrics."""

    def __init__(self, response: _Resp):
        self._response = response

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        return self._response


SGLANG_METRICS = (
    "# HELP sglang:num_running_reqs running\n"
    'sglang:num_running_reqs{model_name="qwen",tp_rank="0",pp_rank="0"} 2.0\n'
)
VLLM_METRICS = 'vllm:num_requests_running{model_name="qwen"} 2.0\n'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, engine",
    [(SGLANG_METRICS, "sglang"), (VLLM_METRICS, "vllm")],
)
async def test_engine_check_accepts_either_namespace(text, engine):
    # The old check required the literal "vllm:", so every SGLang node failed.
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com")
    with patch("preflight.httpx.AsyncClient", _Client(_Resp(text))):
        result = await _check_engine(node)

    assert result.ok
    assert result.check == "engine"
    assert f"{engine} metrics endpoint ready" in result.detail


@pytest.mark.asyncio
async def test_engine_check_accepts_version_bumper_metrics_and_load_api():
    # A server without --enable-metrics serves no /metrics at all but still
    # answers SGLang's load API — healthy, not a hard failure.
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com")
    load = AsyncMock(return_value=EngineLoad(running=2, waiting=1))
    with patch("preflight.httpx.AsyncClient", _Client(_Resp(status_error=RuntimeError("404")))):
        with patch("preflight.fetch_engine_load", load):
            result = await _check_engine(node)

    assert result.ok
    assert "load endpoint ready" in result.detail
    load.assert_awaited_once_with("http://node-a.example.com")


@pytest.mark.asyncio
async def test_engine_check_fails_when_neither_surface_answers():
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com")
    with patch("preflight.httpx.AsyncClient", _Client(_Resp(status_error=RuntimeError("404")))):
        with patch("preflight.fetch_engine_load", AsyncMock(return_value=None)):
            result = await _check_engine(node)

    assert not result.ok
    assert "no metrics and no load endpoint" in result.detail


@pytest.mark.asyncio
async def test_engine_check_skips_the_load_api_for_a_declared_vllm_node():
    # A node declared engine = "vllm" is never probed for SGLang's routes
    # (matching the collector): a dead /metrics is a real failure there, not
    # a hint to look for a load endpoint.
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com", engine="vllm")
    load = AsyncMock(return_value=EngineLoad(running=2, waiting=1))
    with patch("preflight.httpx.AsyncClient", _Client(_Resp(status_error=RuntimeError("404")))):
        with patch("preflight.fetch_engine_load", load):
            result = await _check_engine(node)

    assert not result.ok
    assert "no metrics endpoint" in result.detail
    load.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_check_rejects_an_endpoint_with_only_help_lines():
    # Comment lines are not sample evidence: a 200 with no series is not a
    # metrics endpoint — but it IS recoverable through the load API, so the
    # failure must come after that probe has been tried.
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com")
    load = AsyncMock(return_value=None)
    with patch("preflight.httpx.AsyncClient", _Client(_Resp("# HELP vllm:x help\n"))):
        with patch("preflight.fetch_engine_load", load):
            result = await _check_engine(node)

    assert not result.ok
    assert result.detail == "endpoint returned no engine metrics"
    load.assert_awaited_once_with("http://node-a.example.com")


@pytest.mark.asyncio
async def test_engine_check_uses_the_load_api_when_metrics_carry_no_engine_samples():
    # A 2xx whose body holds no vllm:/sglang: sample (a redirect the client did
    # not follow, an empty registry, a foreign service) is the same situation
    # as a dead /metrics: the collector monitors that node through the load
    # API, so the check must not fail it.
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com")
    load = AsyncMock(return_value=EngineLoad(running=3, waiting=0))
    with patch("preflight.httpx.AsyncClient", _Client(_Resp('{"status": "ok"}\n'))):
        with patch("preflight.fetch_engine_load", load):
            result = await _check_engine(node)

    assert result.ok
    assert result.detail == "load endpoint ready (no engine metrics)"
    load.assert_awaited_once_with("http://node-a.example.com")


@pytest.mark.asyncio
async def test_engine_check_skips_the_load_api_for_declared_vllm_without_samples():
    # The declared-vLLM rule is branch-independent: no load probe, and no
    # success, for a node whose own metrics carry no engine samples.
    node = NodeConfig("node-a", "node-a", "http://node-a.example.com", engine="vllm")
    load = AsyncMock(return_value=EngineLoad(running=3, waiting=0))
    with patch("preflight.httpx.AsyncClient", _Client(_Resp('{"status": "ok"}\n'))):
        with patch("preflight.fetch_engine_load", load):
            result = await _check_engine(node)

    assert not result.ok
    assert result.detail == "endpoint returned no engine metrics"
    load.assert_not_awaited()
