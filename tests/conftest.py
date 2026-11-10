from unittest.mock import AsyncMock

import pytest

import collector


@pytest.fixture(autouse=True)
def _no_stack_log_ssh(monkeypatch):
    """``poll_unit`` reads the engine's startup log over SSH; a test that polls
    a unit must never open a real connection for it. Tests of the fetch itself
    use ``test_collector.REAL_FETCH_STACK_LOG``."""
    monkeypatch.setattr(collector, "_fetch_stack_log", AsyncMock(return_value=(0, 0, 0)))


@pytest.fixture(autouse=True)
def _isolate_poll_state(monkeypatch):
    """``poll_cluster`` keeps each unit's poll and last snapshot at module level,
    and the engine-log read keeps one in-flight read per node: none of it may
    leak from one test into the next."""
    monkeypatch.setattr(collector, "_flights", {})
    monkeypatch.setattr(collector, "_landed", {})
    monkeypatch.setattr(collector, "_stack_log_inflight", {})
