from unittest.mock import AsyncMock

import pytest

import collector
import config
import switches


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
    monkeypatch.setattr(collector, "_switch_flights", {})
    monkeypatch.setattr(collector, "_switch_landed", {})
    monkeypatch.setattr(switches, "_baselines", {})
    monkeypatch.setattr(switches, "_seen_running", {})
    monkeypatch.setattr(config, "SWITCHES", ())
    # _SETTINGS now carries the same switches tuple: reset it too, so a test
    # that configures cannot leak its switches into a later test that reads
    # get_settings() without re-configuring.
    monkeypatch.setattr(config, "_SETTINGS", None)
    monkeypatch.setattr(config, "_CONFIG_PATH", None)
