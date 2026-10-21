"""Behavioral tests for the one-shot endpoint sweep (src/discovery.py)."""

from __future__ import annotations

import pytest

import collector
import config
import discovery
from config import NodeConfig
from discovery import (
    PROC_MARK,
    RANGE_MARK,
    SS_MARK,
    Endpoint,
    NodeScan,
    candidates,
    parse_listen_output,
)

# ─── fixtures ────────────────────────────────────────────────────────

SS_OUTPUT = f"""{RANGE_MARK}
32768	60999
{SS_MARK}
LISTEN 0      4096         0.0.0.0:8000       0.0.0.0:*
LISTEN 0      4096            [::]:8000          [::]:*
LISTEN 0      128          0.0.0.0:22         0.0.0.0:*
LISTEN 0      4096       127.0.0.1:8001       0.0.0.0:*
LISTEN 0      4096           [::1]:8002          [::]:*
LISTEN 0      4096 [::ffff:127.0.0.1]:8003          *:*
LISTEN 0      4096   127.0.0.53%lo:53         0.0.0.0:*
LISTEN 0      4096   198.51.100.21:30000      0.0.0.0:*
LISTEN 0      4096       127.0.0.1:30000      0.0.0.0:*
LISTEN 0      4096               *:9000             *:*
LISTEN 0      4096 [fe80::1]%eth0:8006          [::]:*
LISTEN 0      4096         0.0.0.0:41234      0.0.0.0:*
garbage line that is not a socket
"""

# The same sockets as /proc/net/tcp + tcp6 report them (ss missing).
PROC_OUTPUT = f"""{RANGE_MARK}
32768 60999
{PROC_MARK}
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 1
   1: 0100007F:1F41 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 2
   2: 0100007F:1F41 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000        0 3
   3: 156433C6:7530 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4
  sl  local_address                         remote_address                        st tx_queue
   0: 00000000000000000000000000000000:1F40 00000000000000000000000000000000:0000 0A 0
   1: 00000000000000000000000001000000:1F42 00000000000000000000000000000000:0000 0A 0
"""


def _by_port(listeners):
    return {lst.port: (lst.address, lst.scope) for lst in listeners}


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.text = payload if isinstance(payload, str) else ""

    def raise_for_status(self):
        if isinstance(self._payload, Exception):
            raise self._payload

    def json(self):
        if isinstance(self._payload, (dict, list)):
            return self._payload
        raise ValueError("not JSON")


def _patch_http(monkeypatch, routes: dict[str, object]) -> list[str]:
    """Serve canned bodies keyed by full URL to both discovery's and the
    collector's ``httpx.AsyncClient``; any other URL is a connection error,
    as a dead port is. Returns the list of requested URLs."""
    requested: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            requested.append(url)
            if url in routes:
                return _Resp(routes[url])
            raise ConnectionError(f"connection refused: {url}")

    monkeypatch.setattr(discovery.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(collector.httpx, "AsyncClient", FakeClient)
    return requested


def _patch_ssh(monkeypatch, output: str | Exception) -> list[str]:
    targets: list[str] = []

    async def fake_ssh(target: str, cmd: str) -> str:
        targets.append(target)
        if isinstance(output, Exception):
            raise output
        return output

    monkeypatch.setattr(collector, "_ssh_run", fake_ssh)
    return targets


VLLM_METRICS = (
    "# HELP vllm:num_requests_running running\n"
    'vllm:num_requests_running{model_name="m"} 1.0\n'
    'vllm:generation_tokens_total{model_name="m"} 42.0\n'
)
RAY_METRICS = (
    "# HELP ray_tasks Current number of tasks\n"
    'ray_tasks{State="RUNNING"} 3.0\n'
    'ray_node_cpu_utilization{ip="198.51.100.7"} 12.0\n'
)
SGLANG_LOADS = {"loads": [{"num_running_reqs": 2, "num_waiting_reqs": 1}]}


def _models(name: str) -> dict:
    return {"data": [{"id": name}]}


NODE = NodeConfig("spark-1", "spark@spark01", "http://spark01:8000")


# ─── AC1–AC3: listener parsing and filtering ─────────────────────────


def test_ss_listeners_collapse_per_port_and_classify_scope():
    listeners, port_range = parse_listen_output(SS_OUTPUT)
    ports = _by_port(listeners)

    assert port_range == (32768, 60999)
    # v4 + v6 wildcard binds of one port are one listener.
    assert [lst.port for lst in listeners].count(8000) == 1
    assert ports[8000][1] == "any"
    assert ports[9000][1] == "any"  # `*:port`
    # Every loopback spelling is loopback, including v4-mapped and a %zone.
    assert ports[8001][1] == ports[8002][1] == ports[8003][1] == ports[53][1] == "loopback"
    # A specific bind beats loopback on the same port and keeps its address.
    assert ports[30000] == ("198.51.100.21", "host")
    # ss prints a device zone outside the brackets.
    assert ports[8006] == ("fe80::1", "host")


def test_proc_fallback_parses_only_listen_rows_into_the_same_listeners():
    listeners, port_range = parse_listen_output(PROC_OUTPUT)
    ports = _by_port(listeners)

    assert port_range == (32768, 60999)
    assert ports == {
        8000: ("0.0.0.0", "any"),  # v4 + v6 wildcard collapse
        8001: ("127.0.0.1", "loopback"),  # the ESTABLISHED row is ignored
        8002: ("::1", "loopback"),
        30000: ("198.51.100.21", "host"),
    }


def test_filter_drops_ssh_and_ephemeral_unless_all_ports():
    listeners, port_range = parse_listen_output(SS_OUTPUT)

    default = {lst.port for lst in candidates(listeners, port_range)}
    assert 22 not in default
    assert 41234 not in default  # inside 32768-60999
    assert {8000, 9000, 30000} <= default

    everything = {lst.port for lst in candidates(listeners, port_range, all_ports=True)}
    assert 41234 in everything
    assert 22 not in everything


def test_unknown_ephemeral_range_filters_nothing_but_ssh():
    listeners, port_range = parse_listen_output(SS_OUTPUT.replace("32768\t60999", "garbage"))

    assert port_range is None
    kept = {lst.port for lst in candidates(listeners, port_range)}
    assert 41234 in kept and 22 not in kept


def test_ephemeral_range_is_read_past_unparseable_lines():
    """A junk line before the range (kernel noise, a wrapper banner) must not
    hide a valid range that follows it."""
    listeners, port_range = parse_listen_output(
        SS_OUTPUT.replace("32768\t60999", "cannot read range\n32768 60999")
    )

    assert port_range == (32768, 60999)
    assert 41234 not in {lst.port for lst in candidates(listeners, port_range)}


# ─── AC4: classification ─────────────────────────────────────────────


async def test_probe_reports_engine_from_metrics_with_model_name(monkeypatch):
    base = "http://spark01:8000"
    _patch_http(
        monkeypatch,
        {
            f"{base}/metrics": VLLM_METRICS,
            f"{base}/v1/models": _models("/models/models--Qwen--Qwen3.6-27B/snapshots/abc"),
        },
    )
    async with discovery.httpx.AsyncClient() as client:
        assert await discovery.probe(client, base) == ("vllm", "Qwen3.6-27B")


async def test_probe_rejects_a_non_engine_metrics_registry(monkeypatch):
    base = "http://spark01:8265"
    _patch_http(monkeypatch, {f"{base}/metrics": RAY_METRICS})
    async with discovery.httpx.AsyncClient() as client:
        assert await discovery.probe(client, base) is None


async def test_probe_finds_sglang_without_metrics_through_its_load_api(monkeypatch):
    base = "http://spark01:30000"
    _patch_http(
        monkeypatch,
        {
            f"{base}/metrics": RuntimeError("404 Not Found"),
            f"{base}/v1/loads": SGLANG_LOADS,
        },
    )
    async with discovery.httpx.AsyncClient() as client:
        # No /v1/models route: the name is unknown, not guessed.
        assert await discovery.probe(client, base) == ("sglang", "")


# ─── AC5/AC6: tags, hosts, failures ──────────────────────────────────


async def test_scan_tags_monitored_and_new_and_never_probes_loopback(monkeypatch):
    _patch_ssh(monkeypatch, SS_OUTPUT)
    requested = _patch_http(
        monkeypatch,
        {
            "http://spark01:8000/metrics": VLLM_METRICS,
            "http://spark01:8000/v1/models": _models("Qwen/Qwen3.6-27B"),
            "http://spark01:9000/metrics": "sglang:num_running_reqs 0.0\n",
            "http://198.51.100.21:30000/v1/loads": SGLANG_LOADS,
        },
    )

    scan = await discovery.scan_node(1, NODE)

    assert scan.error is None
    by_port = {ep.port: ep for ep in scan.endpoints}
    assert set(by_port) == {8000, 9000, 30000}
    assert by_port[8000] == Endpoint(
        "http://spark01:8000", 8000, "vllm", "Qwen3.6-27B", "monitored"
    )
    assert by_port[9000].status == "new" and by_port[9000].engine == "sglang"
    # A specific bind is probed at its own address.
    assert by_port[30000].url == "http://198.51.100.21:30000"
    assert by_port[30000].status == "new"
    assert sorted(scan.local_only) == [53, 8001, 8002, 8003]
    assert not [url for url in requested if ":8001/" in url or ":8002/" in url]
    assert not [url for url in requested if ":22/" in url or ":41234/" in url]


async def test_ssh_failure_is_its_own_result(monkeypatch):
    _patch_ssh(monkeypatch, RuntimeError("ssh: connect to host spark01 port 22: timed out"))
    requested = _patch_http(monkeypatch, {})

    scan = await discovery.scan_node(1, NODE)

    assert scan.error == "ssh: connect to host spark01 port 22: timed out"
    assert scan.endpoints == [] and requested == []


async def test_local_configured_url_resolves_the_node_through_ssh_config(monkeypatch):
    node = NodeConfig("spark-1", "spark01-alias", "http://localhost:18000")
    _patch_ssh(monkeypatch, f"{SS_MARK}\nLISTEN 0 4096 [::]:8000 [::]:*\n")

    async def fake_hostname(target: str):
        assert target == "spark01-alias"
        return "spark01.lan"

    monkeypatch.setattr(discovery, "_ssh_hostname", fake_hostname)
    requested = _patch_http(monkeypatch, {"http://spark01.lan:8000/metrics": VLLM_METRICS})

    scan = await discovery.scan_node(1, node)

    assert [ep.url for ep in scan.endpoints] == ["http://spark01.lan:8000"]
    assert scan.endpoints[0].status == "new"  # the tunnel URL is what is monitored
    assert not [url for url in requested if "localhost" in url]


async def test_ipv6_specific_bind_is_bracketed(monkeypatch):
    _patch_ssh(monkeypatch, f"{SS_MARK}\nLISTEN 0 4096 [fd00::5]:8000 [::]:*\n")
    _patch_http(monkeypatch, {"http://[fd00::5]:8000/metrics": VLLM_METRICS})

    scan = await discovery.scan_node(1, NODE)

    assert [ep.url for ep in scan.endpoints] == ["http://[fd00::5]:8000"]


# ─── AC12: simulation ────────────────────────────────────────────────


async def test_simulated_sweep_makes_no_ssh_or_http_call(monkeypatch):
    monkeypatch.setattr(config, "SIMULATION_NODES", 4)
    ssh_calls = _patch_ssh(monkeypatch, RuntimeError("must not ssh"))
    requested = _patch_http(monkeypatch, {})

    scans = [scan async for scan in discovery.sweep(config._synthetic_nodes(4))]

    assert ssh_calls == [] and requested == []
    assert [scan.uid for scan in scans] == [1, 2, 3, 4]
    assert scans[0].endpoints[0].status == "monitored"
    assert scans[2].error is not None


# ─── reporting ───────────────────────────────────────────────────────


def test_nothing_found_counts_only_nodes_that_were_scanned():
    empty = NodeScan(1, "a", False)
    worker = NodeScan(2, "b", True, local_only=[8100])
    failed = NodeScan(3, "c", False, error="ssh: timed out")

    assert (
        discovery.nothing_found_message([empty, worker, failed])
        == "No inference endpoints found on 2 nodes."
    )
    assert discovery.nothing_found_message([empty]) == "No inference endpoints found on 1 node."
    assert discovery.nothing_found_message([failed]) == "No nodes could be scanned."
    found = NodeScan(4, "d", False, endpoints=[Endpoint("http://d:1", 1, "vllm", "", "new")])
    assert discovery.nothing_found_message([empty, found]) is None


@pytest.mark.parametrize("scans", [[], [NodeScan(1, "a", False)]])
def test_table_states_the_empty_sweep(scans):
    table = discovery.format_table(scans)
    assert table.splitlines()[0].split() == ["NODE", "PORT", "ENGINE", "MODEL", "STATUS", "URL"]
    assert table.splitlines()[-1].startswith("No ")
