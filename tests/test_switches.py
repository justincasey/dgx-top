"""RouterOS switch polling: parsing, port selection, rates, findings, and the
end-to-end poll against a mocked REST API. Synthetic data only."""

import asyncio
import base64
import json
import ssl
import types

import httpx
import pytest

import collector
import config
import switches
from config import SwitchConfig
from stats import SwitchPort, SwitchPsu, SwitchStats

RESOURCE = {
    "cpu-load": "7",
    "cpu-frequency": "2000",
    "cpu-count": "4",
    "free-memory": "3000000000",
    "total-memory": "4000000000",
    "uptime": "1w2d3h",
    "version": "7.99.1 (stable)",
    "board-name": "TEST-BOARD",
}
HEALTH = [
    {"name": "cpu-temperature", "value": "47", "type": "C"},
    {"name": "switch-temperature", "value": "62", "type": "C"},
    {"name": "board-temperature1", "value": "40", "type": "C"},
    {"name": "psu1-temperature", "value": "35", "type": "C"},
    {"name": "fan-state", "value": "ok", "type": ""},
    {"name": "fan1-speed", "value": "4800", "type": "RPM"},
    {"name": "psu2-fan-speed", "value": "4900", "type": "RPM"},
    {"name": "psu1-state", "value": "no-input", "type": ""},
    {"name": "psu2-state", "value": "ok", "type": ""},
    {"name": "psu2-power", "value": "38.0", "type": "W"},
    {"name": "psu2-input-voltage", "value": "230", "type": "V"},
]


def _iface(name, running=True, rx=0, tx=0, drops=0, disabled=False, kind="ether"):
    return {
        "name": name,
        "type": kind,
        "running": "true" if running else "false",
        "disabled": "true" if disabled else "false",
        "rx-byte": str(rx),
        "tx-byte": str(tx),
        "rx-packet": str(rx // 100),
        "tx-packet": str(tx // 100),
        "tx-queue-drop": str(drops),
    }


INTERFACES = [
    _iface("qsfp56-dd-2-1"),
    _iface("qsfp56-dd-1-1"),
    _iface("ether1", running=False),
    _iface("bridge", kind="bridge"),
]

CFG = SwitchConfig("fabric", "https://192.0.2.10", "monitor", password_env="TEST_SWITCH_PW")


# ─── Parsing ──────────────────────────────────────────────────────────


def test_parse_resource_reads_every_field():
    s = SwitchStats("fabric")
    switches.parse_resource(RESOURCE, s)
    assert (s.board, s.version, s.uptime) == ("TEST-BOARD", "7.99.1", "1w2d3h")
    assert (s.cpu_pct, s.cpu_mhz, s.cpu_count) == (7.0, 2000.0, 4)
    assert (s.mem_used_bytes, s.mem_total_bytes) == (1_000_000_000, 4_000_000_000)


def test_parse_resource_keeps_sentinels_for_missing_or_garbage_fields():
    s = SwitchStats("fabric")
    switches.parse_resource(
        {"cpu-load": "lots", "free-memory": "9", "total-memory": "x", "board-name": 5}, s
    )
    assert (s.cpu_pct, s.cpu_mhz, s.cpu_count) == (-1.0, 0.0, 0)
    assert (s.mem_used_bytes, s.mem_total_bytes) == (0, 0)
    assert (s.board, s.version, s.uptime) == ("", "", "")
    switches.parse_resource(["not", "a", "dict"], s)  # tolerated
    assert s.cpu_pct == -1.0


def test_parse_health_splits_temps_fans_and_psus():
    s = SwitchStats("fabric")
    switches.parse_health(HEALTH, s)
    assert s.temps == {"cpu": 47.0, "switch": 62.0, "board1": 40.0, "psu1": 35.0}
    assert list(s.temps) == ["cpu", "switch", "board1", "psu1"]  # report order
    assert s.fans == {"fan1": 4800.0, "psu2-fan": 4900.0}
    assert s.fan_state == "ok"
    assert s.psus == [SwitchPsu("psu1", "no-input"), SwitchPsu("psu2", "ok", 38.0, 230.0)]


def test_single_sensor_model_bare_temperature_is_read_and_alarms():
    s = SwitchStats("fabric")
    switches.parse_health([{"name": "temperature", "value": "93"}], s)
    assert s.temps == {"temp": 93.0}
    assert switches.findings(s) == [("crit", "temp 93°")]


def test_parse_health_skips_garbage_rows_without_dropping_the_rest():
    s = SwitchStats("fabric")
    rows = [
        "junk",
        {"name": "cpu-temperature", "value": "n/a"},
        {"value": "12"},
        {"name": "fan1-speed", "value": None},
        {"name": "switch-temperature", "value": "70"},
    ]
    switches.parse_health(rows, s)
    assert s.temps == {"switch": 70.0}
    assert s.fans == {}
    switches.parse_health({"not": "a list"}, s)
    assert s.temps == {"switch": 70.0}


@pytest.mark.parametrize(
    ("rate", "bps"),
    [
        ("200Gbps", 200e9),
        ("1Gbps", 1e9),
        ("2.5Gbps", 2.5e9),
        ("100Mbps", 100e6),
        ("", 0.0),
        ("fast", 0.0),
        (None, 0.0),
    ],
)
def test_parse_rate(rate, bps):
    assert switches.parse_rate(rate) == bps


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("qsfp56-dd-1-1", "dd1"),
        ("qsfp56-dd-1-2", "dd1.2"),
        ("qsfp56-dd-12-4", "dd12.4"),
        ("qsfp28-3-1", "q3"),
        ("sfp28-2", "sfp2"),
        ("sfp-sfpplus2", "sfp2"),
        ("ether1", "eth1"),
        ("vlan-storage", "vlan-sto"),
    ],
)
def test_port_label(name, label):
    assert switches.port_label(name) == label
    assert len(label) <= config.SWITCH_PORT_LABEL_MAX


# ─── Port selection ───────────────────────────────────────────────────


def test_configured_ports_are_shown_exactly_in_config_order():
    cfg = SwitchConfig(
        "fabric",
        "https://192.0.2.10",
        "monitor",
        ports=(("qsfp56-dd-2-1", "spark-2"), ("ether1", "mgmt"), ("gone-9", "gone")),
    )
    ports = switches.select_ports(cfg, INTERFACES)
    assert [(p.name, p.label, p.running) for p in ports] == [
        ("qsfp56-dd-2-1", "spark-2", True),
        ("ether1", "mgmt", False),
        ("gone-9", "gone", False),  # absent on the switch reads as down
    ]


def test_default_selection_is_running_ethernet_in_natural_order():
    rows = [
        _iface("qsfp56-dd-10-1"),
        _iface("qsfp56-dd-2-1"),
        _iface("ether1", running=False),
        _iface("bridge", kind="bridge"),
        _iface("ether2", disabled=True),
    ]
    ports = switches.select_ports(CFG, rows)
    assert [(p.name, p.label) for p in ports] == [
        ("qsfp56-dd-2-1", "dd2"),
        ("qsfp56-dd-10-1", "dd10"),
    ]


def test_a_dropped_link_stays_listed_and_becomes_a_finding():
    switches.select_ports(CFG, [_iface("qsfp56-dd-1-1"), _iface("qsfp56-dd-2-1")])
    ports = switches.select_ports(CFG, [_iface("qsfp56-dd-1-1"), _iface("qsfp56-dd-2-1", False)])
    assert [(p.label, p.running) for p in ports] == [("dd1", True), ("dd2", False)]
    s = SwitchStats("fabric", ports=ports)
    assert switches.findings(s) == [("crit", "dd2 link down")]
    # Disabling the port on purpose retires it.
    ports = switches.select_ports(
        CFG, [_iface("qsfp56-dd-1-1"), _iface("qsfp56-dd-2-1", False, disabled=True)]
    )
    assert [p.label for p in ports] == ["dd1"]


# ─── Rates ────────────────────────────────────────────────────────────


def test_first_observation_seeds_then_deltas_rate():
    port = SwitchPort("p1", "p1", running=True)
    switches.apply_counters("fabric", port, _iface("p1", rx=1000, tx=2000, drops=5), 10.0)
    assert not port.measured and port.rx_bps == 0.0

    port = SwitchPort("p1", "p1", running=True)
    switches.apply_counters("fabric", port, _iface("p1", rx=11000, tx=4000, drops=8), 12.0)
    assert port.measured
    assert port.rx_bps == 10000 * 8 / 2
    assert port.tx_bps == 2000 * 8 / 2
    assert port.rx_pps == 100 / 2  # packets are bytes // 100 in _iface
    assert port.tx_pps == 20 / 2
    assert port.queue_drops == 3


def test_counter_reset_reseeds_without_a_rate():
    switches.apply_counters("fabric", SwitchPort("p1", "p1"), _iface("p1", rx=50000, tx=50000), 1.0)
    port = SwitchPort("p1", "p1")
    switches.apply_counters("fabric", port, _iface("p1", rx=100, tx=100), 2.0)  # reboot
    assert (port.measured, port.rx_bps, port.tx_bps) == (False, 0.0, 0.0)
    port = SwitchPort("p1", "p1")
    switches.apply_counters("fabric", port, _iface("p1", rx=900, tx=100), 4.0)
    assert port.measured and port.rx_bps == 800 * 8 / 2


def test_baselines_are_per_switch():
    switches.apply_counters("a", SwitchPort("p1", "p1"), _iface("p1", rx=0), 1.0)
    port = SwitchPort("p1", "p1")
    switches.apply_counters("b", port, _iface("p1", rx=5000), 2.0)
    assert not port.measured


# ─── Findings ─────────────────────────────────────────────────────────


def test_findings_put_critical_first_in_stable_order():
    s = SwitchStats(
        "fabric",
        temps={"cpu": 50.0, "switch": 85.4, "board1": 80.0},
        fan_state="fan1-fault",
        psus=[SwitchPsu("psu1", "no-input"), SwitchPsu("psu2", "ok")],
        ports=[
            SwitchPort("a", "dd1", running=True, queue_drops=4),
            SwitchPort("b", "dd2", running=False),
        ],
    )
    assert switches.findings(s) == [
        ("crit", "sw 85°"),
        ("crit", "brd1 80°"),
        ("crit", "dd2 link down"),
        ("warn", "psu1 no-input"),
        ("warn", "fan fan1-fault"),
        ("warn", "dd1 drops +4"),
    ]


def test_healthy_switch_has_no_findings():
    s = SwitchStats("fabric", temps={"cpu": 50.0}, fan_state="ok", psus=[SwitchPsu("psu1", "ok")])
    assert switches.findings(s) == []


@pytest.mark.parametrize("state", ["not-present", "idle", "standby"])
def test_empty_or_standby_psu_bay_is_not_a_finding(state):
    s = SwitchStats("fabric", psus=[SwitchPsu("psu1", state), SwitchPsu("psu2", "ok")])
    assert switches.findings(s) == []


@pytest.mark.parametrize("state", ["fail", "no-input"])
def test_failed_or_unpowered_psu_warns(state):
    s = SwitchStats("fabric", psus=[SwitchPsu("psu1", state)])
    assert switches.findings(s) == [("warn", f"psu1 {state}")]


# ─── poll_switch end to end ───────────────────────────────────────────


class FakeSwitch:
    """A mocked RouterOS REST API. Records each request."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.interfaces = INTERFACES
        self.status: dict[str, int] = {}
        self.bodies: dict[str, bytes] = {}
        self.client_kwargs: dict = {}
        self.raise_exc: Exception | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        path = request.url.path
        if path in self.status:
            return httpx.Response(self.status[path])
        if path in self.bodies:
            return httpx.Response(200, content=self.bodies[path])
        if path == "/rest/system/resource":
            return httpx.Response(200, json=RESOURCE)
        if path == "/rest/system/health":
            return httpx.Response(200, json=HEALTH)
        if path == "/rest/interface":
            return httpx.Response(200, json=self.interfaces)
        if path == "/rest/interface/ethernet/monitor" and request.method == "POST":
            names = json.loads(request.content)["numbers"].split(",")
            return httpx.Response(
                200, json=[{"name": n, "rate": "200Gbps", "status": "link-ok"} for n in names]
            )
        if path == "/rest/system/identity":
            return httpx.Response(200, json={"name": "test-switch"})
        return httpx.Response(404)


@pytest.fixture
def switch(monkeypatch):
    fake = FakeSwitch()
    real = httpx.AsyncClient

    def client(**kwargs):
        fake.client_kwargs = dict(kwargs)
        kwargs.pop("verify", None)
        return real(transport=httpx.MockTransport(fake.handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("TEST_SWITCH_PW", "synthetic-secret")
    return fake


async def test_poll_switch_reads_everything_with_basic_auth(switch):
    s = await switches.poll_switch(0, CFG)
    assert s.online and not s.error
    assert s.host == "192.0.2.10"
    assert (s.board, s.version) == ("TEST-BOARD", "7.99.1")
    assert s.temps["switch"] == 62.0
    assert [(p.label, p.rate, p.rate_bps) for p in s.ports] == [
        ("dd1", "200Gbps", 200e9),
        ("dd2", "200Gbps", 200e9),
    ]
    assert s.findings == [("warn", "psu1 no-input")]
    expected = "Basic " + base64.b64encode(b"monitor:synthetic-secret").decode()
    assert all(r.headers["authorization"] == expected for r in switch.requests)
    assert switch.client_kwargs["verify"] is True
    # Read-only: GETs, plus exactly one monitor POST asking for one sample.
    posts = [r for r in switch.requests if r.method != "GET"]
    assert [r.url.path for r in posts] == ["/rest/interface/ethernet/monitor"]
    assert json.loads(posts[0].content) == {
        "numbers": "qsfp56-dd-1-1,qsfp56-dd-2-1",
        "once": "",
        ".proplist": "name,rate,status",
    }


async def test_poll_switch_builds_one_ssl_context_per_ca_file(switch, monkeypatch):
    switches._ca_context.cache_clear()
    built = []

    def fake_context(*, cafile):
        built.append(cafile)
        return f"context:{cafile}"

    monkeypatch.setattr(switches.ssl, "create_default_context", fake_context)
    cfg = SwitchConfig(
        "fabric", "https://192.0.2.10", "monitor", "TEST_SWITCH_PW", ca_file="/ca.pem"
    )
    await switches.poll_switch(0, cfg)
    await switches.poll_switch(0, cfg)
    switches._ca_context.cache_clear()
    assert switch.client_kwargs["verify"] == "context:/ca.pem"  # never the bare path
    assert built == ["/ca.pem"]


async def test_untrusted_certificate_is_reported_as_such(switch):
    try:
        try:
            raise ssl.SSLCertVerificationError("certificate verify failed")
        except ssl.SSLError as cause:
            raise httpx.ConnectError("[SSL] certificate verify failed") from cause
    except httpx.ConnectError as exc:
        switch.raise_exc = exc
    s = await switches.poll_switch(0, CFG)
    assert (s.online, s.error) == (False, "TLS certificate not trusted (see ca_file)")


async def test_first_sample_after_an_outage_is_unmeasured(switch, monkeypatch):
    clock = iter([100.0, 110.0])
    monkeypatch.setattr(switches, "time", types.SimpleNamespace(monotonic=lambda: next(clock)))
    switch.interfaces = [_iface("qsfp56-dd-1-1", rx=0, tx=0, drops=0)]
    await switches.poll_switch(0, CFG)  # seeds
    switch.status["/rest/system/resource"] = 500
    assert not (await switches.poll_switch(0, CFG)).online  # the outage
    del switch.status["/rest/system/resource"]
    switch.interfaces = [_iface("qsfp56-dd-1-1", rx=900_000, tx=900_000, drops=50)]
    s = await switches.poll_switch(0, CFG)
    assert s.online and not s.ports[0].measured
    assert s.ports[0].queue_drops == 0
    assert not any("drops" in text for _, text in s.findings)


async def test_poll_switch_rates_on_the_second_poll(switch, monkeypatch):
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(switches, "time", types.SimpleNamespace(monotonic=lambda: next(clock)))
    switch.interfaces = [_iface("qsfp56-dd-1-1", rx=0, tx=0)]
    first = await switches.poll_switch(0, CFG)
    switch.interfaces = [_iface("qsfp56-dd-1-1", rx=25_000, tx=50_000)]
    second = await switches.poll_switch(0, CFG)
    assert not first.ports[0].measured
    assert second.ports[0].measured
    assert (second.ports[0].rx_bps, second.ports[0].tx_bps) == (100_000.0, 200_000.0)


async def test_missing_password_env_is_offline_without_a_request(switch, monkeypatch):
    monkeypatch.delenv("TEST_SWITCH_PW")
    s = await switches.poll_switch(0, CFG)
    assert (s.online, s.error) == (False, "$TEST_SWITCH_PW is not set")
    assert switch.requests == []


async def test_http_401_is_offline_and_never_echoes_the_password(switch):
    switch.status["/rest/system/resource"] = 401
    s = await switches.poll_switch(0, CFG)
    assert not s.online and not s.pending and s.alarm
    assert "401" in s.error
    assert "synthetic-secret" not in s.error


async def test_monitor_failure_keeps_the_rest_of_the_reading(switch):
    switch.status["/rest/interface/ethernet/monitor"] = 500
    s = await switches.poll_switch(0, CFG)
    assert s.online
    assert s.board == "TEST-BOARD"
    assert [(p.rate, p.rate_bps) for p in s.ports] == [("", 0.0), ("", 0.0)]


async def test_no_selected_ports_skips_the_monitor(switch):
    switch.interfaces = [_iface("ether1", running=False)]
    s = await switches.poll_switch(0, CFG)
    assert s.online and s.ports == []
    assert all(r.method == "GET" for r in switch.requests)


async def test_oversize_body_is_rejected(switch, monkeypatch):
    monkeypatch.setattr(switches, "HTTP_BODY_LIMIT", 64)
    switch.bodies["/rest/system/health"] = b"[" + b" " * 100 + b"]"
    s = await switches.poll_switch(0, CFG)
    assert not s.online
    assert "exceeds" in s.error


async def test_post_body_is_capped_too(monkeypatch):
    monkeypatch.setattr(switches, "HTTP_BODY_LIMIT", 64)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 100))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPError, match="exceeds"):
            await switches._request_capped(client, "POST", "http://192.0.2.10/x", {"a": 1})


async def test_malformed_json_is_offline(switch):
    switch.bodies["/rest/interface"] = b"<html>"
    s = await switches.poll_switch(0, CFG)
    assert (s.online, s.error) == (False, "malformed response")


# ─── poll_cluster integration ─────────────────────────────────────────


def _cluster_with_switch(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[[nodes]]\nlabel = "head"\nssh_target = "head"\nvllm_url = "http://192.0.2.20:8000"\n'
        '[[switches]]\nlabel = "fabric"\nurl = "http://192.0.2.10"\nuser = "monitor"\n'
    )
    config.configure(path)


async def test_slow_switch_never_delays_nodes_and_lands_later(tmp_path, monkeypatch):
    from stats import SparkUnitStats

    _cluster_with_switch(tmp_path)
    monkeypatch.setattr(collector, "POLL_SOFT_DEADLINE", 0.05)
    gate = asyncio.Event()
    polls = []

    async def fake_poll_unit(uid):
        return SparkUnitStats(label="head", online=True)

    async def fake_poll_switch(index, cfg):
        polls.append(index)
        if len(polls) == 1:
            await gate.wait()  # the switch cannot answer yet
        return SwitchStats(cfg.label, "192.0.2.10", online=True)

    monkeypatch.setattr(collector, "poll_unit", fake_poll_unit)
    monkeypatch.setattr(collector, "poll_switch", fake_poll_switch)

    first = await asyncio.wait_for(collector.poll_cluster(), 2.0)
    assert first.units[0].online
    assert [(s.label, s.host, s.pending) for s in first.switches] == [
        ("fabric", "192.0.2.10", True)
    ]
    assert not collector.pending_landed()

    # A carried flight never extends a later call's deadline.
    monkeypatch.setattr(collector, "POLL_SOFT_DEADLINE", 5.0)
    second = await asyncio.wait_for(collector.poll_cluster(), 1.0)
    assert second.switches[0].pending and polls == [0]

    gate.set()
    await asyncio.sleep(0.01)
    assert collector.pending_landed()
    third = await asyncio.wait_for(collector.poll_cluster(), 1.0)
    assert third.switches[0].online and not third.switches[0].pending
    await asyncio.sleep(0)
    assert polls == [0, 0]  # landed between calls: the next poll starts at once


async def test_two_switches_publish_in_config_order_and_one_failure_drops_neither(
    tmp_path, monkeypatch
):
    from stats import SparkUnitStats

    path = tmp_path / "config.toml"
    path.write_text(
        '[[nodes]]\nlabel = "head"\nssh_target = "head"\nvllm_url = "http://192.0.2.20:8000"\n'
        '[[switches]]\nlabel = "spine"\nurl = "https://192.0.2.10"\nuser = "monitor"\n'
        'password_env = "TEST_UNSET_SWITCH_PW"\n'
        '[[switches]]\nlabel = "leaf"\nurl = "https://192.0.2.11"\nuser = "monitor"\n'
        '[[switches]]\nlabel = "edge"\nurl = "https://192.0.2.12"\nuser = "monitor"\n'
    )
    config.configure(path)
    monkeypatch.delenv("TEST_UNSET_SWITCH_PW", raising=False)
    monkeypatch.setattr(collector, "POLL_SOFT_DEADLINE", 1.0)

    async def fake_poll_unit(uid):
        return SparkUnitStats(label="head", online=True)

    async def fake_poll_switch(index, cfg):
        if index == 0:
            return await switches.poll_switch(index, cfg)  # offline: no password, no request
        if index == 1:
            raise RuntimeError("poller bug")
        return SwitchStats(cfg.label, "192.0.2.12", online=True)

    monkeypatch.setattr(collector, "poll_unit", fake_poll_unit)
    monkeypatch.setattr(collector, "poll_switch", fake_poll_switch)
    stats = await asyncio.wait_for(collector.poll_cluster(), 2.0)
    assert [s.label for s in stats.switches] == [c.label for c in config.SWITCHES]
    assert [(s.online, s.pending, s.error) for s in stats.switches] == [
        (False, False, "$TEST_UNSET_SWITCH_PW is not set"),
        (False, False, "RuntimeError"),
        (True, False, ""),
    ]
    assert stats.units[0].online


async def test_simulation_has_no_switches(tmp_path, monkeypatch):
    _cluster_with_switch(tmp_path)
    monkeypatch.setattr(config, "SIMULATION_NODES", 2)
    stats = await collector.poll_cluster()
    assert stats.switches == []
