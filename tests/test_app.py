"""Behavioral tests for the tiling dashboard (src/app.py)."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from config import configure
from stats import ClusterStats, SparkUnitStats, TopologyInfo

# ─── fixtures ────────────────────────────────────────────────────────


def _config(path: Path, theme: str | None = None, n: int = 2) -> None:
    lines = ["[app]", "poll_interval = 5", "history_length = 25"]
    if theme:
        lines.append(f'theme = "{theme}"')
    lines += [
        "[[nodes]]",
        'label = "head"',
        'ssh_target = "head"',
        'vllm_url = "http://192.0.2.10:8000"',
    ]
    if n == 2:
        lines += [
            "[[nodes]]",
            'label = "worker"',
            'ssh_target = "worker"',
            'vllm_url = "http://192.0.2.11:8000"',
            "worker = true",
        ]
    path.write_text("\n".join(lines) + "\n")


def _unit(label: str, worker: bool = False, online: bool = True, hosted: bool = True):
    u = SparkUnitStats(label=label)
    u.is_worker = worker
    u.online = online
    u.model_hosted = hosted
    if hosted:
        u.model_name = "Qwen3.6-27B-Instruct"
    if online:
        u.gpu_util_pct = 73.0
        u.temp_c = 64.0
        u.power_w = 430.0
        u.mem_used_bytes = 62 * 1024**3
        u.mem_total_bytes = 120 * 1024**3
        u.swap_total_kb = 4 * 1024 * 1024
        u.swap_used_kb = 1 * 1024 * 1024
        u.cpu_cores_util = [50.0] * 20
        u.cpu_temp_c = 51.0
        u.gpu_clock_mhz = 2411.0
        u.roce_rx_bps = 3.2e9
        u.roce_tx_bps = 1.1e9
        u.roce_capacity_bps = 5e10
        u.kv_cache_pct = 32.0
        u.kv_total_tokens = 3_800_000
        u.kv_cache_used_tokens = 1_230_000
        u.kv_prefix_hit_rate = 45.0
        u.requests_running = 2
        u.requests_waiting = 1
        u.prompt_gen_ratio = 3.0
        u.throughput_tok_s = 1200.0
        u.prompt_throughput_tok_s = 3600.0
        u.ttft_p50_ms = 700.0
        u.ttft_p95_ms = 20500.0
        u.ttft_p99_ms = 20500.0
    return u


def _cluster(units=None):
    if units is None:
        units = [_unit("head"), _unit("worker", worker=True)]
    return ClusterStats(units=units, topology=TopologyInfo(topology_type="DUAL"))


def _stub(monkeypatch, units=None):
    import app as app_module

    async def fake_poll():
        return _cluster(units)

    monkeypatch.setattr(app_module, "poll_cluster", fake_poll)


async def _resize(pilot, width, height):
    await pilot.resize_terminal(width, height)
    await pilot.pause()
    await pilot.pause()


def _seed_history(app):
    """Give the charts real history so sparklines/line chart populate."""
    import collections

    app.history["throughput"] = collections.deque([40 + (i * 7) % 60 for i in range(24)], maxlen=25)
    app.history["prompt-throughput"] = collections.deque(
        [120 + (i * 11) % 90 for i in range(24)], maxlen=25
    )
    app.history["kv-usage-head"] = collections.deque(
        [18 + (i * 3) % 15 for i in range(24)], maxlen=25
    )
    # the shared model's own time-series (both fixture units host it; the
    # head unit is its authoritative reporter)
    app.history["gen-Qwen3.6-27B-Instruct"] = collections.deque(
        [40 + (i * 7) % 60 for i in range(24)], maxlen=25
    )
    app._update_ui()


def _config_cluster(path: Path, n: int, theme: str | None = None) -> None:
    """Write a config with ``n`` nodes labelled ``node-1..node-n``."""
    lines = ["[app]", "poll_interval = 5", "history_length = 25"]
    if theme:
        lines.append(f'theme = "{theme}"')
    for i in range(1, n + 1):
        lines += [
            "[[nodes]]",
            f'label = "node-{i}"',
            f'ssh_target = "node-{i}"',
            f'vllm_url = "http://192.0.2.{i}:8000"',
        ]
        if i > 1:
            lines.append("worker = true")
    path.write_text("\n".join(lines) + "\n")


def _cluster_n(n: int) -> ClusterStats:
    units = [_unit(f"node-{i}", worker=(i > 1)) for i in range(1, n + 1)]
    return ClusterStats(
        units=units, topology=TopologyInfo(topology_type="SWITCHED" if n >= 3 else "DUAL")
    )


def _stub_n(monkeypatch, n: int) -> None:
    import app as app_module

    async def fake_poll():
        return _cluster_n(n)

    monkeypatch.setattr(app_module, "poll_cluster", fake_poll)


def _seed_history_n(app, n: int) -> None:
    """Give every hosted node's KV series real history."""
    import collections

    app.history["throughput"] = collections.deque([40 + (i * 7) % 60 for i in range(24)], maxlen=25)
    app.history["prompt-throughput"] = collections.deque(
        [120 + (i * 11) % 90 for i in range(24)], maxlen=25
    )
    for i in range(1, n + 1):
        app.history[f"kv-usage-node-{i}"] = collections.deque(
            [18 + (i * 3) % 15 for i in range(24)], maxlen=25
        )
    app._update_ui()


def _style_at(text, idx):
    """Rich span styles covering character offset ``idx``."""
    out = []
    for sp in text.spans:
        if sp.start <= idx < sp.end:
            out.append(str(sp.style))
    return out


# ─── config / theme (module wiring) ──────────────────────────────────


def test_app_uses_configured_poll_interval_and_node_count(tmp_path: Path):
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    app = DGXTop()
    assert app._current_interval() == 5
    assert len(app.settings.nodes) == 2
    # bindings intact (AC12)
    keys = {b.key for b in app.BINDINGS}
    assert {"plus", "minus", "t", "q", "r"} <= keys


async def test_default_theme_and_custom_registration(tmp_path: Path):
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    app = DGXTop()
    async with app.run_test(size=(132, 40)):
        assert app.current_theme.name == "dgx-aeon"


# ─── AC1: caret title-in-border header fidelity ──────────────────────


async def test_node_box_header_matches_caret_pattern(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox
    from themes import build_palette

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        pal = build_palette(app.current_theme)
        node = app.query_one("#node-1", NodeBox)
        text = node.render()
        top = text.plain.split("\n")[0]
        # exact structural pattern from the requested example, and exact width
        assert re.match(r"^╭─┤ \^ \w+ (host|worker) ├─+┤ [0-9.]+ ├─╮$", top), top
        assert len(top) == node.content_size.width
        # worker caret/role are the warn (orange) identity colour
        caret_idx = top.index("^")
        assert any(pal.warn.lower() in s.lower() for s in _style_at(text, caret_idx))


# ─── AC2: heavy focused vs light node charsets ───────────────────────


async def test_serving_heavy_vs_node_light_charsets(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox, ServingBox
    from themes import build_palette

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        pal = build_palette(app.current_theme)
        serv = app.query_one("#serving", ServingBox).render()
        node = app.query_one("#node-0", NodeBox).render()
        s_top = serv.plain.split("\n")[0]
        n_top = node.plain.split("\n")[0]
        # serving uses the same light charset as node containers
        assert s_top.startswith("╭─") and serv.plain.split("\n")[-1].startswith("╰")
        assert n_top.startswith("╭─") and node.plain.split("\n")[-1].startswith("╰")
        # both borders are now dim grey (the focus cyan was removed)
        assert any(pal.dim.lower() in s.lower() for s in _style_at(serv, 0))
        assert not any(pal.cyan.lower() in s.lower() for s in _style_at(serv, 0))
        assert any(pal.dim.lower() in s.lower() for s in _style_at(node, 0))


# ─── AC3: tiling geometry ────────────────────────────────────────────


async def test_tiled_serving_left_nodes_right_at_wide(tmp_path: Path, monkeypatch):
    """Above TILING_WIDTH the SERVING card tiles beside the node column (node
    cards to the RIGHT of the serving card, sharing its top row)."""
    from app import DGXTop, NodeBox, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        assert app.tiled
        serv = app.query_one("#serving", ServingBox).region
        nodes = [app.query_one(f"#node-{i}", NodeBox).region for i in range(2)]
        # SERVING is the left column; every node card sits to its right.
        assert serv.x == 0 and serv.width == 132 * 56 // 100
        for r in nodes:
            assert r.x > serv.right
            assert r.right <= 132
        assert nodes[0].y == serv.y  # the first card shares serving's top row
        assert nodes[1].y > nodes[0].y  # two-node cluster stacks in the column
        for r in (serv, *nodes):
            assert r.bottom <= 40


async def test_stacked_hero_above_node_grid_below_96(tmp_path: Path, monkeypatch):
    """Below TILING_WIDTH the SERVING hero stays full-width on top and the
    node cards wrap below (the stacked arrangement)."""
    from app import DGXTop, NodeBox, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(90, 40)) as pilot:
        await pilot.pause()
        assert not app.tiled
        serv = app.query_one("#serving", ServingBox).region
        n0 = app.query_one("#node-0", NodeBox).region
        n1 = app.query_one("#node-1", NodeBox).region
        assert serv.x == 0 and serv.width == 90  # full-width hero on top
        assert n0.y >= serv.bottom + 1
        assert n0.y == n1.y  # both nodes share one grid row
        for r in (serv, n0, n1):
            assert r.bottom <= 40


# ─── AC4/AC10: every metric survives width sweep, never scroll ───────


async def test_every_metric_survives_and_never_scrolls(tmp_path: Path, monkeypatch):
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        for w, h in [(132, 44), (100, 40), (80, 40), (63, 40), (50, 40)]:
            await _resize(pilot, w, h)
            blob = "\n".join(
                wid.render().plain for wid in app.screen.query("Waybar, ServingBox, NodeBox")
            )
            # The refinement drops the low-value graphics/stats at density:
            # node cards keep the gpu/mem/cpu values and the serving keeps gen/
            # requests/ttft/kv%. RoCE, power and the window stat are dropped.
            for token in (
                "73%",
                "52%",
                "50%",
                "32%",
                "kv",
                "ttft",
            ):
                assert token in blob, (w, h, token)
            assert app.screen.max_scroll_y == 0, (w, h)


async def test_floor_never_scrolls_or_clips(tmp_path: Path, monkeypatch):
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        for w in (40, 50, 63, 80, 96, 132):
            await _resize(pilot, w, 8)
            assert app.floor, (w, "should be floor at h=8")
            assert app.screen.max_scroll_y == 0, (w, "scroll")
            vis = [
                wid for wid in app.screen.query("Waybar, ServingBox, NodeBox") if wid.region.height
            ]
            for wid in vis:
                r = wid.region
                assert r.bottom <= 8, (w, wid.id, r.bottom)
            # No two visible widgets may share screen space: max_scroll_y==0 and
            # bottom<=h both hold even when widgets overlap exactly, so assert
            # rectangle disjointness directly.
            for i, a in enumerate(vis):
                for b in vis[i + 1 :]:
                    ra, rb = a.region, b.region
                    separated = (
                        ra.x + ra.width <= rb.x
                        or rb.x + rb.width <= ra.x
                        or ra.y + ra.height <= rb.y
                        or rb.y + rb.height <= ra.y
                    )
                    assert separated, (w, a.id, b.id, ra, rb)


async def test_density_ladder_steps_down(tmp_path: Path, monkeypatch):
    """Every tier of the ladder is reached as height shrinks. At width 90
    (below TILING_WIDTH) the stacked arrangement exposes the full roomy →
    dense → compact → rail → floor sequence, each one row denser than the
    previous at the calibrated heights."""
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(90, 44)) as pilot:
        await pilot.pause()
        seen = []
        for h in (44, 32, 30, 17, 8):
            await _resize(pilot, 90, h)
            tier = "floor" if app.floor else ("rail" if app.rail else app.density)
            seen.append(tier)
        assert seen == ["roomy", "dense", "compact", "rail", "floor"], seen


# ─── AC5: gradient meters vs single-hue KV ───────────────────────────


async def test_node_meter_is_gradient_kv_is_single_hue(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        node = app.query_one("#node-0", NodeBox).render()
        gpu_meter = node.plain.split("\n")[2]  # top, gpu, meter
        base = sum(len(line) + 1 for line in node.plain.split("\n")[:2])
        fill_positions = [base + i for i, ch in enumerate(gpu_meter) if ch == "█"]
        colors = {tuple(_style_at(node, p)) for p in fill_positions}
        assert len(colors) > 1, "gpu gradient meter should ramp per cell"

        serv = app.query_one("#serving", ServingBox).render()
        lines = serv.plain.split("\n")
        kv_line_idx = next(i for i, ln in enumerate(lines) if "kv%" in ln)
        kbase = sum(len(line) + 1 for line in lines[:kv_line_idx])
        kv_line = lines[kv_line_idx]
        kfill = [kbase + i for i, ch in enumerate(kv_line) if ch == "█"]
        kcolors = {tuple(_style_at(serv, p)) for p in kfill}
        assert len(kcolors) == 1, "kv meter is single hue"


# ─── AC6: serving area chart ─────────────────────────────────────────


async def test_serving_area_chart_present(tmp_path: Path, monkeypatch):
    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        lines = app.query_one("#serving", ServingBox).render().plain.split("\n")
        assert not any("last 24 samples" in ln for ln in lines)  # chart label removed
        # gen/prompt/kv sparklines (blocks) + the multi-series braille chart
        assert any(c in ln for ln in lines for c in "▁▂▃▄▅▆▇█")
        chart_rows = [ln for ln in lines if any(0x2800 <= ord(c) < 0x2900 for c in ln)]
        assert len(chart_rows) >= 2


# ─── AC7: waybar ─────────────────────────────────────────────────────


async def test_waybar_shows_cluster_chrome(tmp_path: Path, monkeypatch):
    from app import DGXTop, Waybar

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        wb = app.query_one("#waybar", Waybar)
        text = wb.render()
        plain = text.plain
        # workspace chips + aggregate temp/power/clock removed; only online count
        assert "● 2/2" in plain
        assert "°C" not in plain and "W" not in plain
        assert " 2 3" not in plain
        assert "Qwen3.6-27B-Instruct" in plain
        assert len(plain) == wb.content_size.width


# ─── AC8: CPU frequency renders MHz ──────────────────────────────────


def test_fmt_freq_renders_mhz():
    from app import _fmt_freq

    assert _fmt_freq(2808.0) == "2808MHz"
    assert _fmt_freq(3900.0) == "3900MHz"
    assert _fmt_freq(0) == ""


async def test_node_gpu_row_shows_sm_clock(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        node = app.query_one("#node-0", NodeBox).render().plain
        gpu_line = next(ln for ln in node.split("\n") if "gpu" in ln)
        assert "2411MHz" in gpu_line


# ─── AC9: bottom bar only in the most compressed tiers ───────────────


async def test_waybar_always_visible_carries_base_stats(tmp_path: Path, monkeypatch):
    from app import DGXTop, Waybar

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        wb = app.query_one("#waybar", Waybar)
        # The footer is gone: the header is the only chrome and stays visible in
        # every tier, carrying the base serving stats (gen, KV, online).
        for w, h in [(132, 44), (100, 40), (63, 20), (40, 8)]:
            await _resize(pilot, w, h)
            assert wb.styles.display == "block", (w, "waybar should never hide")
            text = wb.render().plain
            assert "tok/s" in text, (w, text)
            assert "KV 32%" in text, (w, text)
            assert "● 2/2" in text, (w, text)


# ─── AC8: waybar WARN marker + offline flip ──────────────────────────


async def test_waybar_warn_marker_on_offline_or_hot(tmp_path: Path, monkeypatch):
    from app import DGXTop, Waybar
    from themes import build_palette

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        pal = build_palette(app.current_theme)
        wb = app.query_one("#waybar", Waybar).render()
        assert "KV 32%" in wb.plain
        assert " ! " not in wb.plain  # healthy cluster: no warn marker

    _stub(
        monkeypatch, units=[_unit("head"), _unit("worker", worker=True, online=False, hosted=False)]
    )
    app2 = DGXTop()
    async with app2.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        pal = build_palette(app2.current_theme)
        wb = app2.query_one("#waybar", Waybar).render()
        assert " ! " in wb.plain
        w_idx = wb.plain.index(" ! ")
        assert any(f"on {pal.warn}".lower() in s.lower() for s in _style_at(wb, w_idx))


# ─── AC9: offline node ───────────────────────────────────────────────


async def test_offline_node_dashes_and_glyph(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(
        monkeypatch, units=[_unit("head"), _unit("worker", worker=True, online=False, hosted=False)]
    )
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        lines = app.query_one("#node-1", NodeBox).render().plain.split("\n")
        top = lines[0]
        assert "✗" in top and "worker" in top  # glyph state, label kept
        for label in ("gpu", "mem", "cpu", "roce"):
            row = next(ln for ln in lines if ln.lstrip("│ ").startswith(label))
            assert "—" in row, (label, row)
        assert top.startswith("╭─") and lines[-1].startswith("╰")


# ─── AC11: non-finite gating + theme repaint ─────────────────────────


async def test_non_finite_samples_do_not_crash(tmp_path: Path, monkeypatch):
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")

    bad = _unit("head")
    bad.kv_cache_pct = float("nan")
    bad.throughput_tok_s = float("inf")
    _stub(monkeypatch, units=[bad, _unit("worker", worker=True)])
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        # renders without raising; charts never see NaN/Inf
        app.query_one("#serving").render()
        assert app.screen.max_scroll_y == 0


async def test_theme_switch_repaints(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox
    from themes import build_palette

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        app.theme = "tokyo-night"
        await pilot.pause()
        pal = build_palette(app.current_theme)
        node = app.query_one("#node-0", NodeBox).render()
        # host caret repaints to tokyo-night cyan
        top = node.plain.split("\n")[0]
        assert any(pal.cyan.lower() in s.lower() for s in _style_at(node, top.index("^")))


# ─── helper units ────────────────────────────────────────────────────


def test_ttft_tail_thresholds():
    from app import _ttft_tail
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    assert _ttft_tail(0.9, pal)[0] == ""
    assert _ttft_tail(3.0, pal)[0] == "!"
    assert _ttft_tail(9.0, pal)[0] == "!!"


def test_ramp_moves_green_to_red():
    from app import _ramp

    low = _ramp(0)
    high = _ramp(100)
    assert low != high
    assert low.lower().startswith("#9e") or low.lower() == "#9ece6a"  # green
    assert high.lower() == "#f7768e"  # red


def test_box_lines_are_exact_width():
    from app import _box_lines
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    from rich.text import Text

    rows = [Text("gpu 73%"), Text("mem 50%")]
    for focused in (True, False):
        lines = _box_lines(
            40, [("^", ""), (" head", ""), (" host", "")], [("1.2.3.4", "")], rows, focused, pal
        )
        assert all(len(ln.plain) == 40 for ln in lines), [ln.plain for ln in lines]


def test_box_clamps_overlong_title():
    from rich.text import Text

    from app import _box_lines
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    lines = _box_lines(
        24,
        [("^", ""), (" a-very-long-node-name", ""), (" worker", "")],
        [("198.51.100.200", "")],
        [Text("x")],
        False,
        pal,
    )
    assert all(len(ln.plain) == 24 for ln in lines)


# ─── meter treatments + quiet mode ───────────────────────────────────


def _config_treated(path: Path, treatment: str, quiet: bool = False) -> None:
    lines = [
        "[app]",
        "poll_interval = 5",
        "history_length = 25",
        f'meter_treatment = "{treatment}"',
        f"quiet = {'true' if quiet else 'false'}",
        "[[nodes]]",
        'label = "head"',
        'ssh_target = "head"',
        'vllm_url = "http://192.0.2.10:8000"',
        "[[nodes]]",
        'label = "worker"',
        'ssh_target = "worker"',
        'vllm_url = "http://192.0.2.11:8000"',
        "worker = true",
    ]
    path.write_text("\n".join(lines) + "\n")


def _meter_row(app, idx=0):
    """(row_text, absolute_offset, node_text) of node idx's GPU meter row."""
    from app import NodeBox

    node = app.query_one(f"#node-{idx}", NodeBox).render()
    lines = node.plain.split("\n")
    # top rule, gpu headline, meter
    row_idx = 2
    base = sum(len(ln) + 1 for ln in lines[:row_idx])
    return lines[row_idx], base, node


@contextlib.asynccontextmanager
async def _treated_app(tmp_path, monkeypatch, treatment, quiet=False):
    from app import DGXTop

    _config_treated(tmp_path / "config.toml", treatment, quiet)
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        yield app, pilot


async def test_line_treatment_renders(tmp_path: Path, monkeypatch):
    from app import ServingBox

    async with _treated_app(tmp_path, monkeypatch, "line") as (app, pilot):
        await pilot.pause()
        _seed_history(app)
        row, _, node = _meter_row(app)
        assert "━" in row and "─" in row, row
        assert "█" not in row and "▓" not in row
        serv = app.query_one("#serving", ServingBox).render()
        kv_row = next(ln for ln in serv.plain.split("\n") if "kv%" in ln)
        assert "━" in kv_row, kv_row


async def test_tick_treatment_renders(tmp_path: Path, monkeypatch):
    from app import ServingBox

    async with _treated_app(tmp_path, monkeypatch, "tick") as (app, pilot):
        await pilot.pause()
        _seed_history(app)
        row, _, _ = _meter_row(app)
        assert row.strip(), "tick row renders"
        from app import _meter_line
        from themes import build_palette, get_theme

        pal = build_palette(get_theme("dgx-aeon"))
        text = _meter_line("tick", 50, 20, pal, pal.blue)
        assert text.plain.count("━") == 1, "exactly one bright marker"
        assert "╾" in text.plain and "┈" in text.plain, "dim scale on both sides"
        serv = app.query_one("#serving", ServingBox).render()
        kv_row = next(ln for ln in serv.plain.split("\n") if "kv%" in ln)
        assert kv_row.strip() != ""


async def test_spark_treatment_renders(tmp_path: Path, monkeypatch):
    from app import ServingBox

    async with _treated_app(tmp_path, monkeypatch, "spark") as (app, pilot):
        await pilot.pause()
        _seed_history(app)
        row, _, _ = _meter_row(app)
        spark_chars = set("▁▂▃▄▅▆▇█")
        assert any(ch in spark_chars for ch in row), row
        serv = app.query_one("#serving", ServingBox).render()
        kv_row = next(ln for ln in serv.plain.split("\n") if "kv%" in ln)
        assert any(ch in spark_chars for ch in kv_row), kv_row


async def test_gpu_mem_history_recorded(tmp_path: Path, monkeypatch):
    from app import NodeBox

    async with _treated_app(tmp_path, monkeypatch, "line") as (app, pilot):
        await pilot.pause()
        _seed_history(app)
        assert len(app.history["gpu-head"]) >= 1  # recorded on each poll
        assert 0 <= app.history["gpu-head"][0] <= 100
        assert 0 <= app.history["mem-head"][0] <= 100
        node = app.query_one("#node-0", NodeBox)
        assert node._gpu_history, "gpu history passed into NodeBox"
        assert node._mem_history


async def test_quiet_palette_and_ramp():
    from app import _ramp
    from themes import build_palette, get_theme

    loud = build_palette(get_theme("dgx-aeon"))
    quiet = build_palette(get_theme("dgx-aeon"), quiet=True)
    assert quiet.quiet is True and loud.quiet is False
    for role in ("accent", "ok", "blue", "cyan"):
        assert getattr(quiet, role) == quiet.fg, role
    assert _ramp(40, quiet) == quiet.fg
    assert _ramp(80, quiet) == quiet.warn
    assert _ramp(95, quiet) == "#f7768e"
    # loud ramp unchanged
    assert _ramp(40, loud) == _ramp(40)


async def test_quiet_composes_with_treatment(tmp_path: Path, monkeypatch):
    from app import _palette_for

    async with _treated_app(tmp_path, monkeypatch, "line", quiet=True) as (app, pilot):
        await pilot.pause()
        _seed_history(app)
        assert _palette_for(app).quiet is True
        row, _, node = _meter_row(app)
        assert "━" in row
        # healthy values stay neutral: no accent hue anywhere on the meter row
        offset = node.plain.index(row)
        spans = [sp for sp in node.spans if sp.start >= offset and sp.end <= offset + len(row)]
        assert all("f7768e" not in str(sp.style) for sp in spans)


async def test_meter_escalates_to_crit(tmp_path: Path, monkeypatch):
    from app import NodeBox

    async with _treated_app(tmp_path, monkeypatch, "line") as (app, pilot):
        await pilot.pause()
        unit = _unit("head")
        unit.gpu_util_pct = 94.0
        app.cluster = _cluster([unit, _unit("worker", worker=True)])
        app._update_ui()
        node = app.query_one("#node-0", NodeBox).render()
        assert "f7768e" in str(node.spans), "94% meter escalates to crit red"


# ─── serving top-row alignment ────────────────────────────────────────


async def test_serving_top_rows_aligned(tmp_path: Path, monkeypatch):
    """AC: gen/prompt/kv duo rows share one graph lane (value + dots on the
    line row, the fill row aligned under the dots), tails align, blank
    spacer rows separate the graph blocks; nothing overflows the box."""
    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        box = app.query_one("#serving", ServingBox)
        box.update_throughput(
            [40 + (i * 7) % 60 for i in range(24)], [120 + (i * 11) % 90 for i in range(24)]
        )
        box.update_kv(
            32.0,
            req=2,
            wait=1,
            used_tok=1_230_000,
            total_tok=3_800_000,
            prefix_hit=45.0,
            kv_history=[18 + (i * 3) % 15 for i in range(24)],
            ttft_p50_ms=700.0,
            ttft_p95_ms=20500.0,
        )
        await pilot.pause()
        width = box.content_size.width
        lines = box.render().plain.split("\n")
        interior = lines[1:-1]  # strip heavy borders
        assert all(len(ln) == width for ln in interior), "row not padded to box width"
        by_label = {}
        for i, ln in enumerate(interior):
            m = re.match(r"^\u2502 (gen    |prompt |kv     |kv%    )", ln)
            if m:
                by_label.setdefault(m.group(1), []).append((i, ln))
        assert set(by_label) == {"gen    ", "prompt ", "kv     ", "kv%    "}
        braille = {chr(c) for c in range(0x2801, 0x2900)}
        starts, lens = set(), set()
        for label, entries in by_label.items():
            if label == "kv%    ":
                continue  # meter row, not a duo spark
            i, ln = entries[0]
            # line row: the gen row leads with the value at the lane start,
            # then contiguous braille dots; prompt/kv are pure dot rows
            cols = [j for j, ch in enumerate(ln) if ch in braille]
            assert cols, f"no dot glyphs on {label!r}"
            assert cols == list(range(min(cols), max(cols) + 1)), f"ragged dots {label!r}"
            # fill row beneath: a contiguous █ run aligned with the dots
            fcols = [j for j, ch in enumerate(interior[i + 1]) if ch == "\u2588"]
            assert fcols, f"no fill row under {label!r}"
            assert fcols == list(range(min(fcols), max(fcols) + 1)), f"ragged fill {label!r}"
            assert (min(fcols), len(fcols)) == (min(cols), len(cols)), (
                f"fill misaligned under {label!r}"
            )
            if label != "gen    ":
                starts.add(min(cols))
                lens.add(len(cols))
        assert len(lens) == 1, f"graph lengths differ: {lens}"
        assert len(starts) == 1, f"graph starts differ: {starts}"
        # the gen value sits exactly at the shared graph lane's start
        assert by_label["gen    "][0][1][min(starts)].isdigit(), by_label["gen    "][0][1]
        # each duo block is line row + fill row; blank spacers between blocks
        gen_i = by_label["gen    "][0][0]
        prompt_i = by_label["prompt "][0][0]
        kv_i = by_label["kv     "][0][0]
        assert prompt_i - gen_i == 3 and kv_i - prompt_i == 3

        def blank(ln: str) -> bool:
            return ln[1:-1].strip() == ""

        kvp_i = by_label["kv%    "][0][0]
        assert kv_i - prompt_i == 3 and kvp_i - kv_i == 3
        assert (
            blank(interior[gen_i + 2])
            and blank(interior[prompt_i + 2])
            and blank(interior[kv_i + 2])
        )
        # the widest tail (gen) reaches the interior's right edge
        assert len(by_label["gen    "][0][1][1:-1].rstrip()) == width - 3


# ─── cluster scaling: 1-12 nodes, fluid node grid ────────────────────


async def test_config_and_compose_twelve_nodes(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        assert len(app.settings.nodes) == 12
        assert len(list(app.query(NodeBox))) == 12


async def test_twelve_nodes_wrap_to_usable_cards_at_wide(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        # The refinement keeps the node card as long as possible: 12 nodes that
        # cannot fit a usable single row wrap into usable cards (each tile held
        # at the card minimum), never a single narrow strip row.
        assert app.node_mode == "card"
        assert app.cols < 12
        nodes = [app.query_one(f"#node-{i}", NodeBox).region for i in range(12)]
        assert len({n.y for n in nodes}) > 1  # wrapped into multiple rows
        for n in nodes:
            assert n.width >= 22  # each tile keeps a usable card width
        assert all(n.bottom <= 50 for n in nodes)
        assert app.screen.max_scroll_y == 0


async def test_nodes_wrap_at_narrow_with_min_width(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    async with app.run_test(size=(50, 40)) as pilot:
        await pilot.pause()
        nodes = [app.query_one(f"#node-{i}", NodeBox).region for i in range(12)]
        assert len({n.y for n in nodes}) > 1  # wrapped into multiple rows
        for n in nodes:
            assert n.width >= 10  # each tile keeps a useful minimum width


async def test_never_scroll_or_clip_for_cluster_sizes(tmp_path: Path, monkeypatch):
    from app import DGXTop

    for n in (2, 5, 12):
        _config_cluster(tmp_path / "config.toml", n)
        configure(tmp_path / "config.toml")
        _stub_n(monkeypatch, n)
        app = DGXTop()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            for w, h in [
                (180, 50),
                (132, 40),
                (103, 40),  # lands a 1fr column on the card-format boundary
                (100, 40),
                (90, 42),
                (80, 30),
                (77, 40),  # boundary width (ceil(col) reaches NODE_FULL_MIN)
                (76, 40),
                (70, 50),
                (63, 20),
                (51, 40),  # boundary width (ceil(col) reaches NODE_FULL_MIN)
                (50, 40),
                (45, 42),
                (40, 8),
            ]:
                await _resize(pilot, w, h)
                assert app.screen.max_scroll_y == 0, (n, w, h)
                vis = [
                    wid
                    for wid in app.screen.query("Waybar, ServingBox, NodeBox")
                    if wid.region.height
                ]
                for wid in vis:
                    r = wid.region
                    assert r.bottom <= h, (n, w, h, wid.id, r.bottom)
                    assert r.x + r.width <= w, (n, w, h, wid.id, r)  # no horizontal clip
                for i, a in enumerate(vis):
                    for b in vis[i + 1 :]:
                        ra, rb = a.region, b.region
                        separated = (
                            ra.x + ra.width <= rb.x
                            or rb.x + rb.width <= ra.x
                            or ra.y + ra.height <= rb.y
                            or rb.y + rb.height <= ra.y
                        )
                        assert separated, (n, w, h, a.id, b.id, ra, rb)


async def test_density_ladder_for_twelve_nodes(tmp_path: Path, monkeypatch):
    """A 12-node cluster steps through the whole ladder as height shrinks
    (at width 180 the wide tiled roomy tier and the denser tiers both
    appear; the duo spark rows cost height, so dense now needs ≥33 rows)."""
    from app import DGXTop

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    async with app.run_test(size=(180, 60)) as pilot:
        await pilot.pause()
        seen = []
        for h in (60, 33, 24, 20, 8):
            await _resize(pilot, 180, h)
            tier = "floor" if app.floor else ("rail" if app.rail else app.density)
            seen.append(tier)
        assert seen == ["roomy", "dense", "compact", "rail", "floor"], seen


async def test_condensed_table_row_shows_gpu_mem_cpu(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    async with app.run_test(size=(240, 8)) as pilot:
        await pilot.pause()
        assert app.floor and app.node_mode == "table"
        line = app.query_one("#node-0", NodeBox).render().plain
        assert "\n" not in line  # one aligned row, no window frame
        # a condensed table row favours gpu/mem/cpu with the short identity
        assert "73%" in line  # gpu util
        assert "52%" in line  # mem util
        assert "50%" in line  # cpu util
        assert app.screen.max_scroll_y == 0


async def test_node_text_card_drops_meters(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        await _resize(pilot, 50, 40)
        assert app.node_mode == "card"
        node = app.query_one("#node-0", NodeBox).render().plain
        # text card: gpu/mem/cpu values survive, but the metre/core-grid/RoCE
        # graphs are dropped.
        assert "73%" in node and "52%" in node and "50%" in node
        assert "roce" not in node
        assert not any(ch in node for ch in "▁▂▃▄▅▆▇█▓")


async def test_serving_never_mentions_window(tmp_path: Path, monkeypatch):
    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        for w, h in [(132, 44), (100, 40), (80, 40), (50, 20), (40, 8)]:
            await _resize(pilot, w, h)
            blob = app.query_one("#serving", ServingBox).render().plain
            assert "window" not in blob.lower(), (w, h, blob)


async def test_compact_keeps_meter_cards_and_chart(tmp_path: Path, monkeypatch):
    """AC2: the graphs survive an extra tier. At compact the node cards still
    carry the meters + core grid (RoCE is the first graph to go) and the
    serving keeps a multi-row gen line chart (braille dots)."""
    from app import DGXTop, NodeBox, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        await _resize(pilot, 132, 20)
        assert app.density == "compact" and not app.rail and not app.floor
        assert app._chart_rows >= 2, app._chart_rows  # line chart survives compact
        serv_lines = app.query_one("#serving", ServingBox).render().plain.split("\n")
        # sparkline blocks ∪ braille line-chart dots (U+2800–U+28FF)
        assert (
            sum(
                1
                for ln in serv_lines
                if any(c in "▁▂▃▄▅▆▇█" or 0x2800 <= ord(c) < 0x2900 for c in ln)
            )
            >= 3
        )
        node = app.query_one("#node-0", NodeBox).render().plain
        assert "73%" in node and "52%" in node and "50%" in node
        assert any(c in node for c in "█▓━╾┈"), "meter glyphs survive compact"
        assert "■" in node, "core grid survives compact"
        assert "roce" not in node, "RoCE drops before the meters"
        assert app.screen.max_scroll_y == 0


async def test_serving_chart_grows_with_height(tmp_path: Path, monkeypatch):
    """AC3: the serving area chart grows into the leftover height instead of a
    symmetric pad (bounded by the tier max)."""
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(90, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        await _resize(pilot, 90, 44)
        assert not app.tiled
        assert app._chart_rows >= 12, app._chart_rows
        body = app.query_one("#body")
        assert body.region.y <= 3, body.region  # leftover consumed, not padded
        await _resize(pilot, 132, 44)
        assert app.tiled and app._chart_rows >= 10, app._chart_rows
        assert app.screen.max_scroll_y == 0


async def test_serving_wins_gen_reqs_ttft(tmp_path: Path, monkeypatch):
    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        for w, h in [(132, 44), (100, 40), (80, 40), (50, 20), (40, 8)]:
            await _resize(pilot, w, h)
            blob = app.query_one("#serving", ServingBox).render().plain
            # the base serving surface always keeps gen, the requests line
            # (concurrency) and ttft.
            assert "gen" in blob, (w, h, blob)
            assert "req" in blob or "requests" in blob, (w, h, blob)
            assert "ttft" in blob, (w, h, blob)


async def test_two_models_share_one_serving_pane(tmp_path: Path, monkeypatch):
    """Two endpoints, one pane: a gen row per model (an SGLang endpoint with
    no Prometheus counter says so honestly instead of plotting a fake zero
    line), one shared braille time-series chart, per-model requests/ttft
    rows — and the extra model rows never clip at any tier."""
    import collections

    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    head = _unit("head")
    head.model_name = "qwen3.8-flash-next"
    head.generation_tokens_total = 264348.0
    worker = _unit("worker", worker=True)
    worker.model_name = "NVIDIA-Nemotron-3.5-Lightning"
    worker.model_source = "sglang"
    worker.generation_tokens_total = 0.0
    worker.throughput_tok_s = 0.0
    worker.model_metrics = False
    worker.requests_running = 0
    worker.requests_waiting = 0
    worker.ttft_p50_ms = 0.0
    worker.ttft_p95_ms = 0.0
    _stub(monkeypatch, [head, worker])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app._models_n == 2
        app.history["gen-qwen3.8-flash-next"] = collections.deque(
            [40 + (i * 7) % 60 for i in range(24)], maxlen=25
        )
        app._update_ui()
        blob = app.query_one("#serving", ServingBox).render().plain
        # both models appear in the pane, each with its own row set
        assert "qwen3.8-flash-next" in blob
        assert "NVIDIA-Nemotron" in blob
        assert "no tok/s · sglang" in blob
        braille = [ln for ln in blob.split("\n") if any(0x2800 <= ord(c) < 0x2900 for c in ln)]
        assert len(braille) >= 2, braille  # the shared time-series chart
        assert blob.count("requests") == 2, blob  # one concurrency row per model
        for w, h in [(100, 40), (95, 24), (60, 18), (40, 8)]:
            await _resize(pilot, w, h)
            assert app.screen.max_scroll_y == 0, (w, h)


# ─── line-chart-hires: connected smooth chart + stable per-model hues ──


def _cells(t):
    """(char, style-string) per column of a rendered Text (single row)."""
    from rich.console import Console

    cells = []
    for seg in Console(force_terminal=True, width=4096).render(t):
        cells.extend((ch, str(seg.style) if seg.style else "") for ch in seg.text)
    return cells


def _line_dots(out):
    """Absolute dot rows per braille sub-column: {(cell, side): set(rows)}."""
    from app import _BRAILLE_BITS

    dots = {}
    for r, t in enumerate(out):
        for c, (ch, _st) in enumerate(_cells(t)):
            code = ord(ch)
            if 0x2800 < code < 0x2900:
                bits = code - 0x2800
                for (cc, rr), b in _BRAILLE_BITS.items():
                    if bits & b:
                        dots.setdefault((c, cc), set()).add(r * 4 + rr)
    return dots


def _aeon_palettes():
    from themes import build_palette, get_theme

    return build_palette(get_theme("dgx-aeon")), build_palette(get_theme("dgx-aeon"), quiet=True)


def test_lines_chart_draws_connected_lines_not_dots():
    """AC1: on a steep multi-slope series every sub-column carries dots and
    adjacent sub-columns never jump more than one dot row — the line is
    connected, not the old single-dot-per-sample scatter."""
    from app import _lines_chart_lines

    pal, _ = _aeon_palettes()
    out = _lines_chart_lines([("m", "#8A7CFF", [5, 30, 12, 44, 8, 27])], 6, 40, pal)
    dots = _line_dots(out)
    assert len(dots) == 80, len(dots)  # every sub-column plotted
    seq = sorted(dots)
    for k1, k2 in zip(seq, seq[1:]):
        gap = min(abs(a - b) for a in dots[k1] for b in dots[k2])
        assert gap <= 1, (k1, k2, dots[k1], dots[k2])


def test_lines_chart_step_transition_is_smooth():
    """AC2: a step between two sample levels renders a curved ramp of
    intermediate dot rows, not a nearest-neighbour plateau + jump."""
    from app import _lines_chart_lines

    pal, _ = _aeon_palettes()
    out = _lines_chart_lines([("m", "#8A7CFF", [0, 0, 60, 60])], 8, 40, pal)
    dots = _line_dots(out)
    levels = {row for rows in dots.values() for row in rows}
    assert len(levels) >= 4, len(levels)
    # The step must SPREAD across sub-columns (each column's lowest line dot
    # advances at most a few rows). Nearest-neighbour resampling concentrates
    # the whole 0→max step in one column (a 31-row spike) — the exact
    # staircase the old renderer drew.
    reps = [max(rows) for _k, rows in sorted(dots.items())]
    jumps = [abs(b - a) for a, b in zip(reps, reps[1:])]
    assert max(jumps) <= 8, max(jumps)


def test_lines_chart_flat_composite_fill():
    """AC Y6 (supersedes the old gradient contract): fill is a UNIFORM █
    at 25% of the hue over the background — no partial blocks, no depth
    gradient — and starts directly under the line's cell."""
    from textual.color import Color

    from app import _lines_chart_lines

    pal, _ = _aeon_palettes()
    out = _lines_chart_lines([("m", "#8A7CFF", [8, 34, 44, 12, 5])], 6, 30, pal)
    expect = Color.parse(pal.bg).blend(Color.parse("#8A7CFF"), 0.25)
    seen = 0
    top_fill: dict[int, int] = {}
    for r, t in enumerate(out):
        for c, (ch, st) in enumerate(_cells(t)):
            if "\u2581" <= ch <= "\u2588":
                assert ch == "\u2588", (r, c, ch)  # flat: never a partial block
                rgb = Color.parse(st.split()[-1]).rgb[:3]
                assert all(abs(x - y) <= 1 for x, y in zip(rgb, expect.rgb[:3])), (r, c, st)
                top_fill[c] = min(top_fill.get(c, 99), r)
                seen += 1
    assert seen > 20, seen
    dots = _line_dots(out)
    per_col: dict[int, set[int]] = {}
    for (c, _side), ds in dots.items():
        per_col.setdefault(c, set()).update(ds)
    for c, fr in top_fill.items():
        d = per_col.get(c, set())
        assert d, c
        assert 4 * fr + 3 > min(d), (c, fr, sorted(d))  # never above the line
        assert fr <= (max(d) >> 2) + 1, (c, fr, sorted(d))  # no gap under it


def test_lines_chart_overlapping_fills_composite():
    """AC Y7: where two series' fills share a cell the shade is the
    two-layer 25% composite — visibly its own tone, not the later series
    punching out the earlier one."""
    from textual.color import Color

    from app import _lines_chart_lines

    pal, _ = _aeon_palettes()
    c1, c2 = "#8A7CFF", "#F06AC0"
    out = _lines_chart_lines([("a", c1, [40, 40, 40, 40]), ("b", c2, [20, 20, 20, 20])], 6, 30, pal)
    bg = Color.parse(pal.bg)
    one = bg.blend(Color.parse(c1), 0.25)
    two = one.blend(Color.parse(c2), 0.25)
    tones: dict[tuple, int] = {}
    for t in out:
        for ch, st in _cells(t):
            if ch == "\u2588":
                rgb = Color.parse(st.split()[-1]).rgb[:3]
                which = (
                    "two"
                    if all(abs(x - y) <= 1 for x, y in zip(rgb, two.rgb[:3]))
                    else (
                        "one" if all(abs(x - y) <= 1 for x, y in zip(rgb, one.rgb[:3])) else "other"
                    )
                )
                assert which != "other", st
                tones[which] = tones.get(which, 0) + 1
    assert tones.get("one", 0) > 0 and tones.get("two", 0) > 0, tones


def test_axis_format_and_ticks():
    """AC Y1/Y4: compact tick format and the shared-scale tick rows
    (max on top, 0 on the baseline, half-max only when ≥ 4 rows)."""
    from app import _axis_labels, _fmt_axis

    assert _fmt_axis(999_999) == "1M"  # .1f rounding must not read "1000K"

    assert (_fmt_axis(0), _fmt_axis(650)) == ("0", "650")
    assert (_fmt_axis(5000), _fmt_axis(1349), _fmt_axis(2_500_000)) == ("5K", "1.3K", "2.5M")
    pal, _ = _aeon_palettes()
    pl = [t.plain for t in _axis_labels(1349.0, 6, 4, pal)]
    assert pl[0].strip() == "1.3K" and pl[-1].strip() == "0" and pl[3].strip() == "674"
    assert pl[1].strip() == "" and pl[2].strip() == ""
    assert all(len(p) == 5 for p in pl)  # gutter + trailing space, every row
    short = [t.plain.strip() for t in _axis_labels(900.0, 3, 3, pal)]
    assert short == ["900", "", "0"]  # no mid tick below 4 rows


def test_lines_chart_geometry_boundaries():
    """AC6: rows/width always exact — incl. rows=1 (single-row gradient
    divide) and width=1 — for empty, single-sample, all-zero and spiky
    histories."""
    from app import _lines_chart_lines

    pal, _ = _aeon_palettes()
    for hist in ([7.0], [0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 100.0], [1.0, 2.0]):
        for rows, width in ((1, 1), (1, 20), (6, 1), (2, 17)):
            out = _lines_chart_lines([("a", "#8A7CFF", hist)], rows, width, pal)
            assert len(out) == rows
            assert all(t.cell_len == width for t in out)
    assert _lines_chart_lines([("a", "#8A7CFF", [1, 2])], 0, 5, pal) == []


def test_series_hues_distinct_quiet_and_light():
    """AC4a: identity cycle is pairwise-distinct then deterministic-cycles;
    quiet collapses to fg; a light background gets a contrast blend."""
    from themes import SERIES_HUES, build_palette, get_theme, series_hue

    pal, quiet = _aeon_palettes()
    hues = [series_hue(i, pal) for i in range(len(SERIES_HUES) + 1)]
    assert len(set(hues[: len(SERIES_HUES)])) == len(SERIES_HUES)
    assert hues[len(SERIES_HUES)] == hues[0]
    assert series_hue(3, quiet) == quiet.fg
    light = build_palette(get_theme("tokyo-night-light"))
    assert series_hue(0, light) != SERIES_HUES[0]


def test_quiet_chart_is_monochrome():
    """AC5: quiet identity collapse happens at series_hue — the ONE hue
    assignment site (chart, legend, rows, sparks all read its output) — so
    a quiet palette yields fg for every slot and the chart renders
    monochrome."""
    from app import _lines_chart_lines
    from themes import SERIES_HUES, series_hue

    _, pq = _aeon_palettes()
    assert all(series_hue(i, pq) == pq.fg for i in range(len(SERIES_HUES)))
    # the renderer honors the (already collapsed) hue it is given
    out = _lines_chart_lines([("a", pq.fg, [3, 9, 4])], 5, 20, pq)
    line_sty = {
        st for t in out for _c, (ch, st) in enumerate(_cells(t)) if 0x2800 < ord(ch) < 0x2900
    }
    expected = {f"bold {pq.fg}".lower()}  # rich lowercases hex in styles
    assert {s.lower() for s in line_sty} == expected, line_sty


async def test_model_hues_stable_across_topology_changes(tmp_path: Path, monkeypatch):
    """AC4b: a model's hue is stable for the app's lifetime — a model
    appearing, dropping and re-appearing (even at the front of the unit
    order) never repaints the other models' lines."""
    from app import DGXTop, ServingBox

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[app]\npoll_interval = 5\nhistory_length = 25\n"
        '[[nodes]]\nlabel = "head"\nssh_target = "head"\n'
        'vllm_url = "http://192.0.2.10:8000"\n'
        '[[nodes]]\nlabel = "worker"\nssh_target = "worker"\n'
        'vllm_url = "http://192.0.2.11:8000"\nworker = true\n'
        '[[nodes]]\nlabel = "third"\nssh_target = "third"\n'
        'vllm_url = "http://192.0.2.12:8000"\nworker = true\n'
    )
    configure(cfg)

    def mk(label, name, worker=False, tok=100.0):
        u = _unit(label, worker=worker)
        u.model_name = name
        u.generation_tokens_total = tok
        return u

    a = mk("head", "model-a", tok=300.0)
    b = mk("worker", "model-b", worker=True, tok=200.0)
    c = mk("third", "model-c", worker=True, tok=100.0)
    units = [a, b]
    _stub(monkeypatch, units)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        await pilot.pause()
        serving = app.query_one("#serving", ServingBox)
        base = {m["name"]: m["color"] for m in serving._models}
        assert base["model-a"] != base["model-b"]

        units.append(c)
        app._update_ui()
        got = {m["name"]: m["color"] for m in serving._models}
        assert (got["model-a"], got["model-b"]) == (base["model-a"], base["model-b"]), got
        assert len({got[n] for n in got}) == 3, got

        units.remove(b)
        app._update_ui()
        units.insert(0, mk("worker", "model-b", worker=True, tok=400.0))
        app._update_ui()
        got = {m["name"]: m["color"] for m in serving._models}
        assert got["model-a"] == base["model-a"], got
        assert got["model-b"] == base["model-b"], got

        units.reverse()
        app._update_ui()
        got = {m["name"]: m["color"] for m in serving._models}
        assert (got["model-a"], got["model-b"]) == (base["model-a"], base["model-b"]), got


async def test_serving_title_row_legends_model_hues(tmp_path: Path, monkeypatch):
    """AC4c: the SERVING title row is the chart legend — each model name is
    styled in the exact hue its chart line and gen row carry."""
    import collections

    from app import DGXTop, ServingBox

    cfg = tmp_path / "config.toml"
    _config(cfg)
    configure(cfg)
    head = _unit("head")
    head.model_name = "model-a"
    head.generation_tokens_total = 300.0
    worker = _unit("worker", worker=True)
    worker.model_name = "model-b"
    worker.generation_tokens_total = 100.0
    _stub(monkeypatch, [head, worker])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        await pilot.pause()
        serving = app.query_one("#serving", ServingBox)
        app.history["gen-model-a"] = collections.deque([40 + (i * 7) % 60 for i in range(24)])
        app.history["gen-model-b"] = collections.deque([20 + (i * 5) % 40 for i in range(24)])
        app._update_ui()
        colors = {m["name"]: m["color"] for m in serving._models}
        title = serving.render().split("\n")[0]
        spans = {(title.plain[s.start : s.end], s.style) for s in title._spans}
        for name, hue in colors.items():
            assert (name, f"bold {hue}") in spans, (name, hue, spans)


@contextlib.asynccontextmanager
async def _two_model_app(tmp_path, monkeypatch, hist_a, hist_b):
    """Shared fixture: two served models with seeded gen histories."""
    import collections

    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    head = _unit("head")
    head.model_name = "model-a"
    head.generation_tokens_total = 300.0
    worker = _unit("worker", worker=True)
    worker.model_name = "model-b"
    worker.generation_tokens_total = 100.0
    worker.throughput_tok_s = hist_b[-1] if hist_b else 0.0
    head.throughput_tok_s = hist_a[-1] if hist_a else 0.0
    _stub(monkeypatch, [head, worker])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        await pilot.pause()
        if hist_a:
            app.history["gen-model-a"] = collections.deque(hist_a, maxlen=25)
        if hist_b:
            app.history["gen-model-b"] = collections.deque(hist_b, maxlen=25)
        app._update_ui()
        yield app


async def test_chart_y_axis_gutter_and_suppression(tmp_path: Path, monkeypatch):
    """AC Y1/Y2/Y3: the chart pane gains a right-aligned tick gutter
    (max / half / 0) without changing its total width, and the gutter is
    suppressed when nothing is plotted."""
    from app import ServingBox

    async with _two_model_app(tmp_path, monkeypatch, [100.0] * 24, [10.0] * 24) as app:
        serving = app.query_one("#serving", ServingBox)
        w = serving.content_size.width
        rows = serving.render().split("\n")
        assert all(t.cell_len == w for t in rows)
        content = rows[1:-1]  # strip box top/bottom borders
        chart = [t.plain[2:] for t in content[-app._chart_rows :]]  # strip "│ "
        assert len(chart) == app._chart_rows, (len(chart), app._chart_rows)
        assert chart[0].startswith("100 "), chart[0]
        assert chart[app._chart_rows // 2].startswith(" 50 "), chart[app._chart_rows // 2]
        assert chart[-1].startswith("  0 "), chart[-1]
    # all-zero throughput: no range to label — the pane renders unlabeled
    # (the fixture pins stub throughput to the seed tail, so an empty seed
    # really does leave only 0s in the history)
    async with _two_model_app(tmp_path, monkeypatch, [], []) as app:
        serving = app.query_one("#serving", ServingBox)
        blob = serving.render().plain
        assert not any(ln[2:].startswith(("0 ", " 0 ")) for ln in blob.split("\n")[1:-1]), blob


async def test_gen_rows_lead_with_hue_values_and_duo_fill(tmp_path: Path, monkeypatch):
    """AC Y8 (duo grammar): each gen row leads with that model's CURRENT
    total output in its own hue, then its duo spark — braille dots at the
    full hue — with a fill row beneath in the chart's translucent shade
    (hue blended 75% toward the background)."""
    from app import ServingBox, _palette_for
    from themes import blend_toward

    hist = [20 + (i * 7) % 50 for i in range(24)]  # last=31, avg=44, hi=69
    async with _two_model_app(tmp_path, monkeypatch, hist, hist) as app:
        serving = app.query_one("#serving", ServingBox)
        hue = {m["name"]: m["color"] for m in serving._models}["model-a"]
        pal = _palette_for(app)
        rows = serving.render().split("\n")
        gi = next(i for i, t in enumerate(rows) if "gen" in t.plain and "model-a" in t.plain)
        gen = rows[gi]
        assert "31" in gen.plain and "44" in gen.plain and "69" in gen.plain, gen.plain
        assert f"bold {hue}" in {s.style for s in gen._spans}, gen.plain
        # dots at the full hue share the row with the leading value
        assert any("\u2800" <= c <= "\u28ff" for c in gen.plain), gen.plain
        # the row beneath is the fill: solid blocks in the 75%-lighter shade
        fill = rows[gi + 1]
        assert "\u2588" in fill.plain, fill.plain
        expected = f"bold {blend_toward(hue, pal.bg, 0.75)}"
        assert expected in {s.style for s in fill._spans}, (
            expected,
            [s.style for s in fill._spans],
        )
        # model-b's fill carries ITS hue — the two light fills are distinct
        hue_b = {m["name"]: m["color"] for m in serving._models}["model-b"]
        assert blend_toward(hue_b, pal.bg, 0.75) != blend_toward(hue, pal.bg, 0.75)


# ─── review-hardening: hero honesty, order, hue hygiene, duo sparks ──


@contextlib.asynccontextmanager
async def _served_models_app(tmp_path, monkeypatch, specs, hists, size=(132, 44)):
    """Fixture: N served models with arbitrary gen histories.

    ``specs``: list of (model_name, generation_tokens_total, throughput).
    ``hists``: per-model history (or None to skip seeding), same length.
    """
    import collections

    from app import DGXTop

    _config_cluster(tmp_path / "config.toml", max(2, len(specs)))
    configure(tmp_path / "config.toml")

    units = []
    for i, (name, gen_total, tput) in enumerate(specs):
        u = _unit(f"n{i}", worker=bool(i))
        u.model_name = name
        u.generation_tokens_total = gen_total
        u.throughput_tok_s = tput
        units.append(u)
    _stub(monkeypatch, units)
    app = DGXTop()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.pause()
        for (name, _g, _t), h in zip(specs, hists):
            if h:
                app.history[f"gen-{name}"] = collections.deque(h, maxlen=25)
        app._update_ui()
        yield app


async def test_gen_hero_shows_rep_value_not_cluster_aggregate(tmp_path, monkeypatch):
    """AC6: one model name served by 2 endpoints — the hero reads the
    authoritative rep's own series, not the N× cluster aggregate."""
    from app import ServingBox

    hist = [20 + (i * 7) % 50 for i in range(24)]  # rep tail = 31
    specs = [("model-a", 300.0, 31.0), ("model-a", 100.0, 1000.0)]
    async with _served_models_app(tmp_path, monkeypatch, specs, [hist, hist]) as app:
        serving = app.query_one("#serving", ServingBox)
        assert len(serving._models) == 1  # one pane, not two
        gen_row = next(t for t in serving.render().split("\n") if t.plain.startswith("│ gen"))
        # pin the parsed hero TOKEN (leading value at the graph lane), not a
        # substring that larger numbers could contain
        m = re.search(r"gen\s+(\d+)", gen_row.plain)
        assert m and m.group(1) == "31", gen_row.plain
        # the aggregate (31 + 1000) must not leak into the hero or the tail
        assert "1031" not in gen_row.plain, gen_row.plain
        # the spark scales to the REP's series (peak 69 reaches the top dot
        # row); the N× aggregate scale would flatten it onto the baseline
        from app import _BRAILLE_BITS

        dot_rows = [
            row
            for ch in gen_row.plain
            if 0x2800 < ord(ch) <= 0x28FF
            for (_col, row), bit in _BRAILLE_BITS.items()
            if (ord(ch) - 0x2800) & bit
        ]
        # spread too: the rep series (20..69) spans dot rows; a single
        # self-scaled aggregate sample would be one flat row
        assert dot_rows and min(dot_rows) == 0 and max(dot_rows) >= 1, gen_row.plain


async def test_model_series_sorted_by_name(tmp_path, monkeypatch):
    """AC7: legend and series order is sorted by model name regardless of
    the unit encounter order."""
    from app import ServingBox

    hist = [10.0 + i for i in range(10)]
    specs = [("model-b", 300.0, 10.0), ("model-a", 200.0, 20.0)]
    async with _served_models_app(tmp_path, monkeypatch, specs, [hist, hist]) as app:
        serving = app.query_one("#serving", ServingBox)
        assert [m["name"] for m in serving._models] == ["model-a", "model-b"]
        title = serving.render().split("\n")[0].plain
        assert title.index("model-a") < title.index("model-b"), title


async def test_model_hue_slots_pruned_on_exit(tmp_path, monkeypatch):
    """AC8: a model that disappears releases its hue slot (and its gen
    history, mirroring the existing history pruning)."""
    hist = [10.0 + i for i in range(10)]
    specs = [("model-a", 300.0, 10.0), ("model-b", 200.0, 20.0)]
    async with _served_models_app(tmp_path, monkeypatch, specs, [hist, hist]) as app:
        assert len(app._model_hue) == 2
        # the live fleet stops hosting models: ONE flaky poll must NOT
        # repaint a live model — the slot survives a 3-poll grace window
        app.cluster = _cluster([_unit("head", hosted=False)])
        app._update_ui()
        assert len(app._model_hue) == 2, app._model_hue
        app._update_ui()
        app._update_ui()
        assert app._model_hue == {}, app._model_hue
        assert not [k for k in app.history if k.startswith("gen-")]


def test_fmt_axis_boundary_band():
    """AC9: the 999.5-999.99 band rounds into the K scale, like 1024."""
    from app import _fmt_axis

    assert _fmt_axis(0) == "0"
    assert _fmt_axis(650) == "650"
    assert _fmt_axis(999) == "999"
    assert _fmt_axis(999.5) == "1K"
    assert _fmt_axis(1024) == "1K"
    assert _fmt_axis(1349) == "1.3K"
    assert _fmt_axis(999_999) == "1M"
    assert _fmt_axis(1_500_000) == "1.5M"


async def test_nine_models_exhaust_hue_cycle_without_crash(tmp_path, monkeypatch):
    """AC13: more live models than hues — the cycle repeats
    deterministically (slot = len(slots) - len(SERIES_HUES) for the 9th) and
    the pane renders."""
    from app import ServingBox
    from themes import series_hue

    specs = [(f"model-{i}", 100.0 * (9 - i), 10.0 + i) for i in range(9)]
    hists = [[10.0 + i] * 6 for i in range(9)]
    async with _served_models_app(tmp_path, monkeypatch, specs, hists) as app:
        serving = app.query_one("#serving", ServingBox)
        assert len(serving._models) == 9
        pal = serving.render()  # must not raise
        assert pal is not None
        slot_hues = [series_hue(i, _palette_of(app)) for i in range(9)]
        colors = [m["color"] for m in serving._models]
        assert colors[:8] == slot_hues[:8]
        assert colors[8] == slot_hues[0]  # 9th reuses slot 0 (len(slots)-8)
        assert colors[8] == colors[0]


def _palette_of(app):
    from app import _palette_for

    return _palette_for(app)


def test_chart_and_spark_helpers_tolerate_non_finite_samples():
    """AC14: NaN/Inf samples are dropped at the door — the helpers are
    total over their input (no NaN scale, no divmod-on-NaN crash)."""
    from app import _lines_chart_lines, _spark_duo_lines
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    series = [("m", "#8A7CFF", [float("nan"), 5.0, float("inf"), 0.0])]
    rows = _lines_chart_lines(series, 4, 20, pal)
    assert len(rows) == 4 and all(t.cell_len == 20 for t in rows)
    assert all("nan" not in t.style for t in rows), [t.style for t in rows]
    duo = _spark_duo_lines([float("nan"), 5.0, float("inf")], "#8A7CFF", 12, pal)
    assert len(duo) == 2 and all(t.cell_len == 12 for t in duo)
    all_nan = _spark_duo_lines([float("nan")], "#8A7CFF", 8, pal)
    assert len(all_nan) == 2 and all(t.cell_len == 8 for t in all_nan)
    assert all(not c for t in all_nan for c in t.plain if c == "█")


def test_spark_duo_boundary_inputs():
    """AC11: empty / single-sample / all-equal / all-zero / width-1 inputs
    all render exactly 2 rows x width without crashing."""
    from app import _spark_duo_lines
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    for data, w in (([], 10), ([5.0], 10), ([5.0, 5.0, 5.0], 10), ([1.0], 1), ([0.0, 0.0], 12)):
        rows = _spark_duo_lines(list(data), "#8A7CFF", w, pal)
        assert len(rows) == 2, (data, w)
        assert all(t.cell_len == w for t in rows), (data, w)
