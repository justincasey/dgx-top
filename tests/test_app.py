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
    """Give the charts real history so sparklines/area chart populate."""
    import collections

    app.history["throughput"] = collections.deque([40 + (i * 7) % 60 for i in range(24)], maxlen=25)
    app.history["prompt-throughput"] = collections.deque(
        [120 + (i * 11) % 90 for i in range(24)], maxlen=25
    )
    app.history["kv-usage-head"] = collections.deque(
        [18 + (i * 3) % 15 for i in range(24)], maxlen=25
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


async def test_serving_hero_on_top_with_node_grid_below(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        serv = app.query_one("#serving", ServingBox).region
        n0 = app.query_one("#node-0", NodeBox).region
        n1 = app.query_one("#node-1", NodeBox).region
        # SERVING is a full-width hero on top; the node grid is below.
        assert serv.x == 0 and serv.width == 132
        assert n0.y >= serv.bottom + 1
        assert n0.y == n1.y  # both nodes share one grid row (2 columns)
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
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        seen = []
        for h in (44, 30, 20, 12, 8):
            await _resize(pilot, 132, h)
            tier = "floor" if app.floor else ("rail" if app.rail else app.density)
            seen.append(tier)
        order = ["roomy", "dense", "compact", "rail", "floor"]
        ranks = [order.index(t) for t in seen]
        assert ranks == sorted(ranks), seen  # monotonically denser
        assert seen[0] == "roomy" and seen[-1] == "floor"


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
        chart_rows = [ln for ln in lines if any(c in ln for c in "▁▂▃▄▅▆▇█")]
        # gen/prompt/kv sparklines + a multi-row area chart
        assert len(chart_rows) >= 3


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
    """AC: gen/prompt/kv/kv% graphs share one width and tails align; blank
    spacer rows separate the graph rows; nothing overflows the box."""
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
        graph_glyphs = set("\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588\u2593\u2591")
        starts, lens = set(), set()
        for label, entries in by_label.items():
            i, ln = entries[0]
            cols = [j for j, ch in enumerate(ln) if ch in graph_glyphs]
            assert cols, f"no graph glyphs on {label!r}"
            starts.add(min(cols))
            lens.add(max(cols) - min(cols) + 1)
            assert cols == list(range(min(cols), max(cols) + 1)), f"ragged graph {label!r}"
        assert len(lens) == 1, f"graph lengths differ: {lens}"
        assert len(starts) == 1, f"graph starts differ: {starts}"
        # blank spacer row between the gen/prompt/kv graph rows
        gen_i = by_label["gen    "][0][0]
        prompt_i = by_label["prompt "][0][0]
        kv_i = by_label["kv     "][0][0]
        assert prompt_i - gen_i == 2 and kv_i - prompt_i == 2

        def blank(ln: str) -> bool:
            return ln[1:-1].strip() == ""

        kvp_i = by_label["kv%    "][0][0]
        assert kv_i - prompt_i == 2 and kvp_i - kv_i == 2
        assert (
            blank(interior[gen_i + 1])
            and blank(interior[prompt_i + 1])
            and blank(interior[kv_i + 1])
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
                    assert wid.region.bottom <= h, (n, w, h, wid.id, wid.region.bottom)
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
    from app import DGXTop

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    async with app.run_test(size=(180, 60)) as pilot:
        await pilot.pause()
        seen = []
        for h in (60, 45, 35, 25, 15, 8):
            await _resize(pilot, 180, h)
            tier = "floor" if app.floor else ("rail" if app.rail else app.density)
            seen.append(tier)
        order = ["roomy", "dense", "compact", "rail", "floor"]
        ranks = [order.index(t) for t in seen]
        assert ranks == sorted(ranks), seen  # monotonically denser
        assert seen[0] == "roomy" and seen[-1] == "floor"


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
