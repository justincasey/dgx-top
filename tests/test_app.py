"""Behavioral tests for the tiling dashboard (src/app.py)."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from config import configure, load_config
from stats import ClusterStats, SparkUnitStats, TopologyInfo

# ─── fixtures ────────────────────────────────────────────────────────


def _config(path: Path, theme: str | None = None, n: int = 2, chart: str | None = None) -> None:
    lines = ["[app]", "poll_interval = 5", "history_length = 25"]
    if theme:
        lines.append(f'theme = "{theme}"')
    if chart:
        lines.append(f'chart = "{chart}"')
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
    compact → rail → floor sequence, monotonically, at the calibrated
    heights."""
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    rank = {"roomy": 0, "compact": 1, "rail": 2, "floor": 3}
    async with app.run_test(size=(90, 44)) as pilot:
        await pilot.pause()
        seen = []
        for h in (44, 36, 28, 24, 20, 17, 14, 12, 10, 8):
            await _resize(pilot, 90, h)
            tier = "floor" if app.floor else ("rail" if app.rail else app.density)
            seen.append(tier)
    assert seen[0] == "roomy" and seen[-1] == "floor", seen
    assert [rank[t] for t in seen] == sorted(rank[t] for t in seen), seen
    assert set(seen) == {"roomy", "compact", "rail", "floor"}, seen


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
        # the kv% row is the pane's single-hue meter; it carries the reading
        # as its value, in the accent hue, and never ramps by position
        pct_row = next(ln for ln in lines if ln[2:].startswith("kv%"))
        assert re.search(r"\d+%", pct_row), pct_row
        assert set(pct_row[11:33]) <= {"█", "▓"}, pct_row


async def test_node_meters_fill_the_tile_interior(tmp_path: Path, monkeypatch):
    """Meters span the full tile interior: no trailing blank cells before the
    right border (the old 20-cell cap left dead space at wider tiles)."""
    from app import DGXTop, NodeBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        node = app.query_one("#node-0", NodeBox).render().plain
        lines = node.split("\n")
        for row_idx in (2, 4):  # gpu meter, mem meter (top, gpu, meter, mem, meter)
            content = lines[row_idx][2:-2]  # strip border + one pad cell each side
            assert content == content.rstrip(), (row_idx, repr(lines[row_idx]))
            assert len(content) == len(lines[1][2:-2]), (row_idx, len(content))


# ─── AC6: serving area chart ─────────────────────────────────────────


async def test_serving_area_chart_present(tmp_path: Path, monkeypatch):
    from app import CHART_ROWS, DGXTop, ServingBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        lines = app.query_one("#serving", ServingBox).render().plain.split("\n")
        assert not any("last 24 samples" in ln for ln in lines)  # chart label removed
        # series sparklines (blocks) + the fixed four-row ramp area footer
        assert any(c in ln for ln in lines for c in "▁▂▃▄▅▆▇█")
        chart = lines[-(CHART_ROWS + 2) : -2]
        assert len(chart) == CHART_ROWS
        assert any(c in "▁▂▃▄▅▆▇█" for c in "\n".join(chart)), chart
        assert not any("\u2800" <= c <= "\u28ff" for c in "\n".join(chart)), chart


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
        # Controls left, in display order, with single-word labels.
        assert plain.startswith("s scan   +/- poll   r refresh   c clock   t theme   q quit")
        # Working stats right, fleet chip flush.
        assert "● 2/2" in plain
        assert plain.rstrip().endswith("● 2/2")
        assert "°C" not in plain and "W" not in plain
        assert " 2 3" not in plain
        # The centred model/topology title is gone: the serving pane names it.
        assert "Qwen3.6-27B-Instruct" not in plain and "DUAL" not in plain
        # KV moved to the SERVING kv% row: the bar carries live rates only.
        assert "KV" not in plain
        assert "tok/s" in plain
        assert len(plain) == wb.content_size.width


async def test_waybar_legend_sheds_before_stats(tmp_path: Path, monkeypatch):
    from app import DGXTop, Waybar

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        wb = app.query_one("#waybar", Waybar)
        expected = {
            (132, 40): "s scan   +/- poll   r refresh   c clock   t theme   q quit",
            (80, 40): "s scan   +/- poll   r refresh   c clock   t theme",
            (60, 40): "s scan   +/- poll   r refresh",
            (40, 16): "",  # fewer than two entries fit: the legend hides entirely
        }
        for (w, h), legend in expected.items():
            await _resize(pilot, w, h)
            text = wb.render().plain
            assert len(text) == wb.content_size.width, (w, text)
            # The legend never costs the stats a cell: both survive every
            # width, and the fleet chip stays flush right.
            assert "2400 tok/s" in text, (w, text)
            assert text.rstrip().endswith("● 2/2"), (w, text)
            assert text.startswith(legend), (w, text)
            if legend:
                # A two-cell gap separates the legend from the stats.
                tail = text[len(legend) :]
                assert tail.startswith("  ") and tail.strip(), (w, text)


async def test_waybar_legend_styles_hotkeys(tmp_path: Path, monkeypatch):
    from app import DGXTop, Waybar
    from themes import build_palette

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        pal = build_palette(app.current_theme)
        text = app.query_one("#waybar", Waybar).render()
        plain = text.plain
        word = plain.index("scan")
        key_style = " ".join(_style_at(text, word - 2)).lower()
        label_style = " ".join(_style_at(text, word)).lower()
        assert "underline" in key_style and pal.cyan.lower() in key_style, key_style
        assert pal.dim.lower() in label_style and "underline" not in label_style, label_style


def test_waybar_hints_are_live_single_word_bindings():
    from app import DGXTop, Waybar

    # `+`/`-` are Textual's `plus`/`minus` key names.
    key_names = {"+": "plus", "-": "minus"}
    bound = {b.key for b in DGXTop.BINDINGS} | {b.key.split(",")[0] for b in DGXTop.BINDINGS}
    for key, label in Waybar.HINTS:
        assert label.isalpha() and " " not in label, label
        for glyph in key.split("/"):
            assert key_names.get(glyph, glyph) in bound, key


# ─── AC8: CPU frequency renders MHz ──────────────────────────────────


def test_fmt_freq_renders_mhz():
    from app import _fmt_freq

    assert _fmt_freq(2405.0) == "2405MHz"
    assert _fmt_freq(2808.0) == "2808MHz"
    assert _fmt_freq(999.0) == "999MHz"
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
        assert "MHz" in gpu_line and "GHz" not in gpu_line


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
        assert "tok/s" in wb.plain
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


async def test_deploy_row_names_the_engine_and_never_clips(tmp_path: Path, monkeypatch):
    """The engine family, the TP width and the model's quantisation are one
    ``deploy`` row. At every width above the fused-grammar floor the row fits
    the pane's interior (``_box_lines`` ellipsizes anything wider), and a
    mixed cluster names no single engine."""
    from app import DGXTop, ServingBox, _box_lines, _palette_for

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    head = _unit("head")
    head.model_name = "qwen3.8-flash-next-122b-a10b"
    head.generation_tokens_total = 264348.0
    worker = _unit("worker", worker=True)
    worker.model_name = "NVIDIA-Nemotron-3.5-Lightning-49B"
    worker.model_source = "sglang"
    worker.generation_tokens_total = 0.0
    worker.model_metrics = False
    _stub(monkeypatch, [head, worker])
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        app._update_ui()
        box = app.query_one("#serving", ServingBox)
        pal = _palette_for(app)
        for w in range(52, 161):
            rows = box._rows(pal, w, False)
            assert all(r.cell_len <= w - 4 for r in rows), (w, [r.plain for r in rows])
            painted = [
                line.plain[2:-2]  # the interior: `_fit` has already run
                for line in _box_lines(w, [("^", ""), (" serving", "")], None, rows, False, pal)
            ]
            deploy = next(ln for ln in painted if ln.startswith("deploy"))
            # vllm + sglang are both served: no single engine is claimed.
            assert "mixed" in deploy, (w, deploy)
        # One engine, one quant tag.
        _stub(monkeypatch, [head])
        app2 = DGXTop()
        async with app2.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app2._update_ui()
            interior = [
                ln[2:-2]
                for ln in app2.query_one("#serving", ServingBox).render().plain.splitlines()
            ]
            deploy = next(ln for ln in interior if ln.startswith("deploy"))
            assert deploy.startswith("deploy  vllm"), deploy


async def test_unknown_kv_renders_as_no_reading_not_zero(tmp_path: Path, monkeypatch):
    """A node reached over SGLang's load API states its used count and no
    capacity, so its fill is unknown. The pane must say so: printing the
    stats default (0%) wrote a confident "empty pool" on a node that reports
    4096 used tokens — the same wrong-number class as the fabricated 100%
    denominator this pass removed."""
    from app import DGXTop, ServingBox, Waybar

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    unit = _unit("head")
    unit.kv_cache_pct = -1.0  # the collector's "no reading" sentinel
    unit.kv_total_tokens = 0  # /get_load states no capacity
    unit.kv_cache_used_tokens = 4096
    unit.model_metrics = False
    unit.model_source = "sglang"
    _stub(monkeypatch, [unit])
    for size in ((160, 48), (64, 30)):
        app = DGXTop()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._update_ui()
            box = app.query_one("#serving", ServingBox)
            painted = box.render().plain.splitlines()
            interior = [ln[2:-2] for ln in painted]
            # The meta tab carries capacity, so it is the em dash — not a
            # "0 tok" nobody measured.
            tab = interior[0].split("┤ ")[-1].rstrip(" ├")
            assert tab == "—", (size, interior[0])
            # The kv row publishes the used count and no denominator; the kv%
            # row says there is no reading rather than painting the sentinel.
            kv_row = next(ln for ln in interior if ln.startswith("kv "))
            assert "4K" in kv_row and "/" not in kv_row, (size, kv_row)
            pct_row = next((ln for ln in interior if ln.startswith("kv%")), "")
            assert "%" not in pct_row[9:], (size, pct_row)
            chip = app.query_one("#waybar", Waybar).render().plain
            assert "KV" not in chip, chip
            # The kv spark plots this history: an unknown fill must not seed
            # a sample the node never reported.
            assert all(v >= 0 for v in app.history["kv-usage-head"]), app.history["kv-usage-head"]


async def test_kv_history_follows_the_unit_with_a_reading(tmp_path: Path, monkeypatch):
    """Mixed cluster: the first hosted unit is load-only (states no fill and
    never records), the second reports a real one. The kv spark must plot
    the same pool the headline percentage comes from — keying the history on
    hosted_units[0] fed the chart a buffer that stayed empty forever while
    the headline showed the other unit's fill."""
    from app import DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    head = _load_only(_unit("head"))
    # _load_only clears the token counters but not the fill: a load-only
    # endpoint states no KV reading at all.
    head.kv_cache_pct = -1.0
    head.kv_total_tokens = 0
    head.kv_cache_used_tokens = 0
    worker = _unit("worker", worker=True)
    _stub(monkeypatch, [head, worker])
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        app._update_ui()
        # The kv spark must PLOT the worker's series, not just record it:
        # kv_key picks which history buffer feeds the pane, and history
        # itself is recorded under every hosted label regardless.
        from app import ServingBox

        serving = app.query_one("#serving", ServingBox)
        assert serving._kv_data, serving._kv_data
        assert not app.history["kv-usage-head"], app.history["kv-usage-head"]


async def test_kv_spark_clears_when_the_reading_unit_loses_its_fill(tmp_path: Path, monkeypatch):
    """Poll 1: the only hosted unit reports a fill and the spark records it.
    Poll 2: the same unit's usage gauge is rejected (pct -1). The headline
    flips to "KV —" and the spark must plot nothing — skipping the sample
    but keeping the buffer (or freezing on the falsy kv_history guard)
    painted a fill nobody reported any more."""
    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml", n=1)
    configure(tmp_path / "config.toml")
    head = _unit("head")
    _stub(monkeypatch, [head])
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        app._update_ui()
        serving = app.query_one("#serving", ServingBox)
        assert serving._kv_data, serving._kv_data
        # Poll 2 (and 3): the usage gauge is rejected — the cleared buffer
        # must reach the pane, and stay cleared.
        head.kv_cache_pct = -1.0
        app._update_ui()
        assert serving._kv_data == [], serving._kv_data
        # The kv% row agrees: no reading, not the stale 32%.
        interior = [
            ln[2:-2] for ln in app.query_one("#serving", ServingBox).render().plain.splitlines()
        ]
        pct_row = next((ln for ln in interior if ln.startswith("kv%")), "")
        assert "%" not in pct_row[9:], pct_row
        app._update_ui()
        assert serving._kv_data == [], serving._kv_data


async def test_unknown_used_paints_capacity_without_a_numerator(tmp_path: Path, monkeypatch):
    """A unit whose usage gauge was rejected states its pool but no fill:
    the kv row must not fabricate "0/3.8M tok" — capacity alone, dim —
    beside a kv% row that correctly reads "no reading"."""
    from app import DGXTop, ServingBox

    _config(tmp_path / "config.toml", n=1)
    configure(tmp_path / "config.toml")
    head = _unit("head")
    head.kv_cache_pct = -1.0
    head.kv_total_tokens = 3_800_000
    head.kv_cache_used_tokens = 0
    _stub(monkeypatch, [head])
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        app._update_ui()
        box = app.query_one("#serving", ServingBox)
        painted = box.render().plain.splitlines()
        interior = [ln[2:-2] for ln in painted]
        kv_row = next(ln for ln in interior if ln.startswith("kv "))
        assert "3.8M" in kv_row and "0/" not in kv_row, (kv_row,)
        # Neither figure stated (used at the cluster's -1 sentinel, no
        # capacity): the tail is the em dash, never a literal "-1".
        bare = _unit("head")
        bare.kv_cache_pct = -1.0
        bare.kv_total_tokens = 0
        bare.kv_cache_used_tokens = 0
        _stub(monkeypatch, [bare])
        app2 = DGXTop()
        async with app2.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app2._update_ui()
            box2 = app2.query_one("#serving", ServingBox)
            interior2 = [ln[2:-2] for ln in box2.render().plain.splitlines()]
            kv_row2 = next(ln for ln in interior2 if ln.startswith("kv "))
            assert "-1" not in kv_row2 and "—" in kv_row2, (kv_row2,)


def _load_only(unit, running: int = 2, waiting: int = 1):
    """A node exactly as ``poll_unit`` leaves it when ``/metrics`` 404s and
    the load API answers: real requests and KV, no token counter at all."""
    unit.model_metrics = False
    unit.model_source = "sglang"
    unit.throughput_tok_s = 0.0
    unit.prompt_throughput_tok_s = 0.0
    unit.generation_tokens_total = 0.0
    unit.prompt_tokens_total = 0.0
    unit.ttft_p50_ms = 0.0
    unit.ttft_p95_ms = 0.0
    unit.kv_prefix_hit_rate = -1.0
    unit.requests_running = running
    unit.requests_waiting = waiting
    return unit


async def test_load_only_cluster_paints_no_rate_and_keeps_no_series(tmp_path: Path, monkeypatch):
    """AC3/AC4/AC5: SGLang without ``--enable-metrics`` states no token
    counter, so the 0 it leaves on every rate field is an absent measurement,
    not an idle server. Recording it drew a flat line and a ``0 tok/s`` tail
    on a server generating ~60 tok/s, and retaining earlier samples painted a
    rate nobody was reporting any more."""
    from app import DGXTop, ServingBox, Waybar

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    units = [
        _load_only(_unit("head"), running=2, waiting=1),
        _load_only(_unit("worker", worker=True), running=3, waiting=0),
    ]
    _stub(monkeypatch, units)
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        # A counters-bearing poll preceded this one: those samples must be
        # dropped, or the pane presents them as the current rate.
        app.history["throughput"].extend([900.0, 950.0])
        app.history["prompt-throughput"].extend([1900.0])
        app.history["gen-Qwen3.6-27B-Instruct"].extend([900.0, 950.0])
        app._update_ui()

        assert app.cluster.throughput_measured is False
        assert list(app.history["throughput"]) == []
        assert list(app.history["prompt-throughput"]) == []
        assert list(app.history["gen-Qwen3.6-27B-Instruct"]) == []

        interior = [
            ln[2:-2] for ln in app.query_one("#serving", ServingBox).render().plain.splitlines()
        ]
        # The gen row is always present but states no rate (an absent series
        # must not be painted as a flat zero), and the empty chart stays blank.
        gen = next(ln for ln in interior if ln.startswith("gen "))
        assert "— tok/s" in gen and "0 tok/s" not in gen, gen
        chart = interior[-(app._chart_rows + 2) : -2]
        assert not any("\u2800" <= c <= "\u28ff" for c in "\n".join(chart)), chart
        assert not any(c in "▁▂▃▄▅▆▇█" for c in chart[0]), chart
        assert "— tok/s" in next(ln for ln in interior if ln.startswith("prompt ")), interior
        # the load API's own readings still land
        assert any(ln.startswith("requests") and "2r · 1w" in ln for ln in interior), interior
        assert "—" in next(ln for ln in interior if ln.startswith("cache ")), interior
        assert "1.2M/3.8M tok" in next(ln for ln in interior if ln.startswith("kv ")), interior
        assert "%" in next(ln for ln in interior if ln.startswith("kv%")), interior

        chip = app.query_one("#waybar", Waybar).render().plain
        assert "— tok/s" in chip and "0 tok/s" not in chip, chip

        # positive control (AC6): the same pane records and paints a rate as
        # soon as a counter-bearing unit is there
        for u in units:
            u.model_metrics = True
            u.throughput_tok_s = 42.0
            u.prompt_throughput_tok_s = 7.0
        app._update_ui()
        assert app.cluster.throughput_measured is True
        assert list(app.history["throughput"]) == [84.0]
        assert list(app.history["prompt-throughput"]) == [14.0]
        assert "84 tok/s" in app.query_one("#waybar", Waybar).render().plain
        measured = [
            ln[2:-2] for ln in app.query_one("#serving", ServingBox).render().plain.splitlines()
        ]
        # The counter-bearing poll feeds the chart: a block-glyph area trace
        # appears with the real rate in the gen row.
        chart = measured[-(app._chart_rows + 2) : -2]
        blob = "\n".join(chart)
        assert any(c in "▁▂▃▄▅▆▇█" for c in blob), chart
        assert "84 tok/s" in next(ln for ln in measured if ln.startswith("gen "))


async def test_two_load_only_models_paint_an_unknown_prompt_rate(tmp_path: Path, monkeypatch):
    """The multi-model grammar averages the shared prompt row; an absent
    series averages to 0.0, which is the same manufactured rate in a second
    place — and the engine is only nameable when the models agree on one."""
    from app import DGXTop, ServingBox

    _config_cluster(tmp_path / "config.toml", 2)
    configure(tmp_path / "config.toml")
    a = _load_only(_unit("node-1"))
    a.model_name = "model-a"
    b = _load_only(_unit("node-2", worker=True))
    b.model_name = "model-b"
    _stub(monkeypatch, [a, b])
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        app._update_ui()

        interior = [
            ln[2:-2] for ln in app.query_one("#serving", ServingBox).render().plain.splitlines()
        ]
        assert len(app.query_one("#serving", ServingBox)._models) == 2
        prompt = next(ln for ln in interior if ln.startswith("prompt "))
        assert "— tok/s" in prompt, interior
        assert "0 tok/s" not in prompt, prompt
        # The gen row is always present (the reference grammar) but states no
        # rate, and the absent series leaves the chart an honest empty grid.
        gen = next(ln for ln in interior if ln.startswith("gen "))
        assert "— tok/s" in gen, gen
        assert not any(c in "▁▂▃▄▅▆▇█" for c in gen), gen
        chart = interior[-(app._chart_rows + 2) : -2]
        assert not any("\u2800" <= c <= "\u28ff" for c in "\n".join(chart)), chart
        assert not any(c in "▁▂▃▄▅▆▇█" for c in chart[0]), chart


async def test_per_model_rows_bound_the_name_column(tmp_path: Path, monkeypatch):
    """The gen row is not the only row spending the name column: the per-model
    requests and ttft rows share it. Unaccounted for, a 20-cell name pushed
    `p50 … · p95 … !!` past the interior and `_fit` ate the p95 tail, hiding a
    stalled latency."""
    from app import DGXTop, ServingBox, _palette_for

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    head = _unit("head")
    worker = _unit("worker", worker=True)
    _stub(monkeypatch, [head, worker])
    app = DGXTop()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.pause()
        box = app.query_one("#serving", ServingBox)
        pal = _palette_for(app)
        box.update_models(
            [
                {
                    "name": name,
                    "color": pal.accent,
                    "gen": [0.0, 0.0],  # observed but idle: an empty spark scale
                    "req": 2,
                    "wait": 1,
                    "ttft_p50_ms": 700.0,
                    "ttft_p95_ms": 20500.0,
                    "source": source,
                }
                for name, source in (
                    ("NVIDIA-Nemotron-3.5-Lightning-49B", "vllm"),
                    ("gpt-oss-20b", "sglang"),
                )
            ]
        )
        for w in range(52, 121):
            rows = box._rows(pal, w, False)
            over = [(r.cell_len, r.plain) for r in rows if r.cell_len > w - 4]
            assert not over, (w, over)
        # Every per-model row keeps its tail in full at the narrowest width.
        painted = [r.plain for r in box._rows(pal, 52, False)]
        assert len([ln for ln in painted if ln.startswith("ttft")]) == 2
        assert all("p95" in ln for ln in painted if ln.startswith("ttft")), painted
        assert all("waiting" in ln for ln in painted if ln.startswith("requests")), painted


def test_legend_tag_costs_only_slack():
    """The title legend's engine tag is secondary: it may be dropped for
    width, but adding it must never evict a model name or the KV meta tab,
    and cells never buy it back once it fits."""
    from app import _box_lines, _legend_segs
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    models = [
        {"name": "qwen3.8-flash-next", "color": pal.accent, "source": "vllm"},
        {"name": "Nemotron-3.5-Lightning", "color": pal.ok, "source": "sglang"},
    ]
    head = [("^", ""), (" ", ""), ("serving", ""), (" ", "")]
    rtab = [("1.2M tok", "")]
    names = [m["name"] for m in models]
    # The width `_box_lines` really reserves for the tab, not a guess: one
    # cell short here moves the whole legend boundary.
    rtab_w = sum(len(t) for t, _ in rtab)
    shown = []
    for w in range(20, 201):
        segs = _legend_segs(models, pal, head_w=10, rtab_w=rtab_w, width=w)
        tagged = _box_lines(w, head + segs, rtab, [], False, pal)[0].plain
        # Same names and separators, no engine tags: the legend the tag may
        # only ever be added beside, never at the expense of.
        untagged = _box_lines(w, head + [(" · ".join(names), "")], rtab, [], False, pal)[0].plain
        for name in names:
            if name in untagged:
                assert name in tagged, (w, name, tagged)
        if "tok" in untagged:
            assert "tok" in tagged, (w, tagged)
        shown.append(any("[sglang]" in t for t, _ in segs))
    for w, (before, after) in enumerate(zip(shown, shown[1:])):
        assert not (before and not after), w
    assert any(shown), "tag never fit at any legend width"


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
        pct_row = next(ln for ln in serv.plain.split("\n") if ln[2:].startswith("kv%"))
        assert "%" in pct_row[35:], pct_row


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
        kv_row = next(ln for ln in serv.plain.split("\n") if "kv " in ln)
        assert kv_row.strip() != "", kv_row


async def test_spark_treatment_renders(tmp_path: Path, monkeypatch):
    from app import ServingBox

    async with _treated_app(tmp_path, monkeypatch, "spark") as (app, pilot):
        await pilot.pause()
        _seed_history(app)
        row, _, _ = _meter_row(app)
        spark_chars = set("▁▂▃▄▅▆▇█")
        assert any(ch in spark_chars for ch in row), row
        serv = app.query_one("#serving", ServingBox).render()
        pct_row = next(ln for ln in serv.plain.split("\n") if ln[2:].startswith("kv%"))
        assert set(pct_row[11:33]) <= {"█", "▓"}, pct_row


def test_spark_meter_scales_against_its_value_not_the_series_extremes():
    """A meter block's height must mean the value it prints.

    Live regression: a node's GPU sat at a steady 96% and its spark meter drew
    the LOWEST block across the row — min…max normalization mapped a flat
    series onto the bottom, so the tile read as "no GPU usage" beside its own
    96% label while the other nodes (one unit of jitter) drew █."""
    from app import _meter_line, _spark_line
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))

    def steady(pct: float) -> str:
        return _meter_line("spark", pct, 20, pal, pal.blue, [pct] * 40).plain

    assert steady(96.0) == "█" * 20
    assert steady(50.0) == "▄" * 20
    assert steady(0.0) == "▁" * 20
    # A brand-new/short history fills the row at that height, not one glyph.
    assert _meter_line("spark", 96.0, 20, pal, pal.blue, [96.0]).plain == "█" * 20
    # A series with no percent domain (token rate) keeps its own 0…peak scale.
    assert _spark_line([30, 60], pal.blue, 2).plain == "▄█"
    # Non-finite samples never reach the glyph arithmetic.
    assert _spark_line([float("nan"), float("inf")], pal.blue, 4).plain == " " * 4


async def test_steady_gpu_history_does_not_render_as_an_empty_meter(tmp_path: Path, monkeypatch):
    import collections

    async with _treated_app(tmp_path, monkeypatch, "spark") as (app, pilot):
        await pilot.pause()
        app.history["gpu-head"] = collections.deque([96.0] * 25, maxlen=25)
        app._update_ui()
        await pilot.pause()
        row, _, _ = _meter_row(app)
        assert "▁" not in row, row  # not the empty-looking bottom row
        cells = [ch for ch in row if ch in "▁▂▃▄▅▆▇█"]
        assert cells and cells.count("█") >= len(cells) - 2, row


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
    """The reference column grammar: a 9-cell label gutter, a 22-cell graph,
    a two-space gap, then the value — so every value in the pane starts on
    the same cell. The gen, prompt, kv and kv% rows are always present."""
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
        interior = box.render().plain.split("\n")[1:-1]  # strip the border rules
        assert all(len(ln) == width for ln in interior), "row not padded to box width"
        rows = {}
        for ln in interior:
            for label in ("gen", "prompt", "kv", "kv%"):
                if ln[2:].startswith(label.ljust(9)):
                    rows[label] = ln
        assert set(rows) == {"gen", "prompt", "kv", "kv%"}, rows.keys()
        blocks = "▁▂▃▄▅▆▇█"
        for label in ("gen", "prompt", "kv"):
            ln = rows[label]
            graph = ln[11:33]
            assert len(graph) == 22, (label, repr(ln))
            assert all(ch in blocks for ch in graph), (label, graph)
            assert ln[33:35] == "  ", (label, repr(ln[33:35]))
            assert ln[35:].strip(), (label, "value column empty")
        # The kv% row's graph is a single-hue meter, not a sparkline.
        meter = rows["kv%"][11:33]
        assert set(meter) <= {"█", "▓"}, meter
        assert rows["kv%"][35:].strip().startswith("32")
        # One blank spacer row separates the graph block from the meta rows.
        gen_i = interior.index(rows["gen"])
        assert interior[gen_i + 4][1:-1].strip() == ""
        assert interior[gen_i + 5][1:].strip().startswith("requests")


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
    """A 12-node cluster steps through the whole ladder as height shrinks:
    the tiled roomy tier, then compact, rail and floor."""
    from app import DGXTop

    _config_cluster(tmp_path / "config.toml", 12)
    configure(tmp_path / "config.toml")
    _stub_n(monkeypatch, 12)
    app = DGXTop()
    rank = {"roomy": 0, "compact": 1, "rail": 2, "floor": 3}
    async with app.run_test(size=(180, 60)) as pilot:
        await pilot.pause()
        seen = []
        for h in (60, 50, 44, 40, 36, 32, 29, 24, 20, 16, 8):
            await _resize(pilot, 180, h)
            tier = "floor" if app.floor else ("rail" if app.rail else app.density)
            seen.append(tier)
    assert seen[0] == "roomy" and seen[-1] == "floor", seen
    assert [rank[t] for t in seen] == sorted(rank[t] for t in seen), seen
    assert set(seen) == {"roomy", "compact", "rail", "floor"}, seen


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
        await _resize(pilot, 132, 16)
        assert app.density == "compact" and not app.rail and not app.floor
        assert app._chart_rows == 0, app._chart_rows  # the reference compact drops the chart
        serv_lines = app.query_one("#serving", ServingBox).render().plain.split("\n")
        # the gen and kv spark rows survive; prompt / kv% / the chart do not
        assert any(ln[2:].startswith("gen") for ln in serv_lines), serv_lines
        assert any(ln[2:].startswith("kv ") for ln in serv_lines), serv_lines
        assert not any(ln[2:].startswith("prompt") for ln in serv_lines), serv_lines
        assert not any(ln[2:].startswith("kv%") for ln in serv_lines), serv_lines
        node = app.query_one("#node-0", NodeBox).render().plain
        assert "73%" in node and "52%" in node and "50%" in node
        assert "■" in node, "core grid survives compact"
        assert not any(c in node for c in "█▓"), "the compact card drops its meters"
        assert "roce" not in node, "RoCE drops with the meters"
        assert app.screen.max_scroll_y == 0


async def test_serving_chart_is_a_fixed_footer(tmp_path: Path, monkeypatch):
    """The reference chart is a fixed four-row footer: it never grows with
    height (surplus height stays at the bottom), and the body is anchored to
    the row under the bar rather than centred."""
    from app import CHART_ROWS, DGXTop

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(90, 44)) as pilot:
        await pilot.pause()
        _seed_history(app)
        await _resize(pilot, 90, 44)
        assert not app.tiled
        assert app._chart_rows == CHART_ROWS, app._chart_rows
        body = app.query_one("#body")
        assert body.region.y <= 1, body.region  # top-anchored, not padded
        await _resize(pilot, 132, 44)
        assert app.tiled and app._chart_rows == CHART_ROWS, app._chart_rows
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
            # the gen row is the pane's first row at every tier
            assert "gen" in blob, (w, h, blob)
            assert "req" in blob or "requests" in blob, (w, h, blob)
            assert "ttft" in blob, (w, h, blob)


async def test_two_models_share_one_serving_pane(tmp_path: Path, monkeypatch):
    """Two endpoints, one pane: the shared braille time-series chart carries
    the decode series (an SGLang endpoint with no Prometheus counter plots
    no invented line), the per-model requests/ttft rows carry each model's
    identity — and the extra model rows never clip at any tier."""
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
        # Both models appear in the pane: identity lives in the per-model
        # requests/ttft rows, and the gen row is the shared aggregate.
        assert "qwen3.8-flash-next" in blob
        assert "NVIDIA-Nemotron" in blob
        assert any(ln[2:].startswith("gen") for ln in blob.split("\n")), blob
        req_lines = [ln for ln in blob.split("\n") if "requests" in ln]
        assert any("NVIDIA-Nemotron" in ln for ln in req_lines), req_lines
        assert any("qwen3.8-flash-next" in ln for ln in req_lines), req_lines
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
async def _two_model_app(tmp_path, monkeypatch, hist_a, hist_b, chart=None):
    """Shared fixture: two served models with seeded gen histories."""
    import collections

    from app import DGXTop

    _config(tmp_path / "config.toml", chart=chart)
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

    async with _two_model_app(
        tmp_path, monkeypatch, [100.0] * 24, [10.0] * 24, chart="oscilloscope"
    ) as app:
        serving = app.query_one("#serving", ServingBox)
        w = serving.content_size.width
        rows = serving.render().split("\n")
        assert all(t.cell_len == w for t in rows)
        content = rows[1:-1]  # strip box top/bottom borders
        chart = [t.plain[2:] for t in content[-app._chart_rows - 1 : -1]]  # strip "│ "
        assert len(chart) == app._chart_rows, (len(chart), app._chart_rows)
        assert chart[0].startswith("100 "), chart[0]
        assert chart[app._chart_rows // 2].startswith(" 50 "), chart[app._chart_rows // 2]
        assert chart[-1].startswith("  0 "), chart[-1]
    # all-zero throughput: no range to label — the pane renders unlabeled
    # (the fixture pins stub throughput to the seed tail, so an empty seed
    # really does leave only 0s in the history)
    async with _two_model_app(tmp_path, monkeypatch, [], [], chart="oscilloscope") as app:
        serving = app.query_one("#serving", ServingBox)
        blob = serving.render().plain
        assert not any(ln[2:].startswith(("0 ", " 0 ")) for ln in blob.split("\n")[1:-1]), blob


# ─── oscilloscope decode chart (new default) ─────────────────────────


async def test_oscilloscope_chart_grammar(tmp_path, monkeypatch):
    """AC: the default decode chart is the oscilloscope — dim graticule and
    dotted baseline behind braille traces with NO area fill, one edge
    marker per series with an inline value+name label, and the Δ5 slope
    row under the chart. The kv tail carries the merged percentage."""
    from app import ServingBox

    hist = [20 + (i * 5) % 60 for i in range(24)]  # hi=75, last=75, rising tail
    async with _two_model_app(tmp_path, monkeypatch, hist, hist, chart="oscilloscope") as app:
        serving = app.query_one("#serving", ServingBox)
        rows = serving.render().plain.split("\n")
        interior = rows[1:-1]
        chart = interior[-app._chart_rows - 1 : -1]  # slope row excluded
        blob = "\n".join(chart)
        # the chart rows fit the interior exactly: a stray '…' means the
        # axis/label budget overflowed and _fit truncated the row
        assert not any("…" in ln for ln in chart), "chart rows must not truncate"
        assert any("·" in ln for ln in chart), blob
        assert any("┄" in ln for ln in chart), blob
        assert "███" not in blob, "oscilloscope carries no area fill"
        assert blob.count("◉") == 2, blob
        assert "model-a" in blob and "model-b" in blob, blob
        slope = interior[-1]
        assert "Δ5" in slope and "steady" in slope, slope
        kv_row = next(ln for ln in interior if ln[2:].startswith("kv "))
        assert "tok" in kv_row, kv_row
        assert any(ln[2:].startswith("kv%") for ln in interior), interior


async def test_lines_mode_keeps_legacy_chart(tmp_path, monkeypatch):
    """AC: chart = "lines" restores the legacy fill-line chart — compositing
    area fills under the lines, no graticule, no edge markers, no slope
    row."""
    from app import ServingBox

    hist = [20 + (i * 7) % 50 for i in range(24)]
    async with _two_model_app(tmp_path, monkeypatch, hist, hist, chart="lines") as app:
        serving = app.query_one("#serving", ServingBox)
        rows = serving.render().plain.split("\n")
        interior = rows[1:-1]
        chart = interior[-app._chart_rows :]
        blob = "\n".join(chart)
        assert "███" in blob, "legacy area fill present"
        assert "·" not in blob and "◉" not in blob, blob
        assert "Δ5" not in "\n".join(interior), interior


def test_slope_row_states():
    """AC: the Δ5 slope row escalates to warn when aggregate decode falls
    >8%, reads steady otherwise, and stays an honest em dash with too
    little history."""
    from app import _slope_row
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    rising = _slope_row([("", "#ffffff", [100.0 + i for i in range(24)])], pal)
    assert "steady" in rising.plain and "▼" not in rising.plain
    sag = _slope_row([("", "#ffffff", [200.0 - i * 3 for i in range(24)])], pal)
    assert "▼ sagging" in sag.plain
    short = _slope_row([("", "#ffffff", [1.0] * 5)], pal)
    assert "—" in short.plain and "steady" not in short.plain
    assert "—" in _slope_row([], pal).plain


def test_serving_base_modes():
    """AC: the base is the reference row set (gen, prompt, kv, kv% + the meta
    rows + deploy), plus one rank row per stacked node and one extra
    requests/ttft pair per additional model. Compact drops the prompt and kv%
    graphs; rail/floor keep their fixed surfaces."""
    from app import _serving_base

    assert _serving_base("roomy", 80, 1) == 10
    assert _serving_base("roomy", 80, 3) == 14  # + a requests/ttft pair each
    assert _serving_base("roomy", 80, 1, nodes=2, stack=True) == 12  # two rank rows
    assert _serving_base("roomy", 80, 1, nodes=2, stack=False) == 10  # no stack, no rows
    assert _serving_base("compact", 80, 1) == 7  # gen, kv, requests, cache, ttft
    assert _serving_base("compact", 80, 2) == 9
    assert _serving_base("rail", 80, 1) == 4
    assert _serving_base("floor", 80, 1) == 3


async def test_gen_row_leads_with_the_rate_and_a_block_spark(tmp_path: Path, monkeypatch):
    """The gen row is the pane's first row: the current rate in the body
    foreground, then a single-hue block spark spanning the fixed 22-cell
    graph lane."""
    from app import ServingBox

    hist = [20 + (i * 7) % 50 for i in range(24)]  # last = 31
    async with _two_model_app(tmp_path, monkeypatch, hist, hist) as app:
        serving = app.query_one("#serving", ServingBox)
        rows = serving.render().plain.split("\n")
        gen = next(ln for ln in rows if ln[2:].startswith("gen"))
        assert re.search(r"gen\s+[▁▂▃▄▅▆▇█]+\s+\d+ tok/s", gen), gen
        graph = gen[11:33]
        assert len(graph) == 22 and all(c in "▁▂▃▄▅▆▇█" for c in graph), graph


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


async def test_gen_row_reports_the_cluster_rate(tmp_path, monkeypatch):
    """One model name served by 2 endpoints: the pane's gen row reports the
    cluster's decode rate, and its spark plots the same aggregate series the
    waybar counts."""
    from app import ServingBox

    hist = [20 + (i * 7) % 50 for i in range(24)]  # last = 31
    specs = [("model-a", 300.0, 31.0), ("model-a", 100.0, 1000.0)]
    async with _served_models_app(tmp_path, monkeypatch, specs, [hist, hist]) as app:
        serving = app.query_one("#serving", ServingBox)
        assert len(serving._models) == 1  # one pane, not two
        gen_row = next(t for t in serving.render().split("\n") if t.plain.startswith("│ gen"))
        m = re.search(r"gen\s+[▁▂▃▄▅▆▇█]+\s+(\d+) tok/s", gen_row.plain)
        assert m, gen_row.plain
        # the row plots the recorded aggregate, which is what it prints
        assert m.group(1) == f"{app.cluster.total_throughput:.0f}", gen_row.plain


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
    from app import _lines_chart_lines, _spark_line
    from themes import build_palette, get_theme

    pal = build_palette(get_theme("dgx-aeon"))
    series = [("m", "#8A7CFF", [float("nan"), 5.0, float("inf"), 0.0])]
    rows = _lines_chart_lines(series, 4, 20, pal)
    assert len(rows) == 4 and all(t.cell_len == 20 for t in rows)
    assert all("nan" not in t.style for t in rows), [t.style for t in rows]
    spark = _spark_line([float("nan"), 5.0, float("inf")], "#8A7CFF", 12)
    assert spark.cell_len == 12
    all_nan = _spark_line([float("nan")], "#8A7CFF", 8)
    assert all_nan.cell_len == 8


def test_spark_boundary_inputs():
    """AC11: empty / single-sample / all-equal / all-zero / width-1 inputs
    all render exactly ``width`` cells without crashing."""
    from app import _spark_line

    for data, w in (([], 10), ([5.0], 10), ([5.0, 5.0, 5.0], 10), ([1.0], 1), ([0.0, 0.0], 12)):
        row = _spark_line(list(data), "#8A7CFF", w)
        assert row.cell_len == w, (data, w)


# ─── endpoint scan modal (s) ─────────────────────────────────────────


def _scan_stub(monkeypatch, scans):
    """Replace the sweep with canned results (no SSH/HTTP) and the model-name
    refetch with a constant. Returns the per-press sweep call counter."""
    import collector
    import discovery

    calls = []

    async def fake_sweep(nodes, all_ports=False):
        calls.append(len(nodes))
        for scan in scans:
            yield scan

    async def fake_model(_client, _url):
        return "Gemma-4-26B"

    monkeypatch.setattr(discovery, "sweep", fake_sweep)
    monkeypatch.setattr(collector, "_fetch_model_name", fake_model)
    return calls


def _two_endpoint_scans():
    from discovery import Endpoint, NodeScan

    head = NodeScan(1, "head", False)
    head.endpoints = [
        Endpoint("http://192.0.2.10:8000", 8000, "vllm", "Qwen3.6-27B", "monitored"),
        Endpoint("http://192.0.2.10:8001", 8001, "sglang", "Gemma-4-26B", "new"),
        Endpoint("http://192.0.2.10:30000", 30000, "sglang", "", "new"),
    ]
    worker = NodeScan(2, "worker", True, error="ssh: connect to host worker: timed out")
    return [head, worker]


def _scan_text(app):
    from textual.widgets import OptionList, Static

    screen = app.screen
    rows = [
        (str(option.prompt), option.disabled, option.id)
        for option in screen.query_one("#scan-list", OptionList).options
    ]
    hint = str(screen.query_one("#scan-hint", Static).content)
    return rows, hint


async def test_scan_pick_and_confirm_switches_the_tile_and_saves(tmp_path: Path, monkeypatch):
    import app as app_module
    import config

    path = tmp_path / "c.toml"
    _config(path)
    configure(path)
    _stub(monkeypatch)
    calls = _scan_stub(monkeypatch, _two_endpoint_scans())
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("s")  # never stacks a second scan
        await pilot.pause()
        assert isinstance(app.screen, app_module.ScanScreen)
        assert calls == [2]

        rows, hint = _scan_text(app)
        by_id = {oid: (text, disabled) for text, disabled, oid in rows if oid}
        # Every endpoint of the node is listed; only the new ones are pickable.
        assert set(by_id) == {"1:8000", "1:8001", "1:30000"}
        assert by_id["1:8000"][1] is True and "monitored" in by_id["1:8000"][0]
        assert by_id["1:8001"][1] is False and by_id["1:30000"][1] is False
        assert "—" in by_id["1:30000"][0]  # unknown model is a dash
        assert any("worker" in text and "ssh failed" in text for text, _, _ in rows)

        await pilot.press("enter")  # first pickable endpoint is highlighted
        await pilot.pause()
        _, hint = _scan_text(app)
        assert "Monitor head at http://192.0.2.10:8001" in hint and "[y/n]" in hint

        await pilot.press("n")
        await pilot.pause()
        _, hint = _scan_text(app)
        assert "Monitor" not in hint

        await pilot.press("enter")
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not isinstance(app.screen, app_module.ScanScreen)
        assert config.SPARK_UNITS[1]["vllm_url"] == "http://192.0.2.10:8001"
        assert app.settings.nodes[0].vllm_url == "http://192.0.2.10:8001"
        assert load_config(path).nodes[0].vllm_url == "http://192.0.2.10:8001"
        assert load_config(path).nodes[1].vllm_url == "http://192.0.2.11:8000"
        messages = [n.message for n in app._notifications]
        assert any("saved to" in m for m in messages)


async def test_scan_with_nothing_found_says_so_and_lists_ssh_failures(tmp_path: Path, monkeypatch):
    import app as app_module
    from discovery import NodeScan

    path = tmp_path / "c.toml"
    _config(path)
    configure(path)
    _stub(monkeypatch)
    _scan_stub(
        monkeypatch,
        [NodeScan(1, "head", False), NodeScan(2, "worker", True, error="ssh: timed out")],
    )
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()

        rows, hint = _scan_text(app)
        # One node was scanned; the SSH failure is its own row, not "nothing".
        assert "No inference endpoints found on 1 node." in hint
        assert any("head" in text and "no inference found" in text for text, _, _ in rows)
        assert any("worker" in text and "ssh failed: ssh: timed out" in text for text, _, _ in rows)
        assert not [oid for _, _, oid in rows if oid]

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, app_module.ScanScreen)


async def test_scan_save_failure_keeps_the_switch_and_says_why(tmp_path: Path, monkeypatch):
    import app as app_module
    import config

    path = tmp_path / "c.toml"
    _config(path)
    configure(path)
    # The file changed after it was loaded: node 1 is no longer "head".
    path.write_text(path.read_text().replace('label = "head"', 'label = "renamed"'))
    before = path.read_bytes()
    _stub(monkeypatch)
    _scan_stub(monkeypatch, _two_endpoint_scans())
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert config.SPARK_UNITS[1]["vllm_url"] == "http://192.0.2.10:8001"
        assert path.read_bytes() == before
        messages = [n.message for n in app._notifications]
        assert any("applied; config not saved:" in m for m in messages)


async def test_simulated_scan_switch_is_not_saved(tmp_path: Path, monkeypatch):
    import app as app_module
    import config

    path = tmp_path / "c.toml"
    _config(path)
    before = path.read_bytes()
    configure(path, simulate=2)
    try:
        app = app_module.DGXTop()
        async with app.run_test(size=(132, 40)) as pilot:
            await pilot.press("s")
            for _ in range(10):  # the simulated sweep staggers its nodes
                await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert config.SPARK_UNITS[1]["vllm_url"] == "http://sim-1:8001"
            messages = [n.message for n in app._notifications]
            assert any("applied; simulation: not saved" in m for m in messages)
        assert path.read_bytes() == before
    finally:
        configure(path)


async def test_last_apply_owns_the_live_endpoint_the_settings_and_the_file(
    tmp_path: Path, monkeypatch
):
    """AC1: a second apply on the same node lands while the first one's
    model-name fetch is still waiting. Live unit, ``app.settings`` and the
    config file all end on the second endpoint, and the first one's late
    name is not kept."""
    import asyncio

    import app as app_module
    import collector
    import config
    from discovery import Endpoint

    path = tmp_path / "c.toml"
    _config(path)
    configure(path)
    _stub(monkeypatch)
    monkeypatch.setattr(collector, "_model_names", {})
    monkeypatch.setattr(collector, "_endpoint_epoch", {})
    first_answers = asyncio.Event()

    async def fetch(_client, url):
        if url.endswith(":8001"):
            await first_answers.wait()
        return "Model-" + url.rsplit(":", 1)[1]

    monkeypatch.setattr(collector, "_fetch_model_name", fetch)
    first = Endpoint("http://192.0.2.10:8001", 8001, "sglang", "Model-8001", "new")
    second = Endpoint("http://192.0.2.10:8002", 8002, "vllm", "Model-8002", "new")
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        app.run_worker(app.apply_endpoint(1, first), group="scan-apply")
        await pilot.pause()
        # Still waiting on its model name, the first apply is already recorded.
        assert app.settings.nodes[0].vllm_url == first.url
        assert load_config(path).nodes[0].vllm_url == first.url

        app.run_worker(app.apply_endpoint(1, second), group="scan-apply")
        await pilot.pause()
        first_answers.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert config.SPARK_UNITS[1]["vllm_url"] == second.url
        assert app.settings.nodes[0].vllm_url == second.url
        assert load_config(path).nodes[0].vllm_url == second.url
        assert collector._model_names[1] == "Model-8002"


async def test_closing_the_scan_mid_sweep_cancels_it(tmp_path: Path, monkeypatch):
    """AC11: esc while a node is still scanning closes the window and stops
    the sweep — nothing keeps probing behind a dismissed screen."""
    import asyncio

    import app as app_module
    import discovery
    from discovery import NodeScan

    path = tmp_path / "c.toml"
    _config(path)
    configure(path)
    _stub(monkeypatch)
    never = asyncio.Event()
    seen: list[str] = []

    async def fake_sweep(nodes, all_ports=False):
        yield NodeScan(1, "head", False)
        try:
            await never.wait()  # the worker node is still being scanned
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise
        yield NodeScan(2, "worker", True)

    monkeypatch.setattr(discovery, "sweep", fake_sweep)
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("s")
        await pilot.pause()
        rows, _ = _scan_text(app)
        assert any("worker" in text and "scanning" in text for text, _, _ in rows)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert not isinstance(app.screen, app_module.ScanScreen)
        assert seen == ["cancelled"]


async def test_confirms_on_two_nodes_both_keep_their_model_names(tmp_path: Path, monkeypatch):
    """F1 boundary: an apply on node 2 never cancels node 1's pending
    model-name fetch (the scan-apply worker group is not exclusive)."""
    import asyncio

    import collector
    import discovery
    from discovery import Endpoint, NodeScan

    path = tmp_path / "c.toml"
    _config(path)
    configure(path)
    _stub(monkeypatch)
    monkeypatch.setattr(collector, "_model_names", {})
    monkeypatch.setattr(collector, "_endpoint_epoch", {})
    head_answers = asyncio.Event()

    async def fetch(_client, url):
        if url == "http://192.0.2.10:8001":
            await head_answers.wait()
            return "HeadModel"
        return "WorkerModel" if url == "http://192.0.2.11:8001" else ""

    monkeypatch.setattr(collector, "_fetch_model_name", fetch)
    head = NodeScan(1, "head", False)
    head.endpoints = [Endpoint("http://192.0.2.10:8001", 8001, "sglang", "HeadModel", "new")]
    worker = NodeScan(2, "worker", True)
    worker.endpoints = [Endpoint("http://192.0.2.11:8001", 8001, "vllm", "WorkerModel", "new")]
    rounds = iter([[head], [worker]])

    async def fake_sweep(nodes, all_ports=False):
        for scan in next(rounds):
            yield scan

    monkeypatch.setattr(discovery, "sweep", fake_sweep)
    import app as app_module

    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        for _ in range(2):  # scan, pick the node's new endpoint, confirm
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.press("y")
            await pilot.pause()
        head_answers.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert collector._model_names == {1: "HeadModel", 2: "WorkerModel"}
        assert [n.vllm_url for n in load_config(path).nodes] == [
            "http://192.0.2.10:8001",
            "http://192.0.2.11:8001",
        ]


# ─── GPU clock lock: waybar chip, card marker, `c` menu ──────────────


def _locked(unit: SparkUnitStats, mhz: float) -> SparkUnitStats:
    unit.gpu_clock_lock_mhz = mhz
    unit.gpu_clock_mhz = min(unit.gpu_clock_mhz, mhz - 10)
    return unit


async def test_waybar_clock_lock_chip_states(tmp_path: Path, monkeypatch):
    from app import DGXTop, Waybar
    from themes import build_palette

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    cases = [
        ([_unit("head"), _unit("worker", worker=True)], None, None),
        (
            [_locked(_unit("head"), 2200), _locked(_unit("worker", worker=True), 2200)],
            "≤2200 ",
            "cyan",
        ),
        ([_locked(_unit("head"), 2200), _unit("worker", worker=True)], "≤2200 1/2 ", "warn"),
        (
            [_locked(_unit("head"), 2400), _locked(_unit("worker", worker=True), 2200)],
            "≤2200 ",
            "warn",
        ),
        # An offline node neither counts as locked nor as unlocked.
        (
            [_locked(_unit("head"), 2200), _unit("worker", worker=True, online=False)],
            "≤2200 ",
            "cyan",
        ),
    ]
    for units, chip, role in cases:
        _stub(monkeypatch, units=units)
        app = DGXTop()
        async with app.run_test(size=(132, 40)) as pilot:
            await pilot.pause()
            wb = app.query_one("#waybar", Waybar).render()
            if chip is None:
                assert "≤" not in wb.plain
                continue
            assert f" {chip}" in wb.plain, wb.plain
            assert "≤2200 1/2" not in wb.plain or chip.endswith("1/2 ")
            pal = build_palette(app.current_theme)
            want = pal.cyan if role == "cyan" else pal.warn
            idx = wb.plain.index("≤")
            assert any(want.lower() in s.lower() for s in _style_at(wb, idx)), (chip, role)


async def test_waybar_clock_lock_chip_sheds_value_first_and_goes_after_gen(
    tmp_path: Path, monkeypatch
):
    from app import DGXTop, Waybar

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch, units=[_locked(_unit("head"), 2200), _locked(_unit("worker", True), 2200)])
    app = DGXTop()
    seen = set()
    async with app.run_test(size=(60, 20)) as pilot:
        wb = app.query_one("#waybar", Waybar)
        for width in range(44, 7, -1):
            await _resize(pilot, width, 20)
            plain = wb.render().plain
            full, short = "≤2200" in plain, "≤" in plain
            gen = "tok/s" in plain
            assert "● 2/2" in plain, (width, plain)  # the fleet chip is never dropped
            if gen:
                assert short, (width, plain)  # the chip goes only after the rate
            seen.add((full, short, gen))
    # Every rung of the ladder was observed on the way down: full chip, the
    # value shed to the bare marker, then the chip gone with the rate kept, and
    # finally the rate gone too.
    assert {
        (True, True, True),
        (False, True, True),
        (False, True, False),
        (False, False, False),
    } <= seen


def test_lock_segs_prefer_the_value_then_the_marker():
    from types import SimpleNamespace

    from app import _lock_segs

    pal = SimpleNamespace(dim="dim", cyan="cyan")
    s = _locked(_unit("head"), 2200)
    plain = lambda segs: "".join(t.plain for t in segs)  # noqa: E731
    assert plain(_lock_segs(s, pal, 20, 28)) == " · ≤2200"  # exactly fits
    assert plain(_lock_segs(s, pal, 20, 27)) == " ≤"
    assert plain(_lock_segs(s, pal, 20, 22)) == " ≤"
    assert _lock_segs(s, pal, 20, 21) == []
    assert _lock_segs(_unit("head"), pal, 0, 200) == []  # nothing recorded, nothing shown


async def test_node_card_gpu_row_shows_the_lock_and_leaves_unlocked_rows_alone(
    tmp_path: Path, monkeypatch
):
    from app import DGXTop, NodeBox

    def gpu_line(app, idx):
        node = app.query_one(f"#node-{idx}", NodeBox).render().plain
        return next(ln for ln in node.split("\n") if "gpu" in ln).strip("│ ")

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    head = _unit("head")
    head.gpu_clock_mhz = 2190.0
    _stub(monkeypatch, units=[head, _unit("worker", worker=True)])
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        before = [gpu_line(app, 0), gpu_line(app, 1)]

    head = _locked(_unit("head"), 2200)
    head.gpu_clock_mhz = 2190.0
    _stub(monkeypatch, units=[head, _unit("worker", worker=True)])
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        after = [gpu_line(app, 0), gpu_line(app, 1)]

    assert "≤" not in before[0]
    assert after[0] == before[0] + " · ≤2200"
    assert after[1] == before[1]  # the unlocked node's row is untouched


def _clock_config(path: Path, ceiling: int | None = None) -> None:
    _config(path)
    if ceiling is not None:
        text = path.read_text().replace("[app]\n", f"[app]\nclock_ceiling_mhz = {ceiling}\n", 1)
        path.write_text(text)


def _clock_stub(monkeypatch, result=None, error: Exception | None = None, gate=None) -> list:
    """Fake ``set_clock_lock`` recording each call; ``gate`` (an Event) holds
    the apply in flight until set. Also stubs the startup model-name fetch,
    whose real HTTP timeout would outlive the notifications under test."""
    import app as app_module

    calls: list = []

    async def fake_set_clock_lock(max_mhz):
        calls.append(max_mhz)
        if gate is not None:
            await gate.wait()
        if error is not None:
            raise error
        return result if result is not None else [("head", None), ("worker", None)]

    async def no_model_names():
        return None

    monkeypatch.setattr(app_module, "set_clock_lock", fake_set_clock_lock)
    monkeypatch.setattr(app_module, "_init_model_names", no_model_names)
    return calls


def _clock_text(app):
    from textual.widgets import OptionList, Static

    screen = app.screen
    rows = [
        (str(option.prompt), option.disabled, option.id)
        for option in screen.query_one("#clock-list", OptionList).options
    ]
    return rows, str(screen.query_one("#clock-hint", Static).content)


async def test_clock_menu_locks_the_cluster_after_confirm(tmp_path: Path, monkeypatch):
    import app as app_module

    _clock_config(tmp_path / "c.toml", ceiling=2100)
    configure(tmp_path / "c.toml")
    polls = []

    async def fake_poll():
        polls.append(1)
        return _cluster([_locked(_unit("head"), 2100), _unit("worker", worker=True)])

    monkeypatch.setattr(app_module, "poll_cluster", fake_poll)
    calls = _clock_stub(monkeypatch)
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("c")  # never stacks a second menu
        await pilot.press("s")  # nor a scan on top of it
        await pilot.pause()
        assert isinstance(app.screen, app_module.ClockScreen)
        assert len(app.screen_stack) == 2

        rows, hint = _clock_text(app)
        nodes = [text for text, disabled, oid in rows if oid is None]
        assert all(disabled for _, disabled, oid in rows if oid is None)
        assert "head" in nodes[0] and "locked ≤2100" in nodes[0]
        assert "worker" in nodes[1] and "no lock recorded" in nodes[1]
        actions = {oid: text for text, _, oid in rows if oid}
        assert actions == {
            "lock": "Lock GPU clocks to 0,2100 MHz on 2 nodes",
            "reset": "Reset GPU clocks on 2 nodes",
        }
        assert "Run" not in hint

        await pilot.press("enter")  # the lock action is highlighted first
        await pilot.pause()
        _, hint = _clock_text(app)
        assert "Run sudo -n nvidia-smi -lgc 0,2100 on 2 nodes?" in hint and "[y/n]" in hint

        await pilot.press("n")
        await pilot.pause()
        assert "Run" not in _clock_text(app)[1]
        assert calls == []

        polled = len(polls)
        await pilot.press("enter")
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not isinstance(app.screen, app_module.ClockScreen)
        assert calls == [2100]
        notes = [(n.message, n.severity) for n in app._notifications]
        assert ("GPU clocks locked to 0,2100 MHz on 2 nodes", "information") in notes
        assert len(polls) > polled  # repolled so the tiles show the result


async def test_clock_menu_reset_names_each_failed_node(tmp_path: Path, monkeypatch):
    import app as app_module

    _clock_config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch)
    calls = _clock_stub(
        monkeypatch, result=[("head", None), ("worker", "sudo: a password is required")]
    )
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert "Run sudo -n nvidia-smi -rgc on 2 nodes?" in _clock_text(app)[1]
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert calls == [None]
        notes = [(n.message, n.severity) for n in app._notifications]
        assert (
            "GPU clocks reset on 1/2 nodes; worker: sudo: a password is required",
            "warning",
        ) in notes


async def test_clock_menu_escape_runs_nothing_and_a_failing_apply_keeps_the_app(
    tmp_path: Path, monkeypatch
):
    import app as app_module

    _clock_config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch)
    calls = _clock_stub(monkeypatch, error=RuntimeError("boom"))
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, app_module.ClockScreen)
        assert calls == []

        await pilot.press("c")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.is_running
        assert calls == [2200]
        notes = [(n.message, n.severity) for n in app._notifications]
        assert (
            "GPU clocks locked to 0,2200 MHz on 0/2 nodes; head: boom; worker: boom",
            "warning",
        ) in notes


async def test_clock_menu_refuses_a_second_apply_while_one_is_running(tmp_path: Path, monkeypatch):
    import asyncio

    import app as app_module

    _clock_config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch)
    gate = asyncio.Event()
    calls = _clock_stub(monkeypatch, gate=gate)
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("y")
        await pilot.pause()
        assert calls == [2200]  # the lock is in flight on the nodes

        # Interleaved remote steps could leave a marker that disagrees with
        # the GPU, so no second lock/reset may start until the first lands.
        await pilot.press("c")
        await pilot.pause()
        assert not isinstance(app.screen, app_module.ClockScreen)
        notes = [(n.message, n.severity) for n in app._notifications]
        assert ("GPU clock change still running", "warning") in notes
        app.start_clock_apply(None)
        await pilot.pause()
        assert calls == [2200]

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, app_module.ClockScreen)


async def test_lock_outranks_the_sm_clock_when_only_one_fits(tmp_path: Path, monkeypatch):
    import app as app_module
    from app import DGXTop, NodeBox

    _config(tmp_path / "config.toml")
    configure(tmp_path / "config.toml")
    _stub(monkeypatch)
    app = DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        box = app.query_one("#node-0", NodeBox)
        pal = app_module._palette_for(app)
        # 37 cells → a 33-cell interior: "gpu 73% · 64°C · 430W" (21) has room
        # for the clock (+10) or the lock (+8), not both (+18).
        unlocked = box._interior_rows(_unit("head"), pal, 37, "roomy")[0].plain
        locked = box._interior_rows(_locked(_unit("head"), 2200), pal, 37, "roomy")[0].plain
    assert unlocked == "gpu 73% · 64°C · 430W · 2411MHz"
    assert locked == "gpu 73% · 64°C · 430W · ≤2200"


async def test_clock_menu_steps_the_ceiling_and_applies_the_chosen_value(
    tmp_path: Path, monkeypatch
):
    import app as app_module

    _clock_config(tmp_path / "c.toml")  # configured ceiling: 2200
    configure(tmp_path / "c.toml")
    # The menu opens on the lowest lock recorded on an online node, not the config.
    _stub(
        monkeypatch,
        units=[_locked(_unit("head"), 2400), _locked(_unit("worker", worker=True), 2100)],
    )
    calls = _clock_stub(monkeypatch)
    app = app_module.DGXTop()
    async with app.run_test(size=(132, 40)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        def lock_option():
            return next(text for text, _, oid in _clock_text(app)[0] if oid == "lock")

        assert lock_option() == "Lock GPU clocks to 0,2100 MHz on 2 nodes"
        assert "←→ ceiling ±100 MHz" in _clock_text(app)[1]

        await pilot.press("right", "right", "plus")
        await pilot.pause()
        assert lock_option() == "Lock GPU clocks to 0,2400 MHz on 2 nodes"
        await pilot.press("minus", "left")
        await pilot.pause()
        assert lock_option() == "Lock GPU clocks to 0,2200 MHz on 2 nodes"
        assert app.poll_speeds[app._poll_speed_idx] == 5  # `+`/`-` left polling alone

        # Changing the value drops a confirmation that named the old one.
        await pilot.press("enter")
        await pilot.pause()
        assert "-lgc 0,2200" in _clock_text(app)[1]
        await pilot.press("right")
        await pilot.pause()
        assert "Run" not in _clock_text(app)[1]
        await pilot.press("y")
        await pilot.pause()
        assert calls == []

        await pilot.press("enter")
        await pilot.pause()
        assert "Run sudo -n nvidia-smi -lgc 0,2300 on 2 nodes?" in _clock_text(app)[1]
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == [2300]
    notes = [n.message for n in app._notifications]
    assert "GPU clocks locked to 0,2300 MHz on 2 nodes" in notes


async def test_clock_menu_ceiling_stops_at_the_config_bounds(tmp_path: Path, monkeypatch):
    import app as app_module
    from config import CLOCK_CEILING_MAX, CLOCK_CEILING_MIN

    for ceiling, key, edge in (
        (CLOCK_CEILING_MIN + 50, "left", CLOCK_CEILING_MIN),
        (CLOCK_CEILING_MAX - 50, "right", CLOCK_CEILING_MAX),
    ):
        _clock_config(tmp_path / "c.toml", ceiling=ceiling)
        configure(tmp_path / "c.toml")
        _stub(monkeypatch)
        _clock_stub(monkeypatch)
        app = app_module.DGXTop()
        async with app.run_test(size=(132, 40)) as pilot:
            await pilot.press("c")
            await pilot.pause()
            await pilot.press(key, key, key)
            await pilot.pause()
            options = {oid: text for text, _, oid in _clock_text(app)[0] if oid}
            assert options["lock"] == f"Lock GPU clocks to 0,{edge} MHz on 2 nodes"
