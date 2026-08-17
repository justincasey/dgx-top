from __future__ import annotations

from pathlib import Path

from config import configure


def _config(path: Path, theme: str | None = None) -> None:
    theme_line = (
        f"""theme = "{theme}"
"""
        if theme
        else ""
    )
    path.write_text(
        f"""
[app]
poll_interval = 7
history_length = 25
{theme_line}
[[nodes]]
label = "primary"
ssh_target = "primary"
vllm_url = "http://primary.example.com:8000"
"""
    )


def test_app_uses_configured_poll_interval_and_node_count(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path)
    settings = configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app._current_interval() == 7
    assert len(settings.nodes) == 1


def test_default_theme_applies_through_app(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path)  # no theme key -> default
    configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app.theme == "dgx-dark"


def test_app_applies_configured_theme(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path, theme="tokyo-night-storm")
    configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app.theme == "tokyo-night-storm"
    assert app.current_theme.name == "tokyo-night-storm"


def test_custom_themes_are_registered(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path)
    configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app.get_theme("dgx-dark") is not None
    assert app.get_theme("tokyo-night-storm") is not None
    assert app.get_theme("tokyo-night-light") is not None
    assert app.get_theme("tokyo-night") is not None  # Textual built-in


async def test_meter_bar_keeps_one_base_color_at_all_utilizations():
    from textual.app import App, ComposeResult

    from app import MeterBar
    from themes import CUSTOM_THEMES, build_palette, get_theme

    class MeterHarness(App):
        CSS = "MeterBar { width: 10; height: 1; }"

        def __init__(self):
            super().__init__()
            for theme in CUSTOM_THEMES:
                self.register_theme(theme)
            self.theme = "dgx-dark"

        def compose(self) -> ComposeResult:
            yield MeterBar(metric_color="secondary", id="meter")

    app = MeterHarness()
    async with app.run_test(size=(20, 5)) as pilot:
        meter = app.query_one("#meter", MeterBar)
        palette = build_palette(get_theme("dgx-dark"))

        meter.update_pct(25)
        await pilot.pause()
        low_style = str(meter.render().spans[0].style)

        meter.update_pct(95)
        await pilot.pause()
        high_style = str(meter.render().spans[0].style)

    assert low_style.lower() == palette.secondary.lower()
    assert high_style.lower() == palette.secondary.lower()


def test_cpu_core_cells_keep_one_hue_and_gain_saturation():
    from app import _grid_cell
    from themes import build_palette, get_theme

    palette = build_palette(get_theme("dgx-dark"))

    assert str(_grid_cell(0, palette).style) != palette.accent
    assert str(_grid_cell(100, palette).style).lower() == palette.accent.lower()
    assert str(_grid_cell(100, palette).style).lower() != palette.error.lower()


def test_light_theme_cpu_ramp_remains_distinct():
    from app import _metric_ramp
    from themes import build_palette, get_theme

    palette = build_palette(get_theme("tokyo-night-light"))

    assert _metric_ramp(0, palette, "accent") != palette.background
    assert _metric_ramp(100, palette, "accent") == palette.accent


async def test_theme_binding_opens_picker_and_repaints_title(tmp_path: Path):
    from textual.widgets import Static

    path = tmp_path / "config.toml"
    _config(path)
    configure(path)

    from app import DGXTop

    app = DGXTop()
    async with app.run_test(size=(120, 40)) as pilot:
        title = app.query_one("#title", Static)
        assert "[t]heme" in title.content.plain
        before = [(str(span.style), span.text) for span in title.content.render(app.console)]

        await pilot.press("t")
        await pilot.pause()
        assert type(app.screen).__name__ == "CommandPalette"
        await pilot.press("escape")
        await pilot.pause()

        app.theme = "tokyo-night-storm"
        await pilot.pause()
        after = [(str(span.style), span.text) for span in title.content.render(app.console)]

    assert before != after


def _two_node_config(path: Path) -> None:
    path.write_text(
        """
[app]
poll_interval = 5
history_length = 25

[[nodes]]
label = "head"
ssh_target = "head"
vllm_url = "http://head.example.com:8000"

[[nodes]]
label = "worker"
ssh_target = "worker"
vllm_url = "http://worker.example.com:8000"
"""
    )


def _fake_cluster():
    from stats import ClusterStats, SparkUnitStats, TopologyInfo

    units = []
    for index, label in enumerate(("head", "worker")):
        unit = SparkUnitStats(label=label)
        unit.online = True
        unit.model_hosted = index == 0
        unit.model_name = "Qwen3.6-27B-Instruct"
        unit.gpu_util_pct = 73.0
        unit.temp_c = 64.0
        unit.mem_used_bytes = 62 * 1024**3
        unit.mem_total_bytes = 120 * 1024**3
        unit.swap_total_kb = 4 * 1024 * 1024
        unit.swap_used_kb = 1 * 1024 * 1024
        unit.cpu_cores_util = [50.0] * 20
        unit.cpu_temp_c = 51.0
        unit.kv_cache_pct = 32.0
        unit.kv_total_tokens = 3_800_000
        unit.kv_cache_used_tokens = 1_230_000
        unit.kv_prefix_hit_rate = 45.0
        unit.requests_running = 2
        unit.requests_waiting = 1
        unit.prompt_gen_ratio = 3.0
        unit.throughput_tok_s = 1200.0
        unit.prompt_throughput_tok_s = 3600.0
        units.append(unit)
    return ClusterStats(units=units, topology=TopologyInfo(topology_type="DUAL"))


async def test_compact_tier_fits_a_320x320_viewport(tmp_path: Path, monkeypatch):
    """A ~320x320px viewport (40x21 cells) must show every tile and metric."""
    from textual.widgets import Static

    import app as app_module
    from app import DGXTop, NodeTile, ThroughputTile

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)

    async def fake_poll():
        return _fake_cluster()

    monkeypatch.setattr(app_module, "poll_cluster", fake_poll)

    app = DGXTop()
    async with app.run_test(size=(40, 21)) as pilot:
        await pilot.pause()

        assert app.density == "compact"
        throughput = app.query_one(ThroughputTile)
        nodes = list(app.query(NodeTile))
        assert len(nodes) == 2
        # Throughput is the priority signal: it keeps two-row sparklines and so
        # gets two rows more than a node tile.
        assert throughput.region.height == 7
        for tile in [throughput, *nodes]:
            assert tile.region.bottom <= 21, f"{tile.id} overflows the viewport"
        for node in nodes:
            assert node.region.height == 5

        def text(selector: str) -> str:
            return app.query_one(selector, Static).content.plain

        # Every metric survives the fold: throughput min/avg/max for both
        # streams, requests/waiting/hit-rate/ratio, KV capacity, and the
        # per-node GPU/memory/swap/CPU values plus all 20 core cells.
        prompt_stats = text("#tp-prompt-stats")
        assert prompt_stats.startswith("P ") and "7200" in prompt_stats
        gen_stats = text("#tp-gen-stats")
        assert gen_stats.startswith("G ") and "2400" in gen_stats
        assert text("#kv-header") == "2r  1w  h 45%  3:1"
        assert text("#kv-detail") == "1.2M/3.8M 32%"
        for idx in (0, 1):
            assert text(f"#node-gpu-row-{idx}") == "G 73% 64°C"
            assert text(f"#node-mem-row-{idx}") == "M 62G/120G 52% s1.0G"
            assert text(f"#node-cpu-row-{idx}") == "C 50% 51°C"
            assert text(f"#node-cpu-grid-{idx}").count("\u25a0") == 20


def _stub_polling(monkeypatch) -> None:
    """Serve synthetic telemetry so layout assertions never race a real poll."""
    import app as app_module

    async def fake_poll():
        return _fake_cluster()

    monkeypatch.setattr(app_module, "poll_cluster", fake_poll)


async def _resize(pilot, width: int, height: int) -> None:
    await pilot.resize_terminal(width, height)
    await pilot.pause()
    await pilot.pause()


async def test_medium_tier_shows_every_node_tile(tmp_path: Path, monkeypatch):
    """The two-column tier must wrap onto a second row, not clip a node."""
    from app import DGXTop, NodeTile

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)
    _stub_polling(monkeypatch)

    app = DGXTop()
    async with app.run_test(size=(70, 60)) as pilot:
        await pilot.pause()

        assert app.query_one("#kpis").has_class("medium")
        heights = sorted(tile.region.height for tile in app.query(NodeTile))
        assert heights == [16, 16]


async def test_density_steps_down_one_tier_at_a_time(tmp_path: Path, monkeypatch):
    """Density follows the rows each grid row gets: roomy → dense → compact."""
    from app import DGXTop

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)
    _stub_polling(monkeypatch)

    app = DGXTop()
    async with app.run_test(size=(180, 16)) as pilot:
        await pilot.pause()
        assert app.density == "roomy"

        await _resize(pilot, 180, 14)
        assert app.density == "dense"

        await _resize(pilot, 180, 11)
        assert app.density == "compact"

        # One column of three tiles needs three times the height for the same
        # density, so the same rows buy a looser layout at three columns.
        await _resize(pilot, 40, 40)
        assert app.density == "dense"

        await _resize(pilot, 40, 26)
        assert app.density == "compact"

        await _resize(pilot, 40, 50)
        assert app.density == "roomy"


async def test_tiles_fill_the_viewport_until_they_cap_out(tmp_path: Path, monkeypatch):
    """No dead space: the grid stretches to the bottom until tiles cap out."""
    from app import DGXTop, NodeTile, ThroughputTile

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)
    _stub_polling(monkeypatch)

    app = DGXTop()
    async with app.run_test(size=(180, 12)) as pilot:
        await pilot.pause()

        for height, expected in ((12, 10), (14, 12), (16, 14), (40, 16)):
            await _resize(pilot, 180, height)
            tiles = list(app.query(ThroughputTile)) + list(app.query(NodeTile))
            for tile in tiles:
                assert tile.region.height == expected, (
                    f"{tile.id} is {tile.region.height} rows at height {height}"
                )


async def test_no_row_is_clipped_at_any_viewport(tmp_path: Path, monkeypatch):
    """Fluidity invariant: every row stays inside its tile at every size.

    Covers all three densities, all three column tiers and the scrolling floor.
    """
    from app import DGXTop, NodeTile, ThroughputTile

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)
    _stub_polling(monkeypatch)

    sizes = (
        (180, 40),
        (180, 16),
        (180, 12),
        (120, 22),
        (70, 24),
        (40, 50),
        (40, 40),
        (40, 26),
        (40, 21),
        (26, 16),
    )

    app = DGXTop()
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()

        for width, height in sizes:
            await _resize(pilot, width, height)
            # Compact drops the border, so the last usable row is the tile's own
            # bottom; the bordered densities lose one row to it.
            border = 0 if app.density == "compact" else 1
            for tile in list(app.query(ThroughputTile)) + list(app.query(NodeTile)):
                limit = tile.region.bottom - border
                for child in tile.walk_children():
                    assert child.region.bottom <= limit, (
                        f"{type(child).__name__} in {tile.id} is clipped at "
                        f"{width}x{height} ({app.density})"
                    )


async def test_history_charts_hold_full_pair_width(tmp_path: Path, monkeypatch):
    """All three history bar charts must span the whole tile width at every tier.

    Each label floats over its chart (``position: absolute``) so it takes no
    horizontal space and cannot starve the chart. Every chart -- prompt,
    generation, and KV usage -- must fill the entire tile width, with its label
    overlaid on top rather than sitting beside it.
    """
    from textual.widgets import Sparkline, Static

    from app import DGXTop

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)
    _stub_polling(monkeypatch)

    app = DGXTop()
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()

        # Three columns, two columns, and the compact ~320x320px narrow
        # single-column tier ((40, 21), the same viewport the compact test uses).
        for width, height in ((180, 22), (120, 22), (70, 24), (46, 30), (40, 21)):
            await _resize(pilot, width, height)

            tile = app.query_one("#kpi-throughput")
            charts = [
                app.query_one(cid, Sparkline)
                for cid in ("#tp-prompt-chart", "#tp-gen-chart", "#kv-usage-chart")
            ]
            for chart in charts:
                assert chart.region.width == tile.content_size.width, (
                    f"{chart.id} is {chart.region.width}/{tile.content_size.width} "
                    f"at {width}x{height}"
                )
            # Each label floats on the top row, left-aligned with the chart, and
            # the chart plot is offset one row below it so the bars form a clean
            # rectangle instead of rising past the label. The label shares the
            # chart's left edge and sits exactly one row above the plot.
            labels = ("#tp-prompt-stats", "#tp-gen-stats", "#kv-detail")
            for stats_id, chart in zip(labels, charts):
                stats = app.query_one(stats_id, Static)
                assert stats.styles.position == "absolute"
                assert stats.region.x == chart.region.x, (
                    f"{stats_id} should share {chart.id}'s left edge, not sit beside "
                    f"it, got {stats.region} vs {chart.region} at {width}x{height}"
                )
                assert stats.region.height == 1 and stats.region.y == chart.region.y - 1, (
                    f"{stats_id} should be a one-row label directly above {chart.id}, "
                    f"got {stats.region} vs {chart.region} at {width}x{height}"
                )
                assert stats.region.width < chart.region.width, (
                    f"{stats_id} should be a floating label inside the chart span, "
                    f"got width {stats.region.width} vs {chart.region.width} "
                    f"at {width}x{height}"
                )


async def test_dense_layout_fits_a_180_pixel_tall_viewport(tmp_path: Path, monkeypatch):
    """At three columns the dense layout must fit 12 rows (~180px)."""
    from app import DGXTop, NodeTile, ThroughputTile

    path = tmp_path / "config.toml"
    _two_node_config(path)
    configure(path)
    _stub_polling(monkeypatch)

    app = DGXTop()
    async with app.run_test(size=(180, 12)) as pilot:
        await pilot.pause()

        assert app.density == "dense"
        kpis = app.query_one("#kpis")
        assert not kpis.has_class("narrow") and not kpis.has_class("medium")
        tiles = list(app.query(ThroughputTile)) + list(app.query(NodeTile))
        for tile in tiles:
            assert tile.region.height == 10
            assert tile.region.bottom <= 12, f"{tile.id} overflows the viewport"
        # Every child row is inside the tile, i.e. nothing is clipped by the
        # border the way the KV risk row used to be.
        for tile in tiles:
            for child in tile.walk_children():
                assert child.region.bottom <= tile.region.bottom - 1
