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
