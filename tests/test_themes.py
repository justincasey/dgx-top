from dataclasses import fields

from themes import build_palette, get_theme, theme_names


def _lum(hex_color: str) -> int:
    """Perceptual-ish brightness: weighted RGB sum, higher = brighter."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_theme_names_include_tokyo_family_and_common_themes():
    names = theme_names()
    for expected in (
        "dgx-dark",
        "dgx-aeon",
        "tokyo-night",
        "tokyo-night-storm",
        "tokyo-night-light",
        "nord",
        "gruvbox",
        "dracula",
        "monokai",
        "catppuccin-mocha",
        "textual-dark",
    ):
        assert expected in names


def test_palette_carries_the_tiling_roles():
    """The option-F tiling palette: chrome + identity hues incl. blue/cyan."""
    names = {f.name for f in fields(build_palette(get_theme("dgx-aeon")))}
    assert names == {
        "bg",
        "fg",
        "dim",
        "accent",
        "ok",
        "warn",
        "blue",
        "cyan",
        "track",
        "panel",
        "panel_hi",
        "quiet",
    }


def test_blue_and_cyan_roles_are_pinned_on_customs():
    """GPU-blue and focus-cyan render the design hues under the AEON/tokyo themes."""
    aeon = build_palette(get_theme("dgx-aeon"))
    assert aeon.blue.upper() == "#4AA3FF"
    assert aeon.cyan.upper() == "#57D4F0"
    tokyo = build_palette(get_theme("tokyo-night"))
    assert tokyo.blue.upper() == "#7AA2F7"
    assert tokyo.cyan.upper() == "#7DCFFF"


def test_aeon_default_carries_the_spec_hexes_exactly():
    """dgx-aeon is the CLI Design System palette verbatim (spec §1.1)."""
    palette = build_palette(get_theme("dgx-aeon"))
    expected = {
        "bg": "#0B0E14",
        "fg": "#C8D0DA",
        "dim": "#6B7484",
        "accent": "#7C5CFF",
        "ok": "#3FD07F",
        "warn": "#E8863B",
    }
    for role, hex_value in expected.items():
        assert palette.__getattribute__(role).upper() == hex_value.upper(), role


def test_dark_theme_palette_has_bright_text():
    palette = build_palette(get_theme("tokyo-night-storm"))
    assert _lum(palette.fg) > _lum(palette.dim)


def test_light_theme_palette_has_dark_text():
    theme = get_theme("tokyo-night-light")
    assert theme is not None
    assert not theme.dark
    palette = build_palette(theme)
    assert _lum(palette.fg) < _lum("#e1e2e7")  # dark text on light background


def test_ansi_themes_produce_concrete_hex_palette():
    for name in ("ansi-dark", "ansi-light"):
        palette = build_palette(get_theme(name))
        for field in ("bg", "fg", "dim", "accent", "ok", "warn"):
            assert getattr(palette, field).startswith("#"), f"{name}.{field} not hex"


def test_light_theme_deemphasized_text_stays_darker_than_panel():
    palette = build_palette(get_theme("tokyo-night-light"))
    assert _lum(palette.dim) < _lum("#c4c8da")  # readable against the panel


def test_dark_theme_chrome_is_dimmer_than_text():
    palette = build_palette(get_theme("dgx-dark"))
    assert _lum(palette.fg) > _lum(palette.dim)


def test_roles_are_distinct_in_the_default_theme():
    """Six values that aren't aliases of each other on the default theme."""
    palette = build_palette(get_theme("dgx-aeon"))
    roles = (
        palette.bg,
        palette.fg,
        palette.dim,
        palette.accent,
        palette.ok,
        palette.warn,
    )
    assert len(set(roles)) == 6
