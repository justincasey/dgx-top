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


def test_dark_theme_palette_has_bright_text():
    palette = build_palette(get_theme("tokyo-night-storm"))
    assert _lum(palette.fg) > _lum(palette.muted) > _lum(palette.faint)


def test_light_theme_palette_has_dark_text():
    theme = get_theme("tokyo-night-light")
    assert theme is not None
    assert not theme.dark
    palette = build_palette(theme)
    assert _lum(palette.fg) < _lum("#e1e2e7")  # dark text on light background


def test_palette_ramp_is_monotonic():
    palette = build_palette(get_theme("tokyo-night"))
    lums = [
        _lum(value)
        for value in (palette.fg, palette.mid, palette.dim, palette.muted, palette.faint)
    ]
    assert lums == sorted(lums, reverse=True)
