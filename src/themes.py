"""Theme definitions and palette helpers for dgx-top.

The dgx-top dashboard is themed with Textual's native theme system. All common
Textual themes (tokyo-night, nord, gruvbox, dracula, monokai, catppuccin-*,
solarized-*, rose-pine-*, ...) are available out of the box; this module adds
the dgx-top default theme (which reproduces the classic hardcoded look) and the
missing tokyo-night storm and light variants, and turns any active theme into a
semantic palette for the dashboard's dynamically rendered Text styles.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

DEFAULT_THEME = "dgx-dark"
"""Default theme name; reproduces the classic dgx-top look."""

# Tokyo Night palettes from folke/tokyonight.nvim. ``tokyo-night`` (night) is a
# Textual built-in; storm and light are registered here so the full family is
# available.
_TOKYO_STORM = dict(
    primary="#BB9AF7",  # magenta
    secondary="#7AA2F7",  # blue
    warning="#E0AF68",  # yellow
    error="#F7768E",  # red
    success="#9ECE6A",  # green
    accent="#FF9E64",  # orange
    foreground="#C0CAF5",
    background="#24283B",
    surface="#1F2335",
    panel="#292E42",
)

_TOKYO_LIGHT = dict(
    primary="#2E7DE9",  # blue
    secondary="#9854F1",  # magenta
    warning="#8C6C3E",  # yellow
    error="#F52A65",  # red
    success="#587539",  # green
    accent="#D18616",  # orange
    foreground="#3760BF",
    background="#E1E2E7",
    surface="#D4D6DE",
    panel="#C4C8DA",
)

CUSTOM_THEMES: list[Theme] = [
    Theme(
        name=DEFAULT_THEME,
        primary="#4AA3FF",
        secondary="#33CCFF",
        warning="#E3B341",
        error="#F05050",
        success="#57D787",
        accent="#00E0E0",
        foreground="#FFFFFF",
        background="#0A0A0A",
        surface="#0F0F0F",
        panel="#0F0F0F",
        variables={
            "text-muted": "#9AA0A6",
            "border-blurred": "#2A2A2A",
        },
    ),
    Theme(
        name="tokyo-night-storm",
        **_TOKYO_STORM,
    ),
    Theme(
        name="tokyo-night-light",
        dark=False,
        **_TOKYO_LIGHT,
    ),
]


def get_theme(name: str) -> Theme | None:
    """Look up a supported theme by name (custom first, then built-in)."""
    for custom in CUSTOM_THEMES:
        if custom.name == name:
            return custom
    return BUILTIN_THEMES.get(name)


def theme_names() -> list[str]:
    """Every supported theme name, sorted."""
    return sorted(set(BUILTIN_THEMES) | {custom.name for custom in CUSTOM_THEMES})


@dataclass(frozen=True)
class Palette:
    """Semantic colors resolved from the active theme.

    ``background`` is the concrete theme background. ``fg``/``mid``/``dim``/
    ``muted``/``faint`` are a five-step foreground ramp (bright to dim);
    ``grid_lo`` is retained for other low-emphasis rendering. The remaining
    fields are the theme's accent and status colors. All values are
    ``#rrggbb`` hex strings.
    """

    background: str
    fg: str
    mid: str
    dim: str
    muted: str
    faint: str
    grid_lo: str
    accent: str
    primary: str
    secondary: str
    warn: str
    error: str
    ok: str


def _concrete(value: str | None, fallback: str) -> str:
    """Resolve a color string to a guaranteed ``#rrggbb`` hex value.

    ANSI terminal themes express colors as names such as ``ansi_blue`` or
    ``ansi_default``; Rich text styles cannot parse those, so they are mapped
    to their concrete RGB equivalents (terminal default falls back).
    """
    if not value:
        return fallback
    try:
        parsed = Color.parse(value)
    except Exception:
        return fallback
    if parsed.ansi is None:
        return parsed.hex
    if parsed.ansi == -1:  # terminal default
        return fallback
    truecolor = parsed.rich_color.get_truecolor()
    return truecolor.hex if truecolor else fallback


def _ramp(foreground: Color, background: Color, toward_background: float) -> str:
    """Blend foreground toward background; 0.0 keeps the foreground color."""
    return foreground.blend(background, toward_background).hex


_LIGHT_DEEMPHASIS = Color.parse("#000000")
"""Dark anchor for de-emphasized text on light themes.

On light themes the background is already light, so blending muted text
toward it would collapse the contrast. De-emphasis therefore blends toward a
dark neutral instead, keeping secondary text legible on light panels.
"""


def build_palette(theme: Theme) -> Palette:
    """Derive the dashboard palette from a Textual theme."""
    if getattr(theme, "ansi", False):
        # Terminal-native themes declare their default colors as variables.
        background = _concrete(theme.variables.get("ansi-background"), "#000000")
        foreground = _concrete(theme.variables.get("ansi-foreground"), "#FFFFFF")
    else:
        background = _concrete(theme.background, "#000000")
        foreground = _concrete(theme.foreground, "#FFFFFF")
    background_color = Color.parse(background)
    foreground_color = Color.parse(foreground)

    def color(field: str, fallback: str) -> str:
        return _concrete(getattr(theme, field), fallback)

    dark = getattr(theme, "dark", True)
    if dark:
        mid = _ramp(foreground_color, background_color, 0.15)
        dim = _ramp(foreground_color, background_color, 0.34)
        muted = _ramp(foreground_color, background_color, 0.54)
        faint = _ramp(foreground_color, background_color, 0.68)
        grid_lo = _ramp(foreground_color, background_color, 0.85)
    else:
        # Light themes: keep text dark by blending toward a dark neutral.
        mid = _ramp(foreground_color, _LIGHT_DEEMPHASIS, 0.12)
        dim = _ramp(foreground_color, _LIGHT_DEEMPHASIS, 0.30)
        muted = _ramp(foreground_color, _LIGHT_DEEMPHASIS, 0.45)
        faint = _ramp(foreground_color, _LIGHT_DEEMPHASIS, 0.58)
        grid_lo = _ramp(foreground_color, _LIGHT_DEEMPHASIS, 0.72)

    return Palette(
        background=background_color.hex,
        fg=foreground_color.hex,
        mid=mid,
        dim=dim,
        muted=muted,
        faint=faint,
        grid_lo=grid_lo,
        accent=color("accent", theme.primary or "#00E0E0"),
        primary=color("primary", "#4AA3FF"),
        warn=color("warning", "#E3B341"),
        secondary=color("secondary", "#33CCFF"),
        error=color("error", "#F05050"),
        ok=color("success", "#57D787"),
    )
