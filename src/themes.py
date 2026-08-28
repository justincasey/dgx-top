"""Theme definitions and palette helpers for dgx-top.

The dashboard renders with a tiling-desktop semantic palette:
`bg`, `fg`, `dim`, `track`, `panel`, `panel_hi` (chrome) plus the identity
hues `accent` (KV/RoCE/model), `ok` (health/MEM), `warn` (CPU/worker/caution),
`blue` (GPU util) and `cyan` (focus borders, host caret). The `dgx-aeon`
default theme carries the design's hues; every other theme remaps the same
roles onto its own hues, so switching themes restyles the roles without
introducing new ones.

All common Textual themes (tokyo-night, nord, gruvbox, dracula, ...) are
available out of the box; this module adds the dgx-top default theme (AEON)
and the classic hardcoded-look dgx-dark theme, plus the missing tokyo-night
storm and light variants, and turns any active theme into the semantic palette
the dashboard's dynamically rendered Text styles use.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

DEFAULT_THEME = "dgx-aeon"
"""Default theme name; carries the tiling-desktop palette hues."""

# Tokyo Night palettes from folke/tokyonight.nvim. All three variants are
# registered here: Textual's built-in ``tokyo-night`` uses ``fg_gutter``
# (#414868) as its panel color, which washes out the dashboard tiles against
# the #1A1B26 background, so it is overridden with the darker canonical night
# bg_dark/bg_float pair. ``blue``/``cyan`` pin the option-F hue roles exactly.
_TOKYO_NIGHT = dict(
    primary="#BB9AF7",  # magenta
    secondary="#7AA2F7",  # blue
    warning="#E0AF68",  # yellow
    error="#F7768E",  # red
    success="#9ECE6A",  # green
    accent="#FF9E64",  # orange
    foreground="#C0CAF5",
    background="#1A1B26",
    surface="#16161E",
    panel="#1F2335",
    variables={"blue": "#7AA2F7", "cyan": "#7DCFFF"},
)

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
    variables={"blue": "#7AA2F7", "cyan": "#7DCFFF"},
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
    variables={"blue": "#2E7DE9", "cyan": "#0DB9D7"},
)

CUSTOM_THEMES: list[Theme] = [
    Theme(
        name="dgx-dark",
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
            "text-muted": "#6B7484",
            "border-blurred": "#2A2A2A",
            "blue": "#4AA3FF",
            "cyan": "#33CCFF",
        },
    ),
    Theme(
        name="tokyo-night",
        **_TOKYO_NIGHT,
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
    # The AEON default: the tiling language mapped onto the AEON desert hues.
    # ``text-muted`` pins the ``dim`` step, which build_palette would otherwise
    # derive by blending. ``blue``/``cyan`` give the design's GPU/focus roles a
    # cohesive AEON-blue family so the tiling grammar renders correctly out of
    # the box; ``tokyo-night`` carries the design's exact reference hues.
    Theme(
        name="dgx-aeon",
        primary="#7C5CFF",  # accent
        secondary="#4AA3FF",  # blue (GPU util)
        warning="#E8863B",
        error="#E8863B",  # failures are warn-colored per the glyph table
        success="#3FD07F",  # ok
        accent="#7C5CFF",
        foreground="#C8D0DA",
        background="#0B0E14",
        surface="#0B0E14",
        panel="#12151D",
        variables={
            "text-muted": "#6B7484",  # dim
            "border-blurred": "#6B7484",  # dim
            "blue": "#4AA3FF",
            "cyan": "#57D4F0",
        },
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
    """The eleven semantic roles the dashboard renders with.

    Chrome: ``bg`` (canvas, never drawn explicitly where a terminal background
    can be inherited), ``fg`` (primary text / scanned values), ``dim``
    (labels, units, separators, unfocused borders), ``track`` (meter
    remainder), ``panel`` / ``panel_hi`` (status-bar segment fills). Identity:
    ``accent`` (KV/RoCE/model — the orchestrating hue), ``ok`` (health/MEM),
    ``warn`` (CPU/worker + caution), ``blue`` (GPU util), ``cyan`` (focus
    borders, host caret). ``accent``/``ok``/``warn`` keep the CLI Design System
    semantics; alarm state never relies on colour alone.
    """

    bg: str
    fg: str
    dim: str
    accent: str
    ok: str
    warn: str
    blue: str
    cyan: str
    track: str
    panel: str
    panel_hi: str
    quiet: bool = False


def _concrete(value: str | None, fallback: str | None) -> str | None:
    """Resolve a color string to a guaranteed ``#rrggbb`` hex value.

    ANSI terminal themes express colors as names such as ``ansi_blue`` or
    ``ansi_default``; Rich text styles cannot parse those, so they are mapped
    to their concrete RGB equivalents (terminal default falls back). An
    unparseable or missing value returns ``fallback`` (which may itself be
    ``None`` to mean "no override").
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


_LIGHT_DEEMPHASIS = Color.parse("#000000")
"""Dark anchor for de-emphasized text on light themes.

On light themes the background is already light, so blending muted text
toward it would collapse the contrast. De-emphasis therefore blends toward a
dark neutral instead, keeping secondary text legible on light panels.
"""


def build_palette(theme: Theme, quiet: bool = False) -> Palette:
    """Map a Textual theme onto the eleven semantic palette roles.

    With ``quiet`` the identity hues (accent/ok/blue/cyan) collapse to neutral
    foreground — colour is reserved for caution/critical escalation, calming
    the whole UI. Alarm semantics never rely on hue alone, so nothing else
    changes.
    """
    if getattr(theme, "ansi", False):
        # Terminal-native themes declare their default colors as variables.
        background = _concrete(theme.variables.get("ansi-background"), "#000000") or "#000000"
        foreground = _concrete(theme.variables.get("ansi-foreground"), "#FFFFFF") or "#FFFFFF"
    else:
        background = _concrete(theme.background, "#000000") or "#000000"
        foreground = _concrete(theme.foreground, "#FFFFFF") or "#FFFFFF"
    background_color = Color.parse(background)
    foreground_color = Color.parse(foreground)

    def color(field: str, fallback: str) -> str:
        return _concrete(getattr(theme, field), fallback) or fallback

    dark = getattr(theme, "dark", True)
    if dark:
        dim = foreground_color.blend(background_color, 0.34).hex
    else:
        # Light themes: keep text dark by blending toward a dark neutral.
        dim = foreground_color.blend(_LIGHT_DEEMPHASIS, 0.30).hex
    # A theme may pin ``dim`` exactly via its text-muted variable (the
    # dgx-aeon default does, carrying the spec's #6B7484).
    pinned = _concrete(theme.variables.get("text-muted"), None)
    if pinned is not None:
        dim = pinned

    accent = color("primary", "#7C5CFF")
    # GPU util rides blue; pinned per-theme where the theme's secondary is not
    # already blue, else the theme's own secondary (the design's default).
    blue = _concrete(theme.variables.get("blue"), None) or color("secondary", accent)
    # Focus chrome = cyan; themes that pin it carry the design's exact hue,
    # otherwise the accent is lifted toward white for a legible neon edge.
    cyan = (
        _concrete(theme.variables.get("cyan"), None)
        or Color.parse(accent).blend(Color.parse("#FFFFFF"), 0.28).hex
    )

    identity = foreground if quiet else None
    return Palette(
        bg=background,
        fg=foreground,
        dim=dim,
        accent=identity or accent,
        ok=identity or color("success", "#3FD07F"),
        warn=color("warning", "#E8863B"),
        blue=identity or blue,
        cyan=identity or cyan,
        track=background_color.blend(foreground_color, 0.07).hex,
        panel=_concrete(theme.panel, background) or background,
        panel_hi=Color.parse(_concrete(theme.panel, background) or background)
        .blend(foreground_color, 0.16)
        .hex,
        quiet=quiet,
    )
