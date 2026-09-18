from __future__ import annotations

import collections
import logging
import math
import re
from pathlib import Path
from urllib.parse import urlparse

from rich.text import Text
from textual.app import App, Binding
from textual.containers import Vertical
from textual.drivers.linux_driver import LinuxDriver
from textual.reactive import reactive
from textual.widgets import Static

from collector import _init_model_names, poll_cluster
from config import default_config_path, get_settings
from input_driver import ResilientLinuxDriver
from stats import ClusterStats, SparkUnitStats
from themes import CUSTOM_THEMES, SERIES_HUES, Palette, blend_toward, build_palette, series_hue

log = logging.getLogger("dgx-top")

# ─── Layout constants ────────────────────────────────────────────────

TEMP_ALERT = 80
TEMP_WARM = 60

# Fluid layout: two arrangements chosen by width. At/above TILING_WIDTH the
# SERVING card tiles beside a node column (the reference tiling desktop);
# below it the SERVING hero sits full-width on top and the node grid wraps
# below. In both, the grid picks a column count and a node-tile grammar from
# the available width so a 1-12 Spark cluster keeps the most detail it can
# carry.
WAYBAR_HEIGHT = 1
GRID_GUTTER = 1  # row gutter in the node grid; columns are contiguous
TILING_WIDTH = 96  # width >= this: SERVING left, node column right
SERVING_TILED_FRAC = 56  # serving column width as % of the viewport width
TILING_GUTTER = 1  # blank column between the tiled serving and node column
NODE_CARD_MIN = 22  # a usable card (text) shows gpu/mem/cpu; below this -> table
NODE_FULL_MIN = 26  # a tile at/above this keeps the meter/core/RoCE grammar
NODE_FLOOR_MAX = 20  # condensed table rows prefer this width (keeps gpu/mem/cpu)
SERVING_NARROW_WIDTH = 52  # below this the SERVING hero uses the fused grammar

# Bounded growth (bounded + breathe): the SERVING area chart fills the
# leftover height up to the tier max and never thinner than its tier min; the
# core grid never exceeds CORES_MAX_ROWS rows.
CORES_MAX_ROWS = 2
# Area-chart rows are fit-computed between these bounds; the gen area chart is
# the last serving chart visual and survives compact. Rail/floor carry no area
# chart (the inline gen sparkline is the last chart left there).
CHART_MIN = {"roomy": 5, "dense": 4, "compact": 2, "rail": 0, "floor": 0}
CHART_MAX = {"roomy": 16, "dense": 14, "compact": 6, "rail": 0, "floor": 0}

# Interior rows per node tile (borders excluded). A full card keeps the
# meter/core-grid/RoCE grammar; a compact meter card keeps the meters + core
# grid but drops RoCE; a text card drops the graphs and runs gpu/mem/cpu only.
# The estimator mirrors the render width fold (floor-fit duality lesson).
NODE_ROWS_FULL = 8  # full card interior (gpu/meter+mem/meter+cpu+core+roce)
NODE_ROWS_METER = 6  # compact card interior (headlines + meters + 1 core row)
NODE_ROWS_TEXT = 3  # text card interior (gpu, mem, cpu; no meter/core/RoCE)

# Natural interior rows per SERVING window by density (borders excluded).
# roomy/dense = the wide design row set (no window stat) + area chart; compact =
# the prioritized narrow grammar + the inline gen sparkline (no area chart);
# rail = prioritized grammar, no cache; floor = base gen/req/ttft.
WIDE_BASE = 11  # gen/prompt/kv/kv% rows + requests/cache/ttft (no window)
NARROW_BASE = 5  # gen, requests, ttft, kv%, cache
SERVING_ROWS_RAIL = 4  # gen, requests, ttft, kv%
SERVING_ROWS_FLOOR = 3  # gen, requests, ttft

# Box-painting charsets — HEAVY is the focused window ("neon glow" in pure
# text), LIGHT is every unfocused window (the option-F grammar).
_HEAVY = {"tl": "┏", "tr": "┓", "bl": "┗", "br": "┛", "h": "━", "v": "┃", "jl": "┫", "jr": "┣"}
_LIGHT = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│", "jl": "┤", "jr": "├"}


# btop-style utilisation ramp: green → yellow → orange → red at 0/45/75/100.
# Quiet mode collapses the ramp to neutral below caution, reserving colour for
# the 75/90 escalation.
def _ramp(t: float, pal: Palette | None = None) -> str:
    if pal is not None and pal.quiet:
        if t >= 90:
            return "#f7768e"
        return pal.warn if t >= 75 else pal.fg
    stops = [
        (0, (158, 206, 106)),
        (45, (224, 175, 104)),
        (75, (255, 158, 100)),
        (100, (247, 118, 142)),
    ]
    t = max(0.0, min(100.0, t))
    for i in range(1, len(stops)):
        if t <= stops[i][0]:
            a, ca = stops[i - 1]
            b, cb = stops[i]
            f = (t - a) / ((b - a) or 1)
            c = tuple(round(x + (y - x) * f) for x, y in zip(ca, cb))
            return "#%02x%02x%02x" % c
    return "#f7768e"


def _density(widget) -> str:
    """Active density: ``compact``, ``dense`` or ``roomy``."""
    return getattr(widget.app, "density", "dense")


def _fit(row: Text, width: int) -> Text:
    """Pad or truncate a row to exactly ``width`` cells."""
    if width <= 0:
        return Text()
    if row.cell_len > width:
        row = row.copy()
        row.truncate(width, overflow="ellipsis")
    elif row.cell_len < width:
        row = Text.assemble(row, Text(" " * (width - row.cell_len)))
    return row


def _clamp_segs(segs: list[tuple[str, str]], maxw: int) -> list[tuple[str, str]]:
    """Truncate a (text, style) segment list to ``maxw`` cells (single-width)."""
    out: list[tuple[str, str]] = []
    used = 0
    for t, st in segs:
        if used >= maxw:
            break
        if used + len(t) <= maxw:
            out.append((t, st))
            used += len(t)
        else:
            out.append((t[: maxw - used], st))
            break
    return out


def _legend_segs(
    models: list[dict], pal: Palette, head_w: int, rtab_w: int, width: int
) -> list[tuple[str, str]]:
    """Chart-legend segments for the serving title rule: each model's name in
    its own series hue, followed by the engine family serving it.

    The engine tag is secondary to the legend's own content and to the KV meta
    tab, so it is spent only from slack the top rule has: the untagged legend
    wins whenever the tag would clip a model name (``width - 7`` is what
    ``_box_lines`` leaves the title) or evict the meta tab that the untagged
    legend still fits beside."""
    plain: list[tuple[str, str]] = []
    tagged: list[tuple[str, str]] = []
    for i, m in enumerate(models):
        if i:
            plain.append((" · ", pal.dim))
            tagged.append((" · ", pal.dim))
        plain.append((m["name"], f"bold {m['color']}"))
        tagged.append((m["name"], f"bold {m['color']}"))
        tagged.append((f" [{m['source']}]", pal.dim))
    tagged_w = head_w + sum(len(t) for t, _ in tagged)
    plain_w = head_w + sum(len(t) for t, _ in plain)
    room = width - 12  # corners: 6 cells per cluster, as _box_lines reserves
    # Beside the tab the tag must also leave a few cells of top rule, or the
    # two junction glyphs abut into a visible seam (├┤).
    fits_beside_tab = rtab_w > room - plain_w or rtab_w + 3 <= room - tagged_w
    return tagged if tagged_w <= width - 7 and fits_beside_tab else plain


def _box_lines(
    width: int,
    title: list[tuple[str, str]],
    rtab: list[tuple[str, str]] | None,
    rows: list[Text],
    focused: bool,
    pal: Palette,
) -> list[Text]:
    """Paint a tiling window: border, caret title inset in the top rule, right
    meta tab, content rows, bottom rule. Every returned line is exactly
    ``width`` cells wide. ``title``/``rtab`` are (text, style) segment lists;
    ``title`` renders after ``╭─┤ `` (heavy: ``┏━┣ ``), ``rtab`` before the
    `` ├─╮`` corner so the frame reads ``╭─┤ ^ name role ├───┤ meta ├─╮``.
    """
    cs = _HEAVY if focused else _LIGHT
    bstyle = pal.dim
    # Clamp the tabs so the top rule can never exceed ``width`` and wrap: the
    # fixed cost is 6 cells for the title side (corner + rule + junction + two
    # spaces + junction) and, when present, 6 more for the right tab.
    title = _clamp_segs(title, max(0, width - 6 - 1))
    if rtab is not None:
        rl0 = sum(len(t) for t, _ in rtab)
        tl0 = sum(len(t) for t, _ in title)
        if rl0 + 6 > width - 6 - tl0:  # no room for the tab beside the title
            rtab = None
    if rtab is not None:
        rl = sum(len(t) for t, _ in rtab)
        rtab = _clamp_segs(rtab, max(0, width - 6 - sum(len(t) for t, _ in title) - 6))
        rl = sum(len(t) for t, _ in rtab)
    tl = sum(len(t) for t, _ in title)
    top: list[Text] = [Text(cs["tl"], style=bstyle), Text(cs["h"] + cs["jl"] + " ", style=bstyle)]
    for seg_t, seg_s in title:
        top.append(Text(seg_t, style=seg_s))
    top.append(Text(" " + cs["jr"], style=bstyle))
    if rtab is not None:
        run = max(0, width - (6 + tl) - (rl + 6))
        top.append(Text(cs["h"] * run, style=bstyle))
        top.append(Text(cs["jl"] + " ", style=bstyle))
        for seg_t, seg_s in rtab:
            top.append(Text(seg_t, style=seg_s))
        top.append(Text(" " + cs["jr"] + cs["h"] + cs["tr"], style=bstyle))
    else:
        run = max(0, width - (6 + tl) - 1)
        top.append(Text(cs["h"] * run, style=bstyle))
        top.append(Text(cs["tr"], style=bstyle))
    out = [Text.assemble(*top)]
    iw = width - 4  # two border cells + one padding space on each side
    for row in rows:
        out.append(
            Text.assemble(
                Text(cs["v"] + " ", style=bstyle),
                _fit(row, iw),
                Text(" " + cs["v"], style=bstyle),
            )
        )
    out.append(
        Text.assemble(
            Text(cs["bl"], style=bstyle),
            Text(cs["h"] * max(0, width - 2), style=bstyle),
            Text(cs["br"], style=bstyle),
        )
    )
    return out


def _gmeter_line(pct: float, width: int, pal: Palette) -> Text:
    """btop gradient meter: each filled cell ramps green→red by position."""
    f = round(max(0.0, min(100.0, pct)) / 100 * width)
    parts = [
        Text("█", style=f"bold {_ramp(round((i + 1) / max(1, width) * 100), pal)}")
        for i in range(f)
    ]
    parts.append(Text("▓" * max(0, width - f), style=f"bold {pal.track}"))
    return Text.assemble(*parts)


def _meter_line(
    treatment: str,
    pct: float,
    width: int,
    pal: Palette,
    color: str,
    history: list[float] | None = None,
) -> Text:
    """Configurable meter treatment for one utilisation row.

    ``gradient`` (btop ramp, the default), ``line`` (hairline fill over a
    hairline track), ``tick`` (dim scale + single bright marker at the value)
    or ``spark`` (history sparkline). Fill/marker colour escalates to caution
    at ≥75 and critical at ≥90; under a quiet palette the base colour is
    already neutral, so only escalation carries hue.
    """
    pct = max(0.0, min(100.0, pct))
    if pct >= 90:
        color = "#f7768e"
    elif pct >= 75:
        color = pal.warn
    if treatment == "spark":
        data = list(history) if history else [pct]
        return _spark_line(data, color, max(1, width))
    f = round(pct / 100 * width)
    if treatment == "line":
        return Text.assemble(
            Text("━" * f, style=f"bold {color}"),
            Text("─" * max(0, width - f), style=pal.dim),
        )
    if treatment == "tick":
        parts = [
            Text(
                "━" if i == f - 1 else ("╾" if i < f - 1 else ""),
                style=f"bold {color}" if i == f - 1 else pal.dim,
            )
            for i in range(width)
        ]
        parts.append(Text("┈" * max(0, width - f), style=pal.dim))
        return Text.assemble(*parts)
    return _gmeter_line(pct, width, pal)


def _bar_line(pct: float, width: int, color: str, pal: Palette) -> Text:
    """Single-hue meter (identity metrics): fill in the owning hue, dim track."""
    f = round(max(0.0, min(100.0, pct)) / 100 * width)
    return Text.assemble(
        Text("█" * f, style=f"bold {color}"),
        Text("▓" * max(0, width - f), style=f"bold {pal.track}"),
    )


def _stretch(data: list[float], width: int) -> list[float]:
    """Nearest-neighbour resample so a graph fills the full width even when
    history has fewer points than the graph has columns."""
    n = len(data)
    if n <= 0 or n >= width:
        return data
    return [data[i * n // width] for i in range(width)]


def _spark_line(data: list[float], color: str, width: int) -> Text:
    """Block sparkline (▁…█), single owning hue per series."""
    if not data or width <= 0:
        return Text(" " * width)
    lo, hi = min(data), max(data)
    span = (hi - lo) or 1
    glyphs = "▁▂▃▄▅▆▇█"
    seg = "".join(glyphs[min(7, int((v - lo) / span * 7.999))] for v in data[-width:])
    seg = seg.ljust(width)[:width]
    return Text(seg, style=f"bold {color}")


def _spark_duo_lines(data: list[float], color: str, width: int, pal: Palette) -> list[Text]:
    """Two-row sparkline in the serving chart's grammar: a connected braille
    dot line at the full series hue over a fill row in the chart's translucent
    fill shade — the hue blended 75% toward the background (25% strength, the
    same ``ALPHA`` treatment ``_lines_chart_lines`` applies). With several
    models the light fills sit adjacent without the solid-hue clash the old
    single-row block spark produced, so multi-model sparks read as blended.

    Always exactly 2 rows × ``width`` — the fill row exists under the hero
    variant too, so the serving row budget never depends on data presence.
    Scaled 0…max like the chart (sparks plot throughput; never negative).
    """
    if width <= 0:
        return [Text(""), Text("")]
    # Non-finite samples would poison the scale (max → NaN) and the dot
    # grid — drop them at the door (same policy as _lines_chart_lines).
    data = [v for v in data if math.isfinite(v)]
    if not data:
        return [Text(" " * width), Text(" " * width)]
    hi = max(data)
    span = hi if hi > 0 else 1.0
    ys = _interp_dot_rows(data, width * 2, 4, hi, span)
    cells = [0] * width
    for j, y in enumerate(ys):
        top = y if not j else min(ys[j - 1], y)
        bot = y if not j else max(ys[j - 1], y)
        for dr in range(top, bot + 1):
            cells[j // 2] |= _BRAILLE_BITS[(j % 2, dr)]
    fill = blend_toward(color, pal.bg, 0.75)
    return [
        _fit(
            Text(
                "".join(" " if not c else chr(0x2800 + c) for c in cells),
                style=f"bold {color}",
            ),
            width,
        ),
        _fit(Text("█" * width, style=f"bold {fill}"), width),
    ]


def _cores_line(vals: list[float], pal: Palette, spaced: bool = True) -> Text:
    """Per-core ■ squares, each cell ramped by its own load."""
    parts: list[Text] = []
    for i, v in enumerate(vals):
        if i and spaced:
            parts.append(Text(" "))
        parts.append(Text("■", style=f"bold {_ramp(v, pal)}"))
    return Text.assemble(*parts)


def _area_chart_lines(data: list[float], rows: int, width: int, pal: Palette) -> list[Text]:
    """Multi-row block-glyph area chart coloured per column by height. The
    history is resampled to the chart width so a short history still fills the
    window (no blank columns)."""
    if not data or rows <= 0 or width <= 0:
        return [Text(" " * width) for _ in range(max(0, rows))]
    lo, hi = min(data), max(data)
    span = (hi - lo) or 1
    norm = [(v - lo) / span for v in _stretch(data, width)]
    blk = " ▁▂▃▄▅▆▇█"
    out: list[Text] = []
    for r in range(rows):
        band = rows - r
        parts: list[Text] = []
        for nv in norm:
            lvl = nv * rows
            if lvl >= band:
                ch = "█"
            elif lvl <= band - 1:
                ch = " "
            else:
                ch = blk[max(1, round((lvl - (band - 1)) * 8))]
            parts.append(Text(ch, style=f"bold {_ramp(nv * 100, pal)}") if ch != " " else Text(" "))
        out.append(_fit(Text.assemble(*parts), width))
    return out


def _catmull(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Standard Catmull-Rom: interpolated value at ``t`` in p1→p2."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * p1
        + (p2 - p0) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (p3 - p0 + 3 * (p1 - p2)) * t3
    )


def _interp_dot_rows(
    hist: list[float], subw: int, dotrows: int, hi: float, span: float
) -> list[int]:
    """Resample ``hist`` onto ``subw`` sub-columns with Catmull-Rom smoothing
    and return each sub-column's dot row (0 = top) in a ``dotrows`` grid.

    Values are clamped to ``[0, hi]`` before mapping — the spline may
    overshoot sharp peaks, and an unclamped overshoot would walk the line
    outside the chart. ``span`` is the caller's shared scale denominator
    (never 0)."""
    n = len(hist)
    if n == 0 or subw <= 0 or dotrows <= 0:
        return []
    out: list[int] = []
    last = dotrows - 1
    for j in range(subw):
        t = j * (n - 1) / (subw - 1) if subw > 1 else 0.0
        i = min(int(t), n - 1)
        if i >= n - 1:
            v = hist[-1]
        else:
            v = _catmull(
                hist[i - 1] if i > 0 else hist[0],
                hist[i],
                hist[i + 1],
                hist[min(i + 2, n - 1)],
                t - i,
            )
        v = min(max(v, 0.0), hi)
        out.append(min(max(round(last * (1.0 - v / span)), 0), last))
    return out


# Braille dot bitmaps by (column, row) inside a 2×4 cell (⣿ = all bits).
_BRAILLE_BITS = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}


def _lines_chart_lines(
    series: list[tuple[str, str, list[float]]], rows: int, width: int, pal: Palette
) -> list[Text]:
    """Multi-series line chart on one shared time axis, in the smooth-line +
    translucent-fill grammar the design system's reference chart uses.

    Each series is Catmull-Rom resampled onto the 2-dot-wide braille
    sub-column grid and painted as a CONNECTED staircase: a sub-column
    carries every dot row between its sample and the previous one, so
    slopes read as a line instead of a scatter of single dots. Under each
    line every cell strictly below it is filled with a uniform ``█`` at
    ~25% of the series hue over the background — the terminal stand-in for
    a translucent area fill. Where two series' fills overlap the shades
    COMPOSITE (each layering 25% over the cell's current colour), so
    overlapping regions blend instead of the later series punching out the
    earlier one — several lines can share the same space. The scale is
    0…max(all series) — shared, never per-series — so the lines stay
    comparable, and a line pixel always wins over a fill pixel.
    """
    if rows <= 0:
        return []
    if width <= 0:
        return [Text("") for _ in range(rows)]
    # Non-finite samples would poison the shared scale (max → NaN) and the
    # dot grid (divmod on NaN) — drop them at the door so the renderer is
    # total over its input, not just over what _record happens to pass.
    plotted = [
        (label, color, [v for v in hist if math.isfinite(v)])
        for label, color, hist in series
        if hist
    ]
    plotted = [(label, color, d) for label, color, d in plotted if d]
    hi = max((max(d) for _, _, d in plotted), default=0.0)
    span = hi if hi > 0 else 1.0
    subw = width * 2
    dotrows = rows * 4
    line = [[0] * width for _ in range(rows)]
    line_style: list[list[str | None]] = [[None] * width for _ in range(rows)]
    fill: list[list[list[str]]] = [[[] for _ in range(width)] for _ in range(rows)]
    for _label, color, hist in plotted:
        ys = _interp_dot_rows(hist, subw, dotrows, hi, span)
        for j, y in enumerate(ys):
            cx, dx = divmod(j, 2)
            top = y if not j else min(ys[j - 1], y)
            bot = y if not j else max(ys[j - 1], y)
            for dr in range(top, bot + 1):
                cy, dy = divmod(dr, 4)
                line[cy][cx] |= _BRAILLE_BITS[(dx, dy)]
                if line_style[cy][cx] is None:
                    line_style[cy][cx] = color
        # Flat fill: the cell holding the line is the line's own; every
        # cell strictly below it takes the hue. No partial blocks, no
        # depth gradient — overlaps composite at render time.
        for cx in range(width):
            tl = ys[2 * cx]
            tr = ys[2 * cx + 1] if 2 * cx + 1 < len(ys) else tl
            top_cell = min(tl, tr) >> 2
            if top_cell >= rows - 1:
                continue
            for cy in range(top_cell + 1, rows):
                fill[cy][cx].append(color)
    ALPHA = 0.25
    shades: dict[tuple[str, ...], str] = {}
    out: list[Text] = []
    for r in range(rows):
        parts: list[Text] = []
        run: list[str] = []
        run_style: str | None = None
        for c in range(width):
            if line[r][c]:
                ch, style = chr(0x2800 + line[r][c]), line_style[r][c]
            elif fill[r][c]:
                hues = tuple(fill[r][c])
                shade = shades.get(hues)
                if shade is None:
                    shade = pal.bg
                    for hue in hues:
                        shade = blend_toward(hue, shade, 1.0 - ALPHA)
                    shades[hues] = shade
                ch, style = "█", shade
            else:
                ch, style = " ", None
            if style != run_style:
                if run:
                    parts.append(
                        Text("".join(run))
                        if run_style is None
                        else Text("".join(run), style=f"bold {run_style}")
                    )
                    run = []
                run_style = style
            run.append(ch)
        if run:
            parts.append(
                Text("".join(run))
                if run_style is None
                else Text("".join(run), style=f"bold {run_style}")
            )
        out.append(_fit(Text.assemble(*parts), width))
    return out


# ─── Chrome: waybar (the only chrome; footer removed) ────────────────


class Waybar(Static):
    """One-line waybar chrome: mode marker · centred cluster title · gen · KV ·
    online count (drop-to-fit; the base serving stats survive compression)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cluster: ClusterStats | None = None
        self._interval = 5

    def update_cluster(self, stats: ClusterStats, interval: int) -> None:
        self._cluster = stats
        self._interval = interval
        self.refresh()

    def render(self) -> Text:
        width = self.content_size.width
        if width <= 0:
            return Text()
        pal = _palette_for(self.app)
        c = self._cluster
        if c is None:
            return _fit(Text("…"), width)
        online = sum(1 for u in c.units if u.online)
        total = len(c.units)
        hosted = c.hosted_units
        model = hosted[0].model_name if hosted and hosted[0].model_name else "…"
        topo = c.topology.topology_type if c and c.topology else "…"
        risky = any((not u.online) or (u.temp_c >= TEMP_ALERT) for u in c.units)
        # A cluster with no counter-bearing unit stated no rate; "0 tok/s" would
        # be a manufactured reading of a live server, so the segment says so.
        gen = Text(
            f" {c.total_throughput:.0f} tok/s " if c.throughput_measured else " — tok/s ",
            style=f"bold {pal.fg}",
        )
        kv = Text(
            f" KV {c.kv_cache_pct:.0f}% " if c.kv_cache_pct >= 0 else " KV — ",
            style=f"bold {pal.accent}",
        )
        chip = Text(
            f" ● {online}/{total} ",
            style=f"bold {pal.ok}" if online == total else f"bold {pal.warn}",
        )
        badge = Text(" ! ", style=f"bold {pal.bg} on {pal.warn}") if risky else Text("")
        # The online chip is the last thing kept; KV drops before gen, and the
        # centred title drops before any stat — so the base serving stats (gen,
        # KV, online) always survive the compressed/table layouts.
        stats = [gen, kv, chip]
        avail = width - badge.cell_len
        for idx in (1, 0):  # drop KV before gen; the chip is never dropped
            if sum(s.cell_len for s in stats) <= avail:
                break
            if idx < len(stats):
                stats.pop(idx)
        right = Text.assemble(*stats)
        title = f"{topo} · {model}"
        midw = width - badge.cell_len - right.cell_len
        if midw >= 4:
            mid = title if len(title) <= midw else title[: max(0, midw - 1)] + "…"
            pad = max(0, midw - len(mid))
            mid_text = Text(" " * (pad // 2) + mid, style=pal.dim)
            out = Text.assemble(badge, mid_text, right)
        else:
            out = Text.assemble(badge, right)
        return _fit(out, width)


# ─── Shared metric formatting (ported from the AEON row grammar) ──────


def _fmt_tokens(n: int) -> str:
    """Format a token count compactly (82K, 1.5M, 380000)."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}K"
    return f"{n / 1_000_000:.1f}M"


def _kv_tokens_tail(used: int, total: int, pal: Palette) -> tuple[str, list[tuple[str, str]]]:
    """The kv row's token tail as ``(raw, segments)``.

    ``raw`` is what the caller pads by, so it must be exactly what the
    segments paint: a capacity-less node (the ``/get_load`` route) states its
    used count and no denominator, and padding for a denominator nobody
    reported ran the row one cell past the interior. With neither figure
    known the tail is the same em dash the rest of the pane uses for "no
    reading". A negative ``used`` is the cluster's unknown-used sentinel:
    capacity alone is stated, dim, with no fabricated numerator.
    """
    if total and used >= 0:
        return (
            f"{_fmt_tokens(used)}/{_fmt_tokens(total)} tok",
            [
                (_fmt_tokens(used), f"bold {pal.accent}"),
                (f"/{_fmt_tokens(total)} tok", pal.dim),
            ],
        )
    if total:
        return f"{_fmt_tokens(total)} tok", [(f"{_fmt_tokens(total)} tok", pal.dim)]
    if used > 0:
        return _fmt_tokens(used), [(_fmt_tokens(used), f"bold {pal.accent}")]
    return "—", [("—", pal.dim)]


def _kv_pct_raw(kv_pct: float) -> str:
    """The text the kv percentage tail occupies, for padding purposes.

    A negative percentage is the collector's "no reading" sentinel — the
    gauge was rejected, the route never states one, or no collector branch
    ever wrote the field (the dataclass default IS the sentinel) — so its
    raw width is the em-dash placeholder's.
    The segment itself is built where the row is drawn: an unknown percentage
    renders as the pane's placeholder row, with no tail at all.
    """
    return f"  {kv_pct:.0f}%" if kv_pct >= 0 else "  —"


def _fmt_axis(v: float) -> str:
    """Compact chart-axis tick (``0``, ``650``, ``5K``, ``1.3K``, ``2.5M``).

    Deliberately not ``_fmt_tokens``: an axis needs the decimal that the
    integer-quantized token format drops (1349 → ``1.3K``, not ``1K``)."""
    r = round(v)
    if r >= 1_000_000:
        return f"{r / 1_000_000:.1f}M"
    if r >= 1000:
        k = f"{r / 1000:.1f}K".replace(".0K", "K")
        return "1M" if k == "1000K" else k  # 999_999 must not read 1000K
    return f"{r:.0f}"


def _axis_labels(hi: float, rows: int, gutter: int, pal: Palette) -> list[Text]:
    """Right-aligned y ticks for the chart pane: the shared max on the top
    row, ``0`` on the baseline, and half-max mid-pane when the chart is at
    least four rows. Each label occupies ``gutter`` columns + one space."""
    ticks = {rows - 1: "0", 0: _fmt_axis(hi)}
    if rows >= 4:
        ticks[rows // 2] = _fmt_axis(hi / 2)
    return [Text(f"{ticks.get(r, ''):>{gutter}} ", style=pal.dim) for r in range(rows)]


def _fmt_freq(mhz: float) -> str:
    """Format a clock frequency in MHz as ``2411MHz``."""
    if mhz <= 0:
        return ""
    return f"{mhz:.0f}MHz"


def _fmt_rate(bps: float) -> str:
    """Format a byte rate as 82K, 1.5M or 3.2G bytes/s."""
    if bps < 1_000:
        return f"{bps:.0f}"
    if bps < 1_000_000:
        return f"{bps / 1000:.0f}K"
    if bps < 1_000_000_000:
        return f"{bps / 1_000_000:.1f}M"
    return f"{bps / 1_000_000_000:.1f}G"


def _roce_util_pct(s: SparkUnitStats) -> float:
    """RoCE wire utilization: observed (RX+TX) / full-duplex capacity."""
    if s.roce_capacity_bps <= 0:
        return 0.0
    return min(100.0, (s.roce_rx_bps + s.roce_tx_bps) / s.roce_capacity_bps * 100.0)


def _temp_style(c: float, pal: Palette) -> str:
    """Temperature treatment: dim below warm, warn when warm, bold-warn at alert."""
    if c >= TEMP_ALERT:
        return f"bold {pal.warn}"
    if c >= TEMP_WARM:
        return pal.warn
    return pal.dim


def _ttft_tail(seconds: float, pal: Palette) -> tuple[str, str]:
    """(marker, style) for the TTFT p95 tail: `!` past 2s, `!!` past 8s."""
    if seconds > 8.0:
        return "!!", f"bold {pal.warn}"
    if seconds > 2.0:
        return "!", f"bold {pal.warn}"
    return "", f"bold {pal.fg}"


# ─── ServingBox — the focused SERVING window ─────────────────────────


class ServingBox(Static):
    """The SERVING window: heavy focused border, caret tab inset in the top
    rule, a right meta tab, the design's metric rows, and the time-series
    line chart that fills the window's grown height.

    Every served model contributes one series (per-endpoint throughput
    history) to the same chart pane; with more than one model the wide
    grammar gains a gen/requests/ttft row per extra model (mirrored by the
    fit estimator's ``models`` parameter)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_data: list[float] = []
        self._prompt_data: list[float] = []
        self._kv_data: list[float] = []
        self._kv: dict | None = None
        self._models: list[dict] = []

    def update_throughput(
        self,
        gen_vals: list[float],
        prompt_vals: list[float],
    ):
        self._gen_data = list(gen_vals)
        self._prompt_data = list(prompt_vals)
        self.refresh()

    def update_models(self, models: list[dict]):
        """One dict per served model: name, color, gen (history), req, wait,
        ttft_p50_ms, ttft_p95_ms, source."""
        self._models = [dict(m) for m in models]
        self.refresh()

    def update_kv(
        self,
        pct: float,
        req: int,
        wait: int = 0,
        used_tok: int = 0,
        total_tok: int = 0,
        prefix_hit: float = -1.0,
        kv_history: list[float] | None = None,
        ttft_p50_ms: float = 0.0,
        ttft_p95_ms: float = 0.0,
        ttft_p99_ms: float = 0.0,
    ):
        self._kv = dict(
            pct=pct,
            req=req,
            wait=wait,
            used_tok=used_tok,
            total_tok=total_tok,
            prefix_hit=prefix_hit,
            ttft_p50_ms=ttft_p50_ms,
            ttft_p95_ms=ttft_p95_ms,
            ttft_p99_ms=ttft_p99_ms,
        )
        # The caller always passes a list: an empty (cleared) buffer must
        # empty the pane's series, not freeze the last poll's paint.
        if kv_history is not None:
            self._kv_data = kv_history
        self.refresh()

    def _model(self) -> str:
        return getattr(self.app, "_host_model", "") or "…"

    def render(self) -> Text:
        pal = _palette_for(self.app)
        width = max(1, self.content_size.width)
        if getattr(self.app, "floor", False):
            return Text("\n").join(_fit(r, width) for r in self._floor_rows(pal, width))
        tier = "rail" if getattr(self.app, "rail", False) else _density(self)
        interior = self._interior_rows(pal, width, _density(self), tier)
        rows = list(interior)
        chart_rows = getattr(self.app, "_chart_rows", 0)
        if chart_rows:
            if self._models:
                series = [(m["name"], m["color"], m["gen"]) for m in self._models]
                # Y ticks need the shared scale; with nothing plotted there
                # is no range to label, and a narrow pane cannot spare the
                # gutter — either way the chart renders unlabeled.
                hi = max((max(d) for _, _, d in series if d), default=0.0)
                # the mid tick (hi/2) can be WIDER than the top tick in the
                # ~1000-1049 band ('524' vs '1K') — size the gutter for both
                gutter = len(_fmt_axis(hi)) if hi > 0 else 0
                if hi > 0 and chart_rows >= 4:
                    gutter = max(gutter, len(_fmt_axis(hi / 2)))
                body = max(1, width - 4 - gutter - 1)
                if gutter and body < 12:
                    gutter = 0
                chart = _lines_chart_lines(
                    series, chart_rows, body if gutter else max(1, width - 4), pal
                )
                if gutter:
                    labels = _axis_labels(hi, chart_rows, gutter, pal)
                    chart = [Text.assemble(lab, row) for lab, row in zip(labels, chart)]
                rows.extend(chart)
            else:
                rows.extend(_area_chart_lines(self._gen_data, chart_rows, max(1, width - 4), pal))
        kv = self._kv or {}
        total_tok = kv.get("total_tok", 0)
        rtab = [(f"{_fmt_tokens(total_tok)} tok" if total_tok else "—", pal.dim)]
        title = [
            ("^", f"bold {pal.cyan}"),
            (" ", ""),
            ("serving", f"bold {pal.fg}"),
            (" ", ""),
        ]
        if self._models:
            # The chart legend: each model's name in its own series hue —
            # the same hue its chart line, gen row, and sparkline carry —
            # followed by the engine family serving it, so an SGLang model is
            # identifiable without waiting for a missing token counter.
            title.extend(
                _legend_segs(
                    self._models,
                    pal,
                    sum(len(t) for t, _ in title),
                    sum(len(t) for t, _ in rtab),
                    width,
                )
            )
        else:
            title.append((self._model(), pal.accent))
        return Text("\n").join(_box_lines(width, title, rtab, rows, False, pal))

    def _interior_rows(self, pal: Palette, width: int, density: str, tier: str) -> list[Text]:
        narrow = density == "compact" or width < SERVING_NARROW_WIDTH
        if narrow:
            return self._narrow_rows(pal, width, tier)
        return self._wide_rows(pal, width)

    def _wide_rows(self, pal: Palette, width: int) -> list[Text]:
        """The design's serving rows (4 aligned metric rows with graph
        spacers, then requests/cache/ttft/window). With several served
        models the gen/requests/ttft rows go per-model instead."""
        if len(self._models) > 1:
            return self._multi_rows(pal, width)
        # One model name on N endpoints: the cluster aggregate reads N× the
        # pane's own chart — the gen row's value AND its lo/avg/hi tail all
        # come from the authoritative rep's series.
        rep_gen = (
            self._models[0]["gen"] if self._models and self._models[0]["gen"] else self._gen_data
        )
        gen_avg = sum(rep_gen) / len(rep_gen) if rep_gen else 0.0
        prompt_avg = sum(self._prompt_data) / len(self._prompt_data) if self._prompt_data else 0.0
        lo = min(rep_gen) if rep_gen else 0.0
        hi = max(rep_gen) if rep_gen else 0.0
        s = self._kv or {}
        kv_pct = s.get("pct", -1.0)
        used = s.get("used_tok", 0)
        total = s.get("total_tok", 0)
        r = []
        # Top rows share one graph width and a padded tail so the graphs and
        has = bool(rep_gen)
        has_prompt = bool(self._prompt_data)
        source = self._models[0]["source"] if self._models else ""
        # No token counter (SGLang without --enable-metrics) means the endpoint
        # never stated a rate at all: a numeric tail would be a manufactured
        # zero that reads as "idle" during live generation. Name the reason
        # instead, in the same words the multi-model grammar uses.
        unknown_rate = f"no tok/s · {source}" if source else "no tok/s"
        tail_gen = f"{lo:.0f} · {gen_avg:.0f} · {hi:.0f} tok/s" if has else unknown_rate
        tail_prompt = f"{prompt_avg:.0f} tok/s" if has_prompt else unknown_rate
        tail_kv, kv_segs = _kv_tokens_tail(used, total, pal)
        tail_kvp = _kv_pct_raw(kv_pct)
        kvp_segs = [(tail_kvp, pal.accent)]  # only drawn when the fill is known
        tail_w = max(len(t) for t in (tail_gen, tail_prompt, tail_kv, tail_kvp))
        graph_w = max(3, width - 4 - 7 - 2 - tail_w)

        def tail(segs: list[tuple[str, str]], raw: str) -> list[Text]:
            pad = tail_w - len(raw)
            out = [Text(t, style=st) for t, st in segs]
            if pad > 0:
                out.append(Text(" " * pad, style=""))
            return out

        gen_tail = tail(
            [
                (f"{lo:.0f}", pal.dim),
                (" · ", pal.dim),
                (f"{gen_avg:.0f}", f"bold {pal.ok}"),
                (" · ", pal.dim),
                (f"{hi:.0f}", pal.fg),
                (" tok/s", pal.dim),
            ]
            if has
            else [(tail_gen, pal.dim)],
            tail_gen,
        )
        prompt_tail = tail(
            [
                (f"{prompt_avg:.0f}", f"bold {pal.fg}"),
                (" tok/s", pal.dim),
            ]
            if has_prompt
            else [(tail_prompt, pal.dim)],
            tail_prompt,
        )
        kv_tail = tail(kv_segs, tail_kv)
        kvp_tail = tail(kvp_segs, tail_kvp)

        def graph_row(label: str, graph: Text, tail_segs: list[Text]) -> Text:
            return Text.assemble(
                Text(label, style=pal.dim), graph, Text("  ", style=""), *tail_segs
            )

        # Gen row: the model's current output leads (the authoritative rep's
        # own series — with one model name served by N endpoints the cluster
        # aggregate reads N× what the chart plots), then the duo spark: dots
        # at the full hue over a fill row 75% toward the background — the
        # chart's dot + translucent-fill grammar at sparkline size.
        if self._models and self._models[0]["gen"]:
            hero = f"{self._models[0]['gen'][-1]:.0f}"
        elif self._gen_data:
            hero = f"{self._gen_data[-1]:.0f}"
        else:
            hero = "—"
        hue = self._models[0]["color"] if len(self._models) == 1 else pal.ok
        hero_style = f"bold {hue}" if hero != "—" else pal.dim
        spark_w = max(3, graph_w - len(hero) - 1)
        if has:
            duo = _spark_duo_lines(_stretch(rep_gen, spark_w), hue, spark_w, pal)
            gen_graph = Text.assemble(Text(hero, style=hero_style), Text(" ", style=""), duo[0])
            gen_fill = duo[1]
        else:
            # Absent series is not a flat zero line: blank the cell rather than
            # drawing a baseline the endpoint never reported. Both rows stay,
            # so the fit estimator's budget is unchanged.
            gen_graph = Text.assemble(
                Text(hero, style=hero_style),
                Text(" " * (graph_w - len(hero)), style=""),
            )
            gen_fill = Text(" " * spark_w, style="")
        r.append(graph_row("gen    ", gen_graph, gen_tail))
        r.append(Text.assemble(Text(" " * (7 + len(hero) + 1), style=""), gen_fill))
        r.append(Text("", style=""))
        prompt_duo = (
            _spark_duo_lines(_stretch(self._prompt_data, graph_w), pal.blue, graph_w, pal)
            if has_prompt
            else (Text(" " * graph_w, style=""), Text(" " * graph_w, style=""))
        )
        r.append(graph_row("prompt ", prompt_duo[0], prompt_tail))
        r.append(Text.assemble(Text(" " * 7, style=""), prompt_duo[1]))
        r.append(Text("", style=""))
        kv_duo = _spark_duo_lines(_stretch(self._kv_data, graph_w), pal.accent, graph_w, pal)
        r.append(graph_row("kv     ", kv_duo[0], kv_tail))
        r.append(Text.assemble(Text(" " * 7, style=""), kv_duo[1]))
        r.append(Text("", style=""))
        if self._kv is None or kv_pct < 0:
            # No capacity reading: a bar would have to invent a fill of 0.
            r.append(Text.assemble(Text("kv%    ", style=pal.dim), Text("—", style=pal.dim)))
        else:
            r.append(
                Text.assemble(
                    Text("kv%    ", style=pal.dim),
                    _bar_line(kv_pct, graph_w, pal.accent, pal)
                    if _treatment(self) == "gradient"
                    else _meter_line(
                        _treatment(self), kv_pct, graph_w, pal, pal.accent, list(self._kv_data)
                    ),
                    *kvp_tail,
                )
            )
        r.append(Text("", style=""))
        if self._kv is None:
            r.append(Text.assemble(Text("requests  ", style=pal.dim), Text("—", style=pal.dim)))
        else:
            r.append(self._requests_row(pal, s, width))
        r.append(self._cache_row(pal, s, width))
        r.append(self._ttft_row(pal, s, width))
        return r

    def _multi_rows(self, pal: Palette, width: int) -> list[Text]:
        """Multi-model serving pane: one aligned gen row per served model
        (its hue is its identity in the shared braille chart below), the
        aggregate prompt/kv rows, then per-model requests and ttft rows.
        Row count is ``WIDE_BASE + models + 2 + 4*(models-1)`` — mirrored by
        the fit estimator so nothing ever clips (each gen row is 2 rows: dot
        line + fill; prompt/kv are duo too)."""
        models = self._models
        name_w = min(20, max(len(m["name"]) for m in models))
        s = self._kv or {}
        used = s.get("used_tok", 0)
        total = s.get("total_tok", 0)
        kv_pct = s.get("pct", -1.0)
        prompt_avg = sum(self._prompt_data) / len(self._prompt_data) if self._prompt_data else 0.0
        # The aggregate prompt row speaks for every served model, so it names
        # an engine only when they agree — and it says "no tok/s" at all when
        # the cluster published no counter, rather than the 0 that an absent
        # series averages to.
        sources = {m["source"] for m in models}
        unknown_rate = f"no tok/s · {next(iter(sources))}" if len(sources) == 1 else "no tok/s"

        gen_tails: list[str] = []
        for m in models:
            gen = m["gen"]
            gen_tails.append(
                f"{sum(gen) / len(gen):.0f} · {max(gen):.0f} tok/s"
                if gen
                else f"no tok/s · {m['source']}"
            )
        tail_prompt = f"{prompt_avg:.0f} tok/s" if self._prompt_data else unknown_rate
        tail_kv, kv_segs = _kv_tokens_tail(used, total, pal)
        tail_kvp = _kv_pct_raw(kv_pct)
        kvp_segs = [(tail_kvp, pal.accent)]  # only drawn when the fill is known
        tail_w = max(
            [len(t) for t in gen_tails] + [len(tail_prompt), len(tail_kv), len(tail_kvp), 5]
        )
        # Engine badge — the model row names the engine serving it, so an
        # SGLang model is identifiable without waiting for a missing token
        # counter to give it away. Reserved as a fixed-width column so every
        # model's graph stays aligned, but only when the pane can hold it
        # beside the graph's real minimum width: `graph_w` is a budget the
        # graph content can exceed (a hero number plus the 3-cell spark
        # floor), and spending cells there would ellipsize the tok/s tail
        # that fits exactly without the badge (the title legend carries the
        # same tag when its own segment list has room for it). The badge is
        # spent from slack only, never from the name column below.
        hero_w = max((len(f"{m['gen'][-1]:.0f}") for m in models if m["gen"]), default=0)
        min_graph_w = hero_w + 4 if hero_w else 3
        tag_w = max(len(f" [{m['source']}]") for m in models)
        if tag_w > width - (4 + 7 + name_w + 2 + 2 + tail_w) - min_graph_w:
            tag_w = 0
        # The row budget is `4 + 7 + name_w + tag_w + 2 + graph + 2 + tail_w`
        # = width and the graph cannot shrink past its hero + spark floor, so
        # when the two minima cannot both fit it is the name column — the one
        # column that already truncates — that yields the cells. `_fit` would
        # otherwise ellipsize the tok/s tail, which is the content worth
        # keeping. The badge test above ran first, on the natural name width.
        slack = width - 4 - 7 - name_w - tag_w - 2 - 2 - tail_w - min_graph_w
        if slack < 0:
            name_w = max(4, name_w + slack)  # 3 name cells + the ellipsis
        # The per-model requests and ttft rows spend the same name column, so
        # the widest of them bounds it too. Each is linear in `name_w`, so one
        # measured pass closes its deficit exactly — without it a long name on
        # a narrow pane ellipsizes the p95 tail on those rows even when the gen
        # row above fits.
        for m in models:
            for row in (
                self._model_requests_row(pal, m, name_w),
                self._model_ttft_row(pal, m, name_w),
            ):
                over = row.cell_len - (width - 4)
                if over > 0:
                    name_w = max(4, name_w - over)
        graph_w = max(min_graph_w, width - 4 - 7 - name_w - tag_w - 2 - 2 - tail_w)

        def tail(text: str, segs: list[tuple[str, str]]) -> list[Text]:
            out = [Text(t, style=st) for t, st in segs]
            pad = tail_w - len(text)
            if pad > 0:
                out.append(Text(" " * pad, style=""))
            return out

        r: list[Text] = []
        for m, gen_tail in zip(models, gen_tails):
            name = (m["name"][: name_w - 1] + "…") if len(m["name"]) > name_w else m["name"]
            name = name.ljust(name_w)
            gen = m["gen"]
            if gen:
                avg = sum(gen) / len(gen)
                hi = max(gen)
                # The row leads with the model's current output, then the
                # duo spark: dots at the full hue over a fill row 75% toward
                # the background — the chart's grammar at sparkline size, in
                # the hue that owns this model everywhere else.
                hero = f"{gen[-1]:.0f}"
                spark_w = max(3, graph_w - len(hero) - 1)
                duo = _spark_duo_lines(_stretch(gen, spark_w), m["color"], spark_w, pal)
                graph = Text.assemble(
                    Text(hero, style=f"bold {m['color']}"),
                    Text(" ", style=""),
                    duo[0],
                )
                fill_prefix = 7 + name_w + tag_w + 2 + len(hero) + 1
                segs = [
                    (f"{avg:.0f}", f"bold {m['color']}"),
                    (" · ", pal.dim),
                    (f"{hi:.0f}", m["color"]),
                    (" tok/s", pal.dim),
                ]
            else:
                graph = Text(" " * graph_w)
                fill_prefix = 0
                segs = [(gen_tail, pal.dim)]
            r.append(
                Text.assemble(
                    Text("gen    ", style=pal.dim),
                    Text(name, style=m["color"]),
                    Text(f" [{m['source']}]".ljust(tag_w) if tag_w else "", style=pal.dim),
                    Text("  ", style=""),
                    graph,
                    Text("  ", style=""),
                    *tail(gen_tail, segs),
                )
            )
            if gen:
                # Second row of the duo budget: the model's fill shade,
                # aligned under its dots.
                r.append(
                    Text.assemble(
                        Text(" " * fill_prefix, style=""),
                        duo[1],
                    )
                )
            r.append(Text("", style=""))
        if self._prompt_data:
            prompt_duo = _spark_duo_lines(
                _stretch(self._prompt_data, graph_w), pal.blue, graph_w, pal
            )
            prompt_segs = [(f"{prompt_avg:.0f}", f"bold {pal.fg}"), (" tok/s", pal.dim)]
        else:
            # Absent series, as with the gen rows: a blank graph (a flat
            # baseline reads as "idle") and the reason rather than the 0 an
            # empty average would print.
            prompt_duo = (Text(" " * graph_w, style=""), Text(" " * graph_w, style=""))
            prompt_segs = [(tail_prompt, pal.dim)]
        r.append(
            Text.assemble(
                Text("prompt ", style=pal.dim),
                prompt_duo[0],
                Text("  ", style=""),
                *tail(tail_prompt, prompt_segs),
            )
        )
        r.append(Text.assemble(Text(" " * 7, style=""), prompt_duo[1]))
        r.append(Text("", style=""))
        kv_duo = _spark_duo_lines(_stretch(self._kv_data, graph_w), pal.accent, graph_w, pal)
        r.append(
            Text.assemble(
                Text("kv     ", style=pal.dim),
                kv_duo[0],
                Text("  ", style=""),
                *tail(tail_kv, kv_segs),
            )
        )
        r.append(Text.assemble(Text(" " * 7, style=""), kv_duo[1]))
        r.append(Text("", style=""))
        if kv_pct < 0:
            # No capacity reading: a bar would have to invent a fill of 0.
            r.append(Text.assemble(Text("kv%    ", style=pal.dim), Text("—", style=pal.dim)))
        else:
            r.append(
                Text.assemble(
                    Text("kv%    ", style=pal.dim),
                    _bar_line(kv_pct, graph_w, pal.accent, pal)
                    if _treatment(self) == "gradient"
                    else _meter_line(
                        _treatment(self), kv_pct, graph_w, pal, pal.accent, list(self._kv_data)
                    ),
                    *tail(tail_kvp, kvp_segs),
                )
            )
        r.append(Text("", style=""))
        for m in models:
            r.append(self._model_requests_row(pal, m, name_w))
        r.append(self._cache_row(pal, s, width))
        for m in models:
            r.append(self._model_ttft_row(pal, m, name_w))
        return r

    def _model_requests_row(self, pal: Palette, m: dict, name_w: int) -> Text:
        name = (m["name"][: name_w - 1] + "…") if len(m["name"]) > name_w else m["name"]
        wait = m.get("wait", 0)
        return Text.assemble(
            Text("requests  ", style=pal.dim),
            Text(name.ljust(name_w), style=m["color"]),
            Text("  ", style=""),
            Text(f"{m.get('req', 0)}r", style=f"bold {pal.fg}"),
            Text(" · ", style=pal.dim),
            Text(f"{wait}w waiting", style=pal.dim if wait == 0 else f"bold {pal.warn}"),
        )

    def _model_ttft_row(self, pal: Palette, m: dict, name_w: int) -> Text:
        name = (m["name"][: name_w - 1] + "…") if len(m["name"]) > name_w else m["name"]
        row = Text.assemble(
            Text("ttft      ", style=pal.dim),
            Text(name.ljust(name_w), style=m["color"]),
            Text("  ", style=""),
            Text("p50 ", style=pal.dim),
        )
        p50 = m.get("ttft_p50_ms", 0.0)
        p95 = m.get("ttft_p95_ms", 0.0)
        if p95 <= 0:
            row.append("—", style=pal.dim)
            return row
        marker, tail_style = _ttft_tail(p95 / 1000.0, pal)
        row.append(f"{p50 / 1000:.1f}s", style=f"bold {pal.fg}")
        row.append(" · ", style=pal.dim)
        row.append("p95 ", style=pal.dim)
        row.append(f"{p95 / 1000:.1f}s", style=tail_style)
        if marker:
            row.append(f" {marker}", style=tail_style)
        return row

    def _requests_row(self, pal, s, width) -> Text:
        req = s.get("req", 0)
        wait = s.get("wait", 0)
        return Text.assemble(
            Text("requests  ", style=pal.dim),
            Text(f"{req}r", style=f"bold {pal.fg}"),
            Text(" · ", style=pal.dim),
            Text(f"{wait}w waiting", style=pal.dim if wait == 0 else f"bold {pal.warn}"),
        )

    def _cache_row(self, pal, s, width) -> Text:
        hit = s.get("prefix_hit", -1.0)
        row = Text.assemble(
            Text("cache     ", style=pal.dim),
            (
                Text.assemble(
                    Text("hit ", style=pal.dim), Text(f"{hit:.0f}%", style=f"bold {pal.ok}")
                )
                if hit >= 0
                else Text("—", style=pal.dim)
            ),
            Text("  prefix reuse", style=pal.dim),
        )
        return row

    def _ttft_row(self, pal, s, width) -> Text:
        p50 = s.get("ttft_p50_ms", 0.0)
        p95 = s.get("ttft_p95_ms", 0.0)
        row = Text.assemble(Text("ttft      ", style=pal.dim), Text("p50 ", style=pal.dim))
        if p95 <= 0:
            row.append("—", style=pal.dim)
            return row
        marker, tail_style = _ttft_tail(p95 / 1000.0, pal)
        row.append(f"{p50 / 1000:.1f}s", style=f"bold {pal.fg}")
        row.append(" · ", style=pal.dim)
        row.append("p95 ", style=pal.dim)
        row.append(f"{p95 / 1000:.1f}s", style=tail_style)
        if marker:
            row.append(f" {marker}", style=tail_style)
        return row

    def _narrow_rows(self, pal: Palette, width: int, tier: str) -> list[Text]:
        """Prioritized narrow grammar (compact): gen · requests · ttft · kv% ·
        cache. The inline gen sparkline is the last chart visual; the window
        stat and the per-node serving rates are dropped (aggregate gen is the
        priority)."""
        gen_avg = sum(self._gen_data) / len(self._gen_data) if self._gen_data else 0.0
        s = self._kv or {}
        r = []
        # gen — value + inline sparkline (the last chart visual at compact).
        gen = Text.assemble(
            Text("gen ", style=pal.dim),
            Text(f"{gen_avg:.0f}", style=f"bold {pal.ok}")
            if self._gen_data
            else Text("—", style=pal.dim),
            Text(" tok/s", style=pal.dim),
        )
        if self._gen_data and width > 26:
            spark_w = min(18, max(4, width - 22))
            gen.append(" ", style=pal.dim)
            gen.append(_spark_line(self._gen_data, pal.ok, spark_w))
        r.append(gen)
        # requests — running/waiting concurrency.
        r.append(
            Text.assemble(
                Text("req ", style=pal.dim),
                Text(f"{s.get('req', 0)}r", style=f"bold {pal.fg}"),
                Text(" · ", style=pal.dim),
                Text(
                    f"{s.get('wait', 0)}w waiting",
                    style=pal.dim if s.get("wait", 0) == 0 else f"bold {pal.warn}",
                ),
            )
        )
        # ttft — p50—p95 plus the tail marker.
        p50 = s.get("ttft_p50_ms", 0.0)
        p95 = s.get("ttft_p95_ms", 0.0)
        ttft = Text.assemble(Text("ttft ", style=pal.dim))
        if p95 > 0:
            marker, tail_style = _ttft_tail(p95 / 1000.0, pal)
            ttft.append(f"{p50 / 1000:.1f}—{p95 / 1000:.1f}s", style=f"bold {pal.fg}")
            if marker:
                ttft.append(f" {marker}", style=tail_style)
        else:
            ttft.append("—", style=pal.dim)
        r.append(ttft)
        # kv% — plain value (the meter is dropped from the narrow grammar);
        # a capacity-less node has no fill to state, so the sentinel renders
        # as the same em dash the adjacent cache row uses.
        kv_pct = s.get("pct", -1.0)
        r.append(
            Text.assemble(
                Text("kv ", style=pal.dim),
                Text(
                    f"{kv_pct:.0f}%" if kv_pct >= 0 else "—",
                    style=f"bold {pal.accent}" if kv_pct >= 0 else pal.dim,
                ),
            )
        )
        # cache — dropped at rail so the base gen/req/ttft surface is minimal.
        if tier != "rail":
            hit = s.get("prefix_hit", -1.0)
            cache = Text.assemble(Text("cache ", style=pal.dim))
            cache.append(
                f"{hit:.0f}%" if hit >= 0 else "—",
                style=f"bold {pal.ok}" if hit >= 0 else pal.dim,
            )
            r.append(cache)
        return r

    def _floor_rows(self, pal: Palette, width: int) -> list[Text]:
        """Never-scroll floor: the base serving surface (gen · requests ·
        ttft) with no window frame."""
        gen_avg = sum(self._gen_data) / len(self._gen_data) if self._gen_data else 0.0
        s = self._kv or {}
        r = []
        gen = Text.assemble(
            Text("gen ", style=pal.dim),
            Text(f"{gen_avg:.0f}", style=f"bold {pal.ok}")
            if self._gen_data
            else Text("—", style=pal.dim),
            Text(" tok/s", style=pal.dim),
        )
        r.append(gen)
        r.append(
            Text.assemble(
                Text("req ", style=pal.dim),
                Text(f"{s.get('req', 0)}r", style=f"bold {pal.fg}"),
                Text(" · ", style=pal.dim),
                Text(
                    f"{s.get('wait', 0)}w waiting",
                    style=pal.dim if s.get("wait", 0) == 0 else f"bold {pal.warn}",
                ),
            )
        )
        p50 = s.get("ttft_p50_ms", 0.0)
        p95 = s.get("ttft_p95_ms", 0.0)
        ttft = Text.assemble(Text("ttft ", style=pal.dim))
        if p95 > 0:
            marker, tail_style = _ttft_tail(p95 / 1000.0, pal)
            ttft.append(f"{p50 / 1000:.1f}—{p95 / 1000:.1f}s", style=f"bold {pal.fg}")
            if marker:
                ttft.append(f" {marker}", style=tail_style)
        else:
            ttft.append("—", style=pal.dim)
        r.append(ttft)
        return r


# ─── NodeBox — a per-Spark tiling window ─────────────────────────────


def _short_label(label: str) -> str:
    """A 1-3 cell identity for a node: the trailing number of a ``name-N``
    label (e.g. ``spark-3`` -> ``3``), else the first three characters.
    Used at the condensed-table width where a full label cannot fit."""
    match = re.match(r"^(.*?)(\d+)$", label)
    if match:
        return match.group(2)
    return label[:3]


class NodeBox(Static):
    """A per-node window: light border, caret title (host=cyan, worker=orange)
    inset in the top rule, the configured host as the right meta tab. The tile
    grammar is width-driven: roomy/dense cards keep the configurable meters,
    ramped core grid and RoCE row; the compact meter card keeps meters + core
    grid but drops RoCE first; a narrow card runs gpu/mem/cpu text only; the
    floor table mode collapses the node to a single aligned row."""

    def __init__(self, idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = idx
        self._stats: SparkUnitStats | None = None
        self._gpu_history: list[float] = []
        self._mem_history: list[float] = []

    def update_node(
        self,
        s: SparkUnitStats,
        gpu_history: list[float] | None = None,
        mem_history: list[float] | None = None,
    ) -> None:
        self._stats = s
        self._gpu_history = list(gpu_history or [])
        self._mem_history = list(mem_history or [])
        self.refresh()

    def _host(self) -> str:
        try:
            url = self.app.settings.nodes[self.idx].vllm_url
            host = urlparse(url).hostname
            return host or self.app.settings.nodes[self.idx].label
        except Exception:
            return ""

    def render(self) -> Text:
        s = self._stats
        if s is None:
            node_cfg = self.app.settings.nodes[self.idx]
            s = SparkUnitStats(label=node_cfg.label, is_worker=getattr(node_cfg, "worker", False))
        pal = _palette_for(self.app)
        width = max(1, self.content_size.width)
        if getattr(self.app, "node_mode", "") == "table":
            return self._table_row(s, pal, width)
        density = _density(self)
        tier = "rail" if getattr(self.app, "rail", False) else density
        if width >= NODE_FULL_MIN and tier in ("roomy", "dense"):
            rows = self._interior_rows(s, pal, width, density, roce=True)
        elif width >= NODE_FULL_MIN and tier == "compact":
            rows = self._interior_rows(s, pal, width, density, roce=False)
        else:
            rows = self._text_rows(s, pal, width)
        role = "host" if not s.is_worker else "worker"
        role_style = pal.cyan if not s.is_worker else pal.warn
        caret_style = role_style if s.online else pal.warn
        title = [
            ("^" if s.online else "✗", f"bold {caret_style}"),
            (" ", ""),
            (s.label, f"bold {pal.fg}"),
            (" ", ""),
            (role, role_style),
        ]
        host = self._host()
        rtab = [(host, pal.dim)] if host else None
        return Text("\n").join(_box_lines(width, title, rtab, rows, False, pal))

    def _table_row(self, s: SparkUnitStats, pal: Palette, width: int) -> Text:
        """One aligned condensed-table row: online glyph, short label, then
        right-aligned GPU / MEM / CPU util. No window frame; _fit truncation
        keeps the label and GPU first when the tile is tight."""
        if s.online:
            gpu = f"{s.gpu_util_pct:.0f}%"
            mem = f"{s.mem_used_bytes / s.mem_total_bytes * 100:.0f}%" if s.mem_total_bytes else "—"
            cpu = (
                f"{sum(s.cpu_cores_util) / len(s.cpu_cores_util):.0f}%" if s.cpu_cores_util else "—"
            )
        else:
            gpu = mem = cpu = "—"
        parts = [
            Text("● " if s.online else "✗ ", style=pal.ok if s.online else pal.warn),
            Text(_short_label(s.label), style=f"bold {pal.fg}"),
            Text("  ", style=""),
            Text(f"{gpu:>4}", style=f"bold {pal.blue}"),
            Text(" ", style=pal.dim),
            Text(f"{mem:>4}", style=f"bold {pal.ok}" if s.online else pal.dim),
            Text(" ", style=pal.dim),
            Text(f"{cpu:>4}", style=f"bold {pal.warn}" if s.online else pal.dim),
        ]
        out = Text.assemble()
        for part in parts:
            if out.cell_len + part.cell_len > width:
                break  # drop trailing fields (cpu, then mem) cleanly, no ellipsis
            out.append_text(part)
        return _fit(out, width)

    def _text_rows(self, s: SparkUnitStats, pal: Palette, width: int) -> list[Text]:
        """Text-only node card: gpu/mem/cpu values with no meter, core grid or
        RoCE — the graph is dropped once the tile is tight; the card frame is
        retained."""
        dash = Text("—", style=pal.dim)
        rows: list[Text] = []
        # GPU: util + temp (power/clock drop out in text mode).
        if s.online:
            segs = [
                Text("g ", style=pal.dim),
                Text(f"{s.gpu_util_pct:.0f}%", style=f"bold {pal.blue}"),
            ]
            if s.temp_c:
                segs += [
                    Text(" · ", style=pal.dim),
                    Text(f"{s.temp_c:.0f}°", style=_temp_style(s.temp_c, pal)),
                ]
            rows.append(Text.assemble(*segs))
        else:
            rows.append(Text.assemble(Text("g ", style=pal.dim), dash))
        # MEM: used + pct.
        if s.online and s.mem_total_bytes > 0:
            used_gb = s.mem_used_bytes // (1024**3)
            used_pct = s.mem_used_bytes / s.mem_total_bytes * 100
            rows.append(
                Text.assemble(
                    Text("m ", style=pal.dim),
                    Text(f"{used_gb}G", style=f"bold {pal.fg}"),
                    Text(f" {used_pct:.0f}%", style=f"bold {pal.ok}"),
                )
            )
        else:
            rows.append(Text.assemble(Text("m ", style=pal.dim), dash))
        # CPU: util + temp.
        if s.online and s.cpu_cores_util:
            avg = sum(s.cpu_cores_util) / len(s.cpu_cores_util)
            segs = [
                Text("c ", style=pal.dim),
                Text(f"{avg:.0f}%", style=f"bold {pal.warn}"),
            ]
            if s.cpu_temp_c:
                segs += [
                    Text(" · ", style=pal.dim),
                    Text(f"{s.cpu_temp_c:.0f}°", style=_temp_style(s.cpu_temp_c, pal)),
                ]
            rows.append(Text.assemble(*segs))
        else:
            rows.append(Text.assemble(Text("c ", style=pal.dim), dash))
        return rows

    def _interior_rows(
        self, s: SparkUnitStats, pal: Palette, width: int, density: str, roce: bool = True
    ) -> list[Text]:
        compact = density == "compact" or (width - 4) <= 18
        iw = max(1, width - 4)
        mw = min(20, max(4, iw - 6))
        dash = Text("—", style=pal.dim)
        rows: list[Text] = []
        # GPU: util (headline) + temp + power; gradient meter beneath.
        glabel = "g " if compact else "gpu "
        if s.online:
            segs = [
                Text(glabel, style=pal.dim),
                Text(f"{s.gpu_util_pct:.0f}%", style=f"bold {pal.blue}"),
                Text(" · ", style=pal.dim),
                Text(f"{s.temp_c:.0f}°C", style=_temp_style(s.temp_c, pal)),
            ]
            if s.power_w > 0:
                segs += [Text(" · ", style=pal.dim), Text(f"{s.power_w:.0f}W", style=pal.accent)]
            # Add the SM clock only when it fits beside the headline stats;
            # at narrow tile widths power/temperature outrank the clock.
            if s.gpu_clock_mhz > 0:
                base_len = sum(t.cell_len for t in segs)
                if base_len + 3 + len(_fmt_freq(s.gpu_clock_mhz)) <= iw:
                    segs += [
                        Text(" · ", style=pal.dim),
                        Text(_fmt_freq(s.gpu_clock_mhz), style=pal.accent),
                    ]
            rows.append(Text.assemble(*segs))
            rows.append(
                _meter_line(_treatment(self), s.gpu_util_pct, mw, pal, pal.blue, self._gpu_history)
            )
        else:
            rows.append(Text.assemble(Text(glabel, style=pal.dim), dash))
            rows.append(_meter_line(_treatment(self), 0, mw, pal, pal.blue, self._gpu_history))
        # MEM: used/total + pct (+ swap); gradient meter beneath.
        mlabel = "m " if compact else "mem "
        if s.online and s.mem_total_bytes > 0:
            used_gb = s.mem_used_bytes // (1024**3)
            total_gb = s.mem_total_bytes // (1024**3)
            used_pct = s.mem_used_bytes / s.mem_total_bytes * 100
            mem = Text.assemble(
                Text(mlabel, style=pal.dim),
                Text(f"{used_gb}G", style=f"bold {pal.fg}"),
                Text(f"/{total_gb}G ", style=pal.dim),
                Text(f"{used_pct:.0f}%", style=f"bold {pal.ok}"),
            )
            if s.swap_total_kb > 0:
                swap_pct = s.swap_used_kb / s.swap_total_kb * 100
                swap_gb = s.swap_used_kb / (1024 * 1024)
                mem.append(" sw" if compact else "  swp", style=pal.dim)
                mem.append(
                    f" {swap_gb:.1f}G",
                    style=f"bold {pal.warn}" if swap_pct > 70 else f"bold {pal.fg}",
                )
            rows.append(mem)
            rows.append(_meter_line(_treatment(self), used_pct, mw, pal, pal.ok, self._mem_history))
        elif s.online:
            rows.append(
                Text.assemble(
                    Text(mlabel, style=pal.dim),
                    Text(f"{s.gpu_mem_pct:.0f}%", style=f"bold {pal.ok}"),
                )
            )
            rows.append(
                _meter_line(_treatment(self), s.gpu_mem_pct, mw, pal, pal.ok, self._mem_history)
            )
        else:
            rows.append(Text.assemble(Text(mlabel, style=pal.dim), dash))
            rows.append(_meter_line(_treatment(self), 0, mw, pal, pal.ok, self._mem_history))
        # CPU: util + temp + freq; ramped core grid beneath.
        clabel = "c " if compact else "cpu "
        if s.online and s.cpu_cores_util:
            avg = sum(s.cpu_cores_util) / len(s.cpu_cores_util)
            cpu = Text.assemble(
                Text(clabel, style=pal.dim),
                Text(f"{avg:.0f}%", style=f"bold {pal.warn}"),
                Text(" · ", style=pal.dim),
                Text(f"{s.cpu_temp_c:.0f}°C", style=_temp_style(s.cpu_temp_c, pal)),
            )
            rows.append(cpu)
            cores = list(s.cpu_cores_util[:20])
            spaced = not compact
            per = max(1, (iw + (1 if spaced else 0)) // (2 if spaced else 1))
            per = min(per, 20)
            crows = math.ceil(len(cores) / per) if cores else 0
            crows = min(crows, CORES_MAX_ROWS if not compact else 2)
            per = math.ceil(len(cores) / crows) if crows else per
            for i in range(0, len(cores), per):
                rows.append(_cores_line(cores[i : i + per], pal, spaced))
        else:
            rows.append(Text.assemble(Text(clabel, style=pal.dim), dash))
            rows.append(Text(""))
        # RoCE: RX/TX + wire utilisation, always its own row (accent hue) — but
        # the compact meter card drops it first (the lowest-priority graph).
        if roce:
            roce_row = Text.assemble(Text("roce ", style=pal.dim))
            if s.online and (s.roce_rx_bps > 0 or s.roce_tx_bps > 0):
                pct = _roce_util_pct(s)
                roce_row.append(f"↓{_fmt_rate(s.roce_rx_bps)}", style=f"bold {pal.accent}")
                roce_row.append(f" ↑{_fmt_rate(s.roce_tx_bps)}", style=f"bold {pal.accent}")
                if s.roce_capacity_bps > 0:
                    roce_row.append(f" {pct:.0f}%", style=f"bold {pal.accent}")
            else:
                roce_row.append("—", style=pal.dim)
            rows.append(roce_row)
        return rows


# ─── Theme palette cache ─────────────────────────────────────────────

_palette_cache: dict[tuple[str, bool], Palette] = {}


def _treatment(widget) -> str:
    """Configured meter treatment (batched: GPU/MEM/KV% share one value)."""
    return getattr(getattr(widget, "app", None), "settings", None).meter_treatment


def _palette_for(app: "DGXTop") -> Palette:
    """Resolve (and cache) the semantic palette for the app's active theme."""
    theme = app.current_theme
    quiet = bool(getattr(getattr(app, "settings", None), "quiet", False))
    key = (theme.name, quiet)
    cached = _palette_cache.get(key)
    if cached is None:
        cached = build_palette(theme, quiet=quiet)
        _palette_cache[key] = cached
    return cached


# ─── Fit-driven layout ───────────────────────────────────────────────


def _serving_base(tier: str, serv_width: int, models: int = 1) -> int:
    """SERVING interior rows without the area chart for (tier, serving width,
    model count). Rail/floor are the fixed base surfaces; the prioritized
    fused grammar is used below SERVING_NARROW_WIDTH and always at compact (a
    height-driven fold); the wide design grammar (no window stat) elsewhere.
    Each model beyond the first gains an extra gen row + spacer, its own
    requests row and its own ttft row (4 rows) in the wide grammar only. In
    the wide grammar every gen row is 2 rows (dot line + fill) and the
    prompt/kv sparks are duo as well — the ``+ models + 2`` term."""
    if tier == "rail":
        return SERVING_ROWS_RAIL
    if tier == "floor":
        return SERVING_ROWS_FLOOR
    if tier == "compact" or serv_width < SERVING_NARROW_WIDTH:
        return NARROW_BASE
    return WIDE_BASE + max(1, models) + 2 + 4 * max(0, models - 1)


def _serving_chart(tier: str, room: int) -> int:
    """Area-chart rows for a tier given ``room`` rows of leftover height
    (bounded: never thinner than the tier min, never taller than the tier
    max). Rail/floor carry none; the gen area chart is the last serving chart
    visual and survives compact."""
    if tier in ("rail", "floor"):
        return 0
    return min(CHART_MAX[tier], max(CHART_MIN[tier], room))


def _node_tile_rows(tier: str, tile_w: int) -> int:
    """Rendered height of one node tile (borders included): a 1-row condensed
    table row at floor; a framed full card (meters + core grid + RoCE) when the
    tile is wide enough and roomy/dense; a framed compact meter card (meters +
    core grid, no RoCE); else a framed text card (gpu/mem/cpu only). The
    estimator passes the *widest* grid column (ceil of the column width) so the
    full/meter/text decision matches what NodeBox.render actually does."""
    if tier == "floor":
        return 1
    if tile_w >= NODE_FULL_MIN:
        if tier in ("roomy", "dense"):
            return NODE_ROWS_FULL + 2
        if tier == "compact":
            return NODE_ROWS_METER + 2
    return NODE_ROWS_TEXT + 2


def _floor_cols(grid_w: int, n: int, row_budget: int) -> int:
    """Columns for the floor tier. Condensed table rows are packed as wide as
    the height budget allows (``row_budget`` node rows) and no wider than
    NODE_FLOOR_MAX cells, so a row keeps its gpu/mem/cpu fields when the
    column can carry them and the never-scroll floor holds."""
    cols = max(-(-n // max(1, row_budget)), max(1, grid_w // NODE_FLOOR_MAX))
    return min(n, cols)


def _node_layout(grid_w: int, n: int) -> tuple[int, str]:
    """(columns, node-tile mode) for a full-width grid of ``n`` nodes. Cards
    wrap into usable tiles whenever a single row is too narrow; the table mode
    is selected only at the floor (see _apply_tier)."""
    if n <= 4:
        return min(n, max(1, grid_w // NODE_CARD_MIN)), "card"
    single = grid_w // n
    if single >= NODE_CARD_MIN:
        return n, "card"
    return min(n, max(1, grid_w // NODE_CARD_MIN)), "card"


def _arrangement(width: int) -> bool:
    """True for the wide tiled arrangement (SERVING beside a node column);
    below TILING_WIDTH the SERVING hero stacks above the node grid."""
    return width >= TILING_WIDTH


def _serving_tiled_width(width: int) -> int:
    """Serving column width in cells for the tiled arrangement (explicit, so
    the estimator mirrors the resolved columns exactly — no ``fr`` rounding)."""
    return max(1, width * SERVING_TILED_FRAC // 100)


def _node_grid_columns(n: int, grid_w: int, tiled: bool) -> int:
    """Grid columns for ``n`` nodes in a ``grid_w``-cell container (never the
    floor, which packs via ``_floor_cols``). In the tiled arrangement a small
    cluster (<= 4) stacks its cards in one column beside SERVING (the
    reference tiling); otherwise the width rule applies."""
    if tiled and n <= 4:
        return 1
    return _node_layout(grid_w, n)[0]


def _grid_height(n: int, cols: int, tile: int, tier: str) -> int:
    """Height of a node grid: ``tile`` rows per tile row, GRID_GUTTER blank
    rows between tile rows (floor packs bare table lines flush)."""
    rows = -(-n // cols)
    gutter = 0 if tier == "floor" else GRID_GUTTER
    return rows * tile + max(0, rows - 1) * gutter


def _tier_fit(
    n: int, width: int, avail: int, tier: str, tiled: bool, models: int = 1
) -> tuple[bool, int, int, int, int]:
    """(fits, cols, node_h, serv_h, chart) for one tier in one arrangement.
    Mirrors the CSS row grammar exactly (estimate/layout duality): the stacked
    body is serv_h + GRID_GUTTER + node_h (no gutter at floor); the tiled body
    is max(serv_h, node_h). Textual distributes the width%cols remainder to
    the trailing 1fr columns, so the *widest* column is always reserved (ceil)
    and the floor rows are bare (no +2 window frame)."""
    if tiled:
        sw = _serving_tiled_width(width)
        nw = width - sw - TILING_GUTTER
        if tier == "floor":
            cols = _floor_cols(nw, n, avail - SERVING_ROWS_FLOOR)
        else:
            cols = _node_grid_columns(n, nw, tiled)
        tile = _node_tile_rows(tier, -(-nw // cols))
        node_h = _grid_height(n, cols, tile, tier)
        base = _serving_base(tier, sw, models)
        chart = _serving_chart(tier, avail - base - 2)
        serv_h = base + chart + (2 if tier != "floor" else 0)
        fits = node_h <= avail and serv_h <= avail
        return fits, cols, node_h, serv_h, chart
    if tier == "floor":
        cols = _floor_cols(width, n, avail - SERVING_ROWS_FLOOR)
    else:
        cols = _node_grid_columns(n, width, tiled)
    tile = _node_tile_rows(tier, -(-width // cols))
    node_h = _grid_height(n, cols, tile, tier)
    base = _serving_base(tier, width, models)
    if tier == "floor":
        serv_h = base  # bare floor lines, no window frame
        fits = serv_h + node_h <= avail
        return fits, cols, node_h, serv_h, 0
    body = base + CHART_MIN[tier] + 2 + GRID_GUTTER + node_h
    fits = body <= avail
    chart = _serving_chart(tier, avail - (base + 2 + GRID_GUTTER + node_h)) if fits else 0
    serv_h = base + chart + 2
    return fits, cols, node_h, serv_h, chart


_TIER_RANK = {name: i for i, name in enumerate(("roomy", "dense", "compact", "rail", "floor"))}


def _tier_for(n: int, width: int, height: int, tiled: bool, models: int = 1) -> str:
    """Densest tier whose estimated body fits (loosest first) for one
    arrangement; the floor is the unconditional fallback."""
    avail = height - WAYBAR_HEIGHT
    for tier in ("roomy", "dense", "compact", "rail"):
        if _tier_fit(n, width, avail, tier, tiled, models)[0]:
            return tier
    return "floor"


class DGXTop(App):
    """DGX Spark Cluster Inference Monitor — tiling desktop."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        overflow-y: hidden;
    }

    #waybar { height: 1; padding: 0; text-wrap: nowrap; text-overflow: clip; }

    /* Windows paint exact-width lines; never let Textual reflow/wrap them
       (a measurement-pass width mismatch would otherwise double a box's
       height). */
    ServingBox, NodeBox { text-wrap: nowrap; text-overflow: clip; }

    #body {
        layout: vertical;
        height: 1fr;
    }
    /* SERVING hero full-width on top; the node grid below (the stacked
       arrangement). At/above TILING_WIDTH the fit engine flips #body to
       horizontal and sizes the two columns in cells (tiled arrangement).
       The grid's column count and tile mode are set per-resize in _apply_tier
       (columns-only grid-size so any child count wraps; a fixed rows value
       would clamp and orphan overflow children). */
    #serving { height: auto; width: 1fr; }
    #node-col {
        layout: grid;
        height: auto;
        width: 1fr;
        grid-gutter: 1 0;   /* one blank row between tile rows; columns contiguous */
        margin: 1 0 0 0;     /* one blank row under SERVING (mirrored by the fit estimator) */
    }
    /* The floor packs bare table lines flush (margin and gutter reset). */
    #body.floor #node-col { margin: 0; grid-gutter: 0; }
    #node-col > NodeBox { height: auto; margin: 0; }

    /* The waybar is the only chrome and stays visible in every tier (it is a
       sibling of #body, so a descendant selector could not reach it). */
    """
    BINDINGS = [
        Binding("plus", "poll_faster", "Faster"),
        Binding("minus", "poll_slower", "Slower"),
        Binding("t", "change_theme", "Theme"),
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    cluster: reactive[ClusterStats | None] = reactive(None)

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        for custom in CUSTOM_THEMES:
            self.register_theme(custom)
        self.theme = self.settings.theme
        self.poll_speeds = sorted({1, 2, 5, 10, self.settings.poll_interval})
        self._polling = False
        self._poll_speed_idx = self.poll_speeds.index(self.settings.poll_interval)
        self._poll_timer = None
        self.history: dict[str, collections.deque] = {}
        self._current_topology: str = ""
        self._host_model: str = ""
        self._models_n = 1
        self._model_hue: dict[str, int] = {}
        self._model_hue_miss: dict[str, int] = {}
        self._last_size = (80, 24)
        self.density = ""
        self.cols = 0
        self.node_mode = ""
        self.tiled = False
        self._serv_h = 0
        self._node_h = 0
        self._chart_rows = 0
        self._pad = 0
        self._sw = 0
        self._nw = 0

    def get_driver_class(self):
        """Use resilient input unless Textual selected an explicit driver."""
        driver_class = super().get_driver_class()
        return ResilientLinuxDriver if driver_class is LinuxDriver else driver_class

    def compose(self):
        yield Waybar(id="waybar")
        with Vertical(id="body"):
            yield ServingBox(id="serving")
            with Vertical(id="node-col"):
                for index, _node in enumerate(self.settings.nodes):
                    yield NodeBox(index, id=f"node-{index}")

    def on_mount(self):
        self._poll_timer = self.set_interval(self._current_interval(), self._poll)
        self.run_worker(self._poll())
        self.run_worker(_init_model_names())

    def on_resize(self, event) -> None:
        self._last_size = (event.size.width, event.size.height)
        self._apply_tier(event.size.width, event.size.height)

    def _choose_layout(self, n: int, width: int, height: int) -> tuple[bool, str]:
        """(tiled, tier) for a viewport. The tiled arrangement is used only
        when it reaches a tier at least as loose as the stacked one: a narrow
        right column must never densify the serving surface (the hero chart is
        the priority). Ties prefer the tiled layout the request asks for."""
        tiled = _arrangement(width)
        tier_t = _tier_for(n, width, height, True, self._models_n)
        tier_s = _tier_for(n, width, height, False, self._models_n)
        if tiled and _TIER_RANK[tier_t] <= _TIER_RANK[tier_s]:
            return True, tier_t
        return False, tier_s

    def _apply_tier(self, width: int, height: int) -> None:
        n = len(self.settings.nodes)
        avail = height - WAYBAR_HEIGHT
        tiled, tier = self._choose_layout(n, width, height)
        _fits, cols, node_h, serv_h, chart = _tier_fit(n, width, avail, tier, tiled, self._models_n)
        node_mode = "table" if tier == "floor" else "card"
        rail = tier == "rail"
        floor = tier == "floor"
        density = "compact" if tier in ("compact", "rail", "floor") else tier
        pad = self._body_pad(avail, tier, tiled, node_h, serv_h)
        sw = _serving_tiled_width(width) if tiled else 0
        nw = width - sw - TILING_GUTTER if tiled else 0
        if (
            density == self.density
            and cols == self.cols
            and node_mode == self.node_mode
            and tiled == self.tiled
            and rail == self.rail
            and floor == self.floor
            and serv_h == self._serv_h
            and node_h == self._node_h
            and chart == self._chart_rows
            and pad == self._pad
            and sw == self._sw
            and nw == self._nw
        ):
            return
        self.density = density
        self.cols = cols
        self.node_mode = node_mode
        self.tiled = tiled
        self.rail = rail
        self.floor = floor
        self._serv_h = serv_h
        self._node_h = node_h
        self._chart_rows = chart
        self._pad = pad
        self._sw = sw
        self._nw = nw

        for name in ("compact", "dense", "roomy"):
            self.screen.set_class(name == density, name)
        body = self.query_one("#body", Vertical)
        body.set_class(tiled, "tiled")
        body.set_class(rail, "rail")
        body.set_class(floor, "floor")
        body.styles.layout = "horizontal" if tiled else "vertical"
        node_col = self.query_one("#node-col", Vertical)
        node_col.styles.grid_size_columns = cols
        node_col.styles.grid_columns = " ".join(["1fr"] * cols)
        serving = self.query_one("#serving", ServingBox)
        if tiled:
            sw = _serving_tiled_width(width)
            serving.styles.width = sw
            node_col.styles.width = width - sw - TILING_GUTTER
            node_col.styles.margin = (0, 0, 0, TILING_GUTTER)
        else:
            serving.styles.width = "1fr"
            node_col.styles.width = "1fr"
            node_col.styles.margin = (0, 0, 0, 0) if floor else (1, 0, 0, 0)
        self._apply_fill(pad)
        self._update_ui()

    def _body_pad(self, avail: int, tier: str, tiled: bool, node_h: int, serv_h: int) -> int:
        """Symmetric framing pad (bounded breathe). Rail/floor stay top-anchored
        (pad 0). The tiled body band is max(serv_h, node_h); the stacked body
        adds the grid gap."""
        if tier in ("rail", "floor"):
            return 0
        body = max(serv_h, node_h) if tiled else serv_h + GRID_GUTTER + node_h
        return max(0, (avail - body) // 2)

    def _apply_fill(self, pad: int) -> None:
        """Auto-size every window and centre the body with a top pad."""
        body = self.query_one("#body", Vertical)
        body.styles.margin = (pad, 0, 0, 0) if pad else 0
        self.query_one("#serving", ServingBox).styles.height = "auto"
        for node in self.query(NodeBox):
            node.styles.height = "auto"

    def watch_theme(self, theme_name: str) -> None:
        if not self.is_running:
            return
        self._update_ui()

    def _current_interval(self) -> int:
        return self.poll_speeds[self._poll_speed_idx]

    def _restart_polling(self):
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._polling = False
        self._poll_timer = self.set_interval(self._current_interval(), self._poll)
        self.run_worker(self._poll())
        self._update_ui()

    async def _poll(self):
        if self._polling:
            return
        self._polling = True
        try:
            stats = await poll_cluster()
            self.cluster = stats
        except Exception as e:
            log.warning("poll failed: %s", e)
            stats = ClusterStats()
            stats.units = [
                SparkUnitStats(label=node.label, error=str(e)) for node in self.settings.nodes
            ]
            self.cluster = stats
        finally:
            self._polling = False
        self._update_ui()

    def _update_ui(self):
        stats = self.cluster
        if stats is None:
            return
        self._update_kpis(stats)

    def _update_kpis(self, stats: ClusterStats):
        units = stats.units
        skipped: list[str] = []

        def _record(hist_key: str, value: float) -> None:
            if math.isfinite(value):
                self.history[hist_key].append(value)
            else:
                skipped.append(hist_key)

        self.history.setdefault(
            "throughput", collections.deque(maxlen=self.settings.history_length)
        )
        self.history.setdefault(
            "prompt-throughput", collections.deque(maxlen=self.settings.history_length)
        )
        # A cluster with no counter-bearing unit has no throughput measurement:
        # every hosted unit's `throughput_tok_s` is the dataclass default, and
        # recording it would draw a flat line that reads as "idle" rather than
        # "unknown" — the same reason the per-model series below is gated. The
        # retained samples go with it: an endpoint that stops reporting (a
        # restart without --enable-metrics, a flaky poll) must not leave the
        # pane painting its last-measured rate as the current one.
        if stats.throughput_measured:
            _record("throughput", stats.total_throughput)
            _record("prompt-throughput", stats.total_prompt_throughput)
        else:
            self.history["throughput"].clear()
            self.history["prompt-throughput"].clear()

        hosted_units = stats.hosted_units
        hosted_kv_keys = {f"kv-usage-{u.label}" for u in hosted_units}
        for key in list(self.history):
            if key.startswith("kv-usage-") and key not in hosted_kv_keys:
                self.history.pop(key)
        for u in hosted_units:
            key = f"kv-usage-{u.label}"
            self.history.setdefault(key, collections.deque(maxlen=self.settings.history_length))
            # An unknown fill (negative sentinel) is not a reading: seeding
            # a 0 would draw a valley the node never reported — and keeping
            # the old samples would paint a fill nobody reported any more,
            # the same policy as the throughput series on lost counters.
            if u.kv_cache_pct >= 0:
                _record(key, u.kv_cache_pct)
            else:
                self.history[key].clear()
        live_labels = {u.label for u in units if u.online}
        for key in [
            k
            for k in self.history
            if (k.startswith("gpu-") or k.startswith("mem-")) and k[4:] not in live_labels
        ]:
            self.history.pop(key)
        for u in units:
            if not u.online:
                continue
            mem_pct = u.mem_used_bytes / u.mem_total_bytes * 100 if u.mem_total_bytes else 0.0
            for prefix, value in (("gpu", u.gpu_util_pct), ("mem", mem_pct)):
                key = f"{prefix}-{u.label}"
                self.history.setdefault(key, collections.deque(maxlen=self.settings.history_length))
                _record(key, value)
        if skipped:
            log.warning(
                "dropped %d non-finite sample(s): %s", len(skipped), ", ".join(sorted(set(skipped)))
            )

        # One time-series series per served model: an endpoint re-reporting
        # the same engine (a TP worker sharing the pool, say) must never
        # split or double a model's line, so within each model name the unit
        # carrying the largest cumulative counter is the authoritative
        # reporter.
        reps: dict[str, SparkUnitStats] = {}
        for u in hosted_units:
            name = u.model_name or u.label
            cur = reps.get(name)
            if cur is None or u.generation_tokens_total > cur.generation_tokens_total:
                reps[name] = u
        gen_keys = {f"gen-{name}" for name in reps}
        for key in [k for k in list(self.history) if k.startswith("gen-") and k not in gen_keys]:
            self.history.pop(key)
        # A gone model's hue slot must not linger (a later new model takes
        # the freed slot), but ONE flaky poll that omits a live model must
        # not repaint it either — a model keeps its slot through a 3-poll
        # grace window before its mapping is released.
        for name in [k for k in list(self._model_hue) if k not in gen_keys]:
            self._model_hue_miss[name] = self._model_hue_miss.get(name, 0) + 1
            if self._model_hue_miss[name] >= 3:
                del self._model_hue[name]
                del self._model_hue_miss[name]
        for name in gen_keys:
            self._model_hue_miss.pop(name.removeprefix("gen-"), None)
        for name, u in reps.items():
            key = f"gen-{name}"
            self.history.setdefault(key, collections.deque(maxlen=self.settings.history_length))
            # An endpoint without a token counter (SGLang without
            # --enable-metrics) has NO throughput series — recording its
            # constant 0 would draw a flat line that reads as "idle", not
            # "unknown". The gen row and chart leave it blank instead, and
            # the samples from before it lost its counters are dropped for
            # the same reason the cluster series above are.
            if u.model_metrics:
                _record(key, u.throughput_tok_s)
            else:
                self.history[key].clear()
        pal = _palette_for(self)
        # One distinct identity hue per served model (themes.SERIES_HUES
        # cycle) — the chart line, title legend, per-model rows and inline
        # spark all read m["color"], so this one site coordinates them. A
        # model holds its slot for the app's lifetime (a topology change
        # never repaints an existing line); a first-seen model takes the
        # lowest slot free among the live models, and past eight concurrent
        # models the hue cycle repeats.
        live_used: set[int] = set()
        slots: dict[str, int] = {}
        for name in sorted(reps):
            held = self._model_hue.get(name)
            if held is not None and held not in live_used:
                slots[name] = held
                live_used.add(held)
        for name in sorted(reps):
            if name in slots:
                continue
            slot = next((i for i in range(len(SERIES_HUES)) if i not in live_used), None)
            if slot is None:  # more live models than hues: sharing unavoidable
                slot = len(slots) - len(SERIES_HUES)
            slots[name] = slot
            self._model_hue[name] = slot
            live_used.add(slot)
        models_payload = [
            dict(
                name=name,
                color=series_hue(slots[name], pal),
                gen=list(self.history.get(f"gen-{name}", [])),
                req=u.requests_running,
                wait=u.requests_waiting,
                ttft_p50_ms=u.ttft_p50_ms,
                ttft_p95_ms=u.ttft_p95_ms,
                source=u.model_source,
            )
            for name, u in sorted(reps.items())
        ]
        n_models = max(1, len(models_payload))
        if n_models != self._models_n:
            self._models_n = n_models
            # Each extra model claims 4 serving rows; re-check the tier fit
            # at the current viewport so the grammar never clips.
            self._apply_tier(*self._last_size)

        self._host_model = " · ".join(m["name"] for m in models_payload)

        serving = self.query_one("#serving", ServingBox)
        serving.update_throughput(
            gen_vals=list(self.history["throughput"]),
            prompt_vals=list(self.history["prompt-throughput"]),
        )
        # The kv spark must plot the same pool the headline percentage comes
        # from: the first hosted unit with a known fill (a load-only unit in
        # a mixed cluster states none and never records). No unit with a
        # reading: the first hosted unit, exactly as before this rule.
        kv_unit = next(
            (u for u in hosted_units if u.kv_cache_pct >= 0),
            hosted_units[0] if hosted_units else None,
        )
        kv_key = f"kv-usage-{kv_unit.label}" if kv_unit else ""
        serving.update_kv(
            stats.kv_cache_pct,
            hosted_units[0].requests_running if hosted_units else 0,
            wait=hosted_units[0].requests_waiting if hosted_units else 0,
            used_tok=stats.total_kv_used_tokens,
            total_tok=stats.total_kv_capacity_tokens,
            prefix_hit=stats.kv_prefix_hit_rate,
            kv_history=list(self.history.get(kv_key, [])),
            ttft_p50_ms=hosted_units[0].ttft_p50_ms if hosted_units else 0.0,
            ttft_p95_ms=hosted_units[0].ttft_p95_ms if hosted_units else 0.0,
            ttft_p99_ms=hosted_units[0].ttft_p99_ms if hosted_units else 0.0,
        )
        serving.update_models(models_payload)

        interval = self._current_interval()
        self.query_one("#waybar", Waybar).update_cluster(stats, interval)

        topo_type = stats.topology.topology_type if stats.topology else "UNKNOWN"
        if topo_type != self._current_topology:
            self._current_topology = topo_type

        for idx, s in enumerate(units):
            self.query_one(f"#node-{idx}", NodeBox).update_node(
                s,
                gpu_history=list(self.history.get(f"gpu-{s.label}", [])),
                mem_history=list(self.history.get(f"mem-{s.label}", [])),
            )

    def action_poll_faster(self):
        self._poll_speed_idx = max(0, self._poll_speed_idx - 1)
        self._restart_polling()

    def action_poll_slower(self):
        self._poll_speed_idx = min(len(self.poll_speeds) - 1, self._poll_speed_idx + 1)
        self._restart_polling()

    def action_refresh(self):
        self.run_worker(self._poll())


def run(config_path: str | Path | None = None) -> None:
    """Start the TUI, logging next to the effective config file."""
    if config_path is None:
        config_path = default_config_path()
    if not log.handlers:
        log_path = Path(config_path).expanduser().with_name("dgx-top.log")
        try:
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        except OSError:
            log.warning("cannot open log file %s; diagnostics will not be written", log_path)
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
    log.info("dgx-top starting")
    app = DGXTop()
    app.run()


if __name__ == "__main__":
    run()
