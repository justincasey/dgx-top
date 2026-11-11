from __future__ import annotations

import collections
import logging
import math
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from rich.text import Text
from textual.app import App, Binding
from textual.containers import Vertical
from textual.drivers.linux_driver import LinuxDriver
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

import discovery
from collector import (
    _init_model_names,
    pending_landed,
    poll_cluster,
    refresh_unit_model_name,
    set_clock_lock,
    set_unit_endpoint,
)
from config import (
    CLOCK_CEILING_MAX,
    CLOCK_CEILING_MIN,
    ConfigError,
    config_path,
    default_config_path,
    get_settings,
    save_node_endpoint,
)
from input_driver import ResilientLinuxDriver
from stats import ClusterStats, MemStack, SparkUnitStats, mem_stack
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
FLOOR_GAP = 3  # blank cells that separate two condensed nodes packed side by side
PENDING_RECHECK_S = 0.5  # how often to look for an outstanding first poll having landed
GIB = 1024**3
SERVING_NARROW_WIDTH = 52  # below this the SERVING hero uses the fused grammar

# The reference serving grammar: a fixed label gutter, a fixed-width series
# graph, and a two-space gap before the value column, so every value in the
# pane lands on the same cell regardless of its digits.
SERVING_LABEL_W = 9
SERVING_GRAPH_W = 22
# Wildest reference row: gutter, graph, gap, and the ttft tail value column.
SERVING_ROW_COL_W = 57
SERVING_SIDE_CHART_MIN = 20  # narrowest useful side chart
SERVING_SIDE_MIN_ROWS = 4  # under this, a stacked chart is not worth a row
_TAB_GAP = 3  # rule cells required between a window's title tab and meta tab

# The `area` renderer is the reference's fixed four-row footer: it never
# grows, so the row bands stay identical at every terminal size and the fit
# engine only decides which tier fits. The instrumented renderers keep their
# own behaviour — they FILL the leftover height between these bounds, which is
# the whole point of the taller pane. Rail/floor carry no chart.
AREA_CHART_ROWS = 4
CHART_MIN = {"roomy": 5, "compact": 2, "rail": 0, "floor": 0}
CHART_MAX = {"roomy": 16, "compact": 6, "rail": 0, "floor": 0}

# Interior rows per node tile (borders excluded), matching the reference card:
# gpu / meter / mem / stack-or-meter / legend-or-nothing / cpu / cores / roce.
# A full card keeps the meter, legend and RoCE; the compact card drops the
# meter, legend and RoCE, keeping the headlines, the stack bar and the cores.
NODE_ROWS_FULL = 8  # gpu,meter,mem,stack,legend,cpu,cores,roce (7 without stack)
NODE_ROWS_METER = 5  # gpu, mem, stack|meter, cpu, cores
NODE_ROWS_TEXT = 3  # text card interior (gpu, mem, cpu)

# Natural interior rows per SERVING window by density (borders excluded).
# Reference grammar labels every metric row with a 9-cell gutter, a 22-cell
# graph and a two-space gap before the value column:
#   gen / prompt / kv / kv% / blank / requests / cache / ttft
WIDE_BASE = 8  # roomy: the four graph rows, a spacer, the three meta rows
NARROW_BASE = 6  # compact: gen, kv, spacer, requests, cache, ttft
FUSED_BASE = 5  # below SERVING_NARROW_WIDTH: gen, requests, ttft, kv, cache
SERVING_ROWS_RAIL = 4  # gen, requests, ttft, kv (merged tail) — no chart
SERVING_ROWS_FLOOR = 3  # gen, requests, ttft — no chart

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
    """Active density: ``roomy`` or ``compact`` (rail/floor fold to compact)."""
    return getattr(widget.app, "density", "roomy")


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
        # Beside the tab the title needs a few cells of top rule between the
        # two junctions, or `├` and `┤` abut into a seam.
        if rl0 + 6 + _TAB_GAP > width - 6 - tl0:  # no room for the tab
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
        # A short (or brand-new) history is resampled across the whole meter
        # instead of leaving it nearly blank; the newest sample stays at the
        # right edge and the row still spans the card. Every meter input is a
        # 0-100 percentage, so the row is scaled against 100 rather than the
        # series' own extremes.
        return _spark_line(_stretch(data, max(1, width)), color, max(1, width), ceiling=100.0)
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


def _stretch(data: list[float], width: int) -> list[float]:
    """Nearest-neighbour resample so a graph fills the full width even when
    history has fewer points than the graph has columns."""
    n = len(data)
    if n <= 0 or n >= width:
        return data
    return [data[i * n // width] for i in range(width)]


def _spark_line(data: list[float], color: str, width: int, ceiling: float | None = None) -> Text:
    """Block sparkline (▁…█), single owning hue per series.

    Heights scale from a ZERO baseline, never min…max: the lowest sample of
    a window is a reading, not the floor of the scale. Percent meters pass
    their fixed ``ceiling`` (100) so a block's height means the value it
    prints; a series with no fixed domain (token rate) scales to its own
    peak, the same 0…max scaling the serving chart uses.

    Min-max did neither: a GPU steady at 96% collapsed onto the lowest
    glyph and read as 0% beside its own 96% label, while one unit of jitter
    in the same window saturated the row to 100%."""
    if not data or width <= 0:
        return Text(" " * width)
    values = [v for v in data if math.isfinite(v)][-width:]
    if not values:
        return Text(" " * width)
    hi = ceiling if ceiling and ceiling > 0 else max(values)
    span = hi if hi > 0 else 1.0
    glyphs = "▁▂▃▄▅▆▇█"
    seg = "".join(glyphs[min(7, int(min(1.0, max(0.0, v / span)) * 7.999))] for v in values)
    seg = seg.ljust(width)[:width]
    return Text(seg, style=f"bold {color}")


def _bar_meter(pct: float, width: int, pal: Palette) -> Text:
    """Single-hue utilisation bar: the filled cells in ``accent``, the rest in
    the meter track. The reference kv% row's own meter — it does not ramp by
    position the way the GPU meter does."""
    pct = max(0.0, min(100.0, pct))
    filled = round(pct / 100 * width)
    return Text.assemble(
        Text("█" * filled, style=f"bold {pal.accent}"),
        Text("▓" * max(0, width - filled), style=f"bold {pal.track}"),
    )


def _fit_name(name: str, budget: int) -> str:
    """A model name clipped to ``budget`` cells with an ellipsis (never
    negative, never empty when a name exists)."""
    if budget <= 0:
        return ""
    if len(name) <= budget:
        return name
    if budget <= 1:
        return "…"
    return name[: budget - 1] + "…"


def _kv_reading(used: int, total: int, pct: float, pal: Palette) -> Text:
    """The kv row's value: ``587K/3.8M tok`` when both figures are stated,
    the capacity alone when the fill was rejected (a numerator nobody
    measured is not painted as 0), the used count alone when the endpoint
    states no capacity, and an em dash when neither is known."""
    if total and pct >= 0:
        return Text.assemble(
            Text(f"{_fmt_tokens(used)}/{_fmt_tokens(total)}", style=f"bold {pal.fg}"),
            Text(" tok", style=pal.dim),
        )
    if total:
        return Text.assemble(
            Text(f"{_fmt_tokens(total)}", style=pal.dim), Text(" tok", style=pal.dim)
        )
    if used > 0:
        return Text.assemble(
            Text(f"{_fmt_tokens(used)}", style=f"bold {pal.fg}"), Text(" tok", style=pal.dim)
        )
    return Text("—", style=pal.dim)


def label_text(text: str, pal: Palette, style: str | None = None) -> Text:
    """A metric label padded to the reference gutter width."""
    return Text(text.ljust(SERVING_LABEL_W), style=style or pal.dim)


def _cores_line(vals: list[float], pal: Palette, spaced: bool = True) -> Text:
    """Per-core ■ squares, each cell ramped by its own load."""
    parts: list[Text] = []
    for i, v in enumerate(vals):
        if i and spaced:
            parts.append(Text(" "))
        parts.append(Text("■", style=f"bold {_ramp(v, pal)}"))
    return Text.assemble(*parts)


_EIGHTHS = "▏▎▍▌▋▊▉"  # left-aligned 1/8 … 7/8 blocks: sub-cell segment boundaries


def _fmt_gib(b: float) -> str:
    """Compact GiB figure: one decimal below 10 (``0.3``, ``9.1``), none above."""
    g = b / GIB
    return f"{g:.1f}" if g < 9.95 else f"{g:.0f}"


def _stack_tones(pal: Palette) -> dict[str, str]:
    """Stack segment hues from the semantic palette: weights → ok (the MEM
    hue), graphs → warn, kv → accent (the KV hue) with the pool's unfilled
    remainder the same hue dimmed toward the canvas but never off it,
    other → a lifted dim, free/reserved → the meter track."""
    return {
        "weights": pal.ok,
        "graphs": pal.warn,
        "kv": pal.accent,
        "kv_free": blend_toward(pal.accent, pal.bg, 0.35),
        "other": blend_toward(pal.dim, pal.fg, 0.4),
        "free": pal.track,
        "reserved": pal.track,
    }


def _stack_cells(stack: MemStack, width: int, pal: Palette) -> Text:
    """The node's memory as one segmented bar (weights | graphs | kv | other |
    free | reserved) whose whole width is physical RAM. A cell holding a
    segment boundary is a left-eighth block painted segment-on-segment, so
    boundaries land between cells; a cell spanning more than two segments
    keeps the two that cover most of it. The kv pool splits into its measured
    fill and the unfilled remainder (the whole pool one tone when the fill is
    unknown)."""
    kv_used = stack.kv if stack.kv_fill < 0 else stack.kv * min(100.0, stack.kv_fill) / 100
    sizes = (
        ("weights", stack.weights),
        ("graphs", stack.graphs),
        ("kv", kv_used),
        ("kv_free", stack.kv - kv_used),
        ("other", stack.other),
        ("free", stack.free),
        ("reserved", stack.reserved),
    )
    spans: list[tuple[str, float, float]] = []
    at = 0.0
    for key, size in sizes:
        if size > 0:
            spans.append((key, at, at + size))
            at += size
    if width <= 0 or not spans:
        return Text(" " * max(0, width))
    tone = _stack_tones(pal)
    order = {key: i for i, (key, _s, _e) in enumerate(spans)}
    per = at / width

    def solid(key: str) -> Text:
        if key == "free":
            return Text("▓", style=tone["free"])
        if key == "reserved":
            return Text("╱", style=f"{pal.dim} on {pal.track}")
        return Text("█", style=tone[key])

    cells: list[Text] = []
    for i in range(width):
        lo, hi = i * per, (i + 1) * per
        cover = [
            (key, min(e, hi) - max(s, lo)) for key, s, e in spans if min(e, hi) - max(s, lo) > 1e-9
        ]
        if not cover:  # float edge on the last cell
            cover = [(spans[-1][0], 1.0)]
        if len(cover) == 1:
            cells.append(solid(cover[0][0]))
            continue
        (lk, lw), (rk, rw) = sorted(
            sorted(cover, key=lambda c: -c[1])[:2], key=lambda c: order[c[0]]
        )
        eighths = round(lw / (lw + rw) * 8)
        if eighths <= 0:
            cells.append(solid(rk))
        elif eighths >= 8:
            cells.append(solid(lk))
        else:
            cells.append(Text(_EIGHTHS[eighths - 1], style=f"{tone[lk]} on {tone[rk]}"))
    return Text.assemble(*cells)


def _stack_legend(stack: MemStack, width: int, pal: Palette) -> Text:
    """Colour key for the stack bar, in the reference grammar: one ``■`` per
    segment with no space before its label, ``name`` then the GiB figure, and
    a single space between entries. Entries are ordered weights, graphs, kv,
    other and drop from the tail once the row is full. ``other`` is the only
    derived figure, so it carries the design's ``*`` provenance mark."""
    tone = _stack_tones(pal)
    items = (
        ("wts", stack.weights, tone["weights"], False),
        ("gfx", stack.graphs, tone["graphs"], False),
        ("kv", stack.kv, tone["kv"], False),
        ("oth", stack.other, tone["other"], True),
    )
    out = Text()
    for name, size, color, modelled in items:
        if size <= 0:
            continue
        value = f"{_fmt_gib(size)}" + ("*" if modelled else "")
        part = Text.assemble(Text("■", style=color), Text(f"{name} {value}", style=pal.dim))
        gap = 1 if out.cell_len else 0
        if out.cell_len + gap + part.cell_len > width:
            break
        if gap:
            out.append(" ")
        out.append_text(part)
    return out


def _resample(data: list[float], width: int) -> list[float]:
    """Nearest-neighbour resample of a series onto exactly ``width`` columns,
    preserving order and keeping the newest sample in the last column.

    Unlike ``_stretch`` this also DOWNSAMPLES: a 90-sample history drawn into
    a 57-cell pane must be compressed, not truncated — truncating kept the
    oldest 57 samples on screen and pushed the live end of the series past
    the pane edge."""
    n = len(data)
    if n <= 0 or width <= 0:
        return []
    return [data[i * n // width] for i in range(width)]


def _area_chart_lines(data: list[float], rows: int, width: int, pal: Palette) -> list[Text]:
    """Multi-row block-glyph area chart coloured per column by height, scaled
    from zero to the window's peak (absolute rate, as the reference footer
    plots it — a constant series sits at full height instead of collapsing to
    a baseline). The retained history is resampled onto the chart width (both
    directions), so the pane always shows the whole window newest-at-right."""
    if not data or rows <= 0 or width <= 0:
        return [Text(" " * width) for _ in range(max(0, rows))]
    hi = max(data)
    span = hi if hi > 0 else 1.0
    norm = [max(0.0, min(1.0, v / span)) for v in _resample(data, width)]
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


BRIGHT_CELLS = 24
"""Columns of the oscilloscope's bright leading window — the "now"."""

GRATICULE_STEP = 12
"""Column interval of the oscilloscope's graticule verticals."""


def _oscilloscope_lines(
    series: list[tuple[str, str, list[float]]],
    rows: int,
    width: int,
    pal: Palette,
    label_w: int = 0,
) -> list[Text]:
    """Multi-series decode trace in the oscilloscope grammar the design's
    reference mock validates: a dim graticule (``·`` verticals over a dotted
    ``┄`` baseline) behind connected braille staircases with NO area fill —
    history dimmed 55% toward the background, the trailing ``BRIGHT_CELLS``
    at the full hue — and each series' newest sample marked ``◉`` with an
    inline ``value name`` label on its own row (collision-resolved to one
    series per row, topmost first). ``label_w`` reserves that many right
    columns; 0 disables labels. The scale is shared (0…max over all series)
    so the traces stay comparable; like ``_lines_chart_lines`` the renderer
    is total over non-finite input. Every returned row is exactly ``width``
    cells."""
    if rows <= 0:
        return []
    if width <= 0:
        return [Text("") for _ in range(rows)]
    plotted = [
        (label, color, [v for v in hist if math.isfinite(v)])
        for label, color, hist in series
        if hist
    ]
    plotted = [(label, color, d) for label, color, d in plotted if d]
    hi = max((max(d) for _, _, d in plotted), default=0.0)
    span = hi if hi > 0 else 1.0
    # ``label_w`` counts the whole right label area: one gap cell + the
    # label text (label_w - 1), so the returned rows are exactly ``width``.
    body = max(1, width - label_w) if label_w else width
    subw = body * 2
    dotrows = rows * 4
    line = [[0] * body for _ in range(rows)]
    style: list[list[tuple[str, str] | None]] = [[None] * body for _ in range(rows)]
    marks: list[tuple[int, str, str]] = []
    for label, color, hist in plotted:
        ys = _interp_dot_rows(hist, subw, dotrows, hi, span)
        dim = blend_toward(color, pal.bg, 0.55)
        for j, y in enumerate(ys):
            cx, dx = divmod(j, 2)
            top = y if not j else min(ys[j - 1], y)
            bot = y if not j else max(ys[j - 1], y)
            for dr in range(top, bot + 1):
                cy, dy = divmod(dr, 4)
                line[cy][cx] |= _BRAILLE_BITS[(dx, dy)]
                if style[cy][cx] is None:
                    bright = cx >= body - BRIGHT_CELLS
                    style[cy][cx] = (f"bold {color}" if bright else dim, color)
        y_edge = min(ys[-1] >> 2, rows - 1)
        raw = f"{hist[-1]:.0f} {label}".rstrip() if label else f"{hist[-1]:.0f}"
        marks.append((y_edge, raw, color))
    # Edge markers + labels: one series per row, nearest free row wins.
    taken: set[int] = set()
    labels: dict[int, tuple[str, str]] = {}
    for y_edge, raw, color in sorted(marks, key=lambda m: m[0]):
        row = _free_row(y_edge, taken, rows)
        taken.add(row)
        labels[row] = (raw[: label_w - 1] if label_w else "", color)
    out: list[Text] = []
    for r in range(rows):
        parts: list[Text] = []
        for c in range(body):
            if r in labels and c == body - 1:
                parts.append(Text("◉", style=f"bold {labels[r][1]}"))
            elif line[r][c]:
                style_str, _color = style[r][c]
                parts.append(Text(chr(0x2800 + line[r][c]), style=style_str))
            elif r == rows - 1:
                parts.append(Text("┄", style=blend_toward(pal.dim, pal.bg, 0.4)))
            elif rows >= 4 and c % GRATICULE_STEP == GRATICULE_STEP - 1:
                parts.append(Text("·", style=blend_toward(pal.dim, pal.bg, 0.5)))
            else:
                parts.append(Text(" "))
        if label_w:
            raw, color = labels.get(r, ("", ""))
            text = " " + raw.ljust(label_w - 1)
            parts.append(Text(text, style=f"bold {color}" if color else ""))
        out.append(_fit(Text.assemble(*parts), width))
    return out


def _free_row(want: int, taken: set[int], rows: int) -> int:
    """Nearest row to ``want`` not in ``taken`` (ties downward), in range."""
    for offset in range(rows):
        up = want - offset
        if up >= 0 and up not in taken:
            return up
        down = want + offset
        if down < rows and down not in taken:
            return down
    return want


def _slope_row(series: list[tuple[str, str, list[float]]], pal: Palette) -> Text:
    """The oscilloscope's slope-into-it readout: the aggregate decode delta
    vs 5 ticks ago. Falling >8% of the current rate escalates to the warn
    ``▼ sagging``; anything else reads ``steady``; with under 6 ticks of
    history the row is the honest em dash — a fabricated 0 would read as
    idle."""
    live = [d for _, _, d in series]
    if not live or any(len(d) < 6 for d in live):
        return Text.assemble(Text("Δ5     ", style=pal.dim), Text("—", style=pal.dim))
    cur = sum(d[-1] for d in live)
    delta = cur - sum(d[-6] for d in live)
    falling = delta < -cur * 0.08
    return Text.assemble(
        Text("Δ5     ", style=pal.dim),
        Text(f"{delta:+.0f} tok/s", style=f"bold {pal.warn if falling else pal.ok}"),
        Text("  ▼ sagging" if falling else "  steady", style=pal.warn if falling else pal.dim),
    )


# ─── Chrome: waybar (the only chrome; footer removed) ────────────────


class Waybar(Static):
    """One-line waybar chrome: the key legend left, the working stats right
    (gen · clock lock · online count). Both sides drop to fit: the legend
    sheds single words from its tail before the stats shed anything, so
    ``tok/s`` and the fleet chip survive compression.

    The topology word, the model name and the KV reading are deliberately
    absent: the SERVING window names its model, and its ``kv%`` row carries
    the pool reading, so the bar spends its cells on live rates only."""

    #: Legend order, left to right. Each entry is (key glyph, single-word
    #: label); the labels stay one word so the full legend fits a wide bar.
    HINTS: tuple[tuple[str, str], ...] = (
        ("s", "scan"),
        ("+/-", "poll"),
        ("r", "refresh"),
        ("c", "clock"),
        ("t", "theme"),
        ("q", "quit"),
    )
    HINT_GAP = 3  # blank cells between legend entries
    HINT_MIN_GAP = 2  # blank cells kept between the legend and the stats

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cluster: ClusterStats | None = None
        self._interval = 5

    def update_cluster(self, stats: ClusterStats, interval: int) -> None:
        self._cluster = stats
        self._interval = interval
        self.refresh()

    def _fit_hints(self, budget: int, pal: Palette) -> Text:
        """The key legend, shedding single words from its tail until it fits
        ``budget`` cells (the stats already hold their space). A legend that
        cannot keep two entries is noise, so it hides entirely instead."""
        if budget <= 0:
            return Text()
        segs = [
            Text.assemble(
                (key, f"bold underline {pal.cyan}"),
                (" ", pal.dim),
                (label, pal.dim),
            )
            for key, label in self.HINTS
        ]
        sep = Text(" " * self.HINT_GAP, style=pal.dim)
        for keep in range(len(segs), 1, -1):
            out = Text()
            for i, seg in enumerate(segs[:keep]):
                if i:
                    out.append(sep)
                out.append(seg)
            if out.cell_len <= budget:
                return out
        return Text()

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
        risky = any((not u.online and not u.pending) or (u.temp_c >= TEMP_ALERT) for u in c.units)
        # A cluster with no counter-bearing unit stated no rate; "0 tok/s" would
        # be a manufactured reading of a live server, so the segment says so.
        gen = Text(
            f" {c.total_throughput:.0f} tok/s " if c.throughput_measured else " — tok/s ",
            style=f"bold {pal.fg}",
        )
        chip = Text(
            f" ● {online}/{total} ",
            style=f"bold {pal.ok}" if online == total else f"bold {pal.warn}",
        )
        badge = Text(" ! ", style=f"bold {pal.bg} on {pal.warn}") if risky else Text("")
        lock_full, lock_short = _lock_chips(c, pal)
        lock = lock_full
        # The online chip is the last thing kept; the rate drops before it. A
        # clock-lock chip first sheds its value, and goes only after gen —
        # the fleet chip always survives. KV lives on the SERVING kv% row.
        stats = [gen, chip]
        steps = (["shrink"] if lock_full else []) + ["gen"] + (["lock"] if lock_full else [])
        for step in steps:
            right = Text.assemble(*([lock] if lock else []), *stats)
            if badge.cell_len + right.cell_len + self.HINT_MIN_GAP <= width:
                break
            if step == "shrink":
                lock = lock_short
            elif step == "gen":
                stats.remove(gen)
            else:
                lock = None
        right = Text.assemble(*([lock] if lock else []), *stats)
        hints = self._fit_hints(width - badge.cell_len - right.cell_len - self.HINT_MIN_GAP, pal)
        gap = width - badge.cell_len - hints.cell_len - right.cell_len
        out = Text.assemble(badge, hints, Text(" " * max(0, gap)), right)
        return _fit(out, width)


CLOCK_STEP_MHZ = 100
"""One ``←``/``→`` press in the clocks menu moves the ceiling this far."""


def _lock_label(mhz: float) -> str:
    """A GPU clock ceiling as the waybar/card show it: ``≤2200``."""
    return f"≤{mhz:.0f}"


def _lock_chips(c: ClusterStats, pal: Palette) -> tuple[Text | None, Text | None]:
    """(full, short) waybar clock-lock chips, or (None, None) when no online
    node has a recorded lock. Full: ``≤2200`` when every online node is
    locked at one ceiling, else the lowest ceiling plus ``locked/online`` in
    warn. Short: ``≤`` alone."""
    online = [u for u in c.units if u.online]
    locks = [u.gpu_clock_lock_mhz for u in online if u.gpu_clock_lock_mhz > 0]
    if not locks:
        return None, None
    partial = len(locks) < len(online)
    uniform = len(set(locks)) == 1 and not partial
    style = f"bold {pal.cyan}" if uniform else f"bold {pal.warn}"
    text = f" {_lock_label(min(locks))}"
    if partial:
        text += f" {len(locks)}/{len(online)}"
    return Text(text, style=style), Text(" ≤ ", style=style)


def _lock_segs(s: SparkUnitStats, pal: Palette, used: int, iw: int) -> list[Text]:
    """Clock-lock segments for a node's GPU row after ``used`` cells of an
    ``iw``-cell interior: `` · ≤2200`` when it fits, `` ≤`` when only the
    marker fits, nothing when neither fits or no lock is recorded."""
    if s.gpu_clock_lock_mhz <= 0:
        return []
    full = _lock_label(s.gpu_clock_lock_mhz)
    if used + 3 + len(full) <= iw:
        return [Text(" · ", style=pal.dim), Text(full, style=pal.cyan)]
    if used + 2 <= iw:
        return [Text(" ≤", style=pal.cyan)]
    return []


# ─── Shared metric formatting (ported from the AEON row grammar) ──────


def _fmt_tokens(n: int) -> str:
    """Format a token count compactly (82K, 1.5M, 380000)."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}K"
    return f"{n / 1_000_000:.1f}M"


def _kv_tail(
    used: int, total: int, kv_pct: float, pal: Palette
) -> tuple[str, list[tuple[str, str]]]:
    """The kv row's merged token + percentage tail as ``(raw, segments)``.

    ``raw`` is what the caller pads by, so it must be exactly what the
    segments paint. The percentage rides the same tail — the kv% row is
    folded away — but only when a reading exists: a capacity without a
    usage gauge states capacity alone, a capacity-less node states its used
    count plus the reading it does have, and with neither figure known the
    tail is the same em dash the rest of the pane uses for "no reading".
    A negative ``used`` is the cluster's unknown-used sentinel: capacity is
    stated, dim, with no fabricated numerator.
    """
    if total and used >= 0:
        raw = f"{_fmt_tokens(used)}/{_fmt_tokens(total)} tok"
        segs = [
            (_fmt_tokens(used), f"bold {pal.accent}"),
            (f"/{_fmt_tokens(total)} tok", pal.dim),
        ]
    elif total:
        raw, segs = f"{_fmt_tokens(total)} tok", [(f"{_fmt_tokens(total)} tok", pal.dim)]
    elif used > 0:
        raw, segs = _fmt_tokens(used), [(_fmt_tokens(used), f"bold {pal.accent}")]
    else:
        return "—", [("—", pal.dim)]
    if kv_pct >= 0:
        raw += f" · {kv_pct:.0f}%"
        segs += [(" · ", pal.dim), (f"{kv_pct:.0f}%", f"bold {pal.accent}")]
    return raw, segs


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
    """Format a clock frequency as the reference row does: ``2463MHz``."""
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
        # Row count the frame is asked to occupy; 0 means auto (content).
        self.expected_height = 0

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
        # A pane narrower than the reference grammar carries no chart: the
        # fused rows are the whole surface (the fit engine mirrors this).
        chart_rows = 0 if width < SERVING_NARROW_WIDTH else getattr(self.app, "_chart_rows", 0)
        interior = self._interior_rows(pal, width, _density(self), tier)
        height = self.expected_height
        # Wide and short: the rows cannot stack a chart under them, so the
        # chart takes the pane's right-hand side at the full interior height
        # instead of leaving half the frame empty.
        below = height - 2 - len(interior) - 1 if height else chart_rows
        side = (
            width >= SERVING_NARROW_WIDTH
            and width - 4 >= SERVING_ROW_COL_W + 1 + SERVING_SIDE_CHART_MIN
            and below < SERVING_SIDE_MIN_ROWS
        )
        if side:
            rows = self._side_rows(pal, width, tier)
            chart_rows = 0  # the chart is already in the rows, beside them
        else:
            rows = list(interior)
            if chart_rows:
                rows.append(Text(""))
                rows.extend(self._chart(pal, width, chart_rows))
        kv = self._kv or {}
        total_tok = kv.get("total_tok", 0)
        # A stretched frame (tiled desk) keeps its slack as blank rows inside
        # the border, under the top-anchored grammar. Never measure the
        # widget's own ``size`` here: that height is the padding's own output
        # and would ratchet the frame and never let it shrink again.
        for _ in range(max(0, height - 2 - len(rows))):
            rows.append(Text(""))
        rtab = [(f"{_fmt_tokens(total_tok)} tok" if total_tok else "—", pal.dim)]
        title = [
            ("^", f"bold {pal.cyan}"),
            (" serving ", pal.dim),
        ]
        if self._models:
            # One model reads as the reference title: the name in the body
            # foreground. Several models need their own hues to stay
            # distinguishable, so the legend grammar takes over.
            if len(self._models) == 1:
                title.append((self._models[0]["name"], f"bold {pal.fg}"))
            else:
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

    def _side_rows(self, pal: Palette, width: int, tier: str) -> list[Text]:
        """Rows for a wide, short pane: the metric column keeps its reference
        width and the chart takes every remaining cell at the full interior
        height. The chart is the pane's only elastic element, so on a pane too
        short to stack it under the rows it belongs beside them."""
        inner = max(1, width - 4)
        row_w = SERVING_ROW_COL_W
        chart_w = max(1, inner - row_w - 1)
        interior = self._interior_rows(pal, row_w + 4, _density(self), tier)
        if self.expected_height:
            interior = interior[: max(1, self.expected_height - 2)]
        foot = (
            1 if getattr(getattr(self.app, "settings", None), "chart", "") == "oscilloscope" else 0
        )
        chart = self._chart(pal, chart_w + 4, max(1, len(interior) - foot))
        blank = Text(" " * chart_w)
        rows: list[Text] = []
        for i, row in enumerate(interior):
            right = chart[i] if i < len(chart) else blank
            rows.append(Text.assemble(_fit(row, row_w), Text(" "), _fit(right, chart_w)))
        return rows

    def _chart(self, pal: Palette, width: int, chart_rows: int) -> list[Text]:
        """The pane's footer chart: ``area`` (the reference look) is a ramp-
        coloured area chart with no gutter, axis labels or overlay marks;
        ``oscilloscope`` and ``lines`` keep the instrumented renderers."""
        mode = getattr(getattr(self.app, "settings", None), "chart", "area")
        iw = max(1, width - 4)
        series = [(m["name"], m["color"], m["gen"]) for m in self._models]
        if not series and self._gen_data:
            series = [(self._model(), pal.ok, self._gen_data)]
        if mode == "area":
            data = []
            if len(series) == 1:
                data = series[0][2]
            if not data:
                data = self._gen_data or next((d for _, _, d in series if d), [])
            return _area_chart_lines(data, chart_rows, iw, pal)
        hi = max((max(d) for _, _, d in series if d), default=0.0)
        # the mid tick (hi/2) can be WIDER than the top tick in the
        # ~1000-1049 band ('524' vs '1K') — size the gutter for both.
        gutter = len(_fmt_axis(hi)) if hi > 0 else 0
        if hi > 0 and chart_rows >= 4:
            gutter = max(gutter, len(_fmt_axis(hi / 2)))
        if mode == "oscilloscope":
            # Inline labels ride the chart's right edge (value + name,
            # collision-resolved); they need 3+ rows and spare width.
            label_w = 0
            if series and chart_rows >= 3:
                widest = max(
                    (len(f"{d[-1]:.0f} {name}".rstrip()) for name, _, d in series if d),
                    default=0,
                )
                room = width - 4 - gutter - 1 - 12
                if widest and widest + 1 <= room:
                    label_w = widest + 1
            body = max(1, width - 4 - gutter - 1 - label_w)
            if body < 12:
                if label_w:
                    label_w = 0
                    body = max(1, width - 4 - gutter - 1)
                if gutter and body < 12:
                    gutter = 0
                    body = max(1, width - 4)
            chart = _oscilloscope_lines(series, chart_rows, body + label_w, pal, label_w)
            if gutter:
                labels = _axis_labels(hi, chart_rows, gutter, pal)
                chart = [Text.assemble(lab, row) for lab, row in zip(labels, chart)]
            return chart + [_slope_row(series, pal)]
        body = max(1, width - 4 - gutter - 1)
        if gutter and body < 12:
            gutter = 0
        chart = _lines_chart_lines(series, chart_rows, body if gutter else iw, pal)
        if gutter:
            labels = _axis_labels(hi, chart_rows, gutter, pal)
            chart = [Text.assemble(lab, row) for lab, row in zip(labels, chart)]
        return chart

    def _interior_rows(self, pal: Palette, width: int, density: str, tier: str) -> list[Text]:
        if tier == "rail" or width < SERVING_NARROW_WIDTH:
            return self._narrow_rows(pal, width, tier)
        return self._rows(pal, width, density == "compact")

    def _rows(self, pal: Palette, width: int, compact: bool) -> list[Text]:
        """The reference serving grammar: a 9-cell label gutter, a 22-cell
        graph, a two-space gap, then the value column — so every value in the
        pane starts on the same cell.

        roomy  gen / prompt / kv / kv% / · / requests / cache / ttft / · /
               deploy / rank × node / · / chart
        compact  the same rows with the prompt and kv% graphs and the chart
               removed (gen, kv, ·, requests, cache, ttft, ·, deploy, ranks)

        Per-model rows appear only when more than one model is served: the
        cluster aggregate keeps one row per metric, so the model name rides a
        second requests/ttft row rather than repeating the label."""
        s = self._kv or {}
        iw = max(1, width - 4)
        label_w = SERVING_LABEL_W
        graph_w = max(3, min(SERVING_GRAPH_W, iw - label_w - 2))

        def label(text: str) -> Text:
            return Text(text.ljust(label_w), style=pal.dim)

        def spark(data: list[float], color: str) -> Text:
            """The fixed-width series spark: 0…peak scaling, resampled so a
            short history fills the 22-cell lane at its true height."""
            return _spark_line(_stretch(list(data), graph_w), color, graph_w)

        def graph_row(name: str, graph: Text, value: Text) -> Text:
            return Text.assemble(label(name), graph, Text("  ", style=""), value)

        gen = list(self._gen_data)
        prompt = list(self._prompt_data)
        kv_series = list(self._kv_data)
        used = s.get("used_tok", 0)
        total = s.get("total_tok", 0)
        r: list[Text] = []
        # gen — current decode rate over the sparkline; the value is the last
        # sample (the row reads as "now", the graph as "recently").
        if gen:
            r.append(
                graph_row(
                    "gen",
                    spark(gen, pal.ok),
                    Text.assemble(
                        Text(f"{gen[-1]:.0f}", style=f"bold {pal.fg}"),
                        Text(" tok/s", style=pal.dim),
                    ),
                )
            )
        else:
            r.append(
                graph_row(
                    "gen",
                    Text(" " * graph_w, style=""),
                    Text("— tok/s", style=pal.dim),
                )
            )
        if not compact:
            r.append(
                graph_row(
                    "prompt",
                    spark(prompt, pal.blue) if prompt else Text(" " * graph_w),
                    Text.assemble(
                        (
                            Text(f"{prompt[-1]:.0f}", style=f"bold {pal.fg}")
                            if prompt
                            else Text("—", style=pal.dim)
                        ),
                        Text(" tok/s", style=pal.dim),
                    ),
                )
            )
        kv_value = _kv_reading(used, total, s.get("pct", -1.0), pal)
        r.append(
            graph_row(
                "kv",
                spark(kv_series, pal.accent) if kv_series else Text(" " * graph_w),
                kv_value,
            )
        )
        if not compact:
            pct = s.get("pct", -1.0)
            r.append(
                graph_row(
                    "kv%",
                    _bar_meter(pct, graph_w, pal) if pct >= 0 else Text(" " * graph_w),
                    (
                        Text(f"{pct:.0f}%", style=f"bold {pal.accent}")
                        if pct >= 0
                        else Text("—", style=pal.dim)
                    ),
                )
            )
        r.append(Text(""))
        if self._kv is None:
            r.append(Text.assemble(label("requests"), Text("—", style=pal.dim)))
        elif len(self._models) > 1:
            # One row per model: an aggregate concurrency row beside them
            # would read as a third server.
            for m in self._models:
                r.append(self._requests_row(pal, m, iw, m["name"]))
        else:
            r.append(self._requests_row(pal, s, iw))
        r.append(self._cache_row(pal, s))
        if len(self._models) > 1:
            for m in self._models:
                r.append(self._ttft_row(pal, m, iw, m["name"]))
        else:
            r.append(self._ttft_row(pal, s, iw))
        return r

    def _requests_row(self, pal: Palette, s: dict, iw: int, name: str = "") -> Text:
        """``requests 1r · 0w waiting`` (per-model rows carry the name, fitted
        to the pane so the row can never overflow its interior)."""
        req = s.get("req", 0)
        wait = s.get("wait", 0)
        value = Text.assemble(
            Text(f"{req}r", style=f"bold {pal.fg}"),
            Text(" · ", style=pal.dim),
            Text(f"{wait}w waiting", style=pal.dim if wait == 0 else f"bold {pal.warn}"),
        )
        row = label_text("requests", pal)
        if name:
            row.append_text(
                Text(
                    f"{_fit_name(name, iw - 9 - 2 - value.cell_len)}  ",
                    style=s.get("color", pal.fg),
                )
            )
        row.append_text(value)
        return row

    def _cache_row(self, pal: Palette, s: dict) -> Text:
        """``cache     hit 87%  prefix reuse``."""
        hit = s.get("prefix_hit", -1.0)
        row = label_text("cache", pal)
        if hit >= 0:
            row.append_text(Text("hit ", style=pal.dim))
            row.append_text(Text(f"{hit:.0f}%", style=f"bold {pal.ok}"))
            row.append_text(Text("  prefix reuse", style=pal.dim))
        else:
            row.append_text(Text("—", style=pal.dim))
        return row

    def _ttft_row(self, pal: Palette, s: dict, iw: int, name: str = "") -> Text:
        """``ttft      p50 6.0s · p95 13.1s !!`` — the tail marker escalates
        past 2s/8s, and the p95 (not the median) carries it."""
        p50 = s.get("ttft_p50_ms", 0.0)
        p95 = s.get("ttft_p95_ms", 0.0)
        marker, tail_style = _ttft_tail(p95 / 1000.0, pal) if p95 > 0 else ("", pal.dim)
        value = (
            Text.assemble(
                Text("p50 ", style=pal.dim),
                Text(f"{p50 / 1000:.1f}s", style=f"bold {pal.fg}"),
                Text(" · p95 ", style=pal.dim),
                Text(f"{p95 / 1000:.1f}s", style=tail_style),
                Text(f" {marker}" if marker else "", style=tail_style),
            )
            if p95 > 0
            else Text("—", style=pal.dim)
        )
        row = label_text("ttft", pal)
        if name:
            row.append_text(
                Text(
                    f"{_fit_name(name, iw - 9 - 2 - value.cell_len)}  ",
                    style=s.get("color", pal.fg),
                )
            )
        row.append_text(value)
        return row

    def _narrow_rows(self, pal: Palette, width: int, tier: str) -> list[Text]:
        """Fused narrow grammar, used when the pane is narrower than
        SERVING_NARROW_WIDTH or at rail: gen · requests · ttft · kv (merged
        token + pct tail) · cache. No chart, no deploy/rank block — at this
        width the aggregate rate and latency are the whole surface."""
        s = self._kv or {}
        r = []
        # The gen value is the CURRENT decode rate — the last sample, the same
        # number the waybar prints. Averaging the window here made the row
        # drift slowly upward and disagree with the header.
        gen_now = self._gen_data[-1] if self._gen_data else 0.0
        gen = Text.assemble(
            Text("gen ", style=pal.dim),
            Text(f"{gen_now:.0f}", style=f"bold {pal.ok}")
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
        # kv — the merged token + percentage tail; the kv% row is folded
        # away, so capacity and reading (or the honest em dash when neither
        # is known) live on this one row.
        tail_kv, kv_segs = _kv_tail(
            s.get("used_tok", 0), s.get("total_tok", 0), s.get("pct", -1.0), pal
        )
        r.append(Text.assemble(Text("kv ", style=pal.dim), *kv_segs))
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
        gen_now = self._gen_data[-1] if self._gen_data else 0.0
        s = self._kv or {}
        r = []
        gen = Text.assemble(
            Text("gen ", style=pal.dim),
            Text(f"{gen_now:.0f}", style=f"bold {pal.ok}")
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
    grammar is width-driven: a roomy card keeps the configurable meters,
    ramped core grid and RoCE row; the compact card keeps the headlines, the
    bar and the core grid but drops the meters, legend and RoCE; a narrow card
    runs gpu/mem/cpu text only; the floor table mode collapses the node to a
    single aligned row."""

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
            s = SparkUnitStats(
                label=node_cfg.label, is_worker=getattr(node_cfg, "worker", False), pending=True
            )
        pal = _palette_for(self.app)
        width = max(1, self.content_size.width)
        if getattr(self.app, "node_mode", "") == "table":
            return self._table_row(s, pal, width)
        density = _density(self)
        tier = "rail" if getattr(self.app, "rail", False) else density
        if width >= NODE_FULL_MIN and tier == "roomy":
            rows = self._interior_rows(s, pal, width, density, roce=True)
        elif width >= NODE_FULL_MIN and tier == "compact":
            rows = self._interior_rows(s, pal, width, density, roce=False)
        else:
            rows = self._text_rows(s, pal, width)
        role = "host" if not s.is_worker else "worker"
        role_style = pal.cyan if not s.is_worker else pal.warn
        caret_style = role_style if s.online or s.pending else pal.warn
        title = [
            ("^" if s.online else "…" if s.pending else "✗", f"bold {caret_style}"),
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
        right-aligned GPU / MEM / CPU util and the stack bar. The tile keeps a
        trailing gap so two nodes side by side never read as one run of
        numbers; _fit truncation keeps the label and GPU first when tight."""
        if s.online:
            gpu = f"{s.gpu_util_pct:.0f}%"
            mem = f"{s.mem_used_bytes / s.mem_total_bytes * 100:.0f}%" if s.mem_total_bytes else "—"
            cpu = (
                f"{sum(s.cpu_cores_util) / len(s.cpu_cores_util):.0f}%" if s.cpu_cores_util else "—"
            )
        else:
            gpu = mem = cpu = "—"
        parts = [
            Text(
                "● " if s.online else "… " if s.pending else "✗ ",
                style=pal.ok if s.online else pal.dim if s.pending else pal.warn,
            ),
            Text(_short_label(s.label), style=f"bold {pal.fg}"),
            Text("   ", style=""),
            Text(f"{gpu:>4}", style=f"bold {pal.blue}"),
            Text(" ", style=pal.dim),
            Text(f"{mem:>4}", style=f"bold {pal.ok}" if s.online else pal.dim),
            Text(" ", style=pal.dim),
            Text(f"{cpu:>4}", style=f"bold {pal.warn}" if s.online else pal.dim),
        ]
        stack = mem_stack(s)
        if stack:
            parts += [Text(" ", style=pal.dim), _stack_cells(stack, 6, pal)]
        # The trailing gap is part of the tile: the grid packs tiles flush, so
        # the separation between nodes has to live inside the row.
        avail = max(1, width - FLOOR_GAP)
        out = Text.assemble()
        for part in parts:
            if out.cell_len + part.cell_len > avail:
                break  # drop trailing fields (bar, then cpu, then mem) cleanly
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
            segs += _lock_segs(s, pal, sum(t.cell_len for t in segs), max(1, width - 4))
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
        compact = density == "compact"
        short = (width - 4) <= 18  # only an unusably narrow label abbreviates
        iw = max(1, width - 4)
        mw = iw  # meters fill the interior; _fit pads/truncates to iw either way
        dash = Text("—", style=pal.dim)
        rows: list[Text] = []
        # GPU: util (headline) + temp + power + SM clock; meter beneath. The
        # compact card drops the meter (the reference card keeps graphs only at
        # roomy), so its interior stays five rows.
        glabel = "gpu " if not short else "g "
        if s.online:
            segs = [
                Text(glabel, style=pal.dim),
                Text(f"{s.gpu_util_pct:.0f}%", style=f"bold {pal.blue}"),
                Text(" · ", style=pal.dim),
                Text(f"{s.temp_c:.0f}°C", style=_temp_style(s.temp_c, pal)),
            ]
            if s.power_w > 0:
                segs += [Text(" · ", style=pal.dim), Text(f"{s.power_w:.0f}W", style=pal.accent)]
            # A recorded clock lock outranks the SM clock: its room is reserved
            # first, then the clock is added only when it still fits beside the
            # headline stats (at narrow widths power/temperature outrank it).
            lock = _lock_segs(s, pal, sum(t.cell_len for t in segs), iw)
            if s.gpu_clock_mhz > 0 and not compact:
                base_len = sum(t.cell_len for t in segs + lock)
                if base_len + 3 + len(_fmt_freq(s.gpu_clock_mhz)) <= iw:
                    segs += [
                        Text(" · ", style=pal.dim),
                        Text(_fmt_freq(s.gpu_clock_mhz), style=pal.accent),
                    ]
            rows.append(Text.assemble(*segs, *lock))
            if not compact:
                rows.append(
                    _meter_line(
                        _treatment(self), s.gpu_util_pct, mw, pal, pal.blue, self._gpu_history
                    )
                )
        else:
            rows.append(Text.assemble(Text(glabel, style=pal.dim), dash))
            if not compact:
                rows.append(_meter_line(_treatment(self), 0, mw, pal, pal.blue, self._gpu_history))
        # MEM: used/total + pct, swap, then the modelled total. The bar (or the
        # meter when the engine printed no plan) follows; the legend row is a
        # roomy-only entry so the compact card stops at the bar.
        mlabel = "mem " if not short else "m "
        if s.online and s.mem_total_bytes > 0:
            used_gb = s.mem_used_bytes // (1024**3)
            total_gb = s.mem_total_bytes // (1024**3)
            used_pct = s.mem_used_bytes / s.mem_total_bytes * 100
            stack = mem_stack(s)
            mem = Text.assemble(
                Text(mlabel, style=pal.dim),
                Text(f"{used_gb}G", style=f"bold {pal.fg}"),
                Text(f"/{total_gb}G ", style=pal.dim),
                Text(f"{used_pct:.0f}%", style=f"bold {pal.ok}"),
            )
            if s.swap_total_kb > 0:
                swap_pct = s.swap_used_kb / s.swap_total_kb * 100
                swap_gb = s.swap_used_kb / (1024 * 1024)
                mem.append(" sw" if short else "  swp", style=pal.dim)
                mem.append(
                    f" {swap_gb:.1f}G",
                    style=f"bold {pal.warn}" if swap_pct > 70 else f"bold {pal.fg}",
                )
            if stack:
                # ``Σ`` is the plan total the bar carries; it sums a derived
                # figure (other), so it keeps the design's provenance mark.
                mem.append(f"  Σ{_fmt_gib(stack.used)}*", style=pal.dim)
            rows.append(mem)
            if stack:
                rows.append(_stack_cells(stack, mw, pal))
                if not compact:
                    rows.append(_stack_legend(stack, iw, pal))
            elif not compact:
                rows.append(
                    _meter_line(_treatment(self), used_pct, mw, pal, pal.ok, self._mem_history)
                )
        elif s.online:
            rows.append(
                Text.assemble(
                    Text(mlabel, style=pal.dim),
                    Text(f"{s.gpu_mem_pct:.0f}%", style=f"bold {pal.ok}"),
                )
            )
            if not compact:
                rows.append(
                    _meter_line(_treatment(self), s.gpu_mem_pct, mw, pal, pal.ok, self._mem_history)
                )
        else:
            rows.append(Text.assemble(Text(mlabel, style=pal.dim), dash))
            if not compact:
                rows.append(_meter_line(_treatment(self), 0, mw, pal, pal.ok, self._mem_history))
        # CPU: util + temp; ramped core grid beneath (the card's last graph).
        clabel = "cpu " if not short else "c "
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
            # One row, always: the reference card's core grid is a single
            # spaced strip at roomy and an unspaced strip when the tile is
            # compact or too narrow for the gaps.
            spaced = not compact and iw >= 2 * len(cores) - 1
            if not spaced:
                cores = cores[: max(1, iw)]
            rows.append(_cores_line(cores, pal, spaced))
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


def _serving_base(
    tier: str,
    serv_width: int,
    models: int = 1,
) -> int:
    """SERVING interior rows without the chart, mirroring ``ServingBox._rows``
    exactly: a 9-cell gutter, then

    roomy  gen, prompt, kv, kv%, spacer, requests, cache, ttft
    compact  the same, less the prompt and kv% graphs
    rail   the fused narrow grammar
    floor  the bare base surface
    narrow  the fused grammar (no chart at that width)

    Extra served models add a requests row and a ttft row each."""
    if tier == "rail":
        return SERVING_ROWS_RAIL
    if tier == "floor":
        return SERVING_ROWS_FLOOR
    if serv_width < SERVING_NARROW_WIDTH:
        return FUSED_BASE
    base = WIDE_BASE if tier != "compact" else NARROW_BASE
    return base + 2 * max(0, models - 1)


def _node_tile_rows(tier: str, tile_w: int, stack: bool = False) -> int:
    """Rendered height of one node tile (borders included): a 1-row condensed
    table row at floor; a framed full card (gpu+meter, mem, stack+legend, cpu,
    cores, RoCE) when the tile is wide enough and the tier is roomy; a framed
    compact card (headlines + bar + cores, no meters/legend/RoCE); else a
    framed text card (gpu/mem/cpu only). The estimator passes the *widest*
    grid column (ceil of the column width) so the full/compact/text decision
    matches what NodeBox.render actually does. A full card without a memory
    stack carries one fewer row (the bar and its legend fold into one meter)."""
    if tier == "floor":
        return 1
    if tile_w >= NODE_FULL_MIN:
        if tier == "roomy":
            return (NODE_ROWS_FULL if stack else NODE_ROWS_FULL - 1) + 2
        if tier == "compact":
            return NODE_ROWS_METER + 2
    return NODE_ROWS_TEXT + 2


def _floor_cols(grid_w: int, n: int, row_budget: int) -> int:
    """Columns for the floor tier. Condensed table rows are packed as wide as
    the height budget allows (``row_budget`` node rows) and no wider than
    NODE_FLOOR_MAX cells, so a row keeps its gpu/mem/cpu fields when the
    column can carry them and the never-scroll floor holds."""
    cols = max(
        -(-n // max(1, row_budget)),
        max(1, (grid_w + FLOOR_GAP) // (NODE_FLOOR_MAX + FLOOR_GAP)),
    )
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


def _serving_height(base: int, chart: int, chart_mode: str = "area") -> int:
    """Rendered SERVING height: interior rows, a spacer before the chart, the
    chart itself (plus the oscilloscope's Δ row), and the two border rows."""
    foot = 1 if chart_mode == "oscilloscope" else 0
    return base + (1 + chart + foot if chart else 0) + 2


def _chart_rows_for(tier: str, chart_mode: str, room: int) -> int:
    """Chart rows for one tier and renderer given ``room`` spare rows (the
    spacer already deducted). ``area`` is the reference's fixed footer;
    ``oscilloscope``/``lines`` fill the leftover height between their bounds,
    which is what makes the taller pane worth having. 0 = no chart."""
    if tier in ("rail", "floor") or room <= 0:
        return 0
    if chart_mode == "area":
        # The reference footer is a roomy-tier feature: compact drops it.
        return AREA_CHART_ROWS if tier == "roomy" and room >= AREA_CHART_ROWS else 0
    return min(CHART_MAX[tier], max(CHART_MIN[tier], room))


def _tier_fit(
    n: int,
    width: int,
    avail: int,
    tier: str,
    tiled: bool,
    models: int = 1,
    chart_mode: str = "area",
    stack: bool = False,
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
        tile = _node_tile_rows(tier, -(-nw // cols), stack)
        node_h = _grid_height(n, cols, tile, tier)
        base = _serving_base(tier, sw, models)
        if tier == "floor":
            serv_h = base
            fits = node_h <= avail and serv_h <= avail
            return fits, cols, node_h, serv_h, 0
        chart = (
            0
            if sw < SERVING_NARROW_WIDTH
            else _chart_rows_for(tier, chart_mode, avail - (base + 2) - 1)
        )
        serv_h = _serving_height(base, chart, chart_mode)
        fits = node_h <= avail and serv_h <= avail
        return fits, cols, node_h, serv_h, chart
    if tier == "floor":
        cols = _floor_cols(width, n, avail - SERVING_ROWS_FLOOR)
    else:
        cols = _node_grid_columns(n, width, tiled)
    tile = _node_tile_rows(tier, -(-width // cols), stack)
    node_h = _grid_height(n, cols, tile, tier)
    base = _serving_base(tier, width, models)
    if tier == "floor":
        serv_h = base  # bare floor lines, no window frame
        fits = serv_h + node_h <= avail
        return fits, cols, node_h, serv_h, 0
    spare = avail - (base + 2 + GRID_GUTTER + node_h) - 1
    chart = 0 if width < SERVING_NARROW_WIDTH else _chart_rows_for(tier, chart_mode, spare)
    serv_h = _serving_height(base, chart, chart_mode)
    fits = serv_h + GRID_GUTTER + node_h <= avail
    return fits, cols, node_h, serv_h, chart


_TIER_RANK = {name: i for i, name in enumerate(("roomy", "compact", "rail", "floor"))}


def _tier_for(
    n: int,
    width: int,
    height: int,
    tiled: bool,
    models: int = 1,
    chart_mode: str = "area",
    stack: bool = False,
) -> str:
    """Densest tier whose estimated body fits (loosest first) for one
    arrangement; the floor is the unconditional fallback."""
    avail = height - WAYBAR_HEIGHT
    for tier in ("roomy", "compact", "rail"):
        if _tier_fit(n, width, avail, tier, tiled, models, chart_mode, stack)[0]:
            return tier
    return "floor"


class ScanScreen(ModalScreen[None]):
    """One-shot sweep of the configured nodes for inference endpoints (``s``).

    The sweep runs once in a worker owned by this screen — closing the
    window cancels whatever is still scanning; nothing re-scans on a timer.
    Each node's rows appear as that node finishes while the dashboard keeps
    polling underneath. A ``new`` endpoint can be picked (``enter``) and
    confirmed (``y``) to become the node's monitored endpoint."""

    DEFAULT_CSS = """
    ScanScreen { align: center middle; }
    #scan-box {
        width: 100; max-width: 95%; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 0 1;
    }
    #scan-title { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    #scan-list { height: auto; max-height: 24; border: none; background: $surface; }
    #scan-hint { height: auto; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("y,Y", "confirm", "Yes", show=False),
        Binding("n,N", "decline", "No", show=False),
    ]

    def __init__(self, nodes) -> None:
        super().__init__()
        # Not ``_nodes``: that is Textual's child NodeList on every DOM node.
        self._scan_nodes = tuple(nodes)
        self._scans: dict[int, discovery.NodeScan] = {}
        self._choices: dict[str, tuple[int, discovery.Endpoint]] = {}
        self._pending: tuple[int, discovery.Endpoint] | None = None
        self._applying = False
        self._done = False
        self._started = time.monotonic()
        self._elapsed = 0.0

    def compose(self):
        with Vertical(id="scan-box"):
            yield Static(id="scan-title")
            yield OptionList(id="scan-list")
            yield Static(id="scan-hint")

    def on_mount(self) -> None:
        self._render_rows()
        self.query_one("#scan-list", OptionList).focus()
        self.run_worker(self._sweep(), group="scan", exclusive=True)

    async def _sweep(self) -> None:
        async for scan in discovery.sweep(self._scan_nodes):
            self._scans[scan.uid] = scan
            self._render_rows()
        self._done = True
        self._elapsed = time.monotonic() - self._started
        self._render_rows()

    def _endpoint_row(self, pal: Palette, ep: discovery.Endpoint) -> Text:
        monitored = ep.status == "monitored"
        row = Text("  ● " if monitored else "  ○ ", style=pal.ok if monitored else pal.cyan)
        # URL, engine and model inherit the option's colour: the highlight bar
        # is drawn in the theme's accent, so an accent-coloured span on it
        # would vanish exactly on the row being picked.
        row.append(ep.url)
        row.append(f"  [{ep.engine}]")
        row.append(f"  {ep.model or '—'}")
        if monitored:
            row.append("  monitored", style=pal.dim)
        return row

    def _render_rows(self) -> None:
        pal = _palette_for(self.app)
        option_list = self.query_one("#scan-list", OptionList)
        held = option_list.highlighted_option
        keep = held.id if held is not None else None
        options: list[Option] = []
        self._choices.clear()
        for uid, node in enumerate(self._scan_nodes, 1):
            scan = self._scans.get(uid)
            head = Text(node.label, style=f"bold {pal.fg}")
            if scan is None:
                head.append("  scanning…", style=pal.dim)
            elif scan.error is not None:
                head.append(f"  ssh failed: {scan.error}", style=pal.warn)
            elif not scan.endpoints:
                head.append("  no inference found", style=pal.dim)
                if scan.worker:
                    head.append(" (worker)", style=pal.dim)
            options.append(Option(head, disabled=True))
            if scan is None or scan.error is not None:
                continue
            for ep in scan.endpoints:
                option_id = f"{uid}:{ep.port}"
                monitored = ep.status == "monitored"
                options.append(
                    Option(self._endpoint_row(pal, ep), id=option_id, disabled=monitored)
                )
                if not monitored:
                    self._choices[option_id] = (uid, ep)
        option_list.set_options(options)
        target = keep if keep in self._choices else next(iter(self._choices), None)
        if target is not None:
            option_list.highlighted = option_list.get_option_index(target)

        n = len(self._scan_nodes)
        title = Text("SCAN", style=f"bold {pal.accent}")
        if self._done:
            title.append(
                f" · {n} node{'s' if n != 1 else ''} · {self._elapsed:.1f}s", style=pal.dim
            )
        else:
            title.append(f" · {len(self._scans)}/{n} nodes · scanning…", style=pal.dim)
        self.query_one("#scan-title", Static).update(title)
        self._render_hint()

    def _render_hint(self) -> None:
        pal = _palette_for(self.app)
        hint = Text()
        if self._pending is not None:
            uid, ep = self._pending
            label = self._scan_nodes[uid - 1].label
            path = config_path()
            saving = f"Saves to {path}" if path is not None else "Simulation: not saved"
            hint.append(
                f"Monitor {label} at {ep.url} ({ep.engine} · {ep.model or '—'})? ",
                style=f"bold {pal.fg}",
            )
            hint.append(f"{saving}  ", style=pal.dim)
            hint.append("[y/n]", style=f"bold {pal.cyan}")
        else:
            empty = discovery.nothing_found_message(list(self._scans.values()))
            if self._done and empty is not None:
                hint.append(empty, style=pal.fg)
                hint.append("  · esc close", style=pal.dim)
            elif self._choices:
                hint.append("↑↓ select · enter monitor · esc close", style=pal.dim)
            elif self._done:
                hint.append("Every endpoint found is already monitored · esc close", style=pal.dim)
            else:
                hint.append("esc close", style=pal.dim)
        self.query_one("#scan-hint", Static).update(hint)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        choice = self._choices.get(event.option.id or "")
        if choice is not None and not self._applying:
            self._pending = choice
            self._render_hint()

    def action_confirm(self) -> None:
        if self._pending is None or self._applying:
            return
        self._applying = True
        uid, endpoint = self._pending
        # The app owns the switch: it must finish even though this screen
        # (and every worker it owns) goes away right now.
        self.app.run_worker(self.app.apply_endpoint(uid, endpoint), group="scan-apply")
        self.dismiss()

    def action_decline(self) -> None:
        if self._pending is not None and not self._applying:
            self._pending = None
            self._render_hint()

    def action_close(self) -> None:
        self.dismiss()


class ClockScreen(ModalScreen[None]):
    """GPU clock ceiling menu (``c``): lock every configured node's GPU clocks
    to ``0,<ceiling>`` or reset them, after a ``y`` confirmation. ``←``/``→``
    (or ``-``/``+``) move the ceiling in ``CLOCK_STEP_MHZ`` steps first.

    Node rows show each node's recorded lock from the latest poll (the
    driver cannot read a lock back, so a lock applied outside dgx-top reads
    as "no lock recorded"). Like the scan, the apply is handed to an app
    worker so it finishes after this window closes."""

    DEFAULT_CSS = """
    ClockScreen { align: center middle; }
    #clock-box {
        width: 72; max-width: 95%; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 0 1;
    }
    #clock-title { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    #clock-list { height: auto; max-height: 20; border: none; background: $surface; }
    #clock-hint { height: auto; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("y,Y", "confirm", "Yes", show=False),
        Binding("n,N", "decline", "No", show=False),
        # Screen bindings outrank the app's `+`/`-` (poll speed) while open.
        Binding("right,plus", "ceiling(1)", "Ceiling up", show=False),
        Binding("left,minus", "ceiling(-1)", "Ceiling down", show=False),
    ]

    def __init__(self, ceiling_mhz: int) -> None:
        super().__init__()
        # Prefixed: plain `_nodes`-style names shadow Textual DOM internals.
        self._clock_ceiling = ceiling_mhz
        self._clock_pending: str | None = None  # "lock" | "reset"
        self._clock_applying = False
        self._clock_seen: ClusterStats | None = None

    def compose(self):
        with Vertical(id="clock-box"):
            yield Static(id="clock-title")
            yield OptionList(id="clock-list")
            yield Static(id="clock-hint")

    def on_mount(self) -> None:
        self._render_rows()
        self.query_one("#clock-list", OptionList).focus()
        # Node rows follow the dashboard's polls (it keeps polling underneath).
        self.set_interval(1.0, self._refresh_rows)

    def _refresh_rows(self) -> None:
        if self.app.cluster is not self._clock_seen:
            self._render_rows()

    def _noun(self) -> str:
        n = len(self.app.settings.nodes)
        return f"{n} node{'s' if n != 1 else ''}"

    def _node_row(self, pal: Palette, label: str, unit: SparkUnitStats | None) -> Text:
        row = Text(label, style=f"bold {pal.fg}")
        if unit is None or unit.pending:
            row.append("  waiting for a poll", style=pal.dim)
        elif not unit.online:
            row.append("  offline", style=pal.warn)
        else:
            if unit.gpu_clock_lock_mhz > 0:
                row.append(f"  locked {_lock_label(unit.gpu_clock_lock_mhz)}", style=pal.cyan)
            else:
                row.append("  no lock recorded", style=pal.dim)
            if unit.gpu_clock_mhz > 0:
                row.append(f"  {_fmt_freq(unit.gpu_clock_mhz)} now", style=pal.dim)
        return row

    def _render_rows(self) -> None:
        pal = _palette_for(self.app)
        cluster = self._clock_seen = self.app.cluster
        units = cluster.units if cluster is not None else []
        nodes = self.app.settings.nodes
        pad = max((len(node.label) for node in nodes), default=0)
        option_list = self.query_one("#clock-list", OptionList)
        held = option_list.highlighted_option
        keep = held.id if held is not None and held.id in ("lock", "reset") else "lock"
        options = [
            Option(
                self._node_row(pal, node.label.ljust(pad), units[i] if i < len(units) else None),
                disabled=True,
            )
            for i, node in enumerate(nodes)
        ]
        # Action rows carry no styled spans: the highlight bar is accent-coloured.
        options.append(
            Option(f"Lock GPU clocks to 0,{self._clock_ceiling} MHz on {self._noun()}", id="lock")
        )
        options.append(Option(f"Reset GPU clocks on {self._noun()}", id="reset"))
        option_list.set_options(options)
        option_list.highlighted = option_list.get_option_index(keep)
        title = Text("CLOCKS", style=f"bold {pal.accent}")
        title.append(f" · ceiling 0,{self._clock_ceiling} MHz · {self._noun()}", style=pal.dim)
        self.query_one("#clock-title", Static).update(title)
        self._render_hint()

    def _render_hint(self) -> None:
        pal = _palette_for(self.app)
        hint = Text()
        if self._clock_pending is not None:
            flag = f"-lgc 0,{self._clock_ceiling}" if self._clock_pending == "lock" else "-rgc"
            hint.append(
                f"Run sudo -n nvidia-smi {flag} on {self._noun()}? ", style=f"bold {pal.fg}"
            )
            if config_path() is None:
                hint.append("Simulation: no SSH  ", style=pal.dim)
            hint.append("[y/n]", style=f"bold {pal.cyan}")
        else:
            hint.append(
                f"↑↓ select · ←→ ceiling ±{CLOCK_STEP_MHZ} MHz · enter apply · esc close",
                style=pal.dim,
            )
        self.query_one("#clock-hint", Static).update(hint)

    def action_ceiling(self, steps: int) -> None:
        """Move the ceiling ``steps`` × ``CLOCK_STEP_MHZ`` within the config
        bounds. A pending confirmation is dropped: it named the old value."""
        if self._clock_applying:
            return
        ceiling = min(
            CLOCK_CEILING_MAX, max(CLOCK_CEILING_MIN, self._clock_ceiling + steps * CLOCK_STEP_MHZ)
        )
        if ceiling == self._clock_ceiling:
            self.app.bell()
            return
        self._clock_ceiling = ceiling
        self._clock_pending = None
        self._render_rows()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id in ("lock", "reset") and not self._clock_applying:
            self._clock_pending = event.option.id
            self._render_hint()

    def action_confirm(self) -> None:
        if self._clock_pending is None or self._clock_applying:
            return
        self._clock_applying = True
        max_mhz = self._clock_ceiling if self._clock_pending == "lock" else None
        # The app owns the apply: it must finish after this screen goes away.
        self.app.start_clock_apply(max_mhz)
        self.dismiss()

    def action_decline(self) -> None:
        if self._clock_pending is not None and not self._clock_applying:
            self._clock_pending = None
            self._render_hint()

    def action_close(self) -> None:
        self.dismiss()


class DGXTop(App):
    """DGX Spark Cluster Inference Monitor — tiling desktop."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        overflow-y: hidden;
    }
    /* `Screen` above also matches the app's modals; keep the dashboard
       visible (dimmed) behind them while it keeps polling. */
    ScanScreen, ClockScreen { background: $background 60%; }

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
        Binding("t,T", "change_theme", "Theme"),
        Binding("q,Q", "quit", "Quit"),
        Binding("r,R", "refresh", "Refresh"),
        Binding("s,S", "scan", "Scan"),
        Binding("c,C", "clocks", "Clocks"),
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
        self._repoll = False
        self._poll_speed_idx = self.poll_speeds.index(self.settings.poll_interval)
        self._poll_timer = None
        self._pending_timer = None
        self.history: dict[str, collections.deque] = {}
        self._current_topology: str = ""
        self._host_model: str = ""
        self._models_n = 1
        self._has_stack = False  # some node has a memory stack (adds a legend row)
        self._in_kpis = False  # _update_kpis is re-entered by _apply_tier
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
        self._clock_busy = False  # a clock lock/reset is running on the nodes

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
        tier_t = _tier_for(
            n, width, height, True, self._models_n, self.settings.chart, self._has_stack
        )
        tier_s = _tier_for(
            n, width, height, False, self._models_n, self.settings.chart, self._has_stack
        )
        if tiled and _TIER_RANK[tier_t] <= _TIER_RANK[tier_s]:
            return True, tier_t
        return False, tier_s

    def _apply_tier(self, width: int, height: int) -> None:
        n = len(self.settings.nodes)
        avail = height - WAYBAR_HEIGHT
        tiled, tier = self._choose_layout(n, width, height)
        _fits, cols, node_h, serv_h, chart = _tier_fit(
            n, width, avail, tier, tiled, self._models_n, self.settings.chart, self._has_stack
        )
        node_mode = "table" if tier == "floor" else "card"
        rail = tier == "rail"
        floor = tier == "floor"
        density = "compact" if tier in ("compact", "rail", "floor") else tier
        pad = self._body_pad(avail, tier, tiled, node_h, serv_h)
        sw = _serving_tiled_width(width) if tiled else 0
        nw = width - sw - TILING_GUTTER if tiled else 0
        if tiled and sw >= SERVING_NARROW_WIDTH:
            # The stretched frame hands every interior row it has left to the
            # chart, whatever tier fitted: the desk column (not the fit
            # engine) is what sets the pane's height, so the chart takes the
            # difference and gains the resolution for it.
            base = _serving_base(tier, sw, self._models_n)
            foot = 1 if self.settings.chart == "oscilloscope" else 0
            chart = max(chart, max(serv_h, node_h) - 2 - base - 1 - foot)
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

        for name in ("compact", "roomy"):
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
        """No vertical centring: the reference layout anchors the windows to
        the row under the bar, so surplus height stays at the bottom."""
        return 0

    def _apply_fill(self, pad: int) -> None:
        """Auto-size every window and centre the body with a top pad."""
        body = self.query_one("#body", Vertical)
        body.styles.margin = (pad, 0, 0, 0) if pad else 0
        # The reference desk stretches its columns (``align-items: stretch``):
        # in the tiled arrangement the serving frame reaches the bottom of the
        # node stack and keeps its slack inside the frame, so trimming rows
        # from the pane never punches a hole in the column.
        serving = self.query_one("#serving", ServingBox)
        serving.expected_height = max(self._serv_h, self._node_h) if self.tiled else 0
        serving.styles.height = serving.expected_height or "auto"
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
        self._recheck_while_pending(stats)
        if self._repoll:
            self._repoll = False
            self.run_worker(self._poll())

    def _request_poll(self) -> None:
        """Poll now; if a poll is in flight, poll again the moment it lands
        (``_poll`` drops overlapping calls, which would lose this request)."""
        if self._polling:
            self._repoll = True
        else:
            self.run_worker(self._poll())

    def _recheck_while_pending(self, stats: ClusterStats) -> None:
        """A node whose first poll is still outstanding appears as soon as it
        lands, not a poll interval later. Waiting is cheap: ``_watch_pending``
        polls again only once such a node has landed, so the nodes already on
        screen are neither re-polled nor re-sampled into the history meanwhile."""
        if self._pending_timer is not None:
            self._pending_timer.stop()
            self._pending_timer = None
        if any(u.pending for u in stats.units):
            self._pending_timer = self.set_timer(PENDING_RECHECK_S, self._watch_pending)

    def _watch_pending(self) -> None:
        self._pending_timer = None
        if self.cluster is None or not any(u.pending for u in self.cluster.units):
            return
        if pending_landed():
            self._request_poll()  # _poll arms the next watch if a node is still outstanding
        else:
            self._pending_timer = self.set_timer(PENDING_RECHECK_S, self._watch_pending)

    def _update_ui(self):
        stats = self.cluster
        if stats is None:
            return
        self._update_kpis(stats)

    def _update_kpis(self, stats: ClusterStats):
        if self._in_kpis:
            # _apply_tier() re-enters through _update_ui(); one poll must record
            # one history sample. The outer call repaints after the tier.
            return
        self._in_kpis = True
        try:
            self._update_kpis_body(stats)
        finally:
            self._in_kpis = False

    def _update_kpis_body(self, stats: ClusterStats):
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
        has_stack = any(mem_stack(u) for u in units)
        if has_stack != self._has_stack:
            self._has_stack = has_stack
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
        # Not ``run_worker(self._poll())``: ``_poll`` drops a call that lands
        # while a poll is in flight, so `r` would do nothing for as long as the
        # SSH round trip takes. ``_request_poll`` re-polls the moment it lands.
        self._request_poll()

    def action_scan(self):
        """Open the one-shot endpoint scan (never stacks on an app modal)."""
        if not isinstance(self.screen, (ScanScreen, ClockScreen)):
            self.push_screen(ScanScreen(self.settings.nodes))

    def action_clocks(self):
        """Open the GPU clock ceiling menu (never stacks on an app modal, and
        never while an earlier lock/reset is still running on the nodes)."""
        if isinstance(self.screen, (ScanScreen, ClockScreen)):
            return
        if self._clock_busy:
            self.notify("GPU clock change still running", severity="warning")
            return
        self.push_screen(ClockScreen(self._clock_start_ceiling()))

    def _clock_start_ceiling(self) -> int:
        """The ceiling the clocks menu opens on: the lowest lock recorded on an
        online node (what the waybar shows), else ``clock_ceiling_mhz``."""
        units = self.cluster.units if self.cluster is not None else []
        locks = [u.gpu_clock_lock_mhz for u in units if u.online and u.gpu_clock_lock_mhz > 0]
        if not locks:
            return self.settings.clock_ceiling_mhz
        return min(CLOCK_CEILING_MAX, max(CLOCK_CEILING_MIN, int(min(locks))))

    def start_clock_apply(self, max_mhz: int | None) -> None:
        """Run ``apply_clock_lock`` in an app worker, one at a time: two
        overlapping applies interleave their remote steps per node and can
        leave a marker that disagrees with the GPU."""
        if self._clock_busy:
            return
        self._clock_busy = True
        self.run_worker(self.apply_clock_lock(max_mhz), group="clock-apply")

    async def apply_clock_lock(self, max_mhz: int | None) -> None:
        """Lock every node's GPU clocks to ``0,max_mhz`` (reset when None),
        notify the per-node outcome and repoll. Runs in an app worker, so no
        exception may escape (a failed worker exits the app)."""
        try:
            results = await set_clock_lock(max_mhz)
        except Exception as exc:  # set_clock_lock is total; keep the app alive regardless
            reason = str(exc).strip() or type(exc).__name__
            results = [(node.label, reason) for node in self.settings.nodes]
        finally:
            self._clock_busy = False
        failed = [(label, err) for label, err in results if err is not None]
        n = len(results)
        what = f"locked to 0,{max_mhz} MHz" if max_mhz else "reset"
        if not failed:
            sim = " (simulation)" if config_path() is None else ""
            message = f"GPU clocks {what} on {n} node{'s' if n != 1 else ''}{sim}"
            severity = "information"
        else:
            reasons = "; ".join(f"{label}: {err}" for label, err in failed)
            message = f"GPU clocks {what} on {n - len(failed)}/{n} nodes; {reasons}"
            severity = "warning"
        log.info("clocks: %s", message)
        self.notify(message, severity=severity)
        self._request_poll()

    async def apply_endpoint(self, unit_id: int, endpoint: discovery.Endpoint) -> None:
        """Monitor ``endpoint`` on node ``unit_id`` (1-based) from now on and
        save it to the configuration file. The switch stands even when the
        save fails; the notification says which of the two happened.

        The switch, the settings refresh and the save run with no await
        between them, so of two applies on one node the one that runs last
        owns the live endpoint, ``self.settings`` and the file together; only
        the model-name fetch waits, afterwards."""
        label = self.settings.nodes[unit_id - 1].label
        epoch = set_unit_endpoint(unit_id, endpoint.url, endpoint.engine)
        self.settings = get_settings()
        path = config_path()
        severity = "information"
        if path is None:
            message = f"{label} → {endpoint.url} applied; simulation: not saved"
        else:
            try:
                save_node_endpoint(path, unit_id, label, endpoint.url, endpoint.engine)
            except ConfigError as exc:
                message = f"{label} → {endpoint.url} applied; config not saved: {exc}"
                severity = "warning"
            else:
                message = f"{label} → {endpoint.url} saved to {path}"
        log.info("scan: %s", message)
        self.notify(message, severity=severity)
        self._request_poll()
        if await refresh_unit_model_name(unit_id, epoch):
            self._request_poll()  # paint the tile's model name now, not a poll later


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
