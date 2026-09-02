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
from themes import CUSTOM_THEMES, Palette, build_palette

log = logging.getLogger("dgx-top")

# ─── Layout constants ────────────────────────────────────────────────

TEMP_ALERT = 80
TEMP_WARM = 60

# Fluid node-grid layout: SERVING hero full-width on top, node grid below. The
# grid picks a column count and a node-tile mode (card/compact/strip) from the
# viewport width and node count so a 12-Spark cluster fills a single narrow row
# on a wide terminal and wraps into card tiles on a narrow one.
WAYBAR_HEIGHT = 1
GRID_GUTTER = 1  # row gutter in the node grid; columns are contiguous
NODE_CARD_MIN = 22  # a usable card (text) shows gpu/mem/cpu; below this -> table
NODE_FULL_MIN = 26  # a tile at/above this keeps the meter/core/RoCE grammar
NODE_FLOOR_MIN = 10  # a condensed table tile (used to pack many nodes)
SERVING_NARROW_WIDTH = 52  # below this the SERVING hero uses the fused grammar

# Bounded growth (bounded + breathe): a taller SERVING window is filled with a
# real area chart, never a flat slab; the chart grows to at most
# CHART_MAX_ROWS and the core grid never exceeds CORES_MAX_ROWS rows.
CORES_MAX_ROWS = 2
# Fixed area-chart rows per density (the SERVING window's tall focused chart).
# The large area chart is dropped at compact and below; the inline gen
# sparkline is the last chart visual left, per the density-refinement request.
CHART_ROWS = {"roomy": 5, "dense": 4, "compact": 0, "rail": 0, "floor": 0}

# Interior rows per node tile (borders excluded). A full card keeps the
# meter/core-grid/RoCE grammar; a text card drops the graphs and runs
# gpu/mem/cpu only. The estimator mirrors the render width fold (floor-fit
# duality lesson).
NODE_ROWS_FULL = 8  # full card interior (gpu/meter+mem/meter+cpu+core+roce)
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


def _cores_line(vals: list[float], pal: Palette, spaced: bool = True) -> Text:
    """Per-core ■ squares, each cell ramped by its own load."""
    parts: list[Text] = []
    for i, v in enumerate(vals):
        if i and spaced:
            parts.append(Text(" "))
        parts.append(Text("■", style=f"bold {_ramp(v, pal)}"))
    return Text.assemble(*parts)


def _area_chart_lines(data: list[float], rows: int, width: int, pal: Palette) -> list[Text]:
    """Multi-row block-glyph area chart coloured per column by height."""
    if not data or rows <= 0 or width <= 0:
        return [Text(" " * width) for _ in range(max(0, rows))]
    lo, hi = min(data), max(data)
    span = (hi - lo) or 1
    norm = [(v - lo) / span for v in data[-width:]]
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
        gen = Text(f" {c.total_throughput:.0f} tok/s ", style=f"bold {pal.fg}")
        kv = Text(f" KV {c.kv_cache_pct:.0f}% ", style=f"bold {pal.accent}")
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
    rule, a right meta tab, the design's metric rows, and a gradient area
    chart that fills the window's grown height."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_data: list[float] = []
        self._prompt_data: list[float] = []
        self._kv_data: list[float] = []
        self._kv: dict | None = None

    def update_throughput(
        self,
        gen_vals: list[float],
        prompt_vals: list[float],
    ):
        self._gen_data = list(gen_vals)
        self._prompt_data = list(prompt_vals)
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
        if kv_history:
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
        chart_rows = CHART_ROWS.get(tier, 0)
        if chart_rows:
            rows.extend(_area_chart_lines(self._gen_data, chart_rows, max(1, width - 4), pal))
        kv = self._kv or {}
        rtab = [(f"{_fmt_tokens(kv.get('total_tok', 0))} tok", pal.dim)]
        title = [
            ("^", f"bold {pal.cyan}"),
            (" ", ""),
            ("serving", f"bold {pal.fg}"),
            (" ", ""),
            (self._model(), pal.accent),
        ]
        return Text("\n").join(_box_lines(width, title, rtab, rows, False, pal))

    def _interior_rows(self, pal: Palette, width: int, density: str, tier: str) -> list[Text]:
        narrow = density == "compact" or width < SERVING_NARROW_WIDTH
        if narrow:
            return self._narrow_rows(pal, width, tier)
        return self._wide_rows(pal, width)

    def _wide_rows(self, pal: Palette, width: int) -> list[Text]:
        """The design's serving rows (4 aligned metric rows with graph
        spacers, then requests/cache/ttft/window)."""
        gen_avg = sum(self._gen_data) / len(self._gen_data) if self._gen_data else 0.0
        prompt_avg = sum(self._prompt_data) / len(self._prompt_data) if self._prompt_data else 0.0
        lo = min(self._gen_data) if self._gen_data else 0.0
        hi = max(self._gen_data) if self._gen_data else 0.0
        s = self._kv or {}
        kv_pct = s.get("pct", 0.0)
        used = s.get("used_tok", 0)
        total = s.get("total_tok", 0)
        r = []
        has = self._gen_data
        # Top rows share one graph width and a padded tail so the graphs and
        # their trailing stats align across gen/prompt/kv/kv%.
        tail_gen = f"{lo:.0f} · {gen_avg:.0f} · {hi:.0f} tok/s"
        tail_prompt = f"{prompt_avg:.0f} tok/s"
        tail_kv = f"{_fmt_tokens(used)}/{_fmt_tokens(total)} tok" if total else "—"
        tail_kvp = f"  {kv_pct:.0f}%"
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
                (f"{gen_avg:.0f}", f"bold {pal.ok}" if has else pal.dim),
                (" · ", pal.dim),
                (f"{hi:.0f}", pal.fg),
                (" tok/s", pal.dim),
            ],
            tail_gen,
        )
        prompt_tail = tail(
            [
                (f"{prompt_avg:.0f}", f"bold {pal.fg}" if self._prompt_data else pal.dim),
                (" tok/s", pal.dim),
            ],
            tail_prompt,
        )
        kv_tail = tail(
            [
                (_fmt_tokens(used), f"bold {pal.accent}" if total else pal.dim),
                (f"/{_fmt_tokens(total)} tok" if total else "", pal.dim),
            ],
            tail_kv,
        )
        kvp_tail = tail([(f"  {kv_pct:.0f}%", pal.accent)], tail_kvp)

        def stretched(data: list[float]) -> list[float]:
            """Nearest-neighbour resample so the graph fills the full width
            even when history has fewer points than the graph has columns."""
            n = len(data)
            if n <= 0 or n >= graph_w:
                return data
            return [data[i * n // graph_w] for i in range(graph_w)]

        def graph_row(label: str, graph: Text, tail_segs: list[Text]) -> Text:
            return Text.assemble(
                Text(label, style=pal.dim), graph, Text("  ", style=""), *tail_segs
            )

        r.append(
            graph_row("gen    ", _spark_line(stretched(self._gen_data), pal.ok, graph_w), gen_tail)
        )
        r.append(Text("", style=""))
        r.append(
            graph_row(
                "prompt ", _spark_line(stretched(self._prompt_data), pal.blue, graph_w), prompt_tail
            )
        )
        r.append(Text("", style=""))
        r.append(
            graph_row(
                "kv     ", _spark_line(stretched(self._kv_data), pal.accent, graph_w), kv_tail
            )
        )
        r.append(Text("", style=""))
        if self._kv is None:
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
        # kv% — plain value (the meter is dropped from the narrow grammar).
        r.append(
            Text.assemble(
                Text("kv ", style=pal.dim),
                Text(f"{s.get('pct', 0.0):.0f}%", style=f"bold {pal.accent}"),
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
    inset in the top rule, the configured host as the right meta tab,
    configurable meters (gradient/spark/tick/line), a ramped core grid and the
    RoCE row — when the tile is wide enough. A narrow card drops the graphs and
    runs gpu/mem/cpu text only; the table mode collapses the node to a single
    aligned row."""

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
        full = density in ("roomy", "dense") and width >= NODE_FULL_MIN
        rows = self._interior_rows(s, pal, width, density, text=not full)
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
        self, s: SparkUnitStats, pal: Palette, width: int, density: str, text: bool = False
    ) -> list[Text]:
        if text:
            return self._text_rows(s, pal, width)
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
        # RoCE: RX/TX + wire utilisation, always its own row (accent hue).
        roce = Text.assemble(Text("roce ", style=pal.dim))
        if s.online and (s.roce_rx_bps > 0 or s.roce_tx_bps > 0):
            pct = _roce_util_pct(s)
            roce.append(f"↓{_fmt_rate(s.roce_rx_bps)}", style=f"bold {pal.accent}")
            roce.append(f" ↑{_fmt_rate(s.roce_tx_bps)}", style=f"bold {pal.accent}")
            if s.roce_capacity_bps > 0:
                roce.append(f" {pct:.0f}%", style=f"bold {pal.accent}")
        else:
            roce.append("—", style=pal.dim)
        rows.append(roce)
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


def _serving_rows_for(width: int, tier: str) -> int:
    """SERVING interior rows for (width, tier). The prioritized compact grammar
    (gen/requests/ttft/kv% [+cache]) is used below SERVING_NARROW_WIDTH and at
    the compact tier; the wide grammar (no window stat) elsewhere; rail drops
    cache; floor is the base gen/req/ttft surface."""
    if tier == "rail":
        return SERVING_ROWS_RAIL
    if tier == "floor":
        return SERVING_ROWS_FLOOR
    base = NARROW_BASE if (width < SERVING_NARROW_WIDTH or tier == "compact") else WIDE_BASE
    chart = CHART_ROWS.get(tier, 0)
    return base + chart


def _node_tile_rows(tier: str, node_mode: str, tile_w: int) -> int:
    """Rendered height of one node tile: a 1-row condensed-table row, a framed
    full card (meters + core grid + RoCE) if the tile is wide enough and the
    tier is roomy/dense, else a framed text card (gpu/mem/cpu only). The
    estimator passes the *widest* grid column (ceil of the column width) so the
    full/text decision matches what NodeBox.render actually does."""
    if node_mode == "table" or tier == "floor":
        return 1
    if tile_w >= NODE_FULL_MIN and tier in ("roomy", "dense"):
        return NODE_ROWS_FULL + 2
    return NODE_ROWS_TEXT + 2


def _floor_cols(grid_w: int, n: int) -> int:
    """Columns for the floor tier, whose condensed 1-row table tiles pack
    tightly."""
    return min(n, max(1, grid_w // NODE_FLOOR_MIN))


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


def _layout_height(nodes: int, width: int, cols: int, tier: str, node_mode: str) -> int:
    """Full viewport height (rows) a (cols, tier, node_mode) layout needs."""
    chrome = WAYBAR_HEIGHT
    serv_ir = _serving_rows_for(width, tier)
    # Textual distributes the `width % cols` remainder to the trailing 1fr
    # columns, so the widest column can be one cell wider than `width // cols`.
    # Reserve the widest column's card grammar (full vs text) so the estimator
    # never under-calls the render and clips the grid (never-clip invariant).
    tile_w = -(-width // cols) if cols else width
    tile = _node_tile_rows(tier, node_mode, tile_w)
    grid_rows = -(-nodes // cols) if cols else nodes
    gutter = 0 if tier == "floor" else GRID_GUTTER  # bare floor lines pack tight
    grid_h = grid_rows * tile + max(0, grid_rows - 1) * gutter
    if tier == "floor":
        body = serv_ir + grid_h
    else:
        body = (serv_ir + 2) + GRID_GUTTER + grid_h
    return chrome + body


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

    /* SERVING hero full-width on top; the node grid below. The grid's column
       count and tile mode are set per-resize in _apply_tier (columns-only
       grid-size so any child count wraps; a fixed rows value would clamp and
       orphan overflow children). */
    #serving { height: auto; width: 1fr; }
    #node-col {
        layout: grid;
        height: auto;
        width: 1fr;
        grid-gutter: 1 0;   /* one blank row between tile rows; columns contiguous */
        margin: 1 0 0 0;     /* one blank row under SERVING (mirrored by the fit estimator) */
    }
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
        self.density = ""
        self.cols = 0
        self.node_mode = ""
        self.rail = False
        self.floor = False
        self._serv_h = 0
        self._node_h = 0
        self._pad = 0

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
        self._apply_tier(event.size.width, event.size.height)

    def _pick_tier(self, n: int, width: int, height: int, cols: int, node_mode: str) -> str:
        """Densest tier whose estimated height fits (loosest first)."""
        for tier in ("roomy", "dense", "compact", "rail"):
            if height >= _layout_height(n, width, cols, tier, node_mode):
                return tier
        # The floor packs condensed 1-row table tiles into more columns so
        # every node fits with no scroll.
        if height >= _layout_height(n, width, _floor_cols(width, n), "floor", "table"):
            return "floor"
        return "floor"

    def _apply_tier(self, width: int, height: int) -> None:
        n = len(self.settings.nodes)
        cols, node_mode = _node_layout(width, n)
        tier = self._pick_tier(n, width, height, cols, node_mode)
        if tier == "floor":
            cols = _floor_cols(width, n)
            node_mode = "table"
        rail = tier == "rail"
        floor = tier == "floor"
        density = "compact" if tier in ("compact", "rail", "floor") else tier
        serv_h, node_h, pad = self._fill_heights(height, n, width, cols, tier, node_mode)
        if (
            density == self.density
            and cols == self.cols
            and node_mode == self.node_mode
            and rail == self.rail
            and floor == self.floor
            and serv_h == self._serv_h
            and node_h == self._node_h
            and pad == self._pad
        ):
            return
        self.density = density
        self.cols = cols
        self.node_mode = node_mode
        self.rail = rail
        self.floor = floor
        self._serv_h = serv_h
        self._node_h = node_h
        self._pad = pad

        for name in ("compact", "dense", "roomy"):
            self.screen.set_class(name == density, name)
        body = self.query_one("#body", Vertical)
        body.set_class(rail, "rail")
        body.set_class(floor, "floor")
        node_col = self.query_one("#node-col", Vertical)
        node_col.styles.grid_size_columns = cols
        node_col.styles.grid_columns = " ".join(["1fr"] * cols)
        self._apply_fill(serv_h, node_h, pad)
        self._update_ui()

    def _fill_heights(
        self, height: int, n: int, width: int, cols: int, tier: str, node_mode: str
    ) -> tuple[int, int, int]:
        """Centering pad only: every window auto-sizes to its content (no dead
        space); leftover viewport height frames the dashboard (bounded +
        breathe). Rail/floor are top-anchored (pad 0). Returns ``(0, 0, pad)``
        — the first two are legacy slots."""
        chrome = WAYBAR_HEIGHT
        avail = height - chrome
        body = _layout_height(n, width, cols, tier, node_mode) - chrome
        pad = 0 if tier in ("rail", "floor") else max(0, (avail - body) // 2)
        return 0, 0, pad

    def _apply_fill(self, serv_h: int, node_h: int, pad: int) -> None:
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
        _record("throughput", stats.total_throughput)
        self.history.setdefault(
            "prompt-throughput", collections.deque(maxlen=self.settings.history_length)
        )
        _record("prompt-throughput", stats.total_prompt_throughput)

        hosted_units = stats.hosted_units
        hosted_kv_keys = {f"kv-usage-{u.label}" for u in hosted_units}
        for key in list(self.history):
            if key.startswith("kv-usage-") and key not in hosted_kv_keys:
                self.history.pop(key)
        for u in hosted_units:
            key = f"kv-usage-{u.label}"
            self.history.setdefault(key, collections.deque(maxlen=self.settings.history_length))
            _record(key, u.kv_cache_pct)
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

        self._host_model = (
            hosted_units[0].model_name if hosted_units and hosted_units[0].model_name else ""
        )

        serving = self.query_one("#serving", ServingBox)
        serving.update_throughput(
            gen_vals=list(self.history["throughput"]),
            prompt_vals=list(self.history["prompt-throughput"]),
        )
        kv_key = f"kv-usage-{hosted_units[0].label}" if hosted_units else ""
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
