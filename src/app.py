from __future__ import annotations

import collections

from rich.text import Text
from textual.app import App, Binding
from textual.containers import Grid
from textual.reactive import reactive
from textual.widgets import Sparkline, Static

from collector import _init_model_names, poll_cluster
from config import get_settings
from stats import ClusterStats, SparkUnitStats
from themes import CUSTOM_THEMES, Palette, build_palette

# ─── Theme helpers ───────────────────────────────────────────────────

TEMP_ALERT = 80
TEMP_WARM = 60

_palette_cache: dict[str, Palette] = {}


def _palette_for(app: "DGXTop") -> Palette:
    """Resolve (and cache) the semantic palette for the app's active theme."""
    theme = app.current_theme
    cached = _palette_cache.get(theme.name)
    if cached is None:
        cached = build_palette(theme)
        _palette_cache[theme.name] = cached
    return cached


def _lerp_hex(start: str, end: str, t: float) -> str:
    """Linearly interpolate two #rrggbb colors; t clamps to [0, 1]."""
    t = max(0.0, min(1.0, t))
    a = [int(start[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(end[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def _metric_ramp(value: float, pal: Palette, color: str) -> str:
    """Increase a theme color's saturation as utilization rises."""
    clamped = max(0.0, min(100.0, value))
    hue = getattr(pal, color)
    if clamped == 100:
        return hue
    low = _lerp_hex(hue, pal.background, 0.55)
    return _lerp_hex(low, hue, clamped / 100)


def _grid_cell(util: float, pal: Palette) -> Text:
    """Render a CPU square with saturation proportional to utilization."""
    return Text("\u25a0", style=_metric_ramp(util, pal, "accent"))


def _temp_style(c: float, pal: Palette) -> str:
    if c >= TEMP_ALERT:
        return f"bold {pal.error}"
    if c >= TEMP_WARM:
        return f"bold {pal.fg}"
    return pal.dim


def _fmt_tokens(n: int) -> str:
    """Format token count for compact display (e.g. 82K, 1.5M)."""
    if n < 1000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1000:.0f}K"
    else:
        return f"{n / 1_000_000:.1f}M"


def _compute_kv_risk(
    pct: float, prefix_hit: float, used_tok: int, total_tok: int, pal: Palette
) -> Text:
    """Multi-factor KV cache risk assessment.

    Factors considered:
      - KV cache usage percentage (block-level)
      - Remaining token capacity (relative)
      - Prefix cache hit rate (lower hit = more recomputation)
    """
    factors: list[str] = []
    critical = False

    if pct >= 90:
        factors.append(">90% full")
        critical = True
    elif pct >= 80:
        factors.append(">80% full")
    elif pct >= 70:
        factors.append(">70% full")

    if total_tok > 0:
        used_pct = used_tok / total_tok * 100
        # Remaining fraction as percentage of total pool
        remaining_pct = 100 - used_pct
        if remaining_pct <= 2:
            factors.append("<2% remain")
            critical = True
        elif remaining_pct < 5:
            factors.append("<5% remain")

    if prefix_hit >= 0 and prefix_hit < 10:
        factors.append("low cache hit")

    if not factors:
        return Text("", style=pal.muted)

    label = "KV " + ("!! " if critical else "! ")
    style = f"bold {pal.error}" if critical else f"bold {pal.warn}"
    return Text.assemble(
        Text(label, style=style),
        Text("  ".join(factors), style=pal.dim),
    )


# ─── Widgets ────────────────────────────────────────────────────────────


class MeterBar(Static):
    """A percentage bar that always fills its available content width."""

    def __init__(self, *args, metric_color: str = "primary", **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_color = metric_color
        self._pct = 0.0

    def update_pct(self, pct: float) -> None:
        self._pct = max(0.0, min(100.0, pct))
        self.refresh()

    def render(self) -> Text:
        width = self.content_size.width
        if width <= 0:
            return Text()
        pal = _palette_for(self.app)
        filled = round(self._pct / 100 * width)
        fill_style = getattr(pal, self._metric_color)
        return Text.assemble(
            Text("\u2588" * filled, style=fill_style),
            Text("\u2591" * (width - filled), style=pal.faint),
        )


class ThroughputTile(Static):
    """Throughput panel with KV cache capacity display.

    KV cache values represent *block-allocated token capacity* — vLLM's paged
    attention allocates blocks at block_size granularity, so "used tokens" =
    used_blocks × block_size. This overcounts actual stored tokens (a partially
    filled block counts as fully allocated) but is the correct metric for
    capacity planning: the scheduler cannot use partial blocks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_data: list[float] = []
        self._prompt_data: list[float] = []
        self._prompt_gen_ratio: float = 0.0
        self._kv_data: list[float] = []

    def compose(self):
        yield Static("THROUGHPUT", classes="section-header")
        yield Sparkline(id="tp-prompt-chart")
        yield Static(id="tp-prompt-stats")
        yield Sparkline(id="tp-gen-chart")
        yield Static(id="tp-gen-stats")
        yield Static(id="tp-ratio")
        yield Static("KV CACHE", classes="section-header kv-header")
        yield Static(id="kv-stats")
        yield Static(id="kv-detail")
        yield MeterBar(id="kv-bar", classes="meter", metric_color="primary")
        yield Sparkline(id="kv-usage-chart")
        yield Static(id="kv-risk")

    def on_mount(self):
        self._render_content()

    def update_throughput(
        self, gen_vals: list[float], prompt_vals: list[float], prompt_gen_ratio: float = 0.0
    ):
        self._gen_data = gen_vals
        self._prompt_data = prompt_vals
        self._prompt_gen_ratio = prompt_gen_ratio
        self._render_content()

    def update_kv(
        self,
        pct: float,
        req: int,
        wait: int = 0,
        used_tok: int = 0,
        total_tok: int = 0,
        prefix_hit: float = -1.0,
        kv_history: list[float] | None = None,
    ):
        """Update KV cache display with capacity framing, request counts, and risk.

        Args:
            pct: KV cache usage percentage (0-100)
            req: Running requests count
            wait: Waiting requests count
            used_tok: Block-allocated token capacity used
            total_tok: Total token capacity
            prefix_hit: Prefix cache hit rate (-1 = unavailable)
            kv_history: Per-node KV usage history for sparkline
        """
        pal = _palette_for(self.app)
        # Line 1: request status with prefix hit rate
        hit_str = ""
        if prefix_hit >= 0:
            hit_str = f"  hit {prefix_hit:.0f}%"

        if req > 0 or wait > 0:
            kv_line1 = Text.assemble(
                Text(f"{req}r", style=pal.mid),
                Text(f"  {wait}w", style=pal.muted if wait == 0 else f"bold {pal.fg}"),
                Text(hit_str, style=f"bold {pal.accent}" if prefix_hit > 0 else pal.muted),
            )
        else:
            kv_line1 = Text.assemble(
                Text("idle", style=pal.muted),
                Text(hit_str, style=f"bold {pal.accent}" if prefix_hit > 0 else pal.muted),
            )
        self.query_one("#kv-stats", Static).update(kv_line1)

        # Line 2: capacity-framed token usage (block-allocated capacity)
        # Format: "Capacity: 1.23M / 3.80M tok (32%)"
        if total_tok > 0:
            used_str = _fmt_tokens(used_tok)
            total_str = _fmt_tokens(total_tok)
            capacity_line = Text.assemble(
                Text("Capacity: ", style=pal.faint),
                Text(f"{used_str} / {total_str} tok", style=f"bold {pal.fg}"),
                Text(f"  ({pct:.0f}%)", style=pal.primary),
            )
        else:
            capacity_line = Text(f"Capacity: {pct:.0f}%", style=pal.primary)
        self.query_one("#kv-detail", Static).update(capacity_line)

        # Line 3: MeterBar visual
        self.query_one("#kv-bar", MeterBar).update_pct(pct)

        # KV usage sparkline from per-node history
        if kv_history:
            self._kv_data = kv_history
            chart = self.query_one("#kv-usage-chart", Sparkline)
            chart.data = kv_history

        # Line 4: risk assessment (multi-factor like memory thrash)
        risk = _compute_kv_risk(pct, prefix_hit, used_tok, total_tok, pal)
        self.query_one("#kv-risk", Static).update(risk)

    def _render_content(self):
        pal = _palette_for(self.app)
        prompt_chart = self.query_one("#tp-prompt-chart", Sparkline)
        prompt_stats = self.query_one("#tp-prompt-stats", Static)
        if self._prompt_data:
            p_vals = self._prompt_data
            prompt_chart.data = p_vals
            p_avg = sum(p_vals) / len(p_vals)
            prompt_stats.update(
                Text.assemble(
                    Text(f"min {min(p_vals):.0f}", style=pal.muted),
                    Text("   avg ", style=pal.faint),
                    Text(f"{p_avg:.0f}", style=f"bold {pal.fg}"),
                    Text("   max ", style=pal.faint),
                    Text(f"{max(p_vals):.0f}", style=pal.mid),
                )
            )
        else:
            prompt_chart.data = []
            prompt_stats.update(Text("\u2014", style=pal.muted))

        gen_chart = self.query_one("#tp-gen-chart", Sparkline)
        gen_stats = self.query_one("#tp-gen-stats", Static)
        if self._gen_data:
            g_vals = self._gen_data
            gen_chart.data = g_vals
            g_avg = sum(g_vals) / len(g_vals)
            gen_stats.update(
                Text.assemble(
                    Text(f"min {min(g_vals):.0f}", style=pal.muted),
                    Text("   avg ", style=pal.faint),
                    Text(f"{g_avg:.0f}", style=f"bold {pal.fg}"),
                    Text("   max ", style=pal.faint),
                    Text(f"{max(g_vals):.0f}", style=pal.mid),
                )
            )
        else:
            gen_chart.data = []
            gen_stats.update(Text("\u2014", style=pal.muted))

        ratio_widget = self.query_one("#tp-ratio", Static)
        if self._prompt_gen_ratio > 0:
            ratio_widget.update(
                Text.assemble(
                    Text("ratio ", style=pal.faint),
                    Text(f"{self._prompt_gen_ratio:.0f}:1", style=f"bold {pal.fg}"),
                )
            )
        else:
            ratio_widget.update("")


class NodeTile(Static):
    """Compact per-Spark hardware tile."""

    def __init__(self, idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = idx

    def compose(self):
        yield Static(id=f"node-label-{self.idx}")
        yield Static("GPU", classes="section-header gpu-header")
        yield Static(id=f"node-gpu-row-{self.idx}")
        yield MeterBar(id=f"node-gpu-bar-{self.idx}", classes="meter", metric_color="secondary")
        yield Static("MEMORY", classes="section-header memory-header")
        yield Static(id=f"node-mem-row-{self.idx}")
        yield MeterBar(id=f"node-mem-bar-{self.idx}", classes="meter", metric_color="ok")
        yield Static("CPU", classes="section-header cpu-header")
        yield Static(id=f"node-cpu-row-{self.idx}")
        yield MeterBar(id=f"node-cpu-bar-{self.idx}", classes="meter", metric_color="accent")
        yield Static(id=f"node-cpu-grid-{self.idx}", classes="cores")

    def on_mount(self):
        self._clear()

    def _clear(self):
        for name in ("gpu-row", "mem-row", "cpu-row", "cpu-grid"):
            self.query_one(f"#node-{name}-{self.idx}", Static).update("")
        for name in ("gpu-bar", "mem-bar", "cpu-bar"):
            self.query_one(f"#node-{name}-{self.idx}", MeterBar).update_pct(0)

    def update_node(self, s: SparkUnitStats):
        idx = self.idx
        online = s.online
        pal = _palette_for(self.app)
        dash = Text("—", style=pal.muted)
        label = s.label
        if s.model_name:
            model = Text(s.model_name)
            model.truncate(15, overflow="ellipsis")
            label += f"  {model.plain}"
        self.query_one(f"#node-label-{idx}", Static).update(
            Text(label, style=f"bold {pal.fg}" if online else pal.muted)
        )

        if online:
            gpu = s.gpu_util_pct
            self.query_one(f"#node-gpu-row-{idx}", Static).update(
                Text.assemble(
                    Text(f"{gpu:.0f}%", style=pal.secondary),
                    Text("   "),
                    Text(f"{s.temp_c:.0f}°C", style=_temp_style(s.temp_c, pal)),
                )
            )
            self.query_one(f"#node-gpu-bar-{idx}", MeterBar).update_pct(gpu)
        else:
            self.query_one(f"#node-gpu-row-{idx}", Static).update(dash)
            self.query_one(f"#node-gpu-bar-{idx}", MeterBar).update_pct(0)

        if s.mem_total_bytes > 0:
            used_gb = s.mem_used_bytes // (1024**3)
            total_gb = s.mem_total_bytes // (1024**3)
            used_pct = s.mem_used_bytes / s.mem_total_bytes * 100
            row = Text.assemble(
                Text(f"{used_gb}G", style=f"bold {pal.fg}"),
                Text(f"/{total_gb}G  ", style=pal.muted),
                Text(f"{used_pct:.0f}%", style=pal.ok),
            )
            if s.swap_total_kb > 0:
                swap_pct = s.swap_used_kb / s.swap_total_kb * 100
                row.append(
                    f"  swp {s.swap_used_kb / (1024 * 1024):.1f}G",
                    style=f"bold {pal.error}" if swap_pct > 70 else pal.muted,
                )
            self.query_one(f"#node-mem-row-{idx}", Static).update(row)
            self.query_one(f"#node-mem-bar-{idx}", MeterBar).update_pct(used_pct)
        elif online:
            self.query_one(f"#node-mem-row-{idx}", Static).update(
                Text(
                    f"{s.gpu_mem_pct:.0f}%",
                    style=pal.ok,
                )
            )
            self.query_one(f"#node-mem-bar-{idx}", MeterBar).update_pct(0)
        else:
            self.query_one(f"#node-mem-row-{idx}", Static).update(dash)
            self.query_one(f"#node-mem-bar-{idx}", MeterBar).update_pct(0)

        if online and s.cpu_cores_util:
            avg = sum(s.cpu_cores_util) / len(s.cpu_cores_util)
            self.query_one(f"#node-cpu-row-{idx}", Static).update(
                Text.assemble(
                    Text("util  ", style=pal.muted),
                    Text(f"{avg:.0f}%", style=pal.accent),
                    Text("   temp  ", style=pal.muted),
                    Text(f"{s.cpu_temp_c:.0f}°C", style=_temp_style(s.cpu_temp_c, pal)),
                )
            )
            self.query_one(f"#node-cpu-bar-{idx}", MeterBar).update_pct(avg)
            core_cells = []
            for core_idx, util in enumerate(s.cpu_cores_util[:20]):
                core_cells.append(_grid_cell(util, pal))
                if core_idx < min(len(s.cpu_cores_util), 20) - 1:
                    core_cells.append(Text(" "))
            self.query_one(f"#node-cpu-grid-{idx}", Static).update(Text.assemble(*core_cells))
        else:
            self.query_one(f"#node-cpu-row-{idx}", Static).update(dash if online else "")
            self.query_one(f"#node-cpu-bar-{idx}", MeterBar).update_pct(0)
            self.query_one(f"#node-cpu-grid-{idx}", Static).update("")


# ─── App ────────────────────────────────────────────────────────────────


class DGXTop(App):
    """DGX Spark Cluster Inference Monitor."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        overflow-y: auto;
    }

    #title {
        height: 2;
        padding: 0 1;
        content-align: left middle;
        background: $background;
        border-bottom: solid $border-blurred;
    }

    #kpis {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 18;
        grid-gutter: 0;
        height: 18;
        background: $background;
    }

    ThroughputTile, NodeTile {
        height: 18;
        min-width: 20;
        border: solid $border-blurred;
        padding: 0 1;
        layout: vertical;
        background: $panel;
    }

    .node-header {
        height: 1;
        color: $text;
        text-style: bold;
    }

    .section-header {
        height: 1;
        color: $text-muted;
    }

    .gpu-header {
        color: $secondary 70%;
    }

    .memory-header {
        color: $success 70%;
    }

    .kv-header {
        color: $primary 70%;
    }

    .cpu-header {
        color: $accent 70%;
    }

    ThroughputTile .section-header {
        margin-top: 1;
    }

    ThroughputTile > Static, NodeTile > Static {
        height: 1;
    }

    #tp-prompt-chart, #tp-gen-chart {
        height: 2;
    }

    #tp-prompt-chart > .sparkline--max-color {
        color: $secondary;
    }

    #tp-prompt-chart > .sparkline--min-color {
        color: $secondary 35%;
    }

    #tp-gen-chart > .sparkline--max-color {
        color: $accent;
    }

    #tp-gen-chart > .sparkline--min-color {
        color: $accent 35%;
    }

    #kv-usage-chart {
        height: 2;
    }

    #kv-usage-chart > .sparkline--max-color {
        color: $primary;
    }

    #kv-usage-chart > .sparkline--min-color {
        color: $primary 30%;
    }

    #kv-risk {
        height: 1;
    }

    #tp-prompt-stats, #tp-gen-stats, #tp-ratio, #kv-detail, #kv-stats {
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    .meter {
        height: 1;
        color: $text-muted;
    }

    .cores {
        height: 2;
    }

    #kpis.medium {
        grid-rows: 18 18;
        height: 36;
    }

    #kpis.medium ThroughputTile {
        column-span: 2;
    }

    #kpis.narrow {
        grid-size: 1 3;
        grid-columns: 1fr;
        grid-rows: 18 18 18;
        height: 54;
    }

    #kpis.narrow ThroughputTile {
        column-span: 1;
    }

    #kpis.narrow ThroughputTile, #kpis.narrow NodeTile {
        min-width: 0;
        padding: 0;
    }
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

    def compose(self):
        yield Static(id="title")
        with Grid(id="kpis"):
            yield ThroughputTile(id="kpi-throughput")
            for index, _node in enumerate(self.settings.nodes):
                yield NodeTile(index, id=f"node-{index}")

    def on_mount(self):
        self._set_title()
        self._poll_timer = self.set_interval(self._current_interval(), self._poll)
        self.run_worker(self._poll())
        # Model-name discovery has one startup owner and does not delay polling.
        self.run_worker(_init_model_names())

    def on_resize(self, event) -> None:
        kpis = self.query_one("#kpis", Grid)
        if event.size.width < 46:
            kpis.set_classes("narrow")
        elif event.size.width < 90:
            kpis.set_classes("medium")
        else:
            kpis.set_classes("")

    def watch_theme(self, theme_name: str) -> None:
        """Repaint palette-derived Rich content when the theme changes.

        Textual restyles CSS itself, but the title and tile bodies are
        rendered as ``Text`` with colors resolved from the active theme, so
        they must be rebuilt immediately instead of waiting for the next poll.
        """
        if not self.is_running:
            return
        self._set_title()
        self._update_ui()

    def _current_interval(self) -> int:
        return self.poll_speeds[self._poll_speed_idx]

    def _restart_polling(self):
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._polling = False
        self._poll_timer = self.set_interval(self._current_interval(), self._poll)
        self.run_worker(self._poll())
        self._set_title()

    def _set_title(self):
        pal = _palette_for(self)
        interval = self.poll_speeds[self._poll_speed_idx]
        topo = self._current_topology or "..."
        self.query_one("#title", Static).update(
            Text.assemble(
                Text("dgx-top", style=f"bold {pal.fg}"),
                Text(" \u26a1", style=pal.faint),
                Text(topo, style=f"bold {pal.accent}"),
                Text(" :: ", style=pal.faint),
                Text(f"poll {interval}s  ", style=pal.muted),
                Text("[+/-]speed [t]heme [r]efresh [q]uit", style=pal.faint),
            )
        )

    async def _poll(self):
        if self._polling:
            return
        self._polling = True
        try:
            stats = await poll_cluster()
            self.cluster = stats
        except Exception as e:
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

        # Generation throughput history
        tp = stats.total_throughput
        self.history.setdefault(
            "throughput", collections.deque(maxlen=self.settings.history_length)
        )
        self.history["throughput"].append(tp)

        # Prompt throughput history
        prompt_tp = stats.total_prompt_throughput
        self.history.setdefault(
            "prompt-throughput", collections.deque(maxlen=self.settings.history_length)
        )
        self.history["prompt-throughput"].append(prompt_tp)

        hosted_units = stats.hosted_units

        # Clean up stale per-node throughput keys
        hosted_gen_keys = {f"throughput-{u.label}" for u in hosted_units}
        hosted_prompt_keys = {f"prompt-throughput-{u.label}" for u in hosted_units}
        hosted_kv_keys = {f"kv-usage-{u.label}" for u in hosted_units}
        for key in list(self.history):
            if key.startswith("kv-usage-") and key not in hosted_kv_keys:
                self.history.pop(key)
            if key.startswith("throughput-") and key not in hosted_gen_keys:
                self.history.pop(key)
            if key.startswith("prompt-throughput-") and key not in hosted_prompt_keys:
                self.history.pop(key)

        # Per-node KV cache usage history (for sparklines)
        for u in hosted_units:
            key = f"kv-usage-{u.label}"
            self.history.setdefault(key, collections.deque(maxlen=self.settings.history_length))
            self.history[key].append(u.kv_cache_pct)
        # Aggregated (first-hosted) KV info for display
        kv_pct = stats.kv_cache_pct
        kv_total_tok = stats.total_kv_capacity_tokens
        kv_used_tok = stats.total_kv_used_tokens
        kv_hit = stats.kv_prefix_hit_rate
        kv_req = hosted_units[0].requests_running if hosted_units else 0
        kv_wait = hosted_units[0].requests_waiting if hosted_units else 0
        prompt_gen_ratio = hosted_units[0].prompt_gen_ratio if hosted_units else 0.0

        throughput_tile = self.query_one("#kpi-throughput", ThroughputTile)
        throughput_tile.update_throughput(
            gen_vals=list(self.history["throughput"]),
            prompt_vals=list(self.history["prompt-throughput"]),
            prompt_gen_ratio=prompt_gen_ratio,
        )
        kv_key = f"kv-usage-{hosted_units[0].label}" if hosted_units else ""
        kv_history = list(self.history.get(kv_key, []))
        throughput_tile.update_kv(
            kv_pct,
            kv_req,
            wait=kv_wait,
            used_tok=kv_used_tok,
            total_tok=kv_total_tok,
            prefix_hit=kv_hit,
            kv_history=kv_history,
        )

        # Update topology indicator in title
        topo_type = stats.topology.topology_type if stats.topology else "UNKNOWN"
        if topo_type != self._current_topology:
            self._current_topology = topo_type
            self._set_title()

        for idx, s in enumerate(units):
            self.query_one(f"#node-{idx}", NodeTile).update_node(s)

    def action_poll_faster(self):
        self._poll_speed_idx = max(0, self._poll_speed_idx - 1)
        self._restart_polling()

    def action_poll_slower(self):
        self._poll_speed_idx = min(len(self.poll_speeds) - 1, self._poll_speed_idx + 1)
        self._restart_polling()

    def action_refresh(self):
        self.run_worker(self._poll())


def run():
    app = DGXTop()
    app.run()


if __name__ == "__main__":
    run()
