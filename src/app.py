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

# ─── Greyscale helpers ─────────────────────────────────────────────────

TEMP_ALERT = 80
TEMP_WARM = 60


def _grid_cell(util: float) -> Text:
    """Render a fixed square whose brightness tracks core utilization."""
    clamped = max(0.0, min(100.0, util))
    level = 32 + round(clamped * 223 / 100)
    return Text("■", style=f"rgb({level},{level},{level})")


def _val_style(v: float, hi: float = 70, lo: float = 20) -> str:
    if v >= hi:
        return "bold white"
    if v <= lo:
        return "grey46"
    return "grey85"


def _temp_style(c: float) -> str:
    if c >= TEMP_ALERT:
        return "bold red"
    if c >= TEMP_WARM:
        return "bold white"
    return "grey66"


def _fmt_tokens(n: int) -> str:
    """Format token count for compact display (e.g. 82K, 1.5M)."""
    if n < 1000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1000:.0f}K"
    else:
        return f"{n / 1_000_000:.1f}M"


def _compute_kv_risk(pct: float, prefix_hit: float, used_tok: int, total_tok: int) -> Text:
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
        return Text("", style="grey69")

    label = "KV " + ("!! " if critical else "! ")
    style = "bold red" if critical else "bold yellow"
    return Text.assemble(
        Text(label, style=style),
        Text("  ".join(factors), style="grey69"),
    )


# ─── Widgets ────────────────────────────────────────────────────────────


class MeterBar(Static):
    """A percentage bar that always fills its available content width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pct = 0.0

    def update_pct(self, pct: float) -> None:
        self._pct = max(0.0, min(100.0, pct))
        self.refresh()

    def render(self) -> Text:
        width = self.content_size.width
        if width <= 0:
            return Text()
        filled = round(self._pct / 100 * width)
        fill_style = "bold white" if self._pct >= 80 else "grey74" if self._pct >= 50 else "grey46"
        return Text.assemble(
            Text("█" * filled, style=fill_style),
            Text("░" * (width - filled), style="grey30"),
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
        yield Sparkline(id="tp-prompt-chart", max_color="#aabbcc", min_color="#446688")
        yield Static(id="tp-prompt-stats")
        yield Sparkline(id="tp-gen-chart", max_color="#ccbbaa", min_color="#886644")
        yield Static(id="tp-gen-stats")
        yield Static(id="tp-ratio")
        yield Static("KV CACHE", classes="section-header")
        yield Static(id="kv-stats")
        yield Static(id="kv-detail")
        yield MeterBar(id="kv-bar", classes="meter")
        yield Sparkline(id="kv-usage-chart", max_color="#667788", min_color="#334455")
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
        # Line 1: request status with prefix hit rate
        hit_str = ""
        if prefix_hit >= 0:
            hit_str = f"  hit {prefix_hit:.0f}%"

        if req > 0 or wait > 0:
            kv_line1 = Text.assemble(
                Text(f"{req}r", style="grey85"),
                Text(f"  {wait}w", style="grey50" if wait == 0 else "bold white"),
                Text(hit_str, style="bold cyan" if prefix_hit > 0 else "grey46"),
            )
        else:
            kv_line1 = Text.assemble(
                Text("idle", style="grey46"),
                Text(hit_str, style="bold cyan" if prefix_hit > 0 else "grey46"),
            )
        self.query_one("#kv-stats", Static).update(kv_line1)

        # Line 2: capacity-framed token usage (block-allocated capacity)
        # Format: "Capacity: 1.23M / 3.80M tok (32%)"
        if total_tok > 0:
            used_str = _fmt_tokens(used_tok)
            total_str = _fmt_tokens(total_tok)
            capacity_line = Text.assemble(
                Text("Capacity: ", style="grey35"),
                Text(f"{used_str} / {total_str} tok", style="bold white"),
                Text(f"  ({pct:.0f}%)", style="grey66"),
            )
        else:
            capacity_line = Text(f"Capacity: {pct:.0f}%", style="grey66")
        self.query_one("#kv-detail", Static).update(capacity_line)

        # Line 3: MeterBar visual
        self.query_one("#kv-bar", MeterBar).update_pct(pct)

        # KV usage sparkline from per-node history
        if kv_history:
            self._kv_data = kv_history
            chart = self.query_one("#kv-usage-chart", Sparkline)
            chart.data = kv_history

        # Line 4: risk assessment (multi-factor like memory thrash)
        risk = _compute_kv_risk(pct, prefix_hit, used_tok, total_tok)
        self.query_one("#kv-risk", Static).update(risk)

    def _render_content(self):
        prompt_chart = self.query_one("#tp-prompt-chart", Sparkline)
        prompt_stats = self.query_one("#tp-prompt-stats", Static)
        if self._prompt_data:
            p_vals = self._prompt_data
            prompt_chart.data = p_vals
            p_avg = sum(p_vals) / len(p_vals)
            prompt_stats.update(
                Text.assemble(
                    Text(f"min {min(p_vals):.0f}", style="grey46"),
                    Text("   avg ", style="grey35"),
                    Text(f"{p_avg:.0f}", style="bold white"),
                    Text("   max ", style="grey35"),
                    Text(f"{max(p_vals):.0f}", style="grey85"),
                )
            )
        else:
            prompt_chart.data = []
            prompt_stats.update(Text("\u2014", style="grey46"))

        gen_chart = self.query_one("#tp-gen-chart", Sparkline)
        gen_stats = self.query_one("#tp-gen-stats", Static)
        if self._gen_data:
            g_vals = self._gen_data
            gen_chart.data = g_vals
            g_avg = sum(g_vals) / len(g_vals)
            gen_stats.update(
                Text.assemble(
                    Text(f"min {min(g_vals):.0f}", style="grey46"),
                    Text("   avg ", style="grey35"),
                    Text(f"{g_avg:.0f}", style="bold white"),
                    Text("   max ", style="grey35"),
                    Text(f"{max(g_vals):.0f}", style="grey85"),
                )
            )
        else:
            gen_chart.data = []
            gen_stats.update(Text("\u2014", style="grey46"))

        ratio_widget = self.query_one("#tp-ratio", Static)
        if self._prompt_gen_ratio > 0:
            ratio_widget.update(
                Text.assemble(
                    Text("ratio ", style="grey35"),
                    Text(f"{self._prompt_gen_ratio:.0f}:1", style="bold white"),
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
        yield Static("GPU", classes="section-header")
        yield Static(id=f"node-gpu-row-{self.idx}")
        yield MeterBar(id=f"node-gpu-bar-{self.idx}", classes="meter")
        yield Static("MEMORY", classes="section-header")
        yield Static(id=f"node-mem-row-{self.idx}")
        yield MeterBar(id=f"node-mem-bar-{self.idx}", classes="meter")
        yield Static("CPU", classes="section-header")
        yield Static(id=f"node-cpu-row-{self.idx}")
        yield MeterBar(id=f"node-cpu-bar-{self.idx}", classes="meter")
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
        dash = Text("—", style="grey46")
        label = s.label
        if s.model_name:
            model = Text(s.model_name)
            model.truncate(15, overflow="ellipsis")
            label += f"  {model.plain}"
        self.query_one(f"#node-label-{idx}", Static).update(
            Text(label, style="bold white" if online else "grey46")
        )

        if online:
            gpu = s.gpu_util_pct
            self.query_one(f"#node-gpu-row-{idx}", Static).update(
                Text.assemble(
                    Text(f"{gpu:.0f}%", style=_val_style(gpu)),
                    Text("   "),
                    Text(f"{s.temp_c:.0f}°C", style=_temp_style(s.temp_c)),
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
                Text(f"{used_gb}G", style="bold white"),
                Text(f"/{total_gb}G  ", style="grey46"),
                Text(f"{used_pct:.0f}%", style=_val_style(used_pct, 80)),
            )
            if s.swap_total_kb > 0:
                swap_pct = s.swap_used_kb / s.swap_total_kb * 100
                row.append(
                    f"  swp {s.swap_used_kb / (1024 * 1024):.1f}G",
                    style="bold red" if swap_pct > 70 else "grey46",
                )
            self.query_one(f"#node-mem-row-{idx}", Static).update(row)
            self.query_one(f"#node-mem-bar-{idx}", MeterBar).update_pct(used_pct)
        elif online:
            self.query_one(f"#node-mem-row-{idx}", Static).update(
                Text(f"{s.gpu_mem_pct:.0f}%", style=_val_style(s.gpu_mem_pct, 80))
            )
            self.query_one(f"#node-mem-bar-{idx}", MeterBar).update_pct(0)
        else:
            self.query_one(f"#node-mem-row-{idx}", Static).update(dash)
            self.query_one(f"#node-mem-bar-{idx}", MeterBar).update_pct(0)

        if online and s.cpu_cores_util:
            avg = sum(s.cpu_cores_util) / len(s.cpu_cores_util)
            self.query_one(f"#node-cpu-row-{idx}", Static).update(
                Text.assemble(
                    Text("util  ", style="grey50"),
                    Text(f"{avg:.0f}%", style=_val_style(avg)),
                    Text("   temp  ", style="grey50"),
                    Text(f"{s.cpu_temp_c:.0f}°C", style=_temp_style(s.cpu_temp_c)),
                )
            )
            self.query_one(f"#node-cpu-bar-{idx}", MeterBar).update_pct(avg)
            core_cells = []
            for core_idx, util in enumerate(s.cpu_cores_util[:20]):
                core_cells.append(_grid_cell(util))
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
        background: #0a0a0a;
        overflow-y: auto;
    }

    #title {
        height: 2;
        padding: 0 1;
        content-align: left middle;
        background: #0a0a0a;
        border-bottom: solid #333333;
    }

    #kpis {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 18;
        grid-gutter: 0;
        height: 18;
        background: #0a0a0a;
    }

    ThroughputTile, NodeTile {
        height: 18;
        min-width: 20;
        border: solid #2a2a2a;
        padding: 0 1;
        layout: vertical;
        background: #0f0f0f;
    }


    .node-header {
        height: 1;
        color: #ffffff;
        text-style: bold;
    }

    .section-header {
        height: 1;
        color: #777777;
    }

    ThroughputTile .section-header {
        margin-top: 1;
    }

    ThroughputTile > Static, NodeTile > Static {
        height: 1;
    }

    #tp-prompt-chart, #tp-gen-chart {
        height: 2;
        color: #aaaaaa;
    }

    #kv-usage-chart {
        height: 2;
        color: #777788;
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
        color: #888888;
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
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    cluster: reactive[ClusterStats | None] = reactive(None)

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
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
        interval = self.poll_speeds[self._poll_speed_idx]
        topo = self._current_topology or "..."
        self.query_one("#title", Static).update(
            Text.assemble(
                Text("dgx-top", style="bold white"),
                Text(" \u26a1", style="grey35"),
                Text(topo, style="bold cyan"),
                Text(" :: ", style="grey27"),
                Text(f"poll {interval}s  ", style="grey46"),
                Text("[+/-]speed [r]efresh [q]uit", style="grey30"),
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
