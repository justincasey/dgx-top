"""Memory stack: the stats contract, the segmented bar and its place in the UI."""

from __future__ import annotations

from pathlib import Path

from config import configure
from stats import ClusterStats, SparkUnitStats, TopologyInfo, mem_stack

GIB = 1024**3


def _unit(
    label: str = "head",
    worker: bool = False,
    plan: tuple[int, int, int] | None = (60, 12, 1),
    used: int = 100,
    total: int = 120,
    phys: int = 128,
) -> SparkUnitStats:
    u = SparkUnitStats(label=label)
    u.is_worker = worker
    u.online = True
    u.model_hosted = not worker
    if not worker:
        u.model_name = "demo-model"
    u.gpu_util_pct = 70.0
    u.temp_c = 60.0
    u.power_w = 40.0
    u.mem_used_bytes = used * GIB
    u.mem_total_bytes = total * GIB
    u.mem_phys_bytes = phys * GIB
    u.cpu_cores_util = [30.0] * 20
    u.cpu_temp_c = 50.0
    u.kv_cache_pct = 25.0
    u.kv_total_tokens = 600_000
    u.roce_rx_bps = 1e8
    u.roce_tx_bps = 1e8
    u.roce_capacity_bps = 5e10
    if plan:
        u.mem_weights_bytes = plan[0] * GIB
        u.mem_kv_pool_bytes = plan[1] * GIB
        u.mem_graphs_bytes = plan[2] * GIB
    return u


# ─── stats contract ──────────────────────────────────────────────────


def test_mem_stack_other_is_measured_remainder_and_bar_sums_to_physical():
    st = mem_stack(_unit())
    assert st is not None
    assert (st.weights, st.kv, st.graphs) == (60 * GIB, 12 * GIB, 1 * GIB)
    assert st.other == 27 * GIB  # 100 used - 73 planned
    assert st.free == 20 * GIB  # MemTotal - used
    assert st.reserved == 8 * GIB  # physical - MemTotal
    assert st.used + st.free + st.reserved == 128 * GIB


def test_mem_stack_is_absent_when_it_cannot_be_drawn_honestly():
    assert mem_stack(_unit(plan=None)) is None  # no startup-log plan
    offline = _unit()
    offline.online = False
    assert mem_stack(offline) is None
    assert mem_stack(_unit(total=0)) is None  # no MemTotal reading
    # A plan far above measured use describes another launch (stale log).
    assert mem_stack(_unit(plan=(90, 20, 2), used=60)) is None


def test_mem_stack_clamps_a_plan_slightly_above_use_instead_of_overdrawing():
    st = mem_stack(_unit(plan=(60, 12, 1), used=72))  # plan 73 vs used 72: within tolerance
    assert st is not None
    assert abs(st.used - 72 * GIB) < 1  # never draws more than is in use
    assert st.other == 0
    assert st.weights < 60 * GIB  # scaled down proportionally, not dropped


def test_mem_stack_without_physical_size_has_no_reserved_slice():
    st = mem_stack(_unit(phys=0))
    assert st is not None and st.reserved == 0


# ─── bar renderer ────────────────────────────────────────────────────


def _palette():
    from themes import build_palette, get_theme

    return build_palette(get_theme("dgx-aeon"))


def _style_at(text, i: int) -> str:
    style = ""
    for span in text.spans:
        if span.start <= i < span.end:
            style = str(span.style)
    return style


def test_stack_cells_fill_width_and_order_segments_left_to_right():
    from app import _stack_cells

    pal = _palette()
    st = mem_stack(_unit())
    for width in (6, 22, 40, 97):
        bar = _stack_cells(st, width, pal)
        assert len(bar.plain) == width
    bar = _stack_cells(st, 64, pal)
    assert bar.plain[0] == "█" and _style_at(bar, 0) == pal.ok  # weights first
    assert bar.plain[-1] == "╱"  # reserved last (128 physical vs 120 MemTotal)
    # weights (60 of 128 GiB) cover ~half the bar
    solid = sum(1 for i, c in enumerate(bar.plain) if c == "█" and _style_at(bar, i) == pal.ok)
    assert 28 <= solid <= 32


def test_stack_cells_boundary_cells_use_eighth_blocks_not_a_rounded_cell():
    from app import _EIGHTHS, _stack_cells

    pal = _palette()
    st = mem_stack(_unit())
    # 60/128 GiB of 50 cells = 23.4 cells of weights: the boundary lands inside a cell.
    bar = _stack_cells(st, 50, pal)
    assert any(c in _EIGHTHS for c in bar.plain)


def test_stack_cells_boundary_cell_glyph_and_tones():
    from app import _stack_cells, _stack_tones

    pal = _palette()
    tone = _stack_tones(pal)
    # At width 50 (2.56 GiB per cell) the weights/graphs boundary lands in
    # cell 23 — 1.12 GiB weights over 1.00 GiB graphs — so the glyph is 1/2
    # (round(1.12 / 2.12 * 8) = 4) with both tones painted.
    bar = _stack_cells(mem_stack(_unit()), 50, pal)
    assert bar.plain[23] == "▌"
    assert _style_at(bar, 23) == f"{tone['weights']} on {tone['graphs']}"


def test_stack_legend_drops_whole_items_at_the_width_budget():
    from app import _stack_legend

    pal = _palette()
    st = mem_stack(_unit())
    assert _stack_legend(st, 6, pal).plain == ""  # the first item needs 7
    assert _stack_legend(st, 7, pal).plain == "■wts 60"
    assert _stack_legend(st, 18, pal).plain == "■wts 60 ■gfx 1.0"  # kv needs 24
    assert _stack_legend(st, 40, pal).plain.endswith("■oth 27*")


def test_stack_cells_kv_pool_shows_measured_fill_and_unfilled_remainder():
    from app import _stack_cells, _stack_tones

    pal = _palette()
    tone = _stack_tones(pal)
    low, high = _unit(), _unit()
    low.kv_cache_pct, high.kv_cache_pct = 5.0, 95.0

    def kv_cells(u):
        bar = _stack_cells(mem_stack(u), 128, pal)  # one cell per GiB
        filled = sum(1 for i in range(128) if _style_at(bar, i) == tone["kv"])
        rest = sum(1 for i in range(128) if _style_at(bar, i) == tone["kv_free"])
        return filled, rest

    (lf, lr), (hf, hr) = kv_cells(low), kv_cells(high)
    assert hf > lf and lr > hr  # fill grows, remainder shrinks
    assert abs((lf + lr) - (hf + hr)) <= 2  # the pool itself is the same size


# ─── UI ──────────────────────────────────────────────────────────────


def _config(path: Path, n: int = 2) -> None:
    lines = ["[app]", "poll_interval = 5", "history_length = 25"]
    lines += [
        "[[nodes]]",
        'label = "head"',
        'ssh_target = "head"',
        'vllm_url = "http://192.0.2.10:8000"',
    ]
    if n == 2:
        lines += [
            "[[nodes]]",
            'label = "worker"',
            'ssh_target = "worker"',
            'vllm_url = "http://192.0.2.11:8000"',
            "worker = true",
        ]
    path.write_text("\n".join(lines) + "\n")


def _stub(monkeypatch, units):
    import app as app_module

    async def fake_poll():
        return ClusterStats(units=units, topology=TopologyInfo(topology_type="DUAL"))

    monkeypatch.setattr(app_module, "poll_cluster", fake_poll)


async def test_node_card_replaces_mem_meter_with_stack_bar_and_legend(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox, _node_tile_rows

    _config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch, [_unit("head"), _unit("worker", worker=True, plan=(58, 12, 1), used=98)])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        node = app.query_one("#node-0", NodeBox)
        lines = node.render().plain.split("\n")
        text = "\n".join(lines)
        assert "wts 60" in text and "kv 12" in text and "oth 27*" in text
        assert "Σ100*" in next(ln for ln in lines if ln.startswith("│ mem "))
        mem = next(i for i, ln in enumerate(lines) if ln.startswith("│ mem "))
        assert "╱" in lines[mem + 1]  # the bar (with its reserved slice) sits under the headline
        assert "■wts" in lines[mem + 2]  # the legend row follows the bar
        assert len(lines) <= _node_tile_rows(app.density, node.size.width, True)
        # The one invariant this feature changes: a stacked full card reserves
        # exactly one more row than the same card without the legend.
        assert _node_tile_rows(app.density, node.size.width, True) == (
            _node_tile_rows(app.density, node.size.width, False) + 1
        )


async def test_node_card_without_a_plan_keeps_the_plain_mem_meter(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch, [_unit("head", plan=None), _unit("worker", worker=True, plan=None)])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        lines = app.query_one("#node-0", NodeBox).render().plain.split("\n")
        text = "\n".join(lines)
        assert "wts" not in text and "╱" not in text
        mem = next(i for i, ln in enumerate(lines) if ln.startswith("│ mem "))
        assert lines[mem + 2].startswith("│ cpu ")  # no legend row: the old two-row MEM block
        assert not app._has_stack


async def test_serving_pane_carries_the_rank_block(tmp_path: Path, monkeypatch):
    """The reference serving pane owns the deploy line and one rank row per
    node, each carrying a copy of that node's bar and its plan total — the
    pane's cross-check that every rank accounts for the same model."""
    from app import DGXTop, ServingBox

    _config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch, [_unit("head"), _unit("worker", worker=True, plan=(58, 12, 1), used=98)])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        assert app._has_stack
        text = app.query_one("#serving", ServingBox).render().plain
        assert "deploy  vllm · TP2" in text
        ranks = [ln for ln in text.splitlines() if "rank " in ln]
        assert len(ranks) == 2, ranks
        assert "head" in ranks[0] and "worker" in ranks[1], ranks
        # Each rank row carries the same bar grammar as the node card.
        assert "╱" in ranks[0] and "╱" in ranks[1], ranks
        assert "100G*" in ranks[0] and "98G*" in ranks[1], ranks


async def test_plan_arriving_late_grows_the_node_card_without_clipping(tmp_path: Path, monkeypatch):
    """Row estimate and render agree: a node card rendered with a legend never
    exceeds the tile height the fit estimator reserved for it."""
    from app import DGXTop, NodeBox, _node_tile_rows

    _config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch, [_unit("head", plan=None), _unit("worker", worker=True, plan=None)])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        assert not app._has_stack
        app.cluster = ClusterStats(
            units=[_unit("head"), _unit("worker", worker=True, plan=(58, 12, 1), used=98)],
            topology=TopologyInfo(topology_type="DUAL"),
        )
        app._update_ui()
        await pilot.pause()
        assert app._has_stack
        node = app.query_one("#node-0", NodeBox)
        rendered = len(node.render().plain.split("\n"))
        assert rendered <= _node_tile_rows(app.density, node.size.width, True)
        assert _node_tile_rows(app.density, node.size.width, True) == (
            _node_tile_rows(app.density, node.size.width, False) + 1
        )


async def test_stack_flip_records_one_history_sample_per_poll(tmp_path: Path, monkeypatch):
    """The legend row arriving re-runs the tier fit; the re-entrant KPI pass
    must not append the same poll's samples twice."""
    from app import DGXTop

    _config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch, [_unit("head")])
    app = DGXTop()
    async with app.run_test(size=(132, 44)) as pilot:
        await pilot.pause()
        assert app._has_stack
        assert len(app.history["mem-head"]) == 1


async def test_floor_row_keeps_cpu_when_the_mini_bar_does_not_fit(tmp_path: Path, monkeypatch):
    from app import DGXTop, NodeBox

    _config(tmp_path / "c.toml")
    configure(tmp_path / "c.toml")
    _stub(monkeypatch, [_unit("head"), _unit("worker", worker=True, plan=(58, 12, 1), used=98)])
    app = DGXTop()
    async with app.run_test(size=(70, 12)) as pilot:
        await pilot.pause()
        assert app.floor and app.node_mode == "table"
        wide = app.query_one("#node-0", NodeBox).render().plain
        assert " 30%" in wide and any(c in wide for c in "█▓╱▏▎▍▌▋▊▉")
    narrow_app = DGXTop()
    async with narrow_app.run_test(size=(50, 12)) as pilot:
        await pilot.pause()
        assert narrow_app.floor and narrow_app.node_mode == "table"
        narrow = narrow_app.query_one("#node-0", NodeBox).render().plain
        assert " 30%" in narrow  # cpu survives; the mini bar is dropped first
        assert not any(c in narrow for c in "█▓╱▏▎▍▌▋▊▉")
