"""Synthetic-data tests (src/simulation.py and its collector hook)."""

from __future__ import annotations

import math


def test_simulate_cluster_returns_online_hosted_finite_units():
    from simulation import simulate_cluster

    stats = simulate_cluster(12)
    assert len(stats.units) == 12
    for u in stats.units:
        assert u.online is True
        assert u.model_hosted is True
        assert u.label.startswith("spark-")
        # every plotted/derived value is finite and within a plausible range
        assert math.isfinite(u.gpu_util_pct)
        assert 0 <= u.gpu_util_pct <= 100
        assert math.isfinite(u.mem_used_bytes)
        assert 0 < u.mem_used_bytes < u.mem_total_bytes
        assert math.isfinite(u.kv_cache_pct)
        assert 0 <= u.kv_cache_pct <= 100
        assert math.isfinite(u.throughput_tok_s)
        assert u.throughput_tok_s > 0
        assert math.isfinite(u.roce_rx_bps)
        assert math.isfinite(u.ttft_p95_ms)
        assert all(math.isfinite(v) for v in u.cpu_cores_util)


def test_simulated_units_carry_a_memory_plan_below_measured_use():
    from simulation import simulate_cluster
    from stats import mem_stack

    for _ in range(3):
        for u in simulate_cluster(12).units:
            assert u.mem_phys_bytes == 128 * 1024**3 > u.mem_total_bytes
            plan = u.mem_weights_bytes + u.mem_kv_pool_bytes + u.mem_graphs_bytes
            assert u.mem_weights_bytes > 0 and u.mem_kv_pool_bytes > 0 and u.mem_graphs_bytes > 0
            assert plan < u.mem_used_bytes
            assert mem_stack(u) is not None


def test_simulate_cluster_evolves_between_polls():
    from simulation import simulate_cluster

    a = simulate_cluster(5)
    b = simulate_cluster(5)
    a_gpu = [u.gpu_util_pct for u in a.units]
    b_gpu = [u.gpu_util_pct for u in b.units]
    assert a_gpu != b_gpu, "consecutive polls must differ (series evolve)"


async def test_poll_cluster_returns_simulated_when_enabled():
    import collector
    import config

    config.SIMULATION_NODES = 12
    try:
        stats = await collector.poll_cluster()
    finally:
        config.SIMULATION_NODES = None
    assert len(stats.units) == 12
    assert all(u.online for u in stats.units)
    # no SSH/HTTP: poll_cluster returned synthetic data without touching SPARK_UNITS
    assert all(u.label.startswith("spark-") for u in stats.units)


def test_configure_simulate_owns_routing_flag():
    import config

    config.configure("/tmp/__dgx_top_missing__/x.toml", simulate=5)
    try:
        assert config.SIMULATION_NODES == 5
        assert len(config.SPARK_UNITS) == 5
    finally:
        config.SIMULATION_NODES = None
    assert config.SIMULATION_NODES is None


async def test_collector_poll_cluster_with_simulate_config(tmp_path):
    """configure(simulate=N) must route poll_cluster to the simulator."""
    import collector
    import config

    config.configure(str(tmp_path / "missing.toml"), simulate=3)
    try:
        stats = await collector.poll_cluster()
    finally:
        config.SIMULATION_NODES = None
    assert len(stats.units) == 3
    assert all(u.online and u.model_hosted for u in stats.units)
