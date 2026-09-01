"""Synthetic telemetry for designing and verifying large-cluster layouts.

When simulation is enabled (``--simulate N``) the collector returns plausible,
evolving per-node stats generated here instead of polling real hardware, so a
cluster of 1-12 Sparks can be exercised with no SSH/vLLM access.
"""

from __future__ import annotations

import math
import random

from stats import ClusterStats, SparkUnitStats, TopologyInfo

_MEM_TOTAL_BYTES = 120 * 1024**3
_SWAP_TOTAL_KB = 4 * 1024 * 1024
_ROCE_CAPACITY_BPS = 5e10
_INTERIOR = 20  # per-node cores shown in the core grid


class Simulator:
    """A stateful generator of smooth, evolving per-node Spark stats.

    Each node has its own phase and base offsets; values follow a slow
    sinusoid plus a bounded random walk so the meters and sparklines animate
    and consecutive ``cluster()`` calls differ, while every value stays finite
    and within a plausible range.
    """

    def __init__(self, n: int):
        self.n = n
        rng = random.Random(0xD6A5)
        self._step = 0
        self._phase = [rng.uniform(0.0, math.tau) for _ in range(n)]
        self._gpu_base = [rng.uniform(52.0, 88.0) for _ in range(n)]
        self._load_base = [rng.uniform(28.0, 72.0) for _ in range(n)]
        self._kv_base = [rng.uniform(18.0, 55.0) for _ in range(n)]
        self._roce_base = [rng.uniform(400.0, 2600.0) for _ in range(n)]  # MB/s
        self._tok_base = [rng.uniform(600.0, 1500.0) for _ in range(n)]
        self._drift = [0.0] * n
        self._mem_used = [rng.uniform(30.0, 95.0) for _ in range(n)]  # GiB
        self._temp = [rng.uniform(52.0, 72.0) for _ in range(n)]

    def _bounded(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _walk(self, i: int, width: float) -> None:
        """Advance a per-node random drift and clamp it to +/- ``width``."""
        self._drift[i] += random.Random(self._step * 7919 + i * 104729).uniform(-width, width)
        self._drift[i] = max(-1.0, min(1.0, self._drift[i]))

    def _gpu_util(self, i: int) -> float:
        t = self._step / 8.0
        wave = math.sin(t + self._phase[i]) * 12.0
        spike = 8.0 if (self._step // 40 + i) % 9 == 0 else 0.0
        return self._bounded(self._gpu_base[i] + wave + self._drift[i] * 15.0 + spike, 5.0, 100.0)

    def _mem_util(self, i: int) -> float:
        wave = math.sin(self._step / 12.0 + self._phase[i]) * 3.0
        pct = self._mem_used[i] / (110.0) * 100.0 + wave
        return self._bounded(pct, 20.0, 100.0)

    def _cpu_util(self, i: int) -> float:
        wave = math.sin(self._step / 7.0 + self._phase[i] * 1.3) * 14.0
        return self._bounded(self._load_base[i] + wave + self._drift[i] * 12.0, 5.0, 100.0)

    def _kv_pct(self, i: int) -> float:
        wave = math.sin(self._step / 15.0 + self._phase[i]) * 6.0
        return self._bounded(self._kv_base[i] + wave + self._drift[i] * 8.0, 5.0, 95.0)

    def _roce_rate(self, i: int) -> float:
        wave = math.sin(self._step / 6.0 + self._phase[i]) * 0.35
        return self._bounded(self._roce_base[i] * (1.0 + wave + self._drift[i] * 0.3), 80.0, 4200.0)

    def _throughput(self, i: int) -> float:
        wave = math.sin(self._step / 9.0 + self._phase[i]) * 0.25
        return self._bounded(self._tok_base[i] * (1.0 + wave + self._drift[i] * 0.2), 120.0, 2400.0)

    def unit(self, i: int) -> SparkUnitStats:
        self._walk(i, 0.25)
        idx = i + 1
        s = SparkUnitStats(label=f"spark-{idx}")
        s.is_worker = i > 0
        s.online = True
        s.model_hosted = True
        s.model_name = "Qwen3.6-27B-Instruct" if i == 0 else ""
        gpu = self._gpu_util(i)
        s.gpu_util_pct = gpu
        s.mem_util_pct = self._mem_util(i)
        s.temp_c = self._bounded(self._temp[i] + self._drift[i] * 6.0 + gpu / 20.0, 42.0, 92.0)
        s.power_w = self._bounded(
            self._gpu_base[i] * 5.4 + gpu * 2.2 + self._drift[i] * 40.0, 60.0, 520.0
        )
        s.mem_used_bytes = int(self._mem_used[i] * 1024**3)
        s.mem_total_bytes = _MEM_TOTAL_BYTES
        s.swap_total_kb = _SWAP_TOTAL_KB
        s.swap_used_kb = int(self._bounded(self._drift[i] * 1.5, 0.0, 0.5) * _SWAP_TOTAL_KB)
        s.cpu_cores_total = _INTERIOR
        base_cpu = self._cpu_util(i)
        s.cpu_cores_util = [
            self._bounded(
                base_cpu + math.sin(self._step / 3.0 + c * 0.7 + self._phase[i]) * 22.0, 0.0, 100.0
            )
            for c in range(_INTERIOR)
        ]
        s.cpu_temp_c = self._bounded(45.0 + base_cpu * 0.18 + self._drift[i] * 3.0, 38.0, 88.0)
        s.gpu_clock_mhz = self._bounded(2100.0 + gpu * 3.5 + self._drift[i] * 80.0, 900.0, 2900.0)
        s.roce_rx_bps = self._roce_rate(i) * 1e6
        s.roce_tx_bps = self._roce_rate(i) * 0.4e6
        s.roce_capacity_bps = _ROCE_CAPACITY_BPS
        kv_pct = self._kv_pct(i)
        s.kv_cache_pct = kv_pct
        s.kv_total_tokens = 3_800_000
        s.kv_cache_used_tokens = int(3_800_000 * kv_pct / 100.0)
        s.kv_prefix_hit_rate = self._bounded(30.0 + self._drift[i] * 40.0 + kv_pct * 0.3, 5.0, 95.0)
        s.requests_running = max(
            0,
            int(
                round(
                    self._bounded(
                        self._tok_base[i] / 400.0
                        + math.sin(self._step / 5.0 + self._phase[i]) * 1.5,
                        0.0,
                        24.0,
                    )
                )
            ),
        )
        s.requests_waiting = max(0, int(round(self._bounded(self._drift[i] * 4.0, 0.0, 6.0))))
        s.throughput_tok_s = self._throughput(i)
        s.prompt_throughput_tok_s = s.throughput_tok_s * self._bounded(
            2.2 + self._drift[i], 1.5, 4.0
        )
        s.prompt_gen_ratio = self._bounded(2.0 + self._drift[i] * 0.5, 1.2, 4.5)
        s.ttft_p50_ms = self._bounded(450.0 + self._drift[i] * 100.0, 250.0, 900.0)
        s.ttft_p95_ms = self._bounded(2400.0 + self._drift[i] * 900.0, 1200.0, 5200.0)
        s.ttft_p99_ms = s.ttft_p95_ms * 1.25
        s.mem_avail_kb = int((_MEM_TOTAL_BYTES - s.mem_used_bytes) / 1024)
        s.mem_free_kb = int((_MEM_TOTAL_BYTES - s.mem_used_bytes - 8 * 1024**3) / 1024)
        return s

    def cluster(self) -> ClusterStats:
        self._step += 1
        units = [self.unit(i) for i in range(self.n)]
        topology = TopologyInfo(topology_type="SWITCHED" if self.n >= 3 else "DUAL")
        return ClusterStats(units=units, topology=topology)


_simulator: Simulator | None = None


def simulate_cluster(n: int) -> ClusterStats:
    """Return the next evolving synthetic cluster for ``n`` Spark units."""
    global _simulator
    if _simulator is None or _simulator.n != n:
        _simulator = Simulator(n)
    return _simulator.cluster()
