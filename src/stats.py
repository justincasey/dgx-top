from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass
class TopologyPort:
    """A single InfiniBand or RoCE port detected on a node."""

    device: str = ""
    port: str = ""
    state: str = ""
    link_layer: str = ""


@dataclass
class TopologyInterface:
    """A single network interface with link state detected on a node."""

    name: str = ""
    carrier: str = "0"
    mac: str = ""
    driver: str = ""


@dataclass
class TopologyInfo:
    """Derived cluster topology from per-node sysfs reads."""

    topology_type: str = "UNKNOWN"  # SINGLE, DUAL, RING, SWITCHED, UNKNOWN
    description: str = ""  # Human-readable summary
    ports: list[TopologyPort] = field(default_factory=list)
    interfaces: list[TopologyInterface] = field(default_factory=list)


@dataclass
class SparkUnitStats:
    label: str = ""
    is_worker: bool = False
    model_hosted: bool = False
    model_name: str = ""
    kv_cache_pct: float = 0.0  # 0-100, from vllm:kv_cache_usage_perc × 100
    kv_total_blocks: int = 0  # from cache_config_info num_gpu_blocks (minus null block)
    kv_block_size: int = 0  # tokens per block
    kv_total_tokens: int = 0  # KV capacity: kv_cache_size_tokens, else blocks*block_size
    kv_prefix_hit_rate: float = -1.0  # prefix cache hit rate 0-100, -1 = unavailable
    kv_cache_free_blocks: int = 0  # free blocks = kv_total_blocks * (1 - usage_pct)
    kv_cache_used_tokens: int = 0  # used token capacity = kv_total_tokens * usage_pct
    prefix_queries_total: float = 0.0  # cumulative vllm:prefix_cache_queries_total
    prefix_hits_total: float = 0.0  # cumulative vllm:prefix_cache_hits_total
    requests_running: int = 0
    requests_waiting: int = 0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p99_ms: float = 0.0
    generation_tokens_total: float = 0.0
    throughput_tok_s: float = 0.0
    # Prompt token tracking (from Spark Monitor inspiration)
    prompt_tokens_total: float = 0.0
    prompt_throughput_tok_s: float = 0.0
    prompt_gen_ratio: float = 0.0
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    gpu_mem_pct: float = 0.0
    mem_used_bytes: int = 0
    mem_total_bytes: int = 0
    power_w: float = 0.0
    temp_c: float = 0.0
    cpu_temp_c: float = 0.0
    online: bool = False
    cpu_cores_total: int = 0
    cpu_cores_util: list[float] = field(default_factory=list)
    # Current GPU SM clock (MHz) from nvidia-smi clocks.current.sm; 0 = unavailable.
    gpu_clock_mhz: float = 0.0
    # Aggregate RoCE/InfiniBand RX/TX throughput across active ports (bytes/s).
    roce_rx_bps: float = 0.0
    roce_tx_bps: float = 0.0
    # Full-duplex wire capacity across active ports (bytes/s); 0 = unknown.
    roce_capacity_bps: float = 0.0
    error: str = ""
    # Per-node topology data (raw, for cluster-level derivation)
    topology_ports: list[TopologyPort] = field(default_factory=list)
    topology_interfaces: list[TopologyInterface] = field(default_factory=list)
    swap_total_kb: int = 0
    swap_used_kb: int = 0
    swap_cached_kb: int = 0
    swap_in_rate: float = 0.0
    swap_out_rate: float = 0.0
    majflt_rate: float = 0.0
    psi_some_avg10: float = 0.0
    psi_full_avg10: float = 0.0
    psi_full_total_delta: int = 0
    allocstall_total: int = 0
    allocstall_this_poll: int = 0
    kswapd_scan_rate: float = 0.0
    kswapd_steal_rate: float = 0.0
    workingset_refault_rate: float = 0.0
    mem_avail_kb: int = 0
    mem_free_kb: int = 0


@dataclass
class ClusterStats:
    units: list[SparkUnitStats] = field(default_factory=list)
    topology: TopologyInfo = field(default_factory=TopologyInfo)

    @property
    def total_throughput(self) -> float:
        return sum(u.throughput_tok_s for u in self.units)

    @property
    def total_prompt_throughput(self) -> float:
        return sum(u.prompt_throughput_tok_s for u in self.units)

    @property
    def total_power(self) -> float:
        return sum(u.power_w for u in self.units)

    @property
    def avg_power(self) -> float:
        if not self.units:
            return 0.0
        return self.total_power / len(self.units)

    @property
    def max_temp(self) -> float:
        if not self.units:
            return 0.0
        return max(u.temp_c for u in self.units)

    @property
    def max_gpu_util(self) -> float:
        if not self.units:
            return 0.0
        return max(u.gpu_util_pct for u in self.units)

    @property
    def all_online(self) -> bool:
        return len(self.units) > 0 and all(u.online for u in self.units)

    def unit(self, idx: int) -> SparkUnitStats | None:
        try:
            return self.units[idx]
        except IndexError:
            return None

    @property
    def hosted_units(self) -> list[SparkUnitStats]:
        """Units that are hosting a model (have vLLM metrics)."""
        return [u for u in self.units if u.model_hosted]

    @property
    def total_kv_capacity_tokens(self) -> int:
        """Total KV cache token capacity across all hosted units.
        In TP/DP setups, nodes share the same pool, so this is the
        first hosted unit's total (others would be duplicates)."""
        hosted = self.hosted_units
        if not hosted:
            return 0
        return hosted[0].kv_total_tokens

    @property
    def total_kv_used_tokens(self) -> int:
        """Total KV cache used tokens (block-allocated capacity) from first hosted unit."""
        hosted = self.hosted_units
        if not hosted:
            return 0
        return hosted[0].kv_cache_used_tokens

    @property
    def kv_cache_pct(self) -> float:
        """Aggregate KV cache usage percentage from first hosted unit.
        In TP setups all nodes share the same pool, so this is accurate."""
        hosted = self.hosted_units
        if not hosted:
            return 0.0
        return hosted[0].kv_cache_pct

    @property
    def kv_prefix_hit_rate(self) -> float:
        hosted = self.hosted_units
        if not hosted:
            return -1.0
        return hosted[0].kv_prefix_hit_rate

    @property
    def total_kv_blocks(self) -> int:
        hosted = self.hosted_units
        if not hosted:
            return 0
        return hosted[0].kv_total_blocks


class ThrashLevel(enum.IntEnum):
    OK = 0
    CAUTION = 1
    CRITICAL = 2


def compute_thrash_risk(stats: SparkUnitStats) -> tuple[ThrashLevel, str]:
    """Return (risk_level, reason_string). Multi-factor: any critical signal
    or 3+ caution signals -> CRITICAL. 2+ caution signals -> CAUTION."""
    signals: list[tuple[ThrashLevel, str]] = []

    # MemAvailable ratio
    if stats.mem_total_bytes > 0:
        avail_ratio = stats.mem_avail_kb * 1024 / stats.mem_total_bytes * 100
        if avail_ratio < 5:
            signals.append((ThrashLevel.CRITICAL, f"mem_avail={avail_ratio:.1f}%"))
        elif avail_ratio < 10:
            signals.append((ThrashLevel.CAUTION, f"mem_avail={avail_ratio:.1f}%"))

    # Swap used ratio
    if stats.swap_total_kb > 0:
        swap_used_pct = stats.swap_used_kb / stats.swap_total_kb * 100
        if swap_used_pct > 70:
            signals.append((ThrashLevel.CRITICAL, f"swap_used={swap_used_pct:.0f}%"))
        elif swap_used_pct > 30:
            signals.append((ThrashLevel.CAUTION, f"swap_used={swap_used_pct:.0f}%"))

    # SwapCached (refault churn indicator)
    if stats.swap_cached_kb > 2 * 1024 * 1024:
        signals.append((ThrashLevel.CRITICAL, f"swap_cache={stats.swap_cached_kb / 1e6:.1f}G"))
    elif stats.swap_cached_kb > 512 * 1024:
        signals.append((ThrashLevel.CAUTION, f"swap_cache={stats.swap_cached_kb / 1e6:.1f}G"))

    # PSI full avg10
    if stats.psi_full_avg10 > 1:
        signals.append((ThrashLevel.CRITICAL, f"PSI={stats.psi_full_avg10:.1f}%"))
    elif stats.psi_full_avg10 > 0.1:
        signals.append((ThrashLevel.CAUTION, f"PSI={stats.psi_full_avg10:.1f}%"))

    # Allocstall rate (direct reclaim)
    if stats.allocstall_this_poll > 10:
        signals.append((ThrashLevel.CRITICAL, f"allocstall={stats.allocstall_this_poll}/s"))
    elif stats.allocstall_this_poll > 1:
        signals.append((ThrashLevel.CAUTION, f"allocstall={stats.allocstall_this_poll}/s"))

    # kswapd efficiency (steal/scan ratio <70% means wasted scanning)
    if stats.kswapd_scan_rate > 0:
        ratio = (
            (stats.kswapd_steal_rate / stats.kswapd_scan_rate * 100)
            if stats.kswapd_scan_rate > 0
            else 100
        )
        if ratio < 70:
            signals.append((ThrashLevel.CRITICAL, f"kswapd_eff={ratio:.0f}%"))
        elif ratio < 90:
            signals.append((ThrashLevel.CAUTION, f"kswapd_eff={ratio:.0f}%"))

    # Major fault rate
    if stats.majflt_rate > 100:
        signals.append((ThrashLevel.CRITICAL, f"majflt={stats.majflt_rate:.0f}/s"))
    elif stats.majflt_rate > 10:
        signals.append((ThrashLevel.CAUTION, f"majflt={stats.majflt_rate:.0f}/s"))

    critical_count = sum(1 for lvl, _ in signals if lvl == ThrashLevel.CRITICAL)
    caution_count = sum(1 for lvl, _ in signals if lvl == ThrashLevel.CAUTION)
    reasons = "; ".join(desc for _, desc in signals)

    if critical_count > 0 or caution_count >= 3:
        return (ThrashLevel.CRITICAL, reasons)
    elif caution_count >= 2:
        return (ThrashLevel.CAUTION, reasons)
    return (ThrashLevel.OK, reasons)
