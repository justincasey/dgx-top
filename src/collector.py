from __future__ import annotations

import asyncio
import re
import time
from typing import Dict, List, Tuple

import httpx

from config import NVIDIA_SMI_CMD, SPARK_UNITS
from stats import (
    ClusterStats,
    SparkUnitStats,
    TopologyInfo,
    TopologyInterface,
    TopologyPort,
)

# Cache model names fetched once at startup — they don't change between polls.
_model_names: dict[int, str] = {}


async def _init_model_names() -> None:
    """Fetch model names from all Spark units once at startup."""

    async def fetch_one(client: httpx.AsyncClient, uid: int, vllm_url: str) -> None:
        try:
            resp = await client.get(f"{vllm_url}/v1/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                _model_names[uid] = models[0].get("id") or models[0].get("model", "")
        except Exception:
            pass  # model name is optional

    async with httpx.AsyncClient(timeout=10) as client:
        await asyncio.gather(
            *(fetch_one(client, uid, str(cfg["vllm_url"])) for uid, cfg in SPARK_UNITS.items())
        )


def _parse_prometheus_histogram(
    lines: List[str], metric_name: str
) -> Tuple[Dict[float, float], float]:
    """Parse a Prometheus histogram and return (buckets, count)."""
    buckets: Dict[float, float] = {}
    count = 0.0
    for line in lines:
        if line.startswith(f"{metric_name}_bucket{{"):
            m = re.search(r'le="([^"]+)"', line)
            if m:
                le_str = m.group(1)
                parts = line.split()
                cnt = float(parts[-1]) if len(parts) >= 2 else 0.0
                if le_str == "+Inf":
                    le = float("inf")
                else:
                    le = float(le_str)
                buckets[le] = cnt
        elif line.startswith(f"{metric_name}_count"):
            parts = line.split()
            if len(parts) >= 2:
                count = float(parts[-1])
    return buckets, count


def _estimate_quantile(buckets: Dict[float, float], count: float, q: float) -> float:
    """Estimate quantile from histogram buckets via linear interpolation."""
    if count == 0:
        return 0.0
    target = q * count
    prev_le = 0.0
    prev_cum = 0.0
    for le in sorted(buckets.keys()):
        cum = buckets[le]
        if cum >= target:
            if cum == prev_cum:
                return float(le)
            fraction = (target - prev_cum) / (cum - prev_cum)
            return prev_le + (le - prev_le) * fraction
        prev_le = le
        prev_cum = cum
    return prev_le


def _parse_vllm_metrics(text: str) -> SparkUnitStats:
    """Parse vLLM Prometheus metrics into SparkUnitStats.

    KV cache data available from /metrics:
      - vllm:kv_cache_usage_perc (Gauge): block-level allocation fraction (0-1).
        Usage = 1.0 - (free_blocks / (total_gpu_blocks - 1)). The null block is
        subtracted from total because vLLM reserves 1 block as sentinel. This
        is genuinely accurate at the block-granular level: a partially filled
        block counts as fully allocated, which is the correct framing for
        capacity planning (the scheduler cannot use partial blocks).

      - vllm:cache_config_info{..., num_gpu_blocks="N", block_size="BS", ...}
        Static config set at startup. num_gpu_blocks is the total including the
        null block, so usable = num_gpu_blocks - 1.

      - vllm:prefix_cache_hit_rate or vllm:cache_hit_rate (Gauge): 0-1.

    Derived token counts: we compute total_tokens = usable_blocks * block_size
    and used_tokens = total_tokens * usage_pct/100. These represent the
    *block-allocated token capacity*, not actual stored token count (which
    would require per-block hash-table inspection). For capacity planning this
    is the correct metric: "how much of my pool is consumed" at scheduler
    granularity.
    """
    s = SparkUnitStats()
    lines = text.strip().splitlines()
    # Cache config — block_size and total_blocks from cache_config_info at startup
    for line in lines:
        if "vllm:cache_config_info{" in line:
            m = re.search(r'num_gpu_blocks="(\d+)"', line)
            if m:
                s.kv_total_blocks = int(m.group(1))
            m = re.search(r'block_size="(\d+)"', line)
            if m:
                s.kv_block_size = int(m.group(1))
        if s.kv_total_blocks > 0 and s.kv_block_size > 0:
            s.kv_total_tokens = s.kv_total_blocks * s.kv_block_size
    # KV cache usage — block-level allocation fraction. This is from
    # BlockPool.get_usage(): 1.0 - (free_blocks / (total_gpu_blocks - 1)).
    # vLLM reserves 1 null block, which is already netted out by vLLM
    # in the usage gauge. We do NOT subtract it again.
    for line in lines:
        if line.startswith("vllm:kv_cache_usage_perc"):
            parts = line.split()
            if len(parts) >= 2:
                val = float(parts[-1])
                s.kv_cache_pct = val * 100
                # Derive block and token counts from the usage percentage.
                # usage_pct already accounts for the null block subtraction.
                if s.kv_block_size > 0 and s.kv_total_blocks > 0:
                    s.kv_cache_free_blocks = int(s.kv_total_blocks * (1 - val))
                    used_blocks = s.kv_total_blocks - s.kv_cache_free_blocks
                    s.kv_cache_used_tokens = used_blocks * s.kv_block_size
    for line in lines:
        if line.startswith("vllm:prefix_cache_hit_rate"):
            parts = line.split()
            if len(parts) >= 2:
                s.kv_prefix_hit_rate = float(parts[-1]) * 100
        elif line.startswith("vllm:cache_hit_rate"):
            parts = line.split()
            if len(parts) >= 2:
                s.kv_prefix_hit_rate = float(parts[-1]) * 100

    # Requests running
    for line in lines:
        if line.startswith("vllm:num_requests_running"):
            parts = line.split()
            if len(parts) >= 2:
                s.requests_running = int(float(parts[-1]))
        elif line.startswith("vllm:num_requests_waiting"):
            parts = line.split()
            if len(parts) >= 2:
                s.requests_waiting = int(float(parts[-1]))

    # TTFT histogram
    ttft_lines = [l for l in lines if "vllm:time_to_first_token" in l]
    if ttft_lines:
        buckets, count = _parse_prometheus_histogram(ttft_lines, "vllm:time_to_first_token_seconds")
        s.ttft_p50_ms = _estimate_quantile(buckets, count, 0.50) * 1000
        s.ttft_p99_ms = _estimate_quantile(buckets, count, 0.99) * 1000

    # ITL histogram
    itl_lines = [l for l in lines if "vllm:time_per_output_token" in l]
    if itl_lines:
        buckets, count = _parse_prometheus_histogram(
            itl_lines, "vllm:time_per_output_token_seconds"
        )
        s.itl_p50_ms = _estimate_quantile(buckets, count, 0.50) * 1000
        s.itl_p99_ms = _estimate_quantile(buckets, count, 0.99) * 1000

    # Prefer the live generation token counter. The request histogram fallback only
    # updates when requests finish, so it can lag and spike during long generations.
    generation_tokens_total = 0.0
    has_generation_tokens = False
    request_generation_tokens_total = 0.0
    has_request_generation_tokens = False
    for line in lines:
        if line.startswith("vllm:generation_tokens"):
            parts = line.split()
            if len(parts) >= 2:
                has_generation_tokens = True
                generation_tokens_total += float(parts[-1])
        elif line.startswith("vllm:request_generation_tokens_sum"):
            parts = line.split()
            if len(parts) >= 2:
                has_request_generation_tokens = True
                request_generation_tokens_total += float(parts[-1])
    s.generation_tokens_total = (
        generation_tokens_total if has_generation_tokens else request_generation_tokens_total
    )

    if (
        has_generation_tokens
        or has_request_generation_tokens
        or s.kv_total_blocks > 0
        or s.requests_running > 0
        or s.ttft_p50_ms > 0
    ):
        s.model_hosted = True

    # Prompt tokens (split from generation — Spark Monitor insight)
    for line in lines:
        if line.startswith("vllm:prompt_tokens"):
            parts = line.split()
            if len(parts) >= 2:
                s.prompt_tokens_total += float(parts[-1])

    s.online = True
    return s


def _parse_nvidia_smi(output: str) -> Tuple[float, float, float, float, float]:
    """Parse nvidia-smi output: gpu_util_pct, mem_util_pct, mem_pct, power_w, temp_c."""
    lines = output.strip().splitlines()
    if not lines:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    parts = lines[0].split(", ")
    if len(parts) >= 8:

        def _safe_float(v: str) -> float:
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        gpu_util = _safe_float(parts[0])
        mem_util = _safe_float(parts[1])
        mem_used = _safe_float(parts[2])
        mem_total = _safe_float(parts[3])
        power = _safe_float(parts[4])
        temp = _safe_float(parts[5])
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0.0
        return gpu_util, mem_util, mem_pct, power, temp
    return 0.0, 0.0, 0.0, 0.0, 0.0


_prev_cpu_ticks: Dict[str, Tuple[int, ...]] = {}


def _parse_cpu_stat(host: str, stat_text: str) -> Tuple[int, list[float]]:
    """Parse /proc/stat output into per-core CPU utilization via delta tracking.

    Returns (core_count, [util_0, util_1, ...]) or (0, []) on failure.
    """
    lines = stat_text.strip().splitlines()
    if not lines:
        return 0, []

    core_lines = lines[1:]  # skip aggregate "cpu"
    core_count = len(core_lines)
    if core_count == 0:
        return 0, []

    # Check if core count changed — reset if so
    host_keys = [k for k in _prev_cpu_ticks if k.startswith(f"{host}-cpu")]
    if host_keys:
        prev_count = len(host_keys)
        if prev_count != core_count:
            for k in host_keys:
                del _prev_cpu_ticks[k]

    utilizations: list[float] = []
    for i, line in enumerate(core_lines):
        parts = line.split()
        if len(parts) < 9:
            utilizations.append(0.0)
            continue
        ticks = tuple(int(v) for v in parts[1:9])
        cpu_key = f"{host}-cpu{i}"
        prev = _prev_cpu_ticks.get(cpu_key)
        if prev is not None and len(prev) == len(ticks):
            total_delta = sum(t - p for t, p in zip(ticks, prev))
            idle_delta = ticks[3] - prev[3]
            if total_delta > 0:
                util = (1.0 - idle_delta / total_delta) * 100.0
                utilizations.append(max(0.0, min(100.0, util)))
            else:
                utilizations.append(0.0)
        else:
            utilizations.append(0.0)
        _prev_cpu_ticks[cpu_key] = ticks

    return core_count, utilizations


def _parse_memory_thrash_output(output: str) -> dict:
    """Parse memory-thrash SSH output into a dict.

    The first two lines from meminfo and vmstat are required. The PSI lines are
    optional because /proc/pressure/memory is not available on every host.

    Returns empty dict when a required line cannot be parsed.
    """
    lines = output.strip().splitlines()
    if len(lines) < 2:
        return {}

    # Line 1: /proc/meminfo
    mi = lines[0].split()
    if len(mi) < 6:
        return {}
    result: dict = {}
    result["swap_total_kb"] = int(mi[0])
    swap_free_kb = int(mi[1])
    result["swap_used_kb"] = result["swap_total_kb"] - swap_free_kb
    result["swap_cached_kb"] = int(mi[2])
    result["mem_avail_kb"] = int(mi[3])
    result["mem_free_kb"] = int(mi[4])
    result["mem_total_kb"] = int(mi[5])

    # Line 2: /proc/vmstat counters
    vs = lines[1].split()
    if len(vs) < 11:
        return {}
    result["pswpin"] = int(vs[0])
    result["pswpout"] = int(vs[1])
    result["pgmajfault"] = int(vs[2])
    result["allocstall_dma"] = int(vs[3])
    result["allocstall_normal"] = int(vs[4])
    result["allocstall_movable"] = int(vs[5])
    result["allocstall_device"] = int(vs[6])
    result["pgscan_kswapd"] = int(vs[7])
    result["pgsteal_kswapd"] = int(vs[8])
    result["workingset_refault_anon"] = int(vs[9])
    result["workingset_refault_file"] = int(vs[10])

    # Lines 3 and 4: optional PSI some/full avg10 + total
    if len(lines) >= 3:
        psi_some = lines[2].split()
        if len(psi_some) >= 2:
            result["psi_some_avg10"] = float(psi_some[0])
            result["psi_some_total"] = int(psi_some[1])

    if len(lines) >= 4:
        psi_full = lines[3].split()
        if len(psi_full) >= 2:
            result["psi_full_avg10"] = float(psi_full[0])
            result["psi_full_total"] = int(psi_full[1])

    return result


_SSH_CTRL = "/tmp/dgx-top-ssh-%C"


def _ssh_base_args(target: str) -> list[str]:
    """Common SSH args with ControlMaster multiplexing."""
    return [
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={_SSH_CTRL}",
        "-o",
        "ControlPersist=300",
        target,
    ]


async def _ssh_run(target: str, cmd: str) -> str:
    """Run a remote command via SSH, return stdout or raise on failure."""
    proc = await asyncio.create_subprocess_exec(
        *_ssh_base_args(target),
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or f"ssh exited {proc.returncode}")
    return stdout.decode()


async def _fetch_vllm_metrics(vllm_url: str) -> str:
    """Fetch vLLM metrics via HTTP."""
    url = f"{vllm_url}/metrics"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


def _parse_telemetry_output(stdout: str) -> dict:
    """Parse delimited SSH output into per-section dicts.

    Sections are delimited by lines matching ---<NAME>---.
    Each section is emitted when the next marker is seen,
    so empty sections produce empty-string values.
    """
    result: dict = {}
    prev_section: str | None = None
    prev_sections: list[str] = []

    for line in stdout.splitlines():
        if line == "---GPU---":
            if prev_section is not None:
                result[prev_section] = "\n".join(prev_sections)
            prev_section = "gpu_output"
            prev_sections = []
        elif line == "---CPU_TEMP---":
            if prev_section is not None:
                result[prev_section] = "\n".join(prev_sections)
            prev_section = "cpu_temp"
            prev_sections = []
        elif line == "---CPU_STAT---":
            if prev_section is not None:
                result[prev_section] = "\n".join(prev_sections)
            prev_section = "cpu_stat"
            prev_sections = []
        elif line == "---THRASH---":
            if prev_section is not None:
                result[prev_section] = "\n".join(prev_sections)
            prev_section = "thrash_output"
            prev_sections = []
        elif line == "---TOPOLOGY---":
            if prev_section is not None:
                result[prev_section] = "\n".join(prev_sections)
            prev_section = "topology_output"
            prev_sections = []
        else:
            prev_sections.append(line)
    # Last section
    if prev_section is not None:
        result[prev_section] = "\n".join(prev_sections)

    return result


# ─── Topology (RoCE/InfiniBand) ──────────────────────────────────────────


def _parse_topology_output(output: str) -> dict:
    """Parse topology SSH section output into structured data.

    Format examples:
      ib:mlx5_0:1:4: ACTIVE:InfiniBand
      192.0.2.10/24 dev enp1s0 ...
    """
    result: dict = {
        "ib_ports": [],
        "guid": "",
        "net_interfaces": [],
        "ip_addrs": [],
    }
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("ib:"):
            # ib:device:port:state:link_layer
            parts = line.split(":", 4)
            if len(parts) >= 4:
                result["ib_ports"].append(
                    {
                        "device": parts[1],
                        "port": parts[2],
                        "state": parts[3].strip(),
                        "link_layer": parts[4].strip() if len(parts) >= 5 else "",
                    }
                )
        elif line.startswith("guid:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                result["guid"] = parts[2]
        elif line.startswith("net:"):
            # net:name:carrier:mac:driver
            parts = line.split(":", 4)
            if len(parts) >= 5:
                result["net_interfaces"].append(
                    {
                        "name": parts[1],
                        "carrier": parts[2],
                        "mac": parts[3],
                        "driver": parts[4],
                    }
                )
        else:
            result["ip_addrs"].append(line)
    return result


def _derive_topology(node_topos: dict[int, dict]) -> TopologyInfo:
    """Derive cluster-wide topology from per-node topology data.

    Cross-references InfiniBand port states, GUIDs, and network interface
    carrier/driver info to determine actual interconnect topology.
    """
    ports: list[TopologyPort] = []
    interfaces: list[TopologyInterface] = []
    ib_devices: set[str] = set()
    active_ib_ports = 0
    active_interconnect_ifaces = 0

    for _node_id, topos in node_topos.items():
        for ibp in topos.get("ib_ports", []):
            ib_devices.add(ibp.get("device", ""))
            is_active = "ACTIVE" in ibp.get("state", "")
            if is_active:
                active_ib_ports += 1
            ports.append(
                TopologyPort(
                    device=ibp.get("device", ""),
                    port=ibp.get("port", ""),
                    state=ibp.get("state", ""),
                    link_layer=ibp.get("link_layer", ""),
                )
            )
        for iface in topos.get("net_interfaces", []):
            driver = iface.get("driver", "")
            carrier = iface.get("carrier", "0")
            is_interconnect = "mlx5" in driver or "ib" in iface.get("name", "")
            if is_interconnect and carrier == "1":
                active_interconnect_ifaces += 1
            interfaces.append(
                TopologyInterface(
                    name=iface.get("name", ""),
                    carrier=carrier,
                    mac=iface.get("mac", ""),
                    driver=driver,
                )
            )

    if active_ib_ports >= 2 and len(ib_devices) >= 1:
        return TopologyInfo(
            topology_type="DUAL",
            description=f"{active_ib_ports} active InfiniBand ports",
            ports=ports,
            interfaces=interfaces,
        )
    elif active_interconnect_ifaces >= 2:
        return TopologyInfo(
            topology_type="DUAL",
            description=f"{active_interconnect_ifaces} active RoCE links",
            ports=ports,
            interfaces=interfaces,
        )
    elif active_ib_ports > 0 or active_interconnect_ifaces > 0:
        return TopologyInfo(
            topology_type="SINGLE",
            description="Interconnect present but partial links",
            ports=ports,
            interfaces=interfaces,
        )
    else:
        return TopologyInfo(
            topology_type="SINGLE",
            description="No interconnect (ethernet only)",
            ports=ports,
            interfaces=interfaces,
        )


async def _fetch_telemetry(ssh_target: str) -> dict:
    """Batch all hardware telemetry into one SSH call with delimited output.

    Returns dict with keys:
      gpu_output    — raw nvidia-smi stdout (parsed by _parse_nvidia_smi)
      cpu_stat      — raw /proc/stat lines (parsed by _parse_cpu_cores)
      cpu_temp      — thermal zone temperatures in millidegrees
      thrash_output — 4-line output from memory counters
    Each key is omitted if that section failed.
    """
    cmd = (
        # SECTION: GPU stats
        'echo "---GPU---" ; '
        f"{NVIDIA_SMI_CMD} ; "
        # SECTION: CPU temperature
        'echo "---CPU_TEMP---" ; '
        "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null ; "
        # SECTION: CPU /proc/stat
        'echo "---CPU_STAT---" ; '
        "grep '^cpu' /proc/stat ; "
        # SECTION: Memory thrash counters
        'echo "---THRASH---" ; '
        "awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} /SwapCached/{sc=$2} "
        "/MemAvailable/{a=$2} /MemFree/{mf=$2} /MemTotal/{mt=$2} "
        "END{print t,f,sc,a,mf,mt}' /proc/meminfo ; "
        "awk '/^pswpin/{a=$2} /^pswpout/{b=$2} /^pgmajfault/{c=$2} "
        "/^allocstall_dma/{d1=$2} /^allocstall_normal/{d2=$2} "
        "/^allocstall_movable/{d3=$2} /^allocstall_device/{d4=$2} "
        "/^pgscan_kswapd/{e=$2} /^pgsteal_kswapd/{f2=$2} "
        "/^workingset_refault_anon/{g1=$2} /^workingset_refault_file/{g2=$2} "
        "END{print a,b,c,d1,d2,d3,d4,e,f2,g1,g2}' /proc/vmstat ; "
        "sed -n 's/^some avg10=\\([0-9.]*\\)[^t]* total=\\([0-9]*\\).*/\\1 \\2/p' "
        "/proc/pressure/memory 2>/dev/null || true ; "
        "sed -n 's/^full avg10=\\([0-9.]*\\)[^t]* total=\\([0-9]*\\).*/\\1 \\2/p' "
        "/proc/pressure/memory 2>/dev/null || true ; "
        # SECTION: Topology (RoCE/InfiniBand state from sysfs)
        'echo "---TOPOLOGY---" ; '
        "for d in /sys/class/infiniband/*; do "
        '[ -e "$d" ] || continue; '
        "b=${d##*/}; "
        'for p in "$d"/ports/*; do '
        '[ -e "$p/state" ] || continue; '
        "port=${p##*/}; "
        'echo "ib:$b:$port:$(cat "$p/state" 2>/dev/null | head -1):$(cat "$p/link_layer" 2>/dev/null)"; '
        "done; "
        'echo "guid:$b:$(cat "$d/node_guid" 2>/dev/null)"; '
        "done; "
        "for _if in /sys/class/net/*; do "
        "name=${_if##*/}; "
        '[ "$name" = "lo" ] && continue; '
        'c=$(cat "$_if/carrier" 2>/dev/null || echo 0); '
        'addr=$(cat "$_if/address" 2>/dev/null || echo ""); '
        'drv=$(cat "$_if/device/uevent" 2>/dev/null | grep ^DRIVER= | cut -d= -f2 || echo "unknown"); '
        'echo "net:$name:$c:$addr:$drv"; '
        "done; "
        'ip -br addr 2>/dev/null | grep -v "^lo "'
    )

    try:
        stdout = await _ssh_run(ssh_target, cmd)
    except Exception:
        return {}

    return _parse_telemetry_output(stdout)


async def poll_unit(unit_id: int) -> SparkUnitStats:
    """Poll a single Spark unit for all stats."""
    cfg = SPARK_UNITS[unit_id]
    ssh_target = str(cfg["ssh_target"])
    vllm_url = str(cfg["vllm_url"])
    is_worker = cfg.get("worker", False)
    label = str(cfg["label"])

    s = SparkUnitStats(label=label, is_worker=is_worker)
    errors: list[str] = []

    # Fetch HTTP and SSH telemetry concurrently. All nodes may serve a model.
    metrics_task = asyncio.create_task(_fetch_vllm_metrics(vllm_url))
    telemetry_task = asyncio.create_task(_fetch_telemetry(ssh_target))

    try:
        metrics_text = await metrics_task
        parsed = _parse_vllm_metrics(metrics_text)
        parsed.label = label
        parsed.is_worker = is_worker
        s = parsed
        s.model_name = _model_names.get(unit_id, "")
    except Exception as e:
        errors.append(f"vLLM: {e}")

    # Hardware telemetry started alongside the vLLM request above.
    telemetry = await telemetry_task

    # Parse GPU stats
    if "gpu_output" in telemetry:
        try:
            gpu_util, mem_util, mem_pct, power, temp = _parse_nvidia_smi(telemetry["gpu_output"])
            s.gpu_util_pct = gpu_util
            s.mem_util_pct = mem_util
            s.power_w = power
            s.temp_c = temp
            s.gpu_mem_pct = mem_pct
        except Exception as e:
            errors.append(f"GPU: {e}")
    else:
        errors.append("GPU: no data")

    # Parse CPU per-core utilization
    if "cpu_stat" in telemetry:
        try:
            cores_total, cores_util = _parse_cpu_stat(ssh_target, telemetry["cpu_stat"])
            s.cpu_cores_total = cores_total
            s.cpu_cores_util = cores_util
        except Exception as e:
            errors.append(f"CPU: {e}")
    else:
        errors.append("CPU: no data")

    # Parse CPU temperature
    if "cpu_temp" in telemetry:
        try:
            temps = [
                float(v) / 1000.0 for v in telemetry["cpu_temp"].strip().splitlines() if v.strip()
            ]
            s.cpu_temp_c = max(temps) if temps else 0.0
        except Exception as e:
            errors.append(f"temp: {e}")
    else:
        errors.append("temp: no data")

    # Parse memory thrash counters
    if "thrash_output" in telemetry:
        try:
            thrash_data = _parse_memory_thrash_output(telemetry["thrash_output"])
            if thrash_data:
                now = time.monotonic()
                rates = _update_thrash_rates(ssh_target, thrash_data, now)
                for k, v in rates.items():
                    setattr(s, k, v)
                mem_total_kb = thrash_data.get("mem_total_kb", 0)
                if mem_total_kb > 0:
                    s.mem_total_bytes = mem_total_kb * 1024
                mem_avail_kb = thrash_data.get("mem_avail_kb", 0)
                if mem_avail_kb > 0:
                    s.mem_used_bytes = (mem_total_kb - mem_avail_kb) * 1024
        except Exception:
            pass  # thrash fields stay at defaults

    # Parse topology data
    if "topology_output" in telemetry:
        try:
            topo_data = _parse_topology_output(telemetry["topology_output"])
            for ibp in topo_data.get("ib_ports", []):
                s.topology_ports.append(
                    TopologyPort(
                        device=ibp.get("device", ""),
                        port=ibp.get("port", ""),
                        state=ibp.get("state", ""),
                        link_layer=ibp.get("link_layer", ""),
                    )
                )
            for iface in topo_data.get("net_interfaces", []):
                s.topology_interfaces.append(
                    TopologyInterface(
                        name=iface.get("name", ""),
                        carrier=iface.get("carrier", "0"),
                        mac=iface.get("mac", ""),
                        driver=iface.get("driver", ""),
                    )
                )
        except Exception:
            pass

    # Node is online if any hardware stats (GPU/CPU) were successfully fetched.
    # vLLM-only failures don't mark the node offline — it may just not be serving.
    if s.gpu_util_pct > 0 or s.power_w > 0 or s.cpu_cores_util or s.cpu_temp_c > 0:
        s.online = True
    else:
        s.online = False
        s.error = "; ".join(errors)

    return s


_prev_tokens: Dict[
    int, Tuple[float, float]
] = {}  # unit_id -> (timestamp, cumulative_generation_tokens)
_prev_prompt_tokens: Dict[
    int, Tuple[float, float]
] = {}  # unit_id -> (timestamp, cumulative_prompt_tokens)


def _update_throughput(unit_id: int, s: SparkUnitStats, now: float) -> None:
    if not s.model_hosted or s.generation_tokens_total <= 0:
        _prev_tokens.pop(unit_id, None)
        return

    prev = _prev_tokens.get(unit_id)
    if prev is not None:
        prev_time, prev_tokens = prev
        dt = now - prev_time
        token_delta = s.generation_tokens_total - prev_tokens
        if dt > 0 and token_delta >= 0:
            s.throughput_tok_s = token_delta / dt
    _prev_tokens[unit_id] = (now, s.generation_tokens_total)


def _update_prompt_throughput(unit_id: int, s: SparkUnitStats, now: float) -> None:
    """Track prompt token throughput rate (parallel to generation throughput)."""
    if not s.model_hosted or s.prompt_tokens_total <= 0:
        _prev_prompt_tokens.pop(unit_id, None)
        return

    prev = _prev_prompt_tokens.get(unit_id)
    if prev is not None:
        prev_time, prev_tokens = prev
        dt = now - prev_time
        token_delta = s.prompt_tokens_total - prev_tokens
        if dt > 0 and token_delta >= 0:
            s.prompt_throughput_tok_s = token_delta / dt
    _prev_prompt_tokens[unit_id] = (now, s.prompt_tokens_total)


_prev_thrash: Dict[str, tuple] = {}  # host -> (timestamp, pswpin, pswpout, pgmajfault,
#                                       pgscan_kswapd, pgsteal_kswapd,
#                                       workingset_refault_anon, workingset_refault_file,
#                                       allocstall_dma, allocstall_normal, allocstall_movable,
#                                       allocstall_device, psi_full_total)


def _update_thrash_rates(host: str, data: dict, now: float) -> dict:
    """Compute rate fields from raw thrash fetch data.

    Takes raw fetch dict from _fetch_memory_thrash, returns dict with rate
    fields computed and static fields passed through. Keys match SparkUnitStats
    field names exactly for setattr usage.

    If no prev data for host: all rates = 0.0, store current values.
    """
    result = {
        "swap_total_kb": data.get("swap_total_kb", 0),
        "swap_used_kb": data.get("swap_used_kb", 0),
        "swap_cached_kb": data.get("swap_cached_kb", 0),
        "mem_avail_kb": data.get("mem_avail_kb", 0),
        "mem_free_kb": data.get("mem_free_kb", 0),
        "mem_total_kb": data.get("mem_total_kb", 0),
        "swap_in_rate": 0.0,
        "swap_out_rate": 0.0,
        "majflt_rate": 0.0,
        "psi_some_avg10": data.get("psi_some_avg10", 0.0),
        "psi_full_avg10": data.get("psi_full_avg10", 0.0),
        "psi_full_total_delta": 0,
        "allocstall_total": 0,
        "allocstall_this_poll": 0,
        "kswapd_scan_rate": 0.0,
        "kswapd_steal_rate": 0.0,
        "workingset_refault_rate": 0.0,
    }

    # Compute allocstall_total from per-zone counters
    allocstall_total = (
        data.get("allocstall_dma", 0)
        + data.get("allocstall_normal", 0)
        + data.get("allocstall_movable", 0)
        + data.get("allocstall_device", 0)
    )
    result["allocstall_total"] = allocstall_total

    prev = _prev_thrash.get(host)
    if prev is not None:
        (
            prev_time,
            prev_pswpin,
            prev_pswpout,
            prev_pgmajfault,
            prev_pgscan_kswapd,
            prev_pgsteal_kswapd,
            prev_refault_anon,
            prev_refault_file,
            prev_alloc_dma,
            prev_alloc_normal,
            prev_alloc_movable,
            prev_alloc_device,
            prev_psi_full_total,
        ) = prev

        dt = now - prev_time
        if dt > 0:

            def _rate(current: int, previous: int) -> float:
                d = current - previous
                return max(0.0, d / dt) if d >= 0 else 0.0

            result["swap_in_rate"] = _rate(data.get("pswpin", 0), prev_pswpin)
            result["swap_out_rate"] = _rate(data.get("pswpout", 0), prev_pswpout)
            result["majflt_rate"] = _rate(data.get("pgmajfault", 0), prev_pgmajfault)
            result["kswapd_scan_rate"] = _rate(data.get("pgscan_kswapd", 0), prev_pgscan_kswapd)
            result["kswapd_steal_rate"] = _rate(data.get("pgsteal_kswapd", 0), prev_pgsteal_kswapd)

            # workingset_refault rate = anon + file combined
            cur_refault = data.get("workingset_refault_anon", 0) + data.get(
                "workingset_refault_file", 0
            )
            prev_refault = prev_refault_anon + prev_refault_file
            result["workingset_refault_rate"] = _rate(cur_refault, prev_refault)

            # allocstall_this_poll = delta sum of all allocstall counters
            cur_alloc_dma = data.get("allocstall_dma", 0)
            cur_alloc_normal = data.get("allocstall_normal", 0)
            cur_alloc_movable = data.get("allocstall_movable", 0)
            cur_alloc_device = data.get("allocstall_device", 0)
            alloc_delta = (
                (cur_alloc_dma - prev_alloc_dma)
                + (cur_alloc_normal - prev_alloc_normal)
                + (cur_alloc_movable - prev_alloc_movable)
                + (cur_alloc_device - prev_alloc_device)
            )
            result["allocstall_this_poll"] = max(0, alloc_delta)

            # PSI full total delta
            cur_psi_full_total = data.get("psi_full_total", 0)
            psi_delta = cur_psi_full_total - prev_psi_full_total
            result["psi_full_total_delta"] = max(0, psi_delta)

    # Store current values for next delta
    _prev_thrash[host] = (
        now,
        data.get("pswpin", 0),
        data.get("pswpout", 0),
        data.get("pgmajfault", 0),
        data.get("pgscan_kswapd", 0),
        data.get("pgsteal_kswapd", 0),
        data.get("workingset_refault_anon", 0),
        data.get("workingset_refault_file", 0),
        data.get("allocstall_dma", 0),
        data.get("allocstall_normal", 0),
        data.get("allocstall_movable", 0),
        data.get("allocstall_device", 0),
        data.get("psi_full_total", 0),
    )

    return result


async def poll_cluster() -> ClusterStats:
    """Poll N Spark units in parallel."""
    unit_ids = sorted(SPARK_UNITS.keys())
    tasks = [poll_unit(uid) for uid in unit_ids]
    results = await asyncio.gather(*tasks)

    now = time.monotonic()
    for unit_id, s in zip(unit_ids, results):
        _update_throughput(unit_id, s, now)
        _update_prompt_throughput(unit_id, s, now)
        # Compute prompt:generated ratio (Spark Monitor insight)
        if s.prompt_throughput_tok_s > 0 and s.throughput_tok_s > 0:
            s.prompt_gen_ratio = s.prompt_throughput_tok_s / s.throughput_tok_s

    # Derive cluster topology from per-node topology data
    node_topos: dict[int, dict] = {}
    for unit_id, s in zip(unit_ids, results):
        topos: dict = {"ib_ports": [], "net_interfaces": []}
        for p in s.topology_ports:
            topos["ib_ports"].append(
                {
                    "device": p.device,
                    "port": p.port,
                    "state": p.state,
                    "link_layer": p.link_layer,
                }
            )
        for iface in s.topology_interfaces:
            topos["net_interfaces"].append(
                {
                    "name": iface.name,
                    "carrier": iface.carrier,
                    "mac": iface.mac,
                    "driver": iface.driver,
                }
            )
        node_topos[unit_id] = topos

    topology = _derive_topology(node_topos)

    return ClusterStats(units=results, topology=topology)
