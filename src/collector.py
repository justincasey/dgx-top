from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

import config
from config import NVIDIA_SMI_CMD, SPARK_UNITS
from simulation import simulate_cluster
from stats import (
    ClusterStats,
    SparkUnitStats,
    TopologyInfo,
    TopologyInterface,
    TopologyPort,
)

# Cache model names fetched once at startup — they don't change between polls.
_model_names: dict[int, str] = {}


def _short_model_name(raw: str) -> str:
    """Human display name for a served-model id.

    sglang serves the raw HF cache layout (``…/models--nvidia--NAME/snapshots/
    <sha>``) as the model id; vLLM may serve ``org/model`` paths. Chart labels
    must fit a legend column, so collapse to the bare model name.
    """
    if not raw:
        return raw
    s = raw.rstrip("/")
    if "models--" in s:
        segs = s.split("models--")[-1].replace("--", "/").split("/")
        if "snapshots" in segs:
            i = segs.index("snapshots")
            return segs[i - 1] if i > 0 else segs[-1]
        return segs[-1]
    return s.split("/")[-1]


async def _init_model_names() -> None:
    """Fetch model names from all Spark units once at startup."""
    if config.SIMULATION_NODES:
        return

    async def fetch_one(client: httpx.AsyncClient, uid: int, vllm_url: str) -> None:
        try:
            resp = await client.get(f"{vllm_url}/v1/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                _model_names[uid] = _short_model_name(
                    models[0].get("id") or models[0].get("model", "")
                )
        except Exception:
            pass  # model name is optional

    async with httpx.AsyncClient(timeout=10) as client:
        await asyncio.gather(
            *(fetch_one(client, uid, str(cfg["vllm_url"])) for uid, cfg in SPARK_UNITS.items())
        )


def _parse_prometheus_histogram(
    lines: List[str], metric_name: str
) -> Tuple[Dict[float, float], float]:
    """Parse a Prometheus histogram and return (buckets, count).

    Buckets and counts accumulate across label sets so an endpoint exposing
    several engines reports the summed histogram, never the last engine's."""
    buckets: Dict[float, float] = {}
    count = 0.0
    for line in lines:
        if line.startswith(f"{metric_name}_bucket{{"):
            m = re.search(r'le="([^"]+)"', line)
            if m:
                le_str = m.group(1)
                parts = line.split()
                try:
                    cnt = float(parts[-1]) if len(parts) >= 2 else 0.0
                    le = float("inf") if le_str == "+Inf" else float(le_str)
                except ValueError:
                    # Truncated/garbage bucket line: skip it, keep the rest
                    # of the payload — one bad sample must not discard a
                    # healthy node's whole parse for the poll.
                    continue
                if math.isfinite(cnt):
                    buckets[le] = buckets.get(le, 0.0) + cnt
        elif line.startswith(f"{metric_name}_count"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    c = float(parts[-1])
                except ValueError:
                    continue
                if math.isfinite(c):
                    count += c
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


def _metric_series(lines: List[str], *names: str) -> Tuple[List[float], bool]:
    """Return (finite_values, saw_any_sample) for sample lines whose metric name
    is exactly one of ``names`` (any label set).

    Exact name matching is what keeps sibling series out of the sum. A loose
    ``startswith("vllm:generation_tokens")`` also swallows
    ``vllm:generation_tokens_created`` (a Unix-epoch pseudo-counter),
    ``vllm:prompt_tokens_by_source_*`` and ``vllm:num_requests_waiting_by_reason``
    — mixing an epoch timestamp into a token counter poisons the throughput
    delta baseline with a ~1/s wall-clock drift and double-counts totals.
    """
    out: List[float] = []
    saw = False
    for line in lines:
        for name in names:
            if not line.startswith(name):
                continue
            rest = line[len(name) :]
            if rest and rest[0] not in " {":
                continue  # sibling series: *_created, *_by_reason, *_by_source…
            saw = True
            parts = line.split()
            if len(parts) >= 2:
                try:
                    v = float(parts[-1])
                except ValueError:
                    break
                if math.isfinite(v):
                    out.append(v)
            break
    return out, saw


def _engine_label(line: str, key: str = "engine") -> Optional[str]:
    """The ``key="…"`` label of an exposition sample line, if it carries one."""
    m = re.search(rf'\b{key}="([^"]*)"', line)
    return m.group(1) if m else None


def _label_text(line: str) -> Optional[str]:
    """The raw ``{…}`` label set of an exposition sample line, if it has one.

    SGLang identifies a series by its whole label set (``model_name``,
    ``tp_rank``, ``pp_rank``, optional ``dp_rank``) rather than by a single
    ``engine`` label, and its two KV gauges expose identical label sets — so
    pairing on the full label text is exact there. vLLM cannot use it: its
    usage gauge and its ``cache_config_info`` block carry different label
    sets, so vLLM pairs on the ``engine`` label they share.
    """
    m = re.search(r"\{(.*)\}", line)
    return m.group(1) if m else None


def _labeled_series(
    lines: List[str], name: str, key: Optional[str] = "engine"
) -> List[Tuple[Optional[str], float]]:
    """``_metric_series`` with each sample's series key attached — for
    aggregations that must pair samples across metric families BY SERIES,
    not by line order (two families' label orderings are not guaranteed to
    correspond). ``key`` selects the pairing identity: a label name, or
    ``None`` for the full label set (see ``_label_text``)."""
    out: List[Tuple[Optional[str], float]] = []
    for line in lines:
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest and rest[0] not in " {":
            continue  # sibling series: *_created, *_by_reason, *_by_source…
        parts = line.split()
        if len(parts) >= 2:
            try:
                v = float(parts[-1])
            except ValueError:
                continue
            if math.isfinite(v):
                label = _label_text(line) if key is None else _engine_label(line, key)
                out.append((label, v))
    return out


def _ns_sample_count(lines: List[str], ns: str) -> int:
    """Count sample lines under namespace prefix ``ns`` that carry a value.

    Sample-evidence counting, not substring matching: ``# HELP vllm:...`` /
    ``# TYPE ...`` comment lines start with ``#`` and never count, arbitrary
    body text mentioning the namespace mid-line never counts, and a
    bare/truncated name line with no value token never counts. (Unlike
    ``_metric_series`` there is no sibling exclusion here — at namespace
    granularity every ``ns:*`` sample line IS evidence.)
    """
    return sum(1 for line in lines if line.startswith(ns) and len(line.split()) >= 2)


# Engine metric profile — which metric name carries each semantic, per engine
# family. SGLang shares only the token counters and the TTFT histogram with
# vLLM: its concurrency, KV and prefix-cache series have different NAMES and
# different TYPES (gauges, where vLLM has a config block and cumulative
# counters). Nothing is ever derived by prefixing a vLLM name with another
# namespace — a semantic absent from a profile has no candidates, so an
# engine-specific fallback can never fire on the other engine.
ENGINE_PROFILES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "vllm": {
        "generation_tokens": ("generation_tokens_total", "generation_tokens"),
        "request_generation_tokens": ("request_generation_tokens_sum",),
        "prompt_tokens": ("prompt_tokens_total", "prompt_tokens"),
        "ttft": ("time_to_first_token_seconds",),
        "itl": ("time_per_output_token_seconds",),
        "kv_usage": ("kv_cache_usage_perc",),
        "cache_config": ("cache_config_info",),
        "running": ("num_requests_running",),
        "waiting": ("num_requests_waiting",),
        "prefix_hits": ("prefix_cache_hits_total",),
        "prefix_queries": ("prefix_cache_queries_total",),
    },
    "sglang": {
        "generation_tokens": ("generation_tokens_total",),
        "prompt_tokens": ("prompt_tokens_total",),
        "ttft": ("time_to_first_token_seconds",),
        # v0.4.5 renamed time_per_output_token_seconds ->
        # inter_token_latency_seconds; accept both so either generation reads.
        "itl": ("inter_token_latency_seconds", "time_per_output_token_seconds"),
        # Gauges (0-1 fractions), not vLLM's block percentage / config block.
        "kv_usage": ("token_usage",),
        "kv_capacity": ("max_total_num_tokens",),
        "kv_used": ("num_used_tokens",),
        "running": ("num_running_reqs",),
        "waiting": ("num_queue_reqs",),
        # A gauge (0-1); SGLang exposes no prefix-cache counters at all.
        "prefix_hit_rate": ("cache_hit_rate",),
    },
}

DEFAULT_ENGINE = "vllm"
"""Engine assumed when neither namespace carries sample evidence."""


def metrics_engine(text: str) -> Optional[str]:
    """The engine family a ``/metrics`` payload reports, or None when it
    carries no sample line from either namespace.

    Sample evidence, not substring matching: majority of actual sample lines,
    tie -> vLLM, so one stray sample from the other family can never flip the
    parse and zero out every real series. This is what the *payload* says it
    is — an endpoint's own metrics are authoritative about which engine
    serves it.
    """
    lines = text.splitlines()
    vllm_n = _ns_sample_count(lines, "vllm:")
    sglang_n = _ns_sample_count(lines, "sglang:")
    if sglang_n > vllm_n:
        return "sglang"
    if vllm_n:
        return DEFAULT_ENGINE
    return None


def detect_engine(text: str) -> str:
    """``metrics_engine`` with the neutral default applied for callers that
    need an engine id regardless (metric names must be looked up somewhere)."""
    return metrics_engine(text) or DEFAULT_ENGINE


def _metric_names(engine: str, semantic: str) -> Tuple[str, ...]:
    """Namespace-qualified candidate names for one semantic on one engine."""
    profile = ENGINE_PROFILES.get(engine, ENGINE_PROFILES[DEFAULT_ENGINE])
    return tuple(f"{engine}:{name}" for name in profile.get(semantic, ()))


def _series_ladder(lines: List[str], *names: str) -> Tuple[List[float], bool]:
    """Modern metric name first; a legacy name is tried only when the modern
    one is entirely ABSENT from the payload.

    The distinction matters: a live-but-non-finite modern sample must not
    silently rebase a throughput baseline onto a different source, which
    would paint a false spike when the counter recovers."""
    if not names:
        return [], False
    vals, saw = _metric_series(lines, names[0])
    if vals or saw or len(names) == 1:
        return vals, saw
    return _metric_series(lines, *names[1:])


def _capacity_weighted_mean(pairs: List[Tuple[float, float]]) -> float:
    """Mean of per-series usage fractions weighted by each series' capacity.

    Never one engine's usage over another engine's capacity: if no pair
    carries a usable capacity, falls back to the plain mean."""
    cap_sum = sum(c for _v, c in pairs)
    if cap_sum > 0:
        return sum(v * c for v, c in pairs) / cap_sum
    return sum(v for v, _c in pairs) / len(pairs)


def _parse_engine_metrics(text: str) -> SparkUnitStats:
    """Parse engine Prometheus metrics into SparkUnitStats.

    The engine family is chosen by sample evidence (``detect_engine``) and
    every series is then looked up through that engine's profile
    (``ENGINE_PROFILES``) — the two families are NOT the same shape. Series
    are selected by exact metric name (see ``_metric_series``) and summed
    across label sets, so an endpoint exposing several engines reports one
    honest endpoint-level aggregate instead of whichever engine printed last.

    vLLM KV cache data available from /metrics:
      - vllm:kv_cache_usage_perc (Gauge): block-level allocation fraction (0-1).
        Usage = 1.0 - (free_blocks / (total_gpu_blocks - 1)). The null block is
        subtracted from total because vLLM reserves 1 block as sentinel. This
        is genuinely accurate at the block-granular level: a partially filled
        block counts as fully allocated, which is the correct framing for
        capacity planning (the scheduler cannot use partial blocks).

      - vllm:cache_config_info{..., num_gpu_blocks="N", block_size="BS",
        kv_cache_size_tokens="T", ...} Static config set at startup. Prefer the
        authoritative kv_cache_size_tokens for capacity (MLA packs several
        tokens per block, so num_gpu_blocks × block_size undercounts); fall back
        to the block product when the field is absent or "None".

      - vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total
        (cumulative Counters): hit rate is derived per poll window in
        _update_prefix_hit_rate (no *_hit_rate gauge exists on modern vLLM).

    SGLang exposes the same semantics as GAUGES instead:
      - sglang:token_usage (0-1) with sglang:max_total_num_tokens as capacity
        and sglang:num_used_tokens as the used count; there is no block
        concept, so kv_total_blocks stays 0 and capacity lives in
        kv_total_tokens.
      - sglang:cache_hit_rate (0-1) is a rate, not a counter pair, so it
        feeds kv_prefix_hit_rate directly and bypasses the per-poll delta
        machinery in _update_prefix_hit_rate (which is counter-only and
        early-returns with no queries).
      - SGLang's mutable scheduler metrics are written only by the stats-logging
        rank (attn_tp_rank == 0) unless --enable-metrics-for-all-schedulers
        is set, so summing across their label sets is the correct endpoint
        total; the non-default all-schedulers flag would over-count
        replicated per-rank series and is a documented limitation. Its
        constant gauges are the exception: max_total_num_tokens is published
        by EVERY rank (Scheduler.emit_metrics_constants runs in each rank's
        __init__, gated only on --enable-metrics) and every rank of a replica
        reports the same shared pool, so capacity is collapsed per replica
        (test_sglang_capacity_collapses_ranks_of_one_replica).

    Derived token counts: total_tokens (above) and used_tokens =
    total_tokens * usage_fraction. These are *block-allocated token capacity*,
    not actual stored tokens; the correct "how much of my pool is consumed"
    framing for capacity planning.
    """
    s = SparkUnitStats()
    lines = text.strip().splitlines()
    engine = detect_engine(text)
    profile = ENGINE_PROFILES.get(engine, ENGINE_PROFILES[DEFAULT_ENGINE])
    ns = f"{engine}:"
    s.model_source = engine
    # Which label identifies a series for cross-family pairing: vLLM's two KV
    # families share only engine="…"; SGLang's share their whole label set.
    cap_key: Optional[str] = "engine" if "cache_config" in profile else None
    # Cache config — block_size and total_blocks from cache_config_info at
    # startup. Pool capacity ACCUMULATES across engines (an endpoint exposing
    # several engines owns the summed pool, consistent with the summation
    # policy used for every other metric here).
    size_tokens = 0
    engine_caps: List[float] = []
    caps_by_engine: Dict[str, float] = {}
    for line in lines:
        if "cache_config" in profile and f"{ns}cache_config_info{{" in line:
            m = re.search(r'num_gpu_blocks="(\d+)"', line)
            if m:
                blocks = _bounded(int(m.group(1)))
                s.kv_total_blocks += blocks
                engine_caps.append(float(blocks))
                cap_lbl = _engine_label(line)
                if cap_lbl is not None:
                    caps_by_engine[cap_lbl] = float(blocks)
            m = re.search(r'block_size="(\d+)"', line)
            if m:
                s.kv_block_size = max(s.kv_block_size, _bounded(int(m.group(1))))
            m = re.search(r'kv_cache_size_tokens="(\d+)"', line)
            if m:
                size_tokens += _bounded(int(m.group(1)))
    # Remote text can repeat huge literals; keep every downstream integer
    # float-safe (an unbounded product raises OverflowError mid-parse).
    s.kv_total_blocks = _bounded(s.kv_total_blocks)
    size_tokens = _bounded(size_tokens)
    if size_tokens > 0:
        s.kv_total_tokens = size_tokens
    elif "cache_config" in profile and s.kv_total_blocks > 0 and s.kv_block_size > 0:
        s.kv_total_tokens = s.kv_total_blocks * s.kv_block_size
    elif "kv_capacity" in profile:
        # SGLang has no cache_config_info block: capacity is a per-scheduler
        # gauge. There is no block concept either, so kv_total_blocks stays 0
        # and the token count is the whole pool. Unlike the mutable gauges this
        # one is published by EVERY rank (`Scheduler.emit_metrics_constants`
        # runs in each rank's __init__ and is gated only on --enable-metrics),
        # and every rank of a replica reports its one shared pool — so the
        # label sets of a replica are collapsed to a single capacity before
        # summing the replicas.
        caps_by_replica: Dict[str, float] = {}
        for cap_lbl, cap in _labeled_series(
            lines, *_metric_names(engine, "kv_capacity"), key=cap_key
        ):
            engine_caps.append(cap)
            if cap_lbl is not None:
                caps_by_engine[cap_lbl] = cap
            # Series values are already finite-filtered by _labeled_series.
            key = _replica_key(cap_lbl)
            caps_by_replica[key] = max(caps_by_replica.get(key, 0.0), cap)
        s.kv_total_tokens = int(_finite_sum(caps_by_replica.values()))

    # KV cache usage — an allocation fraction of the pool (0-1; vLLM's null
    # block is already netted out of its block accounting). The honest
    # endpoint number pairs each series' usage with ITS OWN capacity: by the
    # shared series key when both families carry it (line order across two
    # metric families is not guaranteed to correspond), else by line order
    # when the counts pair, else a plain mean — never one engine's usage over
    # another engine's capacity.
    usage_names = _metric_names(engine, "kv_usage")
    usage = _labeled_series(lines, usage_names[0], key=cap_key) if usage_names else []
    val = 0.0
    have_usage = bool(usage)
    if have_usage:
        if all(lbl is not None and lbl in caps_by_engine for lbl, _v in usage):
            pairs = [(v, caps_by_engine[lbl]) for lbl, v in usage]
        elif len(usage) == len(engine_caps) and sum(engine_caps) > 0:
            pairs = [(v, c) for (_l, v), c in zip(usage, engine_caps)]
        else:
            pairs = [(v, 1.0) for _l, v in usage]
        # The series is an allocation fraction by contract — vLLM's
        # ``kv_cache_usage_perc`` and SGLang's ``token_usage`` are both 0-1 —
        # so every sample is banded BEFORE it is averaged: one hostile value
        # (+1e308 beside -1e308) would otherwise launder through the mean into
        # a plausible-looking 0% pool. A value outside the band is a foreign or
        # mis-scaled series, rejected rather than clamped (the policy
        # ``_coerce_fraction`` applies to the load API); the ×100 happens
        # below, so this is where the band belongs.
        pairs = [(v, c) for v, c in pairs if math.isfinite(v) and 0.0 <= v <= 1.0]
        val = _capacity_weighted_mean(pairs) if pairs else 0.0
        if pairs and math.isfinite(val) and 0.0 <= val <= 1.0:
            s.kv_cache_pct = val * 100
        else:
            # Rejected: the pool's fill is unknown, which is not the same
            # reading as empty — the -1 sentinel is what the UI renders as
            # "no reading" (the prefix hit rate already uses it).
            have_usage = False
            s.kv_cache_pct = -1.0
    else:
        # No usage series at all: unknown, not empty.
        s.kv_cache_pct = -1.0
    if "kv_used" in profile:
        # SGLang states the used token count outright, so take it even when
        # the usage fraction is missing, and fall back to the fraction's
        # product only when the gauge is absent. Like the capacity gauge it
        # pairs with, the used count is a shared-pool figure published by
        # every rank of a replica: collapse the replica's ranks to one
        # reading (max) before summing replicas, or a TP>1 deployment
        # multiplies the pool's fill by its rank count.
        used_pairs = _labeled_series(lines, *_metric_names(engine, "kv_used"), key=cap_key)
        if used_pairs:
            used_by_replica: Dict[str, float] = {}
            for _lbl, u in used_pairs:
                key = _replica_key(_lbl)
                used_by_replica[key] = max(used_by_replica.get(key, 0.0), float(u))
            s.kv_cache_used_tokens = _safe_int(_finite_sum(used_by_replica.values()))
        elif have_usage and s.kv_total_tokens > 0:
            s.kv_cache_used_tokens = _safe_int(s.kv_total_tokens * val)
    elif have_usage and s.kv_total_tokens > 0:
        s.kv_cache_free_blocks = _safe_int(s.kv_total_blocks * (1 - val))
        s.kv_cache_used_tokens = _safe_int(s.kv_total_tokens * val)

    # Prefix cache. vLLM exposes cumulative counters, so the hit rate is
    # derived per poll window in _update_prefix_hit_rate. SGLang exposes a
    # 0-1 rate gauge instead and no counters at all: average the label sets
    # (a rate, so capacity weighting would be meaningless) and set the field
    # directly — _update_prefix_hit_rate early-returns with no queries and
    # leaves this value standing. Counters stay 0 so nothing downstream can
    # mistake the gauge for a cumulative pair.
    if "prefix_hits" in profile:
        hits, _ = _metric_series(lines, *_metric_names(engine, "prefix_hits"))
        s.prefix_hits_total = sum(hits)
        queries, _ = _metric_series(lines, *_metric_names(engine, "prefix_queries"))
        s.prefix_queries_total = sum(queries)
    rate_names = _metric_names(engine, "prefix_hit_rate")
    if rate_names:
        rates, _ = _metric_series(lines, *rate_names)
        if rates:
            raw = sum(rates) / len(rates)
            # A rate gauge is a 0-1 fraction; anything else (NaN, a negative
            # or an oversized sample) is not a hit rate, so leave the "no
            # data" sentinel standing rather than render a nonsense percent.
            if math.isfinite(raw) and 0.0 <= raw <= 1.0:
                s.kv_prefix_hit_rate = raw * 100.0

    # Concurrency — summed across engines; *_by_reason breakdowns excluded.
    running, _ = _metric_series(lines, *_metric_names(engine, "running"))
    s.requests_running = int(sum(running))
    waiting, _ = _metric_series(lines, *_metric_names(engine, "waiting"))
    s.requests_waiting = int(sum(waiting))

    # TTFT histogram
    for ttft_name in _metric_names(engine, "ttft"):
        ttft_lines = [l for l in lines if l.startswith(ttft_name)]
        if not ttft_lines:
            continue
        buckets, count = _parse_prometheus_histogram(ttft_lines, ttft_name)
        s.ttft_p50_ms = _estimate_quantile(buckets, count, 0.50) * 1000
        s.ttft_p95_ms = _estimate_quantile(buckets, count, 0.95) * 1000
        s.ttft_p99_ms = _estimate_quantile(buckets, count, 0.99) * 1000
        break

    # ITL histogram — the first candidate name present wins (SGLang's newer
    # inter_token_latency_seconds before the pre-v0.4.5 name).
    for itl_name in _metric_names(engine, "itl"):
        itl_lines = [l for l in lines if l.startswith(itl_name)]
        if not itl_lines:
            continue
        buckets, count = _parse_prometheus_histogram(itl_lines, itl_name)
        s.itl_p50_ms = _estimate_quantile(buckets, count, 0.50) * 1000
        s.itl_p99_ms = _estimate_quantile(buckets, count, 0.99) * 1000
        break

    # Prefer the live generation token counter. The request histogram fallback
    # only updates when requests finish, so it can lag and spike during long
    # generations. The counter being present at all (even non-finite) is
    # distinct from having parsed a finite value: a transient NaN poll must
    # not silently fall back to the request-sum histogram (which would rebase
    # the throughput baseline onto a different source and paint a false spike
    # on recovery).
    has_gen_tokens = False
    gen, saw_gen = _series_ladder(lines, *_metric_names(engine, "generation_tokens"))
    if gen:
        has_gen_tokens = True
        s.generation_tokens_total = sum(gen)
    elif saw_gen:
        # Live counter present but only non-finite: report 0 so the throughput
        # baseline is dropped and recovers cleanly instead of mixing sources.
        s.generation_tokens_total = 0.0
    else:
        fallback, _ = _metric_series(lines, *_metric_names(engine, "request_generation_tokens"))
        if fallback:
            has_gen_tokens = True
            s.generation_tokens_total = sum(fallback)

    # Same baseline-source stability rule as generation_tokens: the legacy
    # bare name is tried only when the modern counter is entirely absent —
    # a transient NaN on the modern counter must not silently switch the
    # prompt-throughput baseline onto a different source (false spike on
    # recovery).
    ptot, _ = _series_ladder(lines, *_metric_names(engine, "prompt_tokens"))
    s.prompt_tokens_total = sum(ptot)

    if has_gen_tokens or s.kv_total_blocks > 0 or s.requests_running > 0 or s.ttft_p50_ms > 0:
        s.model_hosted = True

    s.online = True
    return s


def _parse_nvidia_smi(output: str) -> Tuple[float, float, float, float, float, float]:
    """Parse nvidia-smi: gpu_util_pct, mem_util_pct, mem_pct, power_w, temp_c, sm_clock_mhz."""
    lines = output.strip().splitlines()
    if not lines:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
        sm_clock = _safe_float(parts[6])
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0.0
        return gpu_util, mem_util, mem_pct, power, temp, sm_clock
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


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


async def _fetch_metrics(engine_url: str) -> str:
    """Fetch engine Prometheus metrics via HTTP."""
    url = f"{engine_url}/metrics"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


_MAX_COUNT = 1 << 62  # 4.6e18 tokens: past any real pool, safe in float math


def _bounded(n: int) -> int:
    """Clamp an integer parsed out of remote text to a float-safe magnitude.

    Huge exact integers survive ``int()`` but every later ``float()`` or
    ``int(int * float)`` on them raises OverflowError, so they are admitted
    only up to a bound no real counter reaches."""
    return n if -_MAX_COUNT <= n <= _MAX_COUNT else 0


def _replica_key(label_text: Optional[str]) -> str:
    """Identity of the KV *pool* a labelled series belongs to.

    SGLang's label set carries the parallel ranks (``tp_rank``/``pp_rank``/
    ``moe_ep_rank``) alongside the replica identity. Ranks of one replica
    inside one tensor-parallel group share a single pool with the same token
    capacity, while each replica (``dp_rank``) owns its own, so pool capacity
    must be summed per replica, not per rank."""
    if not label_text:
        return ""
    pairs = re.findall(r'(\w+)="([^"]*)"', label_text)
    drop = {"tp_rank", "pp_rank", "moe_ep_rank", "attn_tp_rank", "attn_cp_rank", "attn_dp_rank"}
    return "|".join(f"{k}={v}" for k, v in sorted(pairs) if k not in drop)


def _safe_int(v: float) -> int:
    """``int(v)`` that degrades to 0 instead of raising.

    A poisoned gauge can make a product non-finite or too large for a float;
    one bad sample must not discard the node's whole metrics parse."""
    try:
        return int(v) if math.isfinite(v) else 0
    except (OverflowError, ValueError):
        return 0


def _finite_sum(values) -> float:
    """Sum that never yields a non-finite value.

    Two poisoned gauges (``1e308``) add up to ``inf``, and every later
    ``int()`` on that raises — one node's whole metrics parse would be
    discarded. Overflow degrades to 0 instead."""
    total = 0.0
    for v in values:
        if not math.isfinite(v):
            continue
        total += v
        if not math.isfinite(total):
            return 0.0
    return total


def _coerce_count(v: object) -> int:
    """Best-effort int for a remote load field. A malformed value (dict, list,
    garbage string) degrades to 0 for that field — it must never raise out
    of the parse, or one bad payload kills the whole cluster poll tick
    (``poll_cluster``'s gather has no ``return_exceptions``). JSON ``1e999``
    parses to float inf, so OverflowError is part of the contract too.

    Exact big integers are part of that contract: a JSON integer literal with
    hundreds of digits parses to a Python int that ``int()`` accepts happily
    and every later ``float()``/division on it raises ``OverflowError``, so
    anything past any plausible token count degrades to 0 here, at the one
    place remote load numbers enter the process."""
    try:
        n = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return _bounded(n)


def _coerce_fraction(v: object) -> float:
    """Best-effort 0-1 fraction for a remote load gauge; -1.0 when absent.

    Out-of-range values are rejected rather than clamped: the load API's
    gauges are fractions, so 42 is a broken payload, and -1.0 (unknown)
    leaves the honest fallback arithmetic in place instead of rendering
    \"4200%\" KV."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return -1.0
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else -1.0


@dataclass(frozen=True)
class EngineLoad:
    """One endpoint's serving load, aggregated across its DP ranks.

    Filled from SGLang's load API when ``/metrics`` is unavailable (metrics
    disabled), which is the only signal source then. ``kv_pct`` is -1.0 when
    the payload does not state it.

    Only the fields the endpoint populates **whatever the launch flags** live
    here. The same payload also carries ``gen_throughput`` and
    ``cache_hit_rate``, but both are written by the metrics reporter only
    inside ``if self.current_scheduler_metrics_enabled:``
    (``srt/managers/scheduler_components/metrics_reporter.py``, the prefill
    and decode stats paths), so on the metrics-disabled server that is the
    only reason the load API is consulted at all they are constant 0 — a
    schema field, not a reading. Consuming them painted ``0 tok/s`` and
    ``hit 0%`` for a server generating ~60 tok/s.
    """

    running: int = 0
    waiting: int = 0
    used_tokens: int = 0
    total_tokens: int = 0
    kv_pct: float = -1.0


def _load_from_v1(data: object) -> EngineLoad | None:
    """Parse ``/v1/loads`` — the supported SGLang load route.

    The body is an envelope dict whose ``loads`` list holds one flat record
    per DP rank, each carrying ``num_running_reqs``/``num_waiting_reqs`` plus
    the KV gauges (``num_used_tokens``, ``max_total_num_tokens``,
    ``token_usage``). Requests come straight from the scheduler's queues and
    the KV figures from its pool observer, so those survive
    ``--enable-metrics`` being off; ``gen_throughput`` and ``cache_hit_rate``
    do not (see ``EngineLoad``) and are deliberately not read. Returns None
    when the body is not that protocol.
    """
    if not isinstance(data, dict):
        return None
    entries = data.get("loads")
    if not isinstance(entries, list) or not entries:
        return None
    running = waiting = used = total = 0
    pairs: List[Tuple[float, float]] = []
    seen = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not ({"num_running_reqs", "num_waiting_reqs"} & entry.keys()):
            continue
        seen = True
        running += _coerce_count(entry.get("num_running_reqs"))
        waiting += _coerce_count(entry.get("num_waiting_reqs"))
        used += _coerce_count(entry.get("num_used_tokens"))
        cap = _coerce_count(entry.get("max_total_num_tokens"))
        total += cap
        usage = _coerce_fraction(entry.get("token_usage"))
        if usage >= 0:
            pairs.append((usage, float(cap)))
    if not seen:
        return None
    kv_pct = -1.0
    if pairs:
        val = _capacity_weighted_mean(pairs)
        if math.isfinite(val):
            kv_pct = val * 100.0
    elif total > 0:
        # ``used`` sums entries that stated a count while ``total`` only
        # accumulates stated capacities, so a partial payload can push the
        # ratio past the pool's own 0-100 envelope — a reading we cannot
        # trust: the -1 sentinel, like every other percentage here.
        ratio = used / total
        if 0.0 <= ratio <= 1.0:
            kv_pct = ratio * 100.0
    return EngineLoad(
        running=running,
        waiting=waiting,
        used_tokens=used,
        total_tokens=total,
        kv_pct=kv_pct,
    )


def _load_from_get_load(data: object) -> EngineLoad | None:
    """Parse ``/get_load`` — SGLang's deprecated load route, kept as fallback.

    The body is a list of per-DP-rank dicts in which ``num_reqs`` is
    ``num_running_reqs + num_waiting_reqs``, so running requests must be
    recovered BY SUBTRACTION (assigning ``num_reqs`` to running inflates it
    by the waiting count). ``num_tokens`` is NOT the pool size: both shapes
    that serve it report in-flight tokens — used in the pool plus queued —
    so it is netted down by ``num_pending_tokens`` where the route states it
    (the deprecation shim in ``http_server.py`` serves
    ``num_tokens = num_total_tokens`` and
    ``num_pending_tokens = num_total_tokens - num_used_tokens`` on every
    release from v0.5.11 on). Releases older than the shim omit
    ``num_pending_tokens``, and there the value is the in-flight total rather
    than an exact used count. The route never reports capacity, so
    ``total_tokens`` stays 0 and the percentage stays -1.0 (unknown): a
    denominator taken from ``num_tokens`` would render a near-idle server as
    a saturated pool.
    """
    entries = data if isinstance(data, list) else [data]
    running = waiting = used = 0
    seen = False
    for entry in entries:
        if not isinstance(entry, dict) or "num_reqs" not in entry:
            continue
        seen = True
        reqs = _coerce_count(entry.get("num_reqs"))
        wait = _coerce_count(entry.get("num_waiting_reqs"))
        waiting += wait
        running += max(0, reqs - wait)
        if "num_tokens" in entry:
            used += max(
                0,
                _coerce_count(entry.get("num_tokens"))
                - _coerce_count(entry.get("num_pending_tokens")),
            )
    if not seen:
        return None
    return EngineLoad(
        running=running,
        waiting=waiting,
        used_tokens=used,
        total_tokens=0,  # /get_load reports no capacity to divide by
        kv_pct=-1.0,
    )


async def fetch_engine_load(engine_url: str) -> EngineLoad | None:
    """Fetch serving load from an SGLang server's load API.

    ``/v1/loads`` first (the supported route), then ``/get_load`` (deprecated,
    removed in a future SGLang version). Returns None when the endpoint speaks
    neither — a dead port or a non-SGLang server. Every failure mode is
    swallowed: the probe is an optional fallback, never a poll-killing path.
    """
    for path, parse in (("/v1/loads", _load_from_v1), ("/get_load", _load_from_get_load)):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{engine_url}{path}")
                resp.raise_for_status()
                data = resp.json()
            # Inside the try on purpose: the probe must not be able to raise
            # past poll_unit, whose awaited call sits outside its own try.
            load = parse(data)
        except Exception:
            continue
        if load is not None:
            return load
    return None


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
        elif line == "---ROCE---":
            if prev_section is not None:
                result[prev_section] = "\n".join(prev_sections)
            prev_section = "roce_output"
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


# ─── RoCE traffic ───────────────────────────────────────────────────────


def _parse_roce_output(
    output: str,
) -> dict[tuple[str, str], tuple[int, int, float]]:
    """Read per-port RoCE/IB RX, TX and link rate for ACTIVE ports.

    Each line is ``roce:<device>:<port>:<rcv>:<xmit>:<rate>`` where rate is
    the raw `rate` sysfs text like "200 Gb/sec (2X NDR)". InfiniBand data
    counters are in units of 4 octets (PortRcvData/PortXmitData per the IB
    spec), so raw values are multiplied by 4. Port capacity is the rate in
    bytes/second (0 when unreadable). Returns an empty dict when no usable
    ports were reported.
    """
    ports: dict[tuple[str, str], tuple[int, int, float]] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("roce:"):
            continue
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            rcv = int(parts[3])
            xmit = int(parts[4])
        except ValueError:
            continue
        cap_bps = 0.0
        if len(parts) >= 6:
            try:
                cap_bps = float(parts[5].split()[0]) * 1_000_000_000 / 8.0
            except (ValueError, IndexError):
                cap_bps = 0.0
        ports[(parts[1], parts[2])] = (rcv * 4, xmit * 4, cap_bps)
    return ports


_PrevPort = Tuple[float, int, int]  # (monotonic time, cumulative rcv, cumulative xmit)
_prev_roce: Dict[str, Dict[tuple[str, str], _PrevPort]] = {}
# ssh_target -> {(device, port): baseline} across ACTIVE ports


def _update_roce_rates(
    ssh_target: str, ports: dict[tuple[str, str], tuple[int, int, float]], now: float
) -> Tuple[float, float]:
    """Turn cumulative per-port RoCE counters into aggregate (rx_bps, tx_bps).

    Deltas are computed per port so a port entering or leaving the ACTIVE set
    does not fabricate traffic: a first-seen port only seeds its baseline
    (zero rate that interval), an absent port is dropped, and a counter reset
    (reboot/wrap) re-seeds with a zero rate. The baseline is refreshed on every
    poll, so an idle gap does not stretch the next delta over a longer window.
    """
    prev_ports = _prev_roce.get(ssh_target)
    if prev_ports is None:
        prev_ports = {}
        _prev_roce[ssh_target] = prev_ports
    rx_bps = 0.0
    tx_bps = 0.0
    for key, (rcv, xmit, _cap_bps) in ports.items():
        prev = prev_ports.get(key)
        if prev is None:
            prev_ports[key] = (now, rcv, xmit)  # first-seen: seed only, no rate
            continue
        prev_time, prev_rcv, prev_xmit = prev
        if rcv < prev_rcv or xmit < prev_xmit:
            # Counter reset (reboot/wrap): re-seed and emit no rate rather
            # than fabricating a negative throughput.
            prev_ports[key] = (now, rcv, xmit)
            continue
        prev_ports[key] = (now, rcv, xmit)
        dt = now - prev_time
        if dt > 0:
            rx_bps += (rcv - prev_rcv) / dt
            tx_bps += (xmit - prev_xmit) / dt
    # Ports that are no longer reported are forgotten so a later return
    # re-seeds instead of counting their lifetime bytes as one interval.
    for key in [k for k in prev_ports if k not in ports]:
        del prev_ports[key]
    return rx_bps, tx_bps


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
        # SECTION: RoCE/IB traffic counters (PortRcvData/PortXmitData per
        # ACTIVE port, cumulative)
        'echo "---ROCE---" ; '
        "for d in /sys/class/infiniband/*; do "
        '[ -e "$d" ] || continue; '
        "b=${d##*/}; "
        'for p in "$d"/ports/*; do '
        '[ -e "$p/state" ] || continue; '
        "port=${p##*/}; "
        # state holds "4: ACTIVE" (numeric code + name); glob-match the name.
        'case "$(cat "$p/state" 2>/dev/null)" in *ACTIVE*) ;; *) continue ;; esac; '
        'echo "roce:$b:$port:$(cat "$p/counters/port_rcv_data" 2>/dev/null || echo 0):$(cat "$p/counters/port_xmit_data" 2>/dev/null || echo 0):$(cat "$p/rate" 2>/dev/null || echo 0)"; '
        "done; "
        "done; "
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
    declared_engine = cfg.get("engine")

    s = SparkUnitStats(label=label, is_worker=is_worker)
    errors: list[str] = []

    # Fetch HTTP and SSH telemetry concurrently. All nodes may serve a model.
    metrics_task = asyncio.create_task(_fetch_metrics(vllm_url))
    telemetry_task = asyncio.create_task(_fetch_telemetry(ssh_target))

    metrics_error: Exception | None = None
    try:
        metrics_text = await metrics_task
        parsed = _parse_engine_metrics(metrics_text)
        if parsed.model_hosted or metrics_engine(metrics_text) is not None:
            parsed.label = label
            parsed.is_worker = is_worker
            s = parsed
            s.model_name = _model_names.get(unit_id, "")
        else:
            metrics_error = RuntimeError("/metrics served no vllm:/sglang: series")
    except Exception as e:
        metrics_error = e
    if metrics_error is not None:
        # Not a metrics endpoint (or metrics are disabled on it): an SGLang
        # server still reports its state over its load API — concurrency and
        # KV tokens from either route, capacity and percentage only from
        # /v1/loads — so a metrics-disabled SGLang is not reduced to a blank
        # node. A node declared as vLLM is never probed for SGLang's routes.
        if declared_engine == "vllm":
            load = None
        else:
            try:
                load = await fetch_engine_load(vllm_url)
            except Exception:
                # fetch_engine_load is total by contract; the belt stays on so
                # telemetry_task below is always awaited.
                load = None
        if load is not None:
            s.model_hosted = True
            s.model_source = declared_engine or "sglang"
            s.model_metrics = False  # no token counters to rate
            s.requests_running = load.running
            s.requests_waiting = load.waiting
            # Used tokens and capacity are independent: /get_load reports the
            # former and no capacity at all, while /v1/loads reports both.
            if load.used_tokens > 0:
                s.kv_cache_used_tokens = load.used_tokens
            if load.total_tokens > 0:
                s.kv_total_tokens = load.total_tokens
            # The sentinel survives the hop: /get_load states no capacity, so
            # an unknown fill travels as the -1 the UI renders as "no
            # reading" — which is also the dataclass default, never a
            # confident 0%.
            s.kv_cache_pct = load.kv_pct if load.kv_pct >= 0 else -1.0
            # Throughput and prefix reuse are NOT set from the load snapshot:
            # the endpoint states those two fields only when it was started
            # with --enable-metrics (EngineLoad), so a value taken from here
            # would be a manufactured 0 rather than the missing reading it is.
            # They stay at the dataclass sentinels, and the UI paints a dash.
            s.model_name = _model_names.get(unit_id, "")
        else:
            errors.append(f"metrics: {metrics_error}")

    # Hardware telemetry started alongside the metrics request above.
    telemetry = await telemetry_task

    # Parse GPU stats
    if "gpu_output" in telemetry:
        try:
            gpu_util, mem_util, mem_pct, power, temp, sm_clock = _parse_nvidia_smi(
                telemetry["gpu_output"]
            )
            s.gpu_util_pct = gpu_util
            s.mem_util_pct = mem_util
            s.power_w = power
            s.temp_c = temp
            s.gpu_mem_pct = mem_pct
            s.gpu_clock_mhz = sm_clock
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

    # Parse RoCE/IB counter deltas into RX/TX rates
    if "roce_output" in telemetry:
        try:
            ports = _parse_roce_output(telemetry["roce_output"])
            if ports:
                now = time.monotonic()
                rx, tx = _update_roce_rates(ssh_target, ports, now)
                s.roce_rx_bps = rx
                s.roce_tx_bps = tx
                # Full-duplex wire capacity: each port can receive and transmit
                # at its rated speed simultaneously, so 2x the sum of rates.
                s.roce_capacity_bps = 2.0 * sum(cap for _, _, cap in ports.values())
        except Exception:
            pass  # RoCE rates stay at 0

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
            # The ratio can overflow to inf with a huge counter and a tiny
            # window; a non-finite rate would crash the chart renderer.
            rate = token_delta / dt
            if math.isfinite(rate):
                s.throughput_tok_s = rate
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
            rate = token_delta / dt
            if math.isfinite(rate):
                s.prompt_throughput_tok_s = rate
    _prev_prompt_tokens[unit_id] = (now, s.prompt_tokens_total)


_prev_prefix: Dict[int, Tuple[float, float]] = {}  # unit_id -> (queries_total, hits_total)


def _update_prefix_hit_rate(unit_id: int, s: SparkUnitStats) -> None:
    """Derive prefix-cache hit rate (%) from the cumulative counters.

    Windowed (delta hits / delta queries) when this poll saw new queries;
    otherwise the cumulative lifetime ratio, so an idle window still shows a
    number instead of flickering to '—'. Leaves -1 only when the server has
    served no prefix-cache queries at all."""
    if not s.model_hosted or s.prefix_queries_total <= 0:
        _prev_prefix.pop(unit_id, None)
        return
    rate = -1.0
    prev = _prev_prefix.get(unit_id)
    if prev is not None:
        prev_q, prev_h = prev
        dq = s.prefix_queries_total - prev_q
        dh = s.prefix_hits_total - prev_h
        if dq > 0 and 0 <= dh <= dq:
            rate = dh / dq * 100.0
    if rate < 0:
        rate = s.prefix_hits_total / s.prefix_queries_total * 100.0
    if math.isfinite(rate):
        s.kv_prefix_hit_rate = rate
    _prev_prefix[unit_id] = (s.prefix_queries_total, s.prefix_hits_total)


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
    if config.SIMULATION_NODES:
        return simulate_cluster(config.SIMULATION_NODES)
    unit_ids = sorted(SPARK_UNITS.keys())
    tasks = [poll_unit(uid) for uid in unit_ids]
    results = await asyncio.gather(*tasks)

    now = time.monotonic()
    for unit_id, s in zip(unit_ids, results):
        _update_throughput(unit_id, s, now)
        _update_prompt_throughput(unit_id, s, now)
        _update_prefix_hit_rate(unit_id, s)
        # Compute prompt:generated ratio (Spark Monitor insight)
        if s.prompt_throughput_tok_s > 0 and s.throughput_tok_s > 0:
            ratio = s.prompt_throughput_tok_s / s.throughput_tok_s
            # finite/finite can still overflow to inf with extreme rates.
            if math.isfinite(ratio):
                s.prompt_gen_ratio = ratio

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
