import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import collector
from stats import ClusterStats, SparkUnitStats


class VllmMetricsTests(unittest.TestCase):
    def test_generation_tokens_counter_is_parsed(self):
        stats = collector._parse_engine_metrics('vllm:generation_tokens{model_name="a"} 123.0\n')

        self.assertEqual(stats.generation_tokens_total, 123.0)

    def test_generation_tokens_counter_sums_multiple_series(self):
        stats = collector._parse_engine_metrics(
            'vllm:generation_tokens{model_name="a"} 100.0\n'
            'vllm:generation_tokens{model_name="b"} 23.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 123.0)

    def test_request_generation_tokens_sum_is_fallback(self):
        stats = collector._parse_engine_metrics(
            'vllm:request_generation_tokens_sum{model_name="a"} 456.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 456.0)
        self.assertTrue(stats.model_hosted)

    def test_zero_request_generation_tokens_fallback_marks_model_hosted(self):
        stats = collector._parse_engine_metrics(
            'vllm:request_generation_tokens_sum{model_name="a"} 0.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertTrue(stats.model_hosted)

    def test_generation_tokens_counter_is_preferred_over_fallback(self):
        stats = collector._parse_engine_metrics(
            'vllm:generation_tokens{model_name="a"} 123.0\n'
            'vllm:request_generation_tokens_sum{model_name="a"} 456.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 123.0)

    def test_zero_generation_tokens_counter_is_preferred_over_fallback(self):
        stats = collector._parse_engine_metrics(
            'vllm:generation_tokens{model_name="a"} 0.0\n'
            'vllm:request_generation_tokens_sum{model_name="a"} 456.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 0.0)

    def test_model_hosted_set_by_vllm_metrics_parser(self):
        stats = collector._parse_engine_metrics('vllm:generation_tokens{model_name="a"} 123.0\n')

        self.assertTrue(stats.model_hosted)

    def test_model_hosted_false_when_metrics_empty(self):
        stats = collector._parse_engine_metrics("")

        self.assertFalse(stats.model_hosted)

    def test_kv_cache_usage_perc_is_parsed(self):
        stats = collector._parse_engine_metrics('vllm:kv_cache_usage_perc{model_name="a"} 0.45\n')

        self.assertAlmostEqual(stats.kv_cache_pct, 45.0)

    def test_kv_cache_config_parses_blocks_and_block_size(self):
        stats = collector._parse_engine_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="1234"} 1.0\n'
        )

        self.assertEqual(stats.kv_total_blocks, 1234)
        self.assertEqual(stats.kv_block_size, 16)
        self.assertEqual(stats.kv_total_tokens, 1234 * 16)

    def test_kv_cache_derives_block_and_token_counts_from_usage(self):
        stats = collector._parse_engine_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="10000"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} 0.32\n'
        )

        self.assertAlmostEqual(stats.kv_cache_pct, 32.0)
        self.assertEqual(stats.kv_total_blocks, 10000)
        # usage_pct accounts for null block; derive free blocks from total * (1-usage)
        self.assertEqual(stats.kv_cache_free_blocks, int(10000 * (1 - 0.32)))
        self.assertEqual(stats.kv_total_tokens, 10000 * 16)
        self.assertEqual(stats.kv_cache_used_tokens, int(10000 * 16 * 0.32))

    def test_kv_cache_no_token_derivation_without_block_size(self):
        stats = collector._parse_engine_metrics('vllm:kv_cache_usage_perc{model_name="a"} 0.5\n')

        self.assertAlmostEqual(stats.kv_cache_pct, 50.0)
        self.assertEqual(stats.kv_total_tokens, 0)
        self.assertEqual(stats.kv_cache_used_tokens, 0)

    def test_kv_cache_size_tokens_is_authoritative(self):
        # MLA packs several tokens per block: prefer kv_cache_size_tokens over
        # num_gpu_blocks * block_size.
        stats = collector._parse_engine_metrics(
            'vllm:cache_config_info{block_size="4",num_gpu_blocks="16677",'
            'kv_cache_size_tokens="1489151"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} 0.5\n'
        )

        self.assertEqual(stats.kv_total_tokens, 1489151)
        self.assertEqual(stats.kv_cache_used_tokens, int(1489151 * 0.5))

    def test_prefix_cache_counters_are_parsed(self):
        stats = collector._parse_engine_metrics(
            'vllm:prefix_cache_hits_total{model_name="a"} 700.0\n'
            'vllm:prefix_cache_queries_total{model_name="a"} 1000.0\n'
        )

        self.assertEqual(stats.prefix_hits_total, 700.0)
        self.assertEqual(stats.prefix_queries_total, 1000.0)

    def test_prefix_created_series_do_not_pollute_counters(self):
        stats = collector._parse_engine_metrics(
            'vllm:prefix_cache_hits_created{model_name="a"} 1.7e9\n'
            'vllm:external_prefix_cache_hits_total{model_name="a"} 5.0\n'
        )

        self.assertEqual(stats.prefix_hits_total, 0.0)
        self.assertEqual(stats.prefix_queries_total, 0.0)

    def test_prefix_hit_rate_cumulative_then_windowed(self):
        collector._prev_prefix.clear()
        s = SparkUnitStats(model_hosted=True)
        s.prefix_hits_total, s.prefix_queries_total = 700.0, 1000.0
        collector._update_prefix_hit_rate(0, s)
        self.assertAlmostEqual(s.kv_prefix_hit_rate, 70.0)  # first poll: cumulative

        s2 = SparkUnitStats(model_hosted=True)
        s2.prefix_hits_total, s2.prefix_queries_total = 790.0, 1100.0
        collector._update_prefix_hit_rate(0, s2)
        self.assertAlmostEqual(s2.kv_prefix_hit_rate, 90.0)  # windowed: 90/100

        s3 = SparkUnitStats(model_hosted=True)
        s3.prefix_hits_total, s3.prefix_queries_total = 790.0, 1100.0
        collector._update_prefix_hit_rate(0, s3)  # idle window → cumulative fallback
        self.assertAlmostEqual(s3.kv_prefix_hit_rate, 790 / 1100 * 100)

    def test_prefix_hit_rate_survives_counter_reset(self):
        collector._prev_prefix.clear()
        s = SparkUnitStats(model_hosted=True)
        s.prefix_hits_total, s.prefix_queries_total = 700.0, 1000.0
        collector._update_prefix_hit_rate(0, s)
        # vLLM restarted: counters dropped → windowed skipped, cumulative used.
        s2 = SparkUnitStats(model_hosted=True)
        s2.prefix_hits_total, s2.prefix_queries_total = 8.0, 10.0
        collector._update_prefix_hit_rate(0, s2)
        self.assertAlmostEqual(s2.kv_prefix_hit_rate, 80.0)

    def test_prefix_hit_rate_unavailable_without_queries(self):
        collector._prev_prefix.clear()
        s = SparkUnitStats(model_hosted=True)
        collector._update_prefix_hit_rate(0, s)
        self.assertEqual(s.kv_prefix_hit_rate, -1.0)

    def test_ttft_histogram_estimates_p50_and_p95(self):
        """The UI renders a p50—p95 range (§7.5); the tail must separate from
        the typical when the histogram is bimodal."""

        text = (
            'vllm:time_to_first_token_seconds_bucket{model_name="a",le="0.5"} 0\n'
            'vllm:time_to_first_token_seconds_bucket{model_name="a",le="1"} 50\n'
            'vllm:time_to_first_token_seconds_bucket{model_name="a",le="20"} 100\n'
            'vllm:time_to_first_token_seconds_bucket{model_name="a",le="+Inf"} 100\n'
            'vllm:time_to_first_token_seconds_count{model_name="a"} 100\n'
        )

        stats = collector._parse_engine_metrics(text)

        # p50 lands in the fast population (~1s); p95 tracks the slow tail.
        self.assertAlmostEqual(stats.ttft_p50_ms, 1000.0, delta=100.0)
        self.assertAlmostEqual(stats.ttft_p95_ms, 18100.0, delta=200.0)
        self.assertGreater(stats.ttft_p95_ms, stats.ttft_p50_ms)

    def test_model_hosted_true_when_kv_blocks_exist(self):
        stats = collector._parse_engine_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="5000"} 1.0\n'
        )

        self.assertTrue(stats.model_hosted)
        self.assertEqual(stats.kv_total_blocks, 5000)


class NonFiniteVllmMetricsTests(unittest.TestCase):
    """Non-finite vLLM metrics (NaN/Inf) must be dropped, never reach the UI.

    A NaN/Inf in the chart history crashes the Textual Sparkline renderer and
    exits the whole app, so the parser has to keep them out.
    """

    def test_nan_kv_usage_perc_is_ignored(self):
        stats = collector._parse_engine_metrics('vllm:kv_cache_usage_perc{model_name="a"} NaN\n')

        # Dropped, and left as the no-reading sentinel: an unusable sample is
        # not the same reading as an empty pool.
        self.assertEqual(stats.kv_cache_pct, -1.0)

    def test_overflowing_kv_usage_is_ignored(self):
        # 1e308 * 100 overflows to inf even though the raw value is finite.
        stats = collector._parse_engine_metrics('vllm:kv_cache_usage_perc{model_name="a"} 1e308\n')

        self.assertEqual(stats.kv_cache_pct, -1.0)

    def test_nan_kv_usage_does_not_derive_block_counts(self):
        stats = collector._parse_engine_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="10000"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} NaN\n'
        )

        self.assertEqual(stats.kv_cache_pct, -1.0)
        self.assertEqual(stats.kv_cache_free_blocks, 0)
        self.assertEqual(stats.kv_cache_used_tokens, 0)
        self.assertEqual(stats.kv_total_tokens, 10000 * 16)

    def test_out_of_band_kv_usage_does_not_derive_block_counts(self):
        # A finite but out-of-contract sample must not leave the derived
        # figures claiming a known pool: free = total × (1 - usage) would read
        # as "every block free" — a confident empty pool — if the rejected
        # value leaked past the band.
        stats = collector._parse_engine_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="10000"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} 42\n'
        )

        self.assertEqual(stats.kv_cache_pct, -1.0)
        self.assertEqual(stats.kv_cache_free_blocks, 0)
        self.assertEqual(stats.kv_cache_used_tokens, 0)

    def test_inf_generation_tokens_are_ignored(self):
        stats = collector._parse_engine_metrics('vllm:generation_tokens{model_name="a"} +Inf\n')

        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertFalse(stats.model_hosted)

    def test_finite_generation_tokens_survive_a_bad_series(self):
        stats = collector._parse_engine_metrics(
            'vllm:generation_tokens{model_name="a"} NaN\n'
            'vllm:generation_tokens{model_name="b"} 100.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 100.0)
        self.assertTrue(stats.model_hosted)

    def test_inf_requests_running_does_not_break_parse(self):
        stats = collector._parse_engine_metrics(
            'vllm:num_requests_running{model_name="a"} +Inf\n'
            'vllm:generation_tokens{model_name="b"} 5.0\n'
        )

        self.assertEqual(stats.requests_running, 0)
        self.assertTrue(stats.model_hosted)

    def test_inf_requests_waiting_does_not_break_parse(self):
        stats = collector._parse_engine_metrics(
            'vllm:num_requests_waiting{model_name="a"} +Inf\n'
            'vllm:generation_tokens{model_name="b"} 5.0\n'
        )

        self.assertEqual(stats.requests_waiting, 0)
        self.assertTrue(stats.model_hosted)

    def test_inf_request_generation_tokens_sum_is_ignored(self):
        stats = collector._parse_engine_metrics(
            'vllm:request_generation_tokens_sum{model_name="a"} +Inf\n'
        )

        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertFalse(stats.model_hosted)

    def test_nan_prefix_counters_are_ignored(self):
        stats = collector._parse_engine_metrics(
            'vllm:prefix_cache_hits_total{model_name="a"} NaN\n'
            'vllm:prefix_cache_queries_total{model_name="a"} NaN\n'
        )

        self.assertEqual(stats.prefix_hits_total, 0.0)
        self.assertEqual(stats.prefix_queries_total, 0.0)

    def test_nan_prompt_tokens_are_ignored(self):
        stats = collector._parse_engine_metrics('vllm:prompt_tokens{model_name="a"} NaN\n')

        self.assertEqual(stats.prompt_tokens_total, 0.0)


class LabelConflationTests(unittest.TestCase):
    """Sibling Prometheus series (``*_created``, ``*_by_reason``,
    ``*_by_source``) and per-engine label sets must never bleed into the
    parent counter — exact-name matching is the whole defence."""

    def test_created_pseudo_counter_is_not_summed(self):
        # *_created values are Unix epoch timestamps; summing one into the
        # counter poisons the throughput delta with ~1/s wall-clock drift.
        stats = collector._parse_engine_metrics(
            'vllm:generation_tokens_total{engine="0",model_name="a"} 264348.0\n'
            'vllm:generation_tokens_created{engine="0",model_name="a"} 1788733924.32\n'
        )
        self.assertEqual(stats.generation_tokens_total, 264348.0)

    def test_prompt_token_variant_series_are_excluded(self):
        stats = collector._parse_engine_metrics(
            'vllm:prompt_tokens_total{model_name="a"} 3.6e+07\n'
            'vllm:prompt_tokens_created{model_name="a"} 1.7887e+09\n'
            'vllm:prompt_tokens_by_source_total{model_name="a",source="local_compute"} 2.5e+06\n'
            'vllm:prompt_tokens_by_source_total{model_name="a",source="local_cache_hit"} 3.3e+07\n'
            'vllm:prompt_tokens_cached_total{model_name="a"} 3.3e+07\n'
        )
        self.assertEqual(stats.prompt_tokens_total, 3.6e7)

    def test_waiting_by_reason_does_not_override_the_gauge(self):
        # The old prefix match let the last by_reason line overwrite the
        # real waiting count.
        stats = collector._parse_engine_metrics(
            'vllm:num_requests_waiting{model_name="a"} 5.0\n'
            'vllm:num_requests_waiting_by_reason{model_name="a",reason="capacity"} 3.0\n'
            'vllm:num_requests_waiting_by_reason{model_name="a",reason="deferred"} 0.0\n'
        )
        self.assertEqual(stats.requests_waiting, 5)

    def test_multi_engine_label_sets_are_summed(self):
        stats = collector._parse_engine_metrics(
            'vllm:num_requests_running{engine="0",model_name="a"} 2.0\n'
            'vllm:num_requests_running{engine="1",model_name="a"} 3.0\n'
            'vllm:generation_tokens_total{engine="0",model_name="a"} 100.0\n'
            'vllm:generation_tokens_total{engine="1",model_name="a"} 50.0\n'
        )
        self.assertEqual(stats.requests_running, 5)
        self.assertEqual(stats.generation_tokens_total, 150.0)

    def test_multi_engine_histograms_accumulate(self):
        # Engine 0 finishes ≤0.5s, engine 1 ≤1s: the merged p50 sits at the
        # boundary (0.5s); last-engine-wins would report 1.0s.
        stats = collector._parse_engine_metrics(
            'vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5"} 5\n'
            'vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 5\n'
            'vllm:time_to_first_token_seconds_count{engine="0"} 5\n'
            'vllm:time_to_first_token_seconds_bucket{engine="1",le="0.5"} 0\n'
            'vllm:time_to_first_token_seconds_bucket{engine="1",le="1"} 5\n'
            'vllm:time_to_first_token_seconds_bucket{engine="1",le="+Inf"} 5\n'
            'vllm:time_to_first_token_seconds_count{engine="1"} 5\n'
        )
        self.assertAlmostEqual(stats.ttft_p50_ms, 500.0)


class ShortModelNameTests(unittest.TestCase):
    def test_hf_cache_snapshot_path(self):
        self.assertEqual(
            collector._short_model_name(
                "/root/.cache/huggingface/hub/models--nvidia--Nemotron-3.5/snapshots/cc84af2"
            ),
            "Nemotron-3.5",
        )

    def test_org_model_path_takes_the_name(self):
        self.assertEqual(collector._short_model_name("Qwen/Qwen3.6-27B"), "Qwen3.6-27B")

    def test_bare_name_is_unchanged(self):
        self.assertEqual(collector._short_model_name("qwen3.8-flash-next"), "qwen3.8-flash-next")


def _patch_http(routes: dict[str, object]):
    """Patch ``collector.httpx.AsyncClient`` with a client that serves canned
    JSON keyed by request path; a path not in ``routes`` raises, as a real
    404 does. Returns (patcher, requested_urls)."""
    requested: list[str] = []

    class FakeResp:
        def __init__(self, payload: object):
            self._payload = payload

        def raise_for_status(self):
            if isinstance(self._payload, Exception):
                raise self._payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            requested.append(url)
            for path, payload in routes.items():
                if url.endswith(path):
                    return FakeResp(payload)
            return FakeResp(RuntimeError(f"404 Not Found for url '{url}'"))

    return patch.object(collector.httpx, "AsyncClient", FakeClient), requested


class SglangMetricsTests(unittest.TestCase):
    """SGLang's ``--enable-metrics`` exposition is NOT the vLLM shape under a
    different namespace: only the token counters and the TTFT histogram
    match. Every semantic is read through the engine's own profile."""

    LABELS = (
        '{model_name="qwen3.6-27b",engine_type="unified",tp_rank="0",pp_rank="0",moe_ep_rank="0"}'
    )

    def _payload(self) -> str:
        lbl = self.LABELS
        return (
            "# HELP sglang:num_running_reqs The number of running requests.\n"
            "# TYPE sglang:num_running_reqs gauge\n"
            f"sglang:num_running_reqs{lbl} 2.0\n"
            f"sglang:num_queue_reqs{lbl} 1.0\n"
            f"sglang:token_usage{lbl} 0.45\n"
            f"sglang:num_used_tokens{lbl} 450000\n"
            f"sglang:max_total_num_tokens{lbl} 1000000\n"
            f"sglang:cache_hit_rate{lbl} 0.62\n"
            f"sglang:prompt_tokens_total{lbl} 4000\n"
            f"sglang:generation_tokens_total{lbl} 9000\n"
            f'sglang:time_to_first_token_seconds_bucket{lbl[:-1]},le="1.0"}} 10\n'
            f'sglang:time_to_first_token_seconds_bucket{lbl[:-1]},le="+Inf"}} 10\n'
            f"sglang:time_to_first_token_seconds_count{lbl} 10\n"
            f'sglang:inter_token_latency_seconds_bucket{lbl[:-1]},le="0.5"}} 10\n'
            f'sglang:inter_token_latency_seconds_bucket{lbl[:-1]},le="+Inf"}} 10\n'
            f"sglang:inter_token_latency_seconds_count{lbl} 10\n"
        )

    def test_sglang_payload_populates_every_semantic(self):
        s = collector._parse_engine_metrics(self._payload())

        self.assertEqual(s.model_source, "sglang")
        self.assertTrue(s.model_hosted)
        self.assertEqual((s.requests_running, s.requests_waiting), (2, 1))
        self.assertAlmostEqual(s.kv_cache_pct, 45.0)
        self.assertEqual(s.kv_total_tokens, 1_000_000)
        self.assertEqual(s.kv_cache_used_tokens, 450_000)
        self.assertAlmostEqual(s.kv_prefix_hit_rate, 62.0)
        self.assertAlmostEqual(s.itl_p50_ms, 250.0)
        self.assertAlmostEqual(s.ttft_p50_ms, 500.0)
        self.assertEqual(s.generation_tokens_total, 9000.0)
        self.assertEqual(s.prompt_tokens_total, 4000.0)
        # SGLang exposes no block concept at all.
        self.assertEqual(s.kv_total_blocks, 0)
        # A metrics-enabled SGLang poll has token counters to rate: this flag
        # is what tells its row apart from a load-only node's, and nothing
        # else pins it.
        self.assertTrue(s.model_metrics)

    def test_sglang_out_of_range_usage_gauge_is_rejected_not_scaled(self):
        # token_usage is a 0-1 fraction and the ×100 happens in the collector,
        # so a hostile or mis-scaled sample must leave the no-reading sentinel
        # rather than paint a 4200% (or -50%) KV meter and seed the Sparkline
        # history with it.
        payload = self._payload()
        line = f"sglang:token_usage{self.LABELS} 0.45\n"
        for value in ("42", "-0.5", "1.0000001", "NaN", "+Inf", "garbage"):
            s = collector._parse_engine_metrics(
                payload.replace(line, f"sglang:token_usage{self.LABELS} {value}\n")
            )
            self.assertEqual(s.kv_cache_pct, -1.0, value)
            # The used count comes from its own gauge, never from the fraction.
            self.assertEqual(s.kv_cache_used_tokens, 450_000, value)
        for value, expected in (("0.0", 0.0), ("1.0", 100.0)):
            s = collector._parse_engine_metrics(
                payload.replace(line, f"sglang:token_usage{self.LABELS} {value}\n")
            )
            self.assertAlmostEqual(s.kv_cache_pct, expected, msg=value)

    def test_sglang_prefix_rate_is_a_gauge_not_a_counter_pair(self):
        s = collector._parse_engine_metrics(self._payload())

        self.assertEqual(s.prefix_hits_total, 0.0)
        self.assertEqual(s.prefix_queries_total, 0.0)
        # The counter-only delta machinery must leave the gauge value alone.
        collector._prev_prefix.pop(77, None)
        collector._update_prefix_hit_rate(77, s)
        self.assertAlmostEqual(s.kv_prefix_hit_rate, 62.0)

    def test_sglang_out_of_range_prefix_gauge_leaves_the_sentinel(self):
        # cache_hit_rate is a 0-1 fraction. A hostile/never-valid sample must
        # not become "-200%" in the UI.
        for value in ("-2.0", "3.5", "NaN", "+Inf", "garbage"):
            s = collector._parse_engine_metrics(
                f'sglang:cache_hit_rate{{model_name="m"}} {value}\n'
            )
            self.assertEqual(s.kv_prefix_hit_rate, -1.0, value)
        s = collector._parse_engine_metrics('sglang:cache_hit_rate{model_name="m"} 1.0\n')
        self.assertEqual(s.kv_prefix_hit_rate, 100.0)

    def test_sglang_poisoned_gauge_sums_do_not_raise(self):
        # Two 1e308 gauges sum to inf, and int(inf) raises — which would
        # discard the node's whole metrics parse. Overflow degrades to 0.
        s = collector._parse_engine_metrics(
            'sglang:token_usage{a="1"} 0.5\n'
            'sglang:max_total_num_tokens{a="1"} 1e308\n'
            'sglang:max_total_num_tokens{a="2"} 1e308\n'
        )
        self.assertEqual(s.kv_total_tokens, 0)
        self.assertAlmostEqual(s.kv_cache_pct, 50.0)

        s = collector._parse_engine_metrics(
            'sglang:token_usage{a="1"} 0.5\n'
            'sglang:max_total_num_tokens{a="1"} 1000\n'
            'sglang:num_used_tokens{a="1"} 1e308\n'
            'sglang:num_used_tokens{a="2"} 1e308\n'
        )
        self.assertEqual(s.kv_cache_used_tokens, 0)
        self.assertEqual(s.kv_total_tokens, 1000)

    def test_sglang_token_usage_pairs_by_label_not_line_order(self):
        # The two families are listed in opposite rank order: pairing by line
        # order would weight each usage by the other rank's capacity.
        s = collector._parse_engine_metrics(
            'sglang:max_total_num_tokens{model_name="m",tp_rank="0",dp_rank="1"} 3000\n'
            'sglang:max_total_num_tokens{model_name="m",tp_rank="0",dp_rank="0"} 1000\n'
            'sglang:token_usage{model_name="m",tp_rank="0",dp_rank="0"} 0.2\n'
            'sglang:token_usage{model_name="m",tp_rank="0",dp_rank="1"} 0.6\n'
        )

        self.assertEqual(s.kv_total_tokens, 4000)
        self.assertAlmostEqual(s.kv_cache_pct, 50.0)  # (0.2·1000 + 0.6·3000)/4000

    def test_sglang_inter_token_latency_accepts_both_generations_of_name(self):
        modern = collector._parse_engine_metrics(
            'sglang:inter_token_latency_seconds_bucket{le="0.5"} 10\n'
            'sglang:inter_token_latency_seconds_bucket{le="+Inf"} 10\n'
            "sglang:inter_token_latency_seconds_count 10\n"
        )
        legacy = collector._parse_engine_metrics(
            'sglang:time_per_output_token_seconds_bucket{le="0.5"} 10\n'
            'sglang:time_per_output_token_seconds_bucket{le="+Inf"} 10\n'
            "sglang:time_per_output_token_seconds_count 10\n"
        )

        self.assertAlmostEqual(modern.itl_p50_ms, 250.0)
        self.assertAlmostEqual(legacy.itl_p50_ms, 250.0)

        # A transitional SGLang may expose BOTH names. The modern one leads
        # the ladder, so it must win: taking the last match or reversing the
        # order would report the legacy value here (250 vs 750 ms).
        both = collector._parse_engine_metrics(
            'sglang:inter_token_latency_seconds_bucket{le="0.5"} 10\n'
            'sglang:inter_token_latency_seconds_bucket{le="+Inf"} 10\n'
            "sglang:inter_token_latency_seconds_count 10\n"
            'sglang:time_per_output_token_seconds_bucket{le="1.5"} 10\n'
            'sglang:time_per_output_token_seconds_bucket{le="+Inf"} 10\n'
            "sglang:time_per_output_token_seconds_count 10\n"
        )
        self.assertAlmostEqual(both.itl_p50_ms, 250.0)

    def test_sglang_dead_metric_names_are_not_read_by_the_vllm_profile(self):
        # The false premise was "one parser, both namespaces". These are
        # SGLang's names; a vLLM payload must never be read through them, and
        # vice versa (their series would be silently dead, not shared).
        s = collector._parse_engine_metrics(
            "vllm:num_running_reqs 5\n"
            "vllm:num_queue_reqs 3\n"
            "vllm:token_usage 0.5\n"
            "vllm:max_total_num_tokens 100\n"
            "vllm:cache_hit_rate 0.9\n"
            # Positive control: the parse DID read this payload — otherwise
            # the sentinel/zero assertions below would also pass on an empty
            # stats object and would pin nothing.
            "vllm:generation_tokens_total 9000\n"
        )
        self.assertEqual(s.generation_tokens_total, 9000.0)
        self.assertEqual(s.requests_running, 0)
        self.assertEqual(s.requests_waiting, 0)
        # No vLLM usage series in the payload: unknown, so the sentinel (not
        # the dataclass default, which would read as a 0% pool).
        self.assertEqual(s.kv_cache_pct, -1.0)
        self.assertEqual(s.kv_total_tokens, 0)
        self.assertEqual(s.kv_prefix_hit_rate, -1.0)

    def test_sglang_capacity_is_summed_across_dp_ranks(self):
        s = collector._parse_engine_metrics(
            'sglang:max_total_num_tokens{model_name="m",dp_rank="0"} 1000\n'
            'sglang:max_total_num_tokens{model_name="m",dp_rank="1"} 3000\n'
        )
        self.assertEqual(s.kv_total_tokens, 4000)

    def test_sglang_capacity_collapses_ranks_of_one_replica(self):
        # max_total_num_tokens is published by EVERY rank (emit_metrics_constants
        # runs in each rank's __init__), and the ranks of a replica share one
        # pool — summing them would report TP/PP × the real capacity while the
        # stats-rank-only usage gauges stay at 1×.
        s = collector._parse_engine_metrics(
            'sglang:max_total_num_tokens{model_name="m",tp_rank="0",pp_rank="0"} 3200\n'
            'sglang:max_total_num_tokens{model_name="m",tp_rank="1",pp_rank="0"} 3200\n'
            'sglang:num_used_tokens{model_name="m",tp_rank="0",pp_rank="0"} 1600\n'
            'sglang:token_usage{model_name="m",tp_rank="0",pp_rank="0"} 0.5\n'
        )

        self.assertEqual(s.kv_total_tokens, 3200)
        self.assertEqual(s.kv_cache_used_tokens, 1600)
        self.assertAlmostEqual(s.kv_cache_pct, 50.0)

        # …but each DP replica owns its own pool: two replicas × two ranks
        # still sum to two pools, not four.
        s = collector._parse_engine_metrics(
            "".join(
                f'sglang:max_total_num_tokens{{model_name="m",tp_rank="{t}",dp_rank="{d}"}} 3200\n'
                for d in (0, 1)
                for t in (0, 1)
            )
            + "".join(
                f'sglang:num_used_tokens{{model_name="m",tp_rank="0",dp_rank="{d}"}} 1600\n'
                for d in (0, 1)
            )
            + "".join(
                f'sglang:token_usage{{model_name="m",tp_rank="0",dp_rank="{d}"}} 0.5\n'
                for d in (0, 1)
            )
        )

        self.assertEqual(s.kv_total_tokens, 6400)
        self.assertEqual(s.kv_cache_used_tokens, 3200)
        self.assertAlmostEqual(s.kv_cache_pct, 50.0)

    def test_sglang_used_tokens_survive_a_missing_usage_gauge(self):
        # num_used_tokens is the authoritative figure; it must not be gated on
        # token_usage being present in the same exposition.
        s = collector._parse_engine_metrics(
            'sglang:max_total_num_tokens{model_name="m"} 1000\n'
            'sglang:num_used_tokens{model_name="m"} 400\n'
        )

        self.assertEqual(s.kv_cache_used_tokens, 400)

    def test_hostile_vllm_cache_config_degrades_not_raises(self):
        # A 309-digit integer literal survives int() and then raises
        # OverflowError from float(); the parse must stay total (HEAD raised
        # here).
        huge = "1" + "0" * 308
        s = collector._parse_engine_metrics(
            f'vllm:cache_config_info{{engine="0",num_gpu_blocks="{huge}",block_size="16"}} 1\n'
            'vllm:kv_cache_usage_perc{engine="0"} 1e308\n'
            'vllm:kv_cache_usage_perc{engine="1"} -1e308\n'
        )
        self.assertEqual((s.kv_total_blocks, s.kv_total_tokens), (0, 0))
        # Both samples are outside the 0-1 contract: pairwise they would
        # average to a confident 0% pool, so they are rejected sample by
        # sample and the no-reading sentinel stands.
        self.assertEqual(s.kv_cache_pct, -1.0)


class EngineLoadTests(unittest.IsolatedAsyncioTestCase):
    """SGLang's load API is the only signal source when ``--enable-metrics``
    is off, so both its routes and every malformed-field path matter."""

    async def test_v1_loads_sums_dp_ranks_and_carries_kv(self):
        envelope = {
            "timestamp": "2026-09-12T00:00:00Z",
            "version": "0.5.0",
            "accelerator": "cuda",
            "num_accelerators": 2,
            "loads": [
                {
                    "dp_rank": 0,
                    "num_running_reqs": 3,
                    "num_waiting_reqs": 2,
                    "num_used_tokens": 100,
                    "max_total_num_tokens": 1000,
                    "token_usage": 0.1,
                    "cache_hit_rate": 0.5,
                },
                {
                    "dp_rank": 1,
                    "num_running_reqs": 1,
                    "num_waiting_reqs": 0,
                    "num_used_tokens": 300,
                    "max_total_num_tokens": 1000,
                    "token_usage": 0.3,
                    "cache_hit_rate": 0.7,
                },
            ],
        }
        patcher, requested = _patch_http({"/v1/loads": envelope})
        with patcher:
            load = await collector.fetch_engine_load("http://spark.test:8888")

        self.assertEqual(requested, ["http://spark.test:8888/v1/loads"])
        self.assertEqual((load.running, load.waiting), (4, 2))
        self.assertEqual((load.used_tokens, load.total_tokens), (400, 2000))
        self.assertAlmostEqual(load.kv_pct, 20.0)
        # The payload's cache_hit_rate is deliberately not carried: the same
        # producer writes it only under current_scheduler_metrics_enabled, so
        # on the metrics-disabled server this route exists for it is a
        # constant 0 — a schema field, not a reading. AC1 pins that at the
        # poll_unit seam, where it is observable.

    async def test_v1_loads_capacity_weights_the_stated_gauge(self):
        # Each rank states the fill of its OWN pool, so the aggregate is the
        # capacity-weighted mean of those gauges: a small pool must not weigh
        # as much as a large one, and a stated gauge is not recomputed from
        # the token counts (they answer a different question — what is
        # resident now, not how full the pool is).
        envelope = {
            "loads": [
                {
                    "num_running_reqs": 2,
                    "num_waiting_reqs": 1,
                    "num_used_tokens": 100,
                    "max_total_num_tokens": 1000,
                    "token_usage": 0.1,
                },
                {
                    "num_running_reqs": 1,
                    "num_waiting_reqs": 0,
                    "num_used_tokens": 200,
                    "max_total_num_tokens": 3000,
                    "token_usage": 0.2,
                },
            ]
        }
        patcher, _ = _patch_http({"/v1/loads": envelope})
        with patcher:
            load = await collector.fetch_engine_load("http://spark.test:8888")

        self.assertEqual((load.running, load.waiting), (3, 1))
        self.assertEqual((load.used_tokens, load.total_tokens), (300, 4000))
        self.assertAlmostEqual(load.kv_pct, 17.5)  # (0.1*1000 + 0.2*3000) / 4000

    async def test_get_load_recovers_running_by_subtraction(self):
        # num_reqs is num_running_reqs + num_waiting_reqs — treating it as
        # running inflates the count by every waiting request.
        legacy = [
            {"dp_rank": 0, "num_reqs": 5, "num_waiting_reqs": 2},
            {"dp_rank": 1, "num_reqs": 9, "num_waiting_reqs": 4},
        ]
        patcher, requested = _patch_http({"/get_load": legacy})
        with patcher:
            load = await collector.fetch_engine_load("http://spark.test:8888")

        self.assertEqual(
            requested, ["http://spark.test:8888/v1/loads", "http://spark.test:8888/get_load"]
        )
        self.assertEqual((load.running, load.waiting), (8, 6))

    async def test_get_load_clamps_a_waiting_count_above_num_reqs(self):
        patcher, _ = _patch_http({"/get_load": [{"num_reqs": 1, "num_waiting_reqs": 4}]})
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertEqual((load.running, load.waiting), (0, 4))

    async def test_get_load_yields_used_tokens_but_no_capacity(self):
        # num_tokens is in-flight tokens, not the pool: upstream computes it
        # as used + queued, and the releases that serve this field report no
        # max_total_num_tokens at all. Deriving capacity from it made a
        # near-idle server read as a saturated pool (used == total).
        patcher, _ = _patch_http(
            {
                "/get_load": [
                    {
                        "num_reqs": 2,
                        "num_waiting_reqs": 0,
                        "num_tokens": 1000,
                        "num_pending_tokens": 700,
                    }
                ]
            }
        )
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertEqual((load.used_tokens, load.total_tokens), (300, 0))
        self.assertEqual(load.kv_pct, -1.0)

    async def test_get_load_without_pending_tokens_does_not_saturate(self):
        # The pre-/v1/loads shape serves num_tokens with no num_pending_tokens:
        # read as the pool, that made used == total — KV 100% on an idle node.
        # It is a bare used count.
        patcher, _ = _patch_http(
            {
                "/get_load": [
                    {"dp_rank": 0, "num_reqs": 3, "num_waiting_reqs": 1, "num_tokens": 4096}
                ]
            }
        )
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertEqual((load.running, load.waiting), (2, 1))
        self.assertEqual((load.used_tokens, load.total_tokens), (4096, 0))
        self.assertEqual(load.kv_pct, -1.0)

    async def test_foreign_endpoint_yields_none(self):
        patcher, _ = _patch_http(
            {"/v1/loads": {"detail": "Not Found"}, "/get_load": {"detail": "Not Found"}}
        )
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertIsNone(load)

    async def test_malformed_load_fields_degrade_not_raise(self):
        # poll_cluster's gather has no return_exceptions, so a non-numeric
        # field (or JSON 1e999 -> inf) must degrade to 0, never escape.
        legacy = [
            {"num_reqs": {"a": 1}, "num_waiting_reqs": [2]},
            {"num_reqs": "3", "num_waiting_reqs": None},
            {"num_reqs": 4, "num_waiting_reqs": 1},
            {"num_reqs": 1e999, "num_waiting_reqs": 0},
        ]
        patcher, _ = _patch_http({"/get_load": legacy})
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertEqual((load.running, load.waiting), (6, 1))

    async def test_huge_integer_load_fields_degrade_not_raise(self):
        # A JSON integer literal with hundreds of digits is an exact Python
        # int: int() accepts it and every later float()/division raises
        # OverflowError, which would escape poll_unit and kill the tick.
        huge = int("9" * 309)
        envelope = {
            "loads": [
                {
                    "num_running_reqs": 1,
                    "num_waiting_reqs": 0,
                    "num_used_tokens": 5,
                    "max_total_num_tokens": huge,
                    "token_usage": 0.5,
                }
            ]
        }
        patcher, _ = _patch_http({"/v1/loads": envelope})
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")
        self.assertEqual((load.running, load.total_tokens), (1, 0))
        self.assertAlmostEqual(load.kv_pct, 50.0)  # falls back to the usage gauge

        patcher, _ = _patch_http(
            {"/get_load": [{"num_reqs": 3, "num_waiting_reqs": 1, "num_tokens": huge}]}
        )
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")
        self.assertEqual((load.running, load.waiting), (2, 1))
        self.assertEqual((load.used_tokens, load.total_tokens), (0, 0))

    async def test_out_of_range_load_gauges_are_rejected_not_scaled(self):
        # token_usage is a 0-1 fraction; rendering a broken 42 as "4200%" KV
        # is worse than leaving the honest fallback. cache_hit_rate is not
        # consumed from this route at all (EngineLoad), out-of-band or not.
        envelope = {
            "loads": [
                {
                    "num_running_reqs": 1,
                    "num_waiting_reqs": 0,
                    "num_used_tokens": 5,
                    "max_total_num_tokens": 100,
                    "token_usage": 42.0,
                    "cache_hit_rate": -1.0,
                }
            ]
        }
        patcher, _ = _patch_http({"/v1/loads": envelope})
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")
        self.assertAlmostEqual(load.kv_pct, 5.0)  # used/total, not 4200

    async def test_v1_loads_ignores_entries_without_load_fields(self):
        patcher, _ = _patch_http(
            {"/v1/loads": {"loads": ["garbage", {"dp_rank": 0}, {"num_running_reqs": "x"}]}}
        )
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertEqual((load.running, load.waiting), (0, 0))
        self.assertEqual(load.kv_pct, -1.0)

    async def test_v1_loads_without_a_single_load_field_is_not_the_protocol(self):
        # A dict envelope whose entries carry no load field at all is a
        # foreign/404-ish body: returning a zero load would mark the node
        # hosted with fabricated figures.
        patcher, _ = _patch_http({"/v1/loads": {"loads": ["garbage", {"dp_rank": 0}]}})
        with patcher:
            load = await collector.fetch_engine_load("http://x:1")

        self.assertIsNone(load)

    async def test_non_json_load_body_is_swallowed(self):
        # A proxy or port-forward can answer 200 with HTML on SGLang's route;
        # resp.json() raising must not escape into poll_unit.
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                return FakeResp()

        with patch.object(collector.httpx, "AsyncClient", FakeClient):
            load = await collector.fetch_engine_load("http://x:1")

        self.assertIsNone(load)


class SglangPollUnitTests(unittest.IsolatedAsyncioTestCase):
    """A metrics-disabled SGLang node must still render concurrency AND KV —
    the load API carries both — instead of a row of zeros."""

    UNITS = {
        9: {
            "label": "spark-sg",
            "ssh_target": "tester@spark.test",
            "vllm_url": "http://spark.test:8888",
            "worker": False,
            "engine": None,
        }
    }

    async def _poll(self, load, units=None):
        async def metrics_404(url):
            raise RuntimeError("404 Not Found for url 'http://spark.test:8888/metrics'")

        collector._model_names.clear()
        collector._model_names[9] = "Nemotron-3.5"
        probe = AsyncMock(return_value=load)
        with patch.object(collector, "SPARK_UNITS", units or self.UNITS):
            with patch.object(collector, "_fetch_metrics", metrics_404):
                with patch.object(collector, "_fetch_telemetry", AsyncMock(return_value={})):
                    with patch.object(collector, "fetch_engine_load", probe):
                        return await collector.poll_unit(9), probe

    async def _poll_http(self, routes, units=None):
        """Poll with the REAL load parser behind a canned HTTP client, so the
        path from a raw ``/v1/loads`` body to the published sentinel is
        exercised rather than a hand-built ``EngineLoad``."""

        async def metrics_404(url):
            raise RuntimeError("404 Not Found for url 'http://spark.test:8888/metrics'")

        collector._model_names.clear()
        collector._model_names[9] = "Nemotron-3.5"
        patcher, requested = _patch_http(routes)
        with patcher:
            with patch.object(collector, "SPARK_UNITS", units or self.UNITS):
                with patch.object(collector, "_fetch_metrics", metrics_404):
                    with patch.object(collector, "_fetch_telemetry", AsyncMock(return_value={})):
                        return await collector.poll_unit(9), requested

    async def test_stated_cache_hit_rate_is_never_published(self):
        # The load API does carry cache_hit_rate, but the metrics reporter
        # writes it only inside `if self.current_scheduler_metrics_enabled:`
        # — on the metrics-disabled server that makes this route the only
        # signal source it is structurally constant 0. A stated fraction is
        # not a reading of this endpoint either, so nothing may reach the
        # cache row: the unit keeps the unknown sentinel and the cluster
        # aggregate stays unknown with it.
        envelope = {
            "loads": [
                {
                    "num_running_reqs": 2,
                    "num_waiting_reqs": 1,
                    "num_used_tokens": 450_000,
                    "max_total_num_tokens": 1_000_000,
                    "token_usage": 0.45,
                    "cache_hit_rate": 0.6,
                }
            ]
        }
        stats, requested = await self._poll_http({"/v1/loads": envelope})

        self.assertEqual(requested, ["http://spark.test:8888/v1/loads"])
        self.assertTrue(stats.model_hosted)
        self.assertEqual(stats.model_source, "sglang")
        self.assertFalse(stats.model_metrics)
        self.assertEqual((stats.requests_running, stats.requests_waiting), (2, 1))
        self.assertAlmostEqual(stats.kv_cache_pct, 45.0)
        self.assertEqual(stats.kv_cache_used_tokens, 450_000)
        self.assertEqual(stats.kv_prefix_hit_rate, -1.0)
        self.assertEqual(ClusterStats(units=[stats]).kv_prefix_hit_rate, -1.0)

    async def test_load_fallback_fills_concurrency_and_kv(self):
        stats, probe = await self._poll(
            collector.EngineLoad(
                running=2,
                waiting=1,
                used_tokens=450_000,
                total_tokens=1_000_000,
                kv_pct=45.0,
            )
        )

        probe.assert_awaited_once_with("http://spark.test:8888")
        self.assertTrue(stats.model_hosted)
        self.assertEqual(stats.model_source, "sglang")
        self.assertFalse(stats.model_metrics)
        self.assertEqual((stats.requests_running, stats.requests_waiting), (2, 1))
        self.assertEqual(stats.kv_total_tokens, 1_000_000)
        self.assertEqual(stats.kv_cache_used_tokens, 450_000)
        self.assertAlmostEqual(stats.kv_cache_pct, 45.0)
        # The load snapshot states no prefix rate this endpoint actually
        # measured (EngineLoad), so the unit keeps the unknown sentinel and
        # the cache row paints "—" rather than a manufactured "hit 0%".
        self.assertEqual(stats.kv_prefix_hit_rate, -1.0)
        self.assertEqual(stats.model_name, "Nemotron-3.5")
        # A healthy load-API fallback reports no metrics error at all.
        self.assertNotIn("metrics:", stats.error)

    async def test_load_fallback_fabricates_no_throughput(self):
        # The load API carries no token counters: a tok/s figure here would be
        # invented, and poll_cluster's delta machinery would publish it.
        stats, _ = await self._poll(
            collector.EngineLoad(running=2, waiting=1, used_tokens=450_000, total_tokens=1_000_000)
        )

        self.assertFalse(stats.model_metrics)
        self.assertEqual(stats.throughput_tok_s, 0.0)
        self.assertEqual(stats.prompt_throughput_tok_s, 0.0)
        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertEqual(stats.prompt_tokens_total, 0.0)

    async def test_load_without_capacity_sets_no_kv_denominator(self):
        # /get_load reports used tokens and no capacity: the used count must
        # land (it is real) while capacity stays absent rather than being
        # invented from the token count. The percentage is a reading of its
        # own — /v1/loads states token_usage, and a fraction needs no
        # denominator — so a stated one survives; only a route that states
        # none leaves the sentinel.
        stats, _ = await self._poll(
            collector.EngineLoad(
                running=2,
                waiting=1,
                used_tokens=5,
                total_tokens=0,
                kv_pct=50.0,
            )
        )

        self.assertEqual(stats.kv_cache_used_tokens, 5)
        self.assertEqual(stats.kv_total_tokens, 0)
        self.assertEqual(stats.kv_cache_pct, 50.0)
        self.assertEqual(stats.kv_prefix_hit_rate, -1.0)

    async def test_load_without_a_percentage_keeps_the_sentinel(self):
        # /get_load states neither capacity nor a fill: the dataclass default
        # (0.0) would reach the pane as a confidently empty pool, so the
        # unknown has to arrive as the sentinel the UI paints as "—".
        stats, _ = await self._poll(collector.EngineLoad(running=2, waiting=1, used_tokens=4096))

        self.assertEqual(stats.kv_cache_used_tokens, 4096)
        self.assertEqual(stats.kv_total_tokens, 0)
        self.assertEqual(stats.kv_cache_pct, -1.0)

    async def test_declared_vllm_node_never_probes_the_load_api(self):
        units = {9: {**self.UNITS[9], "engine": "vllm"}}
        stats, probe = await self._poll(None, units)

        probe.assert_not_awaited()
        self.assertFalse(stats.model_hosted)
        self.assertIn("metrics:", stats.error)


class ClusterStatsKvAggregationTests(unittest.TestCase):
    """Tests for ClusterStats KV aggregation properties."""

    def setUp(self):
        from stats import ClusterStats

        self.ClusterStats = ClusterStats

    def test_hosted_units_returns_model_hosted_units_only(self):
        s1 = SparkUnitStats(label="Spark-0", model_hosted=True)
        s2 = SparkUnitStats(label="Spark-1", model_hosted=False)
        cs = self.ClusterStats(units=[s1, s2])

        hosted = cs.hosted_units
        self.assertEqual(len(hosted), 1)
        self.assertEqual(hosted[0].label, "Spark-0")

    def test_total_kv_capacity_is_from_first_hosted_unit(self):
        s1 = SparkUnitStats(label="Spark-0", model_hosted=True, kv_total_tokens=100000)
        s2 = SparkUnitStats(label="Spark-1", model_hosted=True, kv_total_tokens=200000)
        cs = self.ClusterStats(units=[s1, s2])

        self.assertEqual(cs.total_kv_capacity_tokens, 100000)

    def test_kv_aggregates_without_hosted_units(self):
        s = SparkUnitStats(label="Spark-0", model_hosted=False)
        cs = self.ClusterStats(units=[s])

        self.assertEqual(cs.total_kv_capacity_tokens, 0)
        self.assertEqual(cs.total_kv_used_tokens, 0)
        # Nothing hosted is no reading, not an empty pool.
        self.assertEqual(cs.kv_cache_pct, -1.0)
        self.assertEqual(cs.kv_prefix_hit_rate, -1.0)
        self.assertEqual(cs.total_kv_blocks, 0)

    def test_kv_cache_pct_aggregates_from_hosted(self):
        s1 = SparkUnitStats(label="Spark-0", model_hosted=True, kv_cache_pct=42.0)
        cs = self.ClusterStats(units=[s1])

        self.assertAlmostEqual(cs.kv_cache_pct, 42.0)


class PollUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_hardware_fetch_starts_before_vllm_finishes(self):
        metrics_started = asyncio.Event()
        release_metrics = asyncio.Event()
        telemetry_started = asyncio.Event()

        async def fetch_metrics(vllm_url):
            metrics_started.set()
            await release_metrics.wait()
            return ""

        async def fetch_telemetry(ssh_target):
            telemetry_started.set()
            return {}

        metrics_mock = AsyncMock(side_effect=fetch_metrics)
        telemetry_mock = AsyncMock(side_effect=fetch_telemetry)
        units = {
            7: {
                "label": "test-node",
                "ssh_target": "tester@spark.test",
                "vllm_url": "http://spark.test:8000",
                "worker": True,
            }
        }
        with patch.object(collector, "SPARK_UNITS", units):
            with patch.object(collector, "_fetch_metrics", metrics_mock):
                with patch.object(collector, "_fetch_telemetry", telemetry_mock):
                    poll_task = asyncio.create_task(collector.poll_unit(7))
                    try:
                        await metrics_started.wait()
                        await asyncio.sleep(0)
                        self.assertTrue(telemetry_started.is_set())
                    finally:
                        release_metrics.set()
                        await poll_task

        metrics_mock.assert_awaited_once_with("http://spark.test:8000")
        telemetry_mock.assert_awaited_once_with("tester@spark.test")

    async def test_worker_probe_does_not_repeat_model_name_discovery(self):
        units = {
            7: {
                "label": "test-node",
                "ssh_target": "tester@spark.test",
                "vllm_url": "http://spark.test:8000",
                "worker": True,
            }
        }
        metrics_mock = AsyncMock(
            return_value='vllm:request_generation_tokens_sum{model_name="a"} 0.0\n'
        )
        telemetry_mock = AsyncMock(return_value={})
        collector._model_names.clear()

        with patch.object(collector, "SPARK_UNITS", units):
            with patch.object(collector, "_fetch_metrics", metrics_mock):
                with patch.object(collector, "_fetch_telemetry", telemetry_mock):
                    with patch.object(collector.httpx, "AsyncClient") as http_client:
                        stats = await collector.poll_unit(7)

        self.assertTrue(stats.model_hosted)
        metrics_mock.assert_awaited_once_with("http://spark.test:8000")
        http_client.assert_not_called()


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        collector._prev_tokens.clear()

    def tearDown(self):
        collector._prev_tokens.clear()

    def test_throughput_uses_token_delta_over_time(self):
        first = SparkUnitStats(model_hosted=True, generation_tokens_total=100.0)
        second = SparkUnitStats(model_hosted=True, generation_tokens_total=160.0)

        collector._update_throughput(1, first, 10.0)
        collector._update_throughput(1, second, 12.0)

        self.assertEqual(second.throughput_tok_s, 30.0)

    def test_counter_reset_does_not_emit_negative_throughput(self):
        first = SparkUnitStats(model_hosted=True, generation_tokens_total=100.0)
        second = SparkUnitStats(model_hosted=True, generation_tokens_total=10.0)

        collector._update_throughput(1, first, 10.0)
        collector._update_throughput(1, second, 12.0)

        self.assertEqual(second.throughput_tok_s, 0.0)

    def test_throughput_zero_when_model_not_hosted(self):
        first = SparkUnitStats(model_hosted=False, generation_tokens_total=100.0)
        second = SparkUnitStats(model_hosted=False, generation_tokens_total=160.0)

        collector._update_throughput(1, first, 10.0)
        collector._update_throughput(1, second, 12.0)

        self.assertEqual(second.throughput_tok_s, 0.0)

    def test_not_hosted_prunes_prev_tokens(self):
        seeded = SparkUnitStats(model_hosted=True, generation_tokens_total=100.0)
        collector._update_throughput(99, seeded, 10.0)
        self.assertIn(99, collector._prev_tokens)

        unstaged = SparkUnitStats(model_hosted=False, generation_tokens_total=160.0)
        collector._update_throughput(99, unstaged, 12.0)

        self.assertNotIn(99, collector._prev_tokens)
        self.assertEqual(unstaged.throughput_tok_s, 0.0)

    def test_rate_overflow_drops_infinite_rate(self):
        # A huge-but-finite delta (5e307 tokens over a 1e-6 s window) overflows
        # to inf — it must not reach the charts.
        collector._prev_tokens[1] = (10.0 - 1e-6, 1e308)
        s = SparkUnitStats(model_hosted=True, generation_tokens_total=1.5e308)

        collector._update_throughput(1, s, 10.0)

        self.assertEqual(s.throughput_tok_s, 0.0)

    def test_prompt_rate_overflow_drops_infinite_rate(self):
        collector._prev_prompt_tokens[1] = (10.0 - 1e-6, 1e308)
        s = SparkUnitStats(model_hosted=True, prompt_tokens_total=1.5e308)

        collector._update_prompt_throughput(1, s, 10.0)

        self.assertEqual(s.prompt_throughput_tok_s, 0.0)


class MemoryThrashParseTests(unittest.TestCase):
    def test_parse_known_output(self):
        output = (
            "33554432 10737418 1048576 6145728 10485760 128849018\n"
            "1200 800 450 0 5 0 0 150000 118500 200 300\n"
            "0.05 2800000\n"
            "0.02 1200000\n"
        )
        result = collector._parse_memory_thrash_output(output)
        self.assertEqual(result["swap_total_kb"], 33554432)
        self.assertEqual(result["swap_used_kb"], 22817014)  # 33554432 - 10737418
        self.assertEqual(result["swap_cached_kb"], 1048576)
        self.assertEqual(result["mem_avail_kb"], 6145728)
        self.assertEqual(result["mem_free_kb"], 10485760)
        self.assertEqual(result["mem_total_kb"], 128849018)
        self.assertEqual(result["pswpin"], 1200)
        self.assertEqual(result["pswpout"], 800)
        self.assertEqual(result["pgmajfault"], 450)
        self.assertEqual(result["allocstall_dma"], 0)
        self.assertEqual(result["allocstall_normal"], 5)
        self.assertEqual(result["allocstall_movable"], 0)
        self.assertEqual(result["allocstall_device"], 0)
        self.assertEqual(result["pgscan_kswapd"], 150000)
        self.assertEqual(result["pgsteal_kswapd"], 118500)
        self.assertEqual(result["workingset_refault_anon"], 200)
        self.assertEqual(result["workingset_refault_file"], 300)
        self.assertEqual(result["psi_some_avg10"], 0.05)
        self.assertEqual(result["psi_some_total"], 2800000)
        self.assertEqual(result["psi_full_avg10"], 0.02)
        self.assertEqual(result["psi_full_total"], 1200000)

    def test_parse_without_optional_psi_lines(self):
        output = (
            "33554432 10737418 1048576 6145728 10485760 128849018\n"
            "1200 800 450 0 5 0 0 150000 118500 200 300\n"
        )

        result = collector._parse_memory_thrash_output(output)

        self.assertEqual(result["swap_total_kb"], 33554432)
        self.assertEqual(result["pswpin"], 1200)
        self.assertNotIn("psi_some_avg10", result)
        self.assertNotIn("psi_full_avg10", result)

    def test_parse_truncated_output_returns_empty(self):
        output = "33554432 10737418\n"
        result = collector._parse_memory_thrash_output(output)
        self.assertEqual(result, {})

    def test_parse_empty_string_returns_empty(self):
        self.assertEqual(collector._parse_memory_thrash_output(""), {})


class ThrashRateComputationTests(unittest.TestCase):
    def setUp(self):
        collector._prev_thrash.clear()

    def tearDown(self):
        collector._prev_thrash.clear()

    def test_first_call_returns_zero_rates(self):
        data = {
            "swap_total_kb": 33554432,
            "swap_used_kb": 22817014,
            "swap_cached_kb": 1048576,
            "mem_avail_kb": 6145728,
            "mem_free_kb": 10485760,
            "mem_total_kb": 128849018,
            "pswpin": 1200,
            "pswpout": 800,
            "pgmajfault": 450,
            "allocstall_dma": 0,
            "allocstall_normal": 5,
            "allocstall_movable": 0,
            "allocstall_device": 0,
            "pgscan_kswapd": 150000,
            "pgsteal_kswapd": 118500,
            "workingset_refault_anon": 200,
            "workingset_refault_file": 300,
            "psi_some_avg10": 0.05,
            "psi_full_avg10": 0.02,
            "psi_some_total": 2800000,
            "psi_full_total": 1200000,
        }
        result = collector._update_thrash_rates("test-host", data, 100.0)
        self.assertEqual(result["swap_in_rate"], 0.0)
        self.assertEqual(result["swap_out_rate"], 0.0)
        self.assertEqual(result["majflt_rate"], 0.0)
        self.assertEqual(result["kswapd_scan_rate"], 0.0)
        self.assertEqual(result["kswapd_steal_rate"], 0.0)
        self.assertEqual(result["workingset_refault_rate"], 0.0)
        self.assertEqual(result["allocstall_this_poll"], 0)
        self.assertEqual(result["psi_full_total_delta"], 0)
        self.assertEqual(result["swap_total_kb"], 33554432)
        self.assertEqual(result["allocstall_total"], 5)

    def test_second_call_computes_rates(self):
        # First call — seed baseline
        data1 = {
            "swap_total_kb": 33554432,
            "swap_used_kb": 22817014,
            "swap_cached_kb": 1048576,
            "mem_avail_kb": 6145728,
            "mem_free_kb": 10485760,
            "mem_total_kb": 128849018,
            "pswpin": 1200,
            "pswpout": 800,
            "pgmajfault": 450,
            "allocstall_dma": 0,
            "allocstall_normal": 5,
            "allocstall_movable": 0,
            "allocstall_device": 0,
            "pgscan_kswapd": 150000,
            "pgsteal_kswapd": 118500,
            "workingset_refault_anon": 200,
            "workingset_refault_file": 300,
            "psi_some_avg10": 0.05,
            "psi_full_avg10": 0.02,
            "psi_some_total": 2800000,
            "psi_full_total": 1200000,
        }
        collector._update_thrash_rates("test-host", data1, 100.0)

        # Second call — 5 seconds later with increased counters
        data2 = {
            "swap_total_kb": 33554432,
            "swap_used_kb": 23000000,
            "swap_cached_kb": 1048576,
            "mem_avail_kb": 6000000,
            "mem_free_kb": 10400000,
            "mem_total_kb": 128849018,
            "pswpin": 1250,  # +50 in 5s = 10/s
            "pswpout": 900,  # +100 in 5s = 20/s
            "pgmajfault": 500,  # +50 in 5s = 10/s
            "allocstall_dma": 0,  # +0
            "allocstall_normal": 8,  # +3
            "allocstall_movable": 0,  # +0
            "allocstall_device": 1,  # +1
            "pgscan_kswapd": 152000,  # +2000 in 5s = 400/s
            "pgsteal_kswapd": 120000,  # +1500 in 5s = 300/s
            "workingset_refault_anon": 250,  # +50
            "workingset_refault_file": 350,  # +50
            "psi_some_avg10": 0.06,
            "psi_full_avg10": 0.03,
            "psi_some_total": 2800100,
            "psi_full_total": 1200100,  # +100
        }
        result = collector._update_thrash_rates("test-host", data2, 105.0)
        self.assertAlmostEqual(result["swap_in_rate"], 10.0)
        self.assertAlmostEqual(result["swap_out_rate"], 20.0)
        self.assertAlmostEqual(result["majflt_rate"], 10.0)
        self.assertAlmostEqual(result["kswapd_scan_rate"], 400.0)
        self.assertAlmostEqual(result["kswapd_steal_rate"], 300.0)
        # workingset_refault: (250-200)+(350-300) = 100 / 5 = 20/s
        self.assertAlmostEqual(result["workingset_refault_rate"], 20.0)
        # allocstall_this_poll: (8-5)+(1-0) = 4
        self.assertEqual(result["allocstall_this_poll"], 4)
        self.assertEqual(result["psi_full_total_delta"], 100)

    def test_counter_reset_does_not_produce_negative_rates(self):
        data1 = {
            "swap_total_kb": 33554432,
            "swap_used_kb": 22817014,
            "swap_cached_kb": 1048576,
            "mem_avail_kb": 6145728,
            "mem_free_kb": 10485760,
            "mem_total_kb": 128849018,
            "pswpin": 1200,
            "pswpout": 800,
            "pgmajfault": 450,
            "allocstall_dma": 0,
            "allocstall_normal": 5,
            "allocstall_movable": 0,
            "allocstall_device": 0,
            "pgscan_kswapd": 150000,
            "pgsteal_kswapd": 118500,
            "workingset_refault_anon": 200,
            "workingset_refault_file": 300,
            "psi_some_avg10": 0.05,
            "psi_full_avg10": 0.02,
            "psi_some_total": 2800000,
            "psi_full_total": 1200000,
        }
        collector._update_thrash_rates("test-host", data1, 100.0)

        # Counter reset — values go backwards
        data2 = {
            "swap_total_kb": 33554432,
            "swap_used_kb": 22817014,
            "swap_cached_kb": 1048576,
            "mem_avail_kb": 6145728,
            "mem_free_kb": 10485760,
            "mem_total_kb": 128849018,
            "pswpin": 100,
            "pswpout": 50,
            "pgmajfault": 30,
            "allocstall_dma": 0,
            "allocstall_normal": 0,
            "allocstall_movable": 0,
            "allocstall_device": 0,
            "pgscan_kswapd": 100,
            "pgsteal_kswapd": 50,
            "workingset_refault_anon": 10,
            "workingset_refault_file": 20,
            "psi_some_avg10": 0.00,
            "psi_full_avg10": 0.00,
            "psi_some_total": 0,
            "psi_full_total": 0,
        }
        result = collector._update_thrash_rates("test-host", data2, 105.0)
        self.assertEqual(result["swap_in_rate"], 0.0)
        self.assertEqual(result["swap_out_rate"], 0.0)
        self.assertEqual(result["majflt_rate"], 0.0)
        self.assertEqual(result["kswapd_scan_rate"], 0.0)
        self.assertEqual(result["kswapd_steal_rate"], 0.0)
        self.assertEqual(result["workingset_refault_rate"], 0.0)
        self.assertEqual(result["allocstall_this_poll"], 0)
        self.assertEqual(result["psi_full_total_delta"], 0)


class ComputeThrashRiskTests(unittest.TestCase):
    def test_s2_like_state_returns_critical(self):
        """S2 has: swap_used=68% (caution), mem_avail=5.3% (caution),
        kswapd_eff=79% (caution), swap_cached=0.9G (caution) => 4 caution => CRITICAL"""
        import stats as st

        s = st.SparkUnitStats(
            mem_total_bytes=128849018 * 1024,  # ~128GB
            mem_avail_kb=6145728,  # ~5.7% (S1-like)
            swap_total_kb=33554432,  # 32GB swap
            swap_used_kb=0.68 * 33554432,  # 68% used
            swap_cached_kb=950 * 1024,  # ~0.9GB
            kswapd_scan_rate=1000,
            kswapd_steal_rate=790,  # 79% efficiency
        )
        level, reason = st.compute_thrash_risk(s)
        self.assertEqual(level, st.ThrashLevel.CRITICAL)
        self.assertIn("swap_used", reason)

    def test_healthy_state_returns_ok(self):
        import stats as st

        s = st.SparkUnitStats(
            mem_total_bytes=128849018 * 1024,
            mem_avail_kb=20000000,  # ~15% available
            swap_total_kb=33554432,
            swap_used_kb=1 * 1024 * 1024,  # 1GB used, ~3%
            swap_cached_kb=100 * 1024,  # 100MB
        )
        level, reason = st.compute_thrash_risk(s)
        self.assertEqual(level, st.ThrashLevel.OK)

    def test_single_critical_triggers_critical(self):
        """One critical signal alone (PSI=5%) triggers CRITICAL."""
        import stats as st

        s = st.SparkUnitStats(
            mem_total_bytes=128849018 * 1024,
            mem_avail_kb=20000000,
            swap_total_kb=33554432,
            swap_used_kb=1 * 1024 * 1024,
            psi_full_avg10=5.0,  # > 1% => CRITICAL
        )
        level, reason = st.compute_thrash_risk(s)
        self.assertEqual(level, st.ThrashLevel.CRITICAL)
        self.assertIn("PSI", reason)


class TelemetryParseTests(unittest.TestCase):
    """Tests for _fetch_telemetry section-parsing logic (pure function)."""

    def test_four_sections_parsed_correctly(self):
        output = (
            "---GPU---\n"
            "gpu line 1\n"
            "---CPU_TEMP---\n"
            "65000\n"
            "---CPU_STAT---\n"
            "cpu  100 200 300 400 500 600 700 800 900\n"
            "---THRASH---\n"
            "128849018 10737418 1048576 6145728 10485760 128849018\n"
        )
        result = collector._parse_telemetry_output(output)
        self.assertEqual(result["gpu_output"], "gpu line 1")
        self.assertEqual(result["cpu_temp"], "65000")
        self.assertEqual(result["cpu_stat"], "cpu  100 200 300 400 500 600 700 800 900")
        self.assertEqual(
            result["thrash_output"], "128849018 10737418 1048576 6145728 10485760 128849018"
        )

    def test_empty_section_does_not_bleed_into_next(self):
        """When a section marker is present but has no output, subsequent sections
        must not bleed into the previous section."""
        output = (
            "---GPU---\n"
            "---CPU_TEMP---\n"
            "65000\n"
            "---CPU_STAT---\n"
            "cpu  100 200\n"
            "---THRASH---\n"
            "100 200 300\n"
        )
        result = collector._parse_telemetry_output(output)
        self.assertEqual(result["gpu_output"], "")
        self.assertEqual(result["cpu_temp"], "65000")
        self.assertEqual(result["cpu_stat"], "cpu  100 200")
        self.assertEqual(result["thrash_output"], "100 200 300")

    def test_missing_section_is_omitted_without_corrupting_neighbors(self):
        output = "---GPU---\nsmi data\n---CPU_STAT---\ncpu  100 200\n---THRASH---\n100 200\n"

        result = collector._parse_telemetry_output(output)

        self.assertEqual(result["gpu_output"], "smi data")
        self.assertNotIn("cpu_temp", result)
        self.assertEqual(result["cpu_stat"], "cpu  100 200")
        self.assertEqual(result["thrash_output"], "100 200")

    def test_sections_are_parsed(self):
        output = (
            "---CPU_STAT---\ncpu  1 2 3\n"
            "---ROCE---\nroce:mlx5_0:1:100:200\n"
            "---TOPOLOGY---\nib:mlx5_0:1:4: ACTIVE:InfiniBand\n"
        )
        result = collector._parse_telemetry_output(output)

        self.assertEqual(result["cpu_stat"], "cpu  1 2 3")
        self.assertEqual(result["roce_output"], "roce:mlx5_0:1:100:200")
        self.assertEqual(result["topology_output"], "ib:mlx5_0:1:4: ACTIVE:InfiniBand")


class NvidiaSmiParseTests(unittest.TestCase):
    """Tests for _parse_nvidia_smi pure function."""

    def test_parses_sm_clock_from_field_seven(self):
        gpu_util, _mem_util, _mem_pct, power, _temp, sm = collector._parse_nvidia_smi(
            "73, 50, 62000, 120000, 430, 64, 2411, 0x0000000000000000\n"
        )
        self.assertEqual(sm, 2411.0)
        self.assertEqual(gpu_util, 73.0)
        self.assertEqual(power, 430.0)

    def test_na_memory_fields_do_not_break_sm_clock(self):
        # GB10 reports [N/A] for FB memory; SM clock must still parse.
        _u, _mu, mem_pct, _p, _t, sm = collector._parse_nvidia_smi(
            "0, 0, [N/A], [N/A], 11, 41, 2411, 0x0\n"
        )
        self.assertEqual(mem_pct, 0.0)
        self.assertEqual(sm, 2411.0)


class RoceParseTests(unittest.TestCase):
    """Tests for _parse_roce_output and _update_roce_rates."""

    def setUp(self):
        collector._prev_roce.clear()

    def test_parses_each_port_with_4x_octet_scaling_and_rate(self):
        output = (
            "roce:mlx5_0:1:100:200:200 Gb/sec (2X NDR)\nroce:mlx5_0:2:50:25:40 Gb/sec (4X QDR)\n"
        )
        ports = collector._parse_roce_output(output)
        self.assertEqual(ports[("mlx5_0", "1")], (100 * 4, 200 * 4, 200e9 / 8))
        self.assertEqual(ports[("mlx5_0", "2")], (50 * 4, 25 * 4, 40e9 / 8))

    def test_unreadable_rate_yields_zero_capacity(self):
        ports = collector._parse_roce_output("roce:mlx5_0:1:100:200\n")
        self.assertEqual(ports[("mlx5_0", "1")], (400, 800, 0.0))

    def test_skips_non_roce_lines(self):
        ports = collector._parse_roce_output("guid:foo:bar\nib:mlx5:1:4: ACTIVE:InfiniBand\n")
        self.assertEqual(ports, {})

    def test_no_usable_ports_returns_empty(self):
        self.assertEqual(collector._parse_roce_output(""), {})

    def test_rates_derive_from_cumulative_delta(self):
        ports = {("mlx5_0", "1"): (800, 400, 25e9)}
        rx1, tx1 = collector._update_roce_rates("h", ports, 10.0)
        self.assertEqual((rx1, tx1), (0.0, 0.0))  # first observation seeds only
        ports = {("mlx5_0", "1"): (1600, 800, 25e9)}
        rx2, tx2 = collector._update_roce_rates("h", ports, 12.0)
        self.assertEqual((rx2, tx2), (400.0, 200.0))

    def test_counter_reset_yields_zero_rate_without_negative(self):
        collector._update_roce_rates("h", {("mlx5_0", "1"): (800, 400, 25e9)}, 10.0)
        rx, tx = collector._update_roce_rates("h", {("mlx5_0", "1"): (0, 0, 25e9)}, 12.0)
        self.assertEqual((rx, tx), (0.0, 0.0))
        # baseline re-seeded, next poll with a fresh counter derives again
        rx2, tx2 = collector._update_roce_rates("h", {("mlx5_0", "1"): (1600, 800, 25e9)}, 14.0)
        self.assertEqual((rx2, tx2), (800.0, 400.0))

    def test_idle_poll_refreshes_baseline_time(self):
        collector._update_roce_rates("h", {("mlx5_0", "1"): (800, 400, 25e9)}, 0.0)
        # No counter movement for 10s: the next nonzero delta is measured
        # against the LATEST poll, not the last busy one.
        collector._update_roce_rates("h", {("mlx5_0", "1"): (800, 400, 25e9)}, 10.0)
        rx, tx = collector._update_roce_rates("h", {("mlx5_0", "1"): (1200, 600, 25e9)}, 11.0)
        self.assertEqual((rx, tx), (400.0, 200.0))

    def test_new_port_seeds_only_and_absent_port_is_dropped(self):
        # Active-set change: a port's lifetime counters appear only now; they
        # must seed the baseline, not count as one interval of traffic.
        rx, tx = collector._update_roce_rates(
            "h", {("mlx5_0", "2"): (90_000_000_000, 80_000_000_000, 25e9)}, 5.0
        )
        self.assertEqual((rx, tx), (0.0, 0.0))
        # The port goes away and the surviving port keeps its own rates.
        rx, tx = collector._update_roce_rates(
            "h", {("mlx5_0", "2"): (90_000_000_100, 80_000_000_100, 25e9)}, 7.0
        )
        self.assertEqual((rx, tx), (50.0, 50.0))
        # A previously unknown port appearing is again seed-only.
        rx, tx = collector._update_roce_rates(
            "h",
            {
                ("mlx5_0", "2"): (90_000_000_200, 80_000_000_200, 25e9),
                ("mlx5_0", "3"): (5, 6, 10e9),
            },
            9.0,
        )
        self.assertEqual((rx, tx), (50.0, 50.0))
        # Port 3 vanished: no negative rate, no stuck baseline.
        rx, tx = collector._update_roce_rates(
            "h", {("mlx5_0", "2"): (90_000_000_400, 80_000_000_400, 25e9)}, 11.0
        )
        self.assertEqual((rx, tx), (100.0, 100.0))


class TelemetryFetchTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the _SSH_ batch command and end-to-end fetch wiring."""

    async def test_roce_command_matches_ACTIVE_state_by_glob_not_exact(self):
        """state holds '4: ACTIVE' — exact '= ACTIVE' matches skip every port."""
        captured: dict = {}

        async def fake_ssh_run(target, cmd):
            captured["cmd"] = cmd
            return (
                "---ROCE---\n"
                "roce:rocep1s0f1:1:381880659154:381825975309:200 Gb/sec (2X NDR)\n"
                "---TOPOLOGY---\n"
                "ib:rocep1s0f1:1:4: ACTIVE:InfiniBand\n"
            )

        with patch.object(collector, "_ssh_run", side_effect=fake_ssh_run):
            telemetry = await collector._fetch_telemetry("host")

        self.assertIn("*ACTIVE*", captured["cmd"])
        self.assertNotIn('" = "ACTIVE"', captured["cmd"])
        ports = collector._parse_roce_output(telemetry["roce_output"])
        rcv, xmit, cap = ports[("rocep1s0f1", "1")]
        self.assertTrue(rcv > 0 and xmit > 0 and cap > 0)

    async def test_poll_unit_fills_roce_from_telemetry(self):
        units = {
            7: {
                "label": "test-node",
                "ssh_target": "tester@spark.test",
                "vllm_url": "http://spark.test:8000",
                "worker": False,
            }
        }
        metrics_mock = AsyncMock(
            return_value='vllm:request_generation_tokens_sum{model_name="a"} 0.0\n'
        )
        telemetry_mock = AsyncMock(
            return_value={
                "roce_output": "roce:rocep1s0f1:1:80000:70000:200 Gb/sec (2X NDR)\n",
            }
        )
        collector._prev_roce.clear()
        collector._prev_roce["tester@spark.test"] = {
            ("rocep1s0f1", "1"): (time.monotonic() - 1.0, 0, 0)
        }

        with patch.object(collector, "SPARK_UNITS", units):
            with patch.object(collector, "_fetch_metrics", metrics_mock):
                with patch.object(collector, "_fetch_telemetry", telemetry_mock):
                    with patch.object(collector.httpx, "AsyncClient") as http_client:
                        stats = await collector.poll_unit(7)

        self.assertTrue(stats.roce_rx_bps > 0)  # 80000*4 bytes over ~1s
        self.assertTrue(stats.roce_tx_bps > 0)
        # One port at 200 Gb/s -> full-duplex capacity of 2 * 25e9 B/s.
        self.assertAlmostEqual(stats.roce_capacity_bps, 2 * 25e9)
        http_client.assert_not_called()


class CpuStatParseTests(unittest.TestCase):
    """Tests for _parse_cpu_stat pure function."""

    def test_normal_dual_core_output(self):
        stat_text = (
            "cpu  100 200 300 400 500 600 700 800 900\n"
            "cpu0 10 20 30 40 50 60 70 80 90\n"
            "cpu1 100 200 300 400 500 600 700 800 900\n"
        )
        core_count, utils = collector._parse_cpu_stat("test-host", stat_text)
        self.assertEqual(core_count, 2)
        self.assertEqual(len(utils), 2)
        self.assertEqual(utils[0], 0.0)  # first call, no previous
        self.assertEqual(utils[1], 0.0)

    def test_empty_output(self):
        core_count, utils = collector._parse_cpu_stat("test-host", "")
        self.assertEqual(core_count, 0)
        self.assertEqual(utils, [])

    def test_single_core(self):
        stat_text = "cpu  100 200 300 400 500 600 700 800 900\ncpu0 10 20 30 40 50 60 70 80 90\n"
        core_count, utils = collector._parse_cpu_stat("test-host", stat_text)
        self.assertEqual(core_count, 1)
        self.assertEqual(len(utils), 1)


class InitModelNamesTests(unittest.IsolatedAsyncioTestCase):
    """Tests for one-shot startup model-name discovery."""

    def setUp(self):
        collector._model_names.clear()

    async def test_populates_cache_with_one_shared_client(self):
        first_response = MagicMock()
        first_response.json.return_value = {"data": [{"id": "model-a"}]}
        second_response = MagicMock()
        second_response.json.return_value = {"data": [{"model": "model-b"}]}
        client = MagicMock()
        client.get = AsyncMock(side_effect=[first_response, second_response])
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=False)
        units = {
            1: {"vllm_url": "http://one.test:8000"},
            2: {"vllm_url": "http://two.test:8000"},
        }

        with patch.object(collector, "SPARK_UNITS", units):
            with patch.object(
                collector.httpx, "AsyncClient", return_value=client_context
            ) as async_client:
                await collector._init_model_names()

        self.assertEqual(collector._model_names, {1: "model-a", 2: "model-b"})
        async_client.assert_called_once_with(timeout=10)
        self.assertEqual(client.get.await_count, 2)

    async def test_failed_lookup_does_not_block_other_cache_entries(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "model-a"}]}
        client = MagicMock()
        client.get = AsyncMock(side_effect=[response, RuntimeError("unavailable")])
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=False)
        units = {
            1: {"vllm_url": "http://one.test:8000"},
            2: {"vllm_url": "http://two.test:8000"},
        }

        with patch.object(collector, "SPARK_UNITS", units):
            with patch.object(collector.httpx, "AsyncClient", return_value=client_context):
                await collector._init_model_names()

        self.assertEqual(collector._model_names, {1: "model-a"})


if __name__ == "__main__":
    unittest.main()


class ReviewHardeningTests(unittest.TestCase):
    """Robustness contract: a malformed remote payload degrades one field or
    one node — it never raises past the parser or kills the cluster tick."""

    def test_coerce_count_degrades_malformed_values(self):
        self.assertEqual(collector._coerce_count({"a": 1}), 0)
        self.assertEqual(collector._coerce_count([2]), 0)
        self.assertEqual(collector._coerce_count("garbage"), 0)
        self.assertEqual(collector._coerce_count(None), 0)
        self.assertEqual(collector._coerce_count("3"), 3)
        self.assertEqual(collector._coerce_count(2.7), 2)

    def test_get_load_malformed_entries_degrade_not_raise(self):
        # A non-numeric field must never raise out of the parse: poll_cluster's
        # gather has no return_exceptions, so one bad payload would kill the
        # whole cluster poll tick.
        entries = [
            {"num_reqs": {"a": 1}, "num_waiting_reqs": [2]},
            {"num_reqs": "3", "num_waiting_reqs": None},
            {"num_reqs": 4, "num_waiting_reqs": 1},
            # JSON 1e999 parses to float inf — int(inf) raises OverflowError,
            # which must degrade to 0, not escape.
            {"num_reqs": 1e999, "num_waiting_reqs": 0},
            {"num_reqs": 2, "num_waiting_reqs": 1, "num_tokens": "x", "num_pending_tokens": 1e999},
        ]
        patcher, _ = _patch_http({"/get_load": entries})
        with patcher:
            load = asyncio.run(collector.fetch_engine_load("http://x:1"))
        self.assertEqual((load.running, load.waiting), (7, 2))
        self.assertEqual((load.used_tokens, load.total_tokens), (0, 0))

    def test_histogram_skips_garbage_bucket_lines(self):
        lines = [
            'vllm:ttft_bucket{le="0.1"} 10',
            'vllm:ttft_bucket{le="0.5"} abc',  # garbage count
            'vllm:ttft_bucket{le="x"} 20',  # garbage bound
            'vllm:ttft_bucket{le="+Inf"} 40',
            "vllm:ttft_count 40",
        ]
        buckets, count = collector._parse_prometheus_histogram(lines, "vllm:ttft")
        self.assertEqual(count, 40.0)
        # the garbage line is skipped whole — its bound goes with it
        self.assertEqual(buckets, {0.1: 10.0, float("inf"): 40.0})
        # quantiles still computable from the surviving buckets
        q = collector._estimate_quantile(buckets, count, 0.2)
        self.assertGreater(q, 0.0)
        self.assertLessEqual(q, 0.1)

    def test_multi_engine_kv_capacity_weighted(self):
        text = (
            'vllm:cache_config_info{engine="0",num_gpu_blocks="100",'
            'block_size="16",kv_cache_size_tokens="1600"} 0\n'
            'vllm:cache_config_info{engine="1",num_gpu_blocks="300",'
            'block_size="16",kv_cache_size_tokens="4800"} 0\n'
            'vllm:kv_cache_usage_perc{engine="0"} 0.2\n'
            'vllm:kv_cache_usage_perc{engine="1"} 0.6\n'
        )
        s = collector._parse_engine_metrics(text)
        self.assertEqual(s.kv_total_blocks, 400)  # summed, not last-engine
        self.assertEqual(s.kv_total_tokens, 6400)
        self.assertAlmostEqual(s.kv_cache_pct, 50.0)  # (0.2*100 + 0.6*300)/400
        self.assertEqual(s.kv_cache_used_tokens, int(6400 * 0.5))
        self.assertEqual(s.kv_cache_free_blocks, int(400 * 0.5))

    def test_multi_engine_kv_unpaired_counts_use_plain_mean(self):
        text = (
            'vllm:cache_config_info{engine="0",num_gpu_blocks="100",'
            'block_size="16",kv_cache_size_tokens="1600"} 0\n'
            'vllm:kv_cache_usage_perc{engine="0"} 0.2\n'
            'vllm:kv_cache_usage_perc{engine="1"} 0.6\n'
        )
        s = collector._parse_engine_metrics(text)
        self.assertAlmostEqual(s.kv_cache_pct, 40.0)
        self.assertEqual(s.kv_total_tokens, 1600)

    def test_namespace_picked_by_sample_evidence(self):
        vllm_majority = (
            "vllm:generation_tokens_total 5.0\n"
            "vllm:num_requests_running 1.0\n"
            "vllm:prompt_tokens_total 2.0\n"
            "sglang:generation_tokens_total 9.0\n"
        )
        s = collector._parse_engine_metrics(vllm_majority)
        self.assertEqual(s.model_source, "vllm")
        self.assertEqual(s.generation_tokens_total, 5.0)
        self.assertEqual(s.requests_running, 1)
        # SGLang's concurrency names are not read under the vLLM profile, so
        # an SGLang-majority payload must be parsed through SGLang's own.
        sglang_majority = (
            "vllm:generation_tokens_total 5.0\n"
            "sglang:generation_tokens_total 9.0\n"
            "sglang:num_running_reqs 2.0\n"
            "sglang:prompt_tokens_total 3.0\n"
        )
        s2 = collector._parse_engine_metrics(sglang_majority)
        self.assertEqual(s2.model_source, "sglang")
        self.assertEqual(s2.generation_tokens_total, 9.0)
        self.assertEqual(s2.requests_running, 2)
        self.assertEqual(s2.prompt_tokens_total, 3.0)

    def test_metrics_engine_is_none_without_sample_evidence(self):
        self.assertIsNone(collector.metrics_engine(""))
        self.assertIsNone(collector.metrics_engine("# HELP vllm:x help\n"))
        self.assertEqual(collector.detect_engine("# HELP vllm:x help\n"), "vllm")
        self.assertEqual(collector.metrics_engine("sglang:num_running_reqs 1\n"), "sglang")
        self.assertEqual(collector.metrics_engine("vllm:prompt_tokens_total 1\n"), "vllm")

    def test_tie_on_sample_evidence_resolves_to_vllm(self):
        # Locked rule: majority of sample lines, tie -> vLLM. A one-line-each
        # payload must not flip the parse to SGLang (which would zero every
        # vLLM series on a mixed/half-scraped exposition).
        tie = "vllm:prompt_tokens_total 1\nsglang:num_running_reqs 1\n"
        self.assertEqual(collector.metrics_engine(tie), "vllm")
        self.assertEqual(collector._parse_engine_metrics(tie).model_source, "vllm")

    def test_help_only_namespace_mention_is_not_sample_evidence(self):
        text = (
            "# HELP vllm:generation_tokens_total some help text\n"
            "# TYPE vllm:generation_tokens_total counter\n"
        )
        self.assertEqual(collector._ns_sample_count(text.splitlines(), "vllm:"), 0)
        s = collector._parse_engine_metrics(text)
        self.assertFalse(s.model_hosted)

    def test_prompt_nan_only_keeps_baseline_source(self):
        # A transient NaN on the modern counter must NOT switch the prompt
        # baseline onto the legacy name (the generation_tokens rule).
        text = "vllm:prompt_tokens_total  NaN\nvllm:prompt_tokens  77.0\n"
        s = collector._parse_engine_metrics(text)
        self.assertEqual(s.prompt_tokens_total, 0.0)
        # legacy-only payloads still read the legacy name
        s2 = collector._parse_engine_metrics("vllm:prompt_tokens  77.0\n")
        self.assertEqual(s2.prompt_tokens_total, 77.0)
