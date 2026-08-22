import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import collector
from stats import SparkUnitStats


class VllmMetricsTests(unittest.TestCase):
    def test_generation_tokens_counter_is_parsed(self):
        stats = collector._parse_vllm_metrics('vllm:generation_tokens{model_name="a"} 123.0\n')

        self.assertEqual(stats.generation_tokens_total, 123.0)

    def test_generation_tokens_counter_sums_multiple_series(self):
        stats = collector._parse_vllm_metrics(
            'vllm:generation_tokens{model_name="a"} 100.0\n'
            'vllm:generation_tokens{model_name="b"} 23.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 123.0)

    def test_request_generation_tokens_sum_is_fallback(self):
        stats = collector._parse_vllm_metrics(
            'vllm:request_generation_tokens_sum{model_name="a"} 456.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 456.0)
        self.assertTrue(stats.model_hosted)

    def test_zero_request_generation_tokens_fallback_marks_model_hosted(self):
        stats = collector._parse_vllm_metrics(
            'vllm:request_generation_tokens_sum{model_name="a"} 0.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertTrue(stats.model_hosted)

    def test_generation_tokens_counter_is_preferred_over_fallback(self):
        stats = collector._parse_vllm_metrics(
            'vllm:generation_tokens{model_name="a"} 123.0\n'
            'vllm:request_generation_tokens_sum{model_name="a"} 456.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 123.0)

    def test_zero_generation_tokens_counter_is_preferred_over_fallback(self):
        stats = collector._parse_vllm_metrics(
            'vllm:generation_tokens{model_name="a"} 0.0\n'
            'vllm:request_generation_tokens_sum{model_name="a"} 456.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 0.0)

    def test_model_hosted_set_by_vllm_metrics_parser(self):
        stats = collector._parse_vllm_metrics('vllm:generation_tokens{model_name="a"} 123.0\n')

        self.assertTrue(stats.model_hosted)

    def test_model_hosted_false_when_metrics_empty(self):
        stats = collector._parse_vllm_metrics("")

        self.assertFalse(stats.model_hosted)

    def test_kv_cache_usage_perc_is_parsed(self):
        stats = collector._parse_vllm_metrics('vllm:kv_cache_usage_perc{model_name="a"} 0.45\n')

        self.assertAlmostEqual(stats.kv_cache_pct, 45.0)

    def test_kv_cache_config_parses_blocks_and_block_size(self):
        stats = collector._parse_vllm_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="1234"} 1.0\n'
        )

        self.assertEqual(stats.kv_total_blocks, 1234)
        self.assertEqual(stats.kv_block_size, 16)
        self.assertEqual(stats.kv_total_tokens, 1234 * 16)

    def test_kv_cache_derives_block_and_token_counts_from_usage(self):
        stats = collector._parse_vllm_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="10000"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} 0.32\n'
        )

        self.assertAlmostEqual(stats.kv_cache_pct, 32.0)
        self.assertEqual(stats.kv_total_blocks, 10000)
        # usage_pct accounts for null block; derive free blocks from total * (1-usage)
        self.assertEqual(stats.kv_cache_free_blocks, int(10000 * (1 - 0.32)))
        self.assertEqual(stats.kv_total_tokens, 10000 * 16)
        used_blocks = 10000 - int(10000 * (1 - 0.32))
        self.assertEqual(stats.kv_cache_used_tokens, used_blocks * 16)

    def test_kv_cache_no_token_derivation_without_block_size(self):
        stats = collector._parse_vllm_metrics('vllm:kv_cache_usage_perc{model_name="a"} 0.5\n')

        self.assertAlmostEqual(stats.kv_cache_pct, 50.0)
        self.assertEqual(stats.kv_total_tokens, 0)
        self.assertEqual(stats.kv_cache_used_tokens, 0)

    def test_kv_prefix_hit_rate_is_parsed(self):
        stats = collector._parse_vllm_metrics('vllm:prefix_cache_hit_rate{model_name="a"} 0.875\n')

        self.assertAlmostEqual(stats.kv_prefix_hit_rate, 87.5)

    def test_kv_cache_hit_rate_fallback_is_parsed(self):
        stats = collector._parse_vllm_metrics('vllm:cache_hit_rate{model_name="a"} 0.655\n')

        self.assertAlmostEqual(stats.kv_prefix_hit_rate, 65.5)

    def test_model_hosted_true_when_kv_blocks_exist(self):
        stats = collector._parse_vllm_metrics(
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
        stats = collector._parse_vllm_metrics('vllm:kv_cache_usage_perc{model_name="a"} NaN\n')

        self.assertEqual(stats.kv_cache_pct, 0.0)

    def test_overflowing_kv_usage_is_ignored(self):
        # 1e308 * 100 overflows to inf even though the raw value is finite.
        stats = collector._parse_vllm_metrics('vllm:kv_cache_usage_perc{model_name="a"} 1e308\n')

        self.assertEqual(stats.kv_cache_pct, 0.0)

    def test_nan_kv_usage_does_not_derive_block_counts(self):
        stats = collector._parse_vllm_metrics(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="10000"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} NaN\n'
        )

        self.assertEqual(stats.kv_cache_pct, 0.0)
        self.assertEqual(stats.kv_cache_free_blocks, 0)
        self.assertEqual(stats.kv_cache_used_tokens, 0)
        self.assertEqual(stats.kv_total_tokens, 10000 * 16)

    def test_inf_generation_tokens_are_ignored(self):
        stats = collector._parse_vllm_metrics('vllm:generation_tokens{model_name="a"} +Inf\n')

        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertFalse(stats.model_hosted)

    def test_finite_generation_tokens_survive_a_bad_series(self):
        stats = collector._parse_vllm_metrics(
            'vllm:generation_tokens{model_name="a"} NaN\n'
            'vllm:generation_tokens{model_name="b"} 100.0\n'
        )

        self.assertEqual(stats.generation_tokens_total, 100.0)
        self.assertTrue(stats.model_hosted)

    def test_inf_requests_running_does_not_break_parse(self):
        stats = collector._parse_vllm_metrics(
            'vllm:num_requests_running{model_name="a"} +Inf\n'
            'vllm:generation_tokens{model_name="b"} 5.0\n'
        )

        self.assertEqual(stats.requests_running, 0)
        self.assertTrue(stats.model_hosted)

    def test_inf_requests_waiting_does_not_break_parse(self):
        stats = collector._parse_vllm_metrics(
            'vllm:num_requests_waiting{model_name="a"} +Inf\n'
            'vllm:generation_tokens{model_name="b"} 5.0\n'
        )

        self.assertEqual(stats.requests_waiting, 0)
        self.assertTrue(stats.model_hosted)

    def test_inf_request_generation_tokens_sum_is_ignored(self):
        stats = collector._parse_vllm_metrics(
            'vllm:request_generation_tokens_sum{model_name="a"} +Inf\n'
        )

        self.assertEqual(stats.generation_tokens_total, 0.0)
        self.assertFalse(stats.model_hosted)

    def test_nan_prefix_hit_rate_is_ignored(self):
        stats = collector._parse_vllm_metrics('vllm:prefix_cache_hit_rate{model_name="a"} NaN\n')

        self.assertEqual(stats.kv_prefix_hit_rate, -1.0)

    def test_nan_prompt_tokens_are_ignored(self):
        stats = collector._parse_vllm_metrics('vllm:prompt_tokens{model_name="a"} NaN\n')

        self.assertEqual(stats.prompt_tokens_total, 0.0)


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

    def test_kv_aggregates_zero_when_no_hosted_units(self):
        s = SparkUnitStats(label="Spark-0", model_hosted=False)
        cs = self.ClusterStats(units=[s])

        self.assertEqual(cs.total_kv_capacity_tokens, 0)
        self.assertEqual(cs.total_kv_used_tokens, 0)
        self.assertEqual(cs.kv_cache_pct, 0.0)
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
            with patch.object(collector, "_fetch_vllm_metrics", metrics_mock):
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
            with patch.object(collector, "_fetch_vllm_metrics", metrics_mock):
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

    def test_cpu_freq_and_roce_sections_are_parsed(self):
        output = (
            "---CPU_STAT---\ncpu  1 2 3\n"
            "---CPU_FREQ---\n2700000\n2800000\n"
            "---ROCE---\nroce:mlx5_0:1:100:200\n"
            "---TOPOLOGY---\nib:mlx5_0:1:4: ACTIVE:InfiniBand\n"
        )
        result = collector._parse_telemetry_output(output)

        self.assertEqual(result["cpu_freq"], "2700000\n2800000")
        self.assertEqual(result["roce_output"], "roce:mlx5_0:1:100:200")
        self.assertEqual(result["topology_output"], "ib:mlx5_0:1:4: ACTIVE:InfiniBand")


class CpuFreqParseTests(unittest.TestCase):
    """Tests for _parse_cpu_freq pure function."""

    def test_average_of_scaling_cur_freq_khz_values(self):
        self.assertAlmostEqual(collector._parse_cpu_freq("2700000\n2900000\n2500000\n"), 2700.0)

    def test_cpuinfo_mhz_values(self):
        self.assertAlmostEqual(collector._parse_cpu_freq("cpu MHz\t: 2700.000\n"), 2700.0)

    def test_empty_output_returns_zero(self):
        self.assertEqual(collector._parse_cpu_freq(""), 0.0)

    def test_garbage_lines_are_skipped(self):
        self.assertEqual(collector._parse_cpu_freq("not a number\n2700000\n"), 2700.0)


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

    async def test_poll_unit_fills_freq_and_roce_from_telemetry(self):
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
                "cpu_freq": "2700000\n2900000\n",
                "roce_output": "roce:rocep1s0f1:1:80000:70000:200 Gb/sec (2X NDR)\n",
            }
        )
        collector._prev_roce.clear()
        collector._prev_roce["tester@spark.test"] = {
            ("rocep1s0f1", "1"): (time.monotonic() - 1.0, 0, 0)
        }

        with patch.object(collector, "SPARK_UNITS", units):
            with patch.object(collector, "_fetch_vllm_metrics", metrics_mock):
                with patch.object(collector, "_fetch_telemetry", telemetry_mock):
                    with patch.object(collector.httpx, "AsyncClient") as http_client:
                        stats = await collector.poll_unit(7)

        self.assertAlmostEqual(stats.cpu_freq_mhz, 2800.0)
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
