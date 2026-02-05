"""Tests for latency metrics module."""

import pytest

from mcpbr.latency_metrics import (
    LatencyTracker,
    compute_latency_stats,
    format_latency_report,
    percentile,
)


class TestPercentile:
    """Tests for the percentile helper function."""

    def test_median_odd_count(self) -> None:
        """Test p50 with odd number of elements."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(data, 50) == pytest.approx(3.0)

    def test_median_even_count(self) -> None:
        """Test p50 with even number of elements interpolates correctly."""
        data = [1.0, 2.0, 3.0, 4.0]
        assert percentile(data, 50) == pytest.approx(2.5)

    def test_p0_returns_minimum(self) -> None:
        """Test that p0 returns the minimum value."""
        data = [5.0, 3.0, 1.0, 4.0, 2.0]
        assert percentile(data, 0) == pytest.approx(1.0)

    def test_p100_returns_maximum(self) -> None:
        """Test that p100 returns the maximum value."""
        data = [5.0, 3.0, 1.0, 4.0, 2.0]
        assert percentile(data, 100) == pytest.approx(5.0)

    def test_p95_large_dataset(self) -> None:
        """Test p95 with a larger dataset."""
        data = [float(i) for i in range(1, 101)]  # 1.0 to 100.0
        result = percentile(data, 95)
        assert result == pytest.approx(95.05, rel=1e-2)

    def test_p99_large_dataset(self) -> None:
        """Test p99 with a larger dataset."""
        data = [float(i) for i in range(1, 101)]
        result = percentile(data, 99)
        assert result == pytest.approx(99.01, rel=1e-2)

    def test_single_value(self) -> None:
        """Test percentile with a single data point."""
        data = [42.0]
        assert percentile(data, 0) == 42.0
        assert percentile(data, 50) == 42.0
        assert percentile(data, 100) == 42.0

    def test_two_values(self) -> None:
        """Test percentile with two data points."""
        data = [10.0, 20.0]
        assert percentile(data, 0) == pytest.approx(10.0)
        assert percentile(data, 50) == pytest.approx(15.0)
        assert percentile(data, 100) == pytest.approx(20.0)

    def test_unsorted_input(self) -> None:
        """Test that percentile handles unsorted input correctly."""
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        assert percentile(data, 50) == pytest.approx(3.0)

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute percentile of empty data"):
            percentile([], 50)

    def test_negative_percentile_raises_error(self) -> None:
        """Test that negative percentile raises ValueError."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            percentile([1.0], -1)

    def test_over_100_percentile_raises_error(self) -> None:
        """Test that percentile > 100 raises ValueError."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            percentile([1.0], 101)

    def test_identical_values(self) -> None:
        """Test percentile when all values are the same."""
        data = [7.0, 7.0, 7.0, 7.0, 7.0]
        assert percentile(data, 25) == 7.0
        assert percentile(data, 50) == 7.0
        assert percentile(data, 99) == 7.0


class TestLatencyTracker:
    """Tests for the LatencyTracker class."""

    def test_default_initialization(self) -> None:
        """Test that tracker initializes with None timestamps and empty lists."""
        tracker = LatencyTracker()
        assert tracker.task_id == ""
        assert tracker.task_start is None
        assert tracker.first_tool_call is None
        assert tracker.first_response is None
        assert tracker.task_end is None
        assert tracker.tool_call_latencies == []
        assert tracker.total_tokens == 0

    def test_record_task_start(self) -> None:
        """Test recording task start timestamp."""
        tracker = LatencyTracker(task_id="task-1")
        tracker.record_task_start(1000.0)
        assert tracker.task_start == 1000.0

    def test_record_task_end(self) -> None:
        """Test recording task end timestamp."""
        tracker = LatencyTracker()
        tracker.record_task_end(2000.0)
        assert tracker.task_end == 2000.0

    def test_record_first_tool_call_only_once(self) -> None:
        """Test that first_tool_call is only set on the first recording."""
        tracker = LatencyTracker()
        tracker.record_first_tool_call(100.0)
        tracker.record_first_tool_call(200.0)
        assert tracker.first_tool_call == 100.0

    def test_record_first_response_only_once(self) -> None:
        """Test that first_response is only set on the first recording."""
        tracker = LatencyTracker()
        tracker.record_first_response(150.0)
        tracker.record_first_response(250.0)
        assert tracker.first_response == 150.0

    def test_record_tool_call_latency(self) -> None:
        """Test recording individual tool call latencies."""
        tracker = LatencyTracker()
        tracker.record_tool_call_latency(0.5)
        tracker.record_tool_call_latency(1.2)
        tracker.record_tool_call_latency(0.3)
        assert tracker.tool_call_latencies == [0.5, 1.2, 0.3]

    def test_time_to_first_tool_call_property(self) -> None:
        """Test time_to_first_tool_call computed property."""
        tracker = LatencyTracker()
        tracker.record_task_start(100.0)
        tracker.record_first_tool_call(103.5)
        assert tracker.time_to_first_tool_call == pytest.approx(3.5)

    def test_time_to_first_tool_call_missing_start(self) -> None:
        """Test time_to_first_tool_call returns None if start is missing."""
        tracker = LatencyTracker()
        tracker.record_first_tool_call(103.5)
        assert tracker.time_to_first_tool_call is None

    def test_time_to_first_tool_call_missing_tool_call(self) -> None:
        """Test time_to_first_tool_call returns None if no tool call recorded."""
        tracker = LatencyTracker()
        tracker.record_task_start(100.0)
        assert tracker.time_to_first_tool_call is None

    def test_total_task_duration_property(self) -> None:
        """Test total_task_duration computed property."""
        tracker = LatencyTracker()
        tracker.record_task_start(100.0)
        tracker.record_task_end(145.0)
        assert tracker.total_task_duration == pytest.approx(45.0)

    def test_total_task_duration_missing_end(self) -> None:
        """Test total_task_duration returns None if end is missing."""
        tracker = LatencyTracker()
        tracker.record_task_start(100.0)
        assert tracker.total_task_duration is None

    def test_tokens_per_second_property(self) -> None:
        """Test tokens_per_second throughput computation."""
        tracker = LatencyTracker()
        tracker.record_task_start(0.0)
        tracker.record_task_end(10.0)
        tracker.total_tokens = 5000
        assert tracker.tokens_per_second == pytest.approx(500.0)

    def test_tokens_per_second_zero_duration(self) -> None:
        """Test tokens_per_second returns None for zero-duration task."""
        tracker = LatencyTracker()
        tracker.record_task_start(100.0)
        tracker.record_task_end(100.0)
        tracker.total_tokens = 1000
        assert tracker.tokens_per_second is None

    def test_tokens_per_second_no_tokens(self) -> None:
        """Test tokens_per_second returns None when no tokens recorded."""
        tracker = LatencyTracker()
        tracker.record_task_start(0.0)
        tracker.record_task_end(10.0)
        assert tracker.tokens_per_second is None

    def test_tokens_per_second_missing_timestamps(self) -> None:
        """Test tokens_per_second returns None if timestamps are missing."""
        tracker = LatencyTracker()
        tracker.total_tokens = 1000
        assert tracker.tokens_per_second is None

    def test_full_lifecycle(self) -> None:
        """Test a complete task lifecycle with all events recorded."""
        tracker = LatencyTracker(task_id="lifecycle-test")
        tracker.record_task_start(1000.0)
        tracker.record_first_tool_call(1002.0)
        tracker.record_tool_call_latency(0.8)
        tracker.record_first_response(1003.0)
        tracker.record_tool_call_latency(1.1)
        tracker.record_tool_call_latency(0.5)
        tracker.record_task_end(1030.0)
        tracker.total_tokens = 15000

        assert tracker.time_to_first_tool_call == pytest.approx(2.0)
        assert tracker.total_task_duration == pytest.approx(30.0)
        assert tracker.tokens_per_second == pytest.approx(500.0)
        assert len(tracker.tool_call_latencies) == 3


class TestComputeLatencyStats:
    """Tests for compute_latency_stats function."""

    def test_empty_trackers(self) -> None:
        """Test stats with no trackers returns zero task_count and None metrics."""
        stats = compute_latency_stats([])
        assert stats["task_count"] == 0
        assert stats["time_to_first_tool_call"] is None
        assert stats["total_task_duration"] is None
        assert stats["tool_call_latency"] is None
        assert stats["tokens_per_second"] is None

    def test_single_tracker(self) -> None:
        """Test stats with a single fully-populated tracker."""
        tracker = LatencyTracker(task_id="single")
        tracker.record_task_start(0.0)
        tracker.record_first_tool_call(2.0)
        tracker.record_tool_call_latency(0.5)
        tracker.record_task_end(10.0)
        tracker.total_tokens = 5000

        stats = compute_latency_stats([tracker])

        assert stats["task_count"] == 1

        # With a single value, all percentiles and mean should be the same
        ttftc = stats["time_to_first_tool_call"]
        assert ttftc is not None
        assert ttftc["p50"] == pytest.approx(2.0)
        assert ttftc["p95"] == pytest.approx(2.0)
        assert ttftc["p99"] == pytest.approx(2.0)
        assert ttftc["mean"] == pytest.approx(2.0)

        duration = stats["total_task_duration"]
        assert duration is not None
        assert duration["mean"] == pytest.approx(10.0)

        tool_lat = stats["tool_call_latency"]
        assert tool_lat is not None
        assert tool_lat["mean"] == pytest.approx(0.5)

        tps = stats["tokens_per_second"]
        assert tps is not None
        assert tps["mean"] == pytest.approx(500.0)

    def test_multiple_trackers(self) -> None:
        """Test stats aggregation across multiple trackers."""
        trackers = []
        for i in range(5):
            t = LatencyTracker(task_id=f"task-{i}")
            t.record_task_start(0.0)
            t.record_first_tool_call(float(i + 1))  # 1, 2, 3, 4, 5
            t.record_tool_call_latency(0.1 * (i + 1))  # 0.1, 0.2, 0.3, 0.4, 0.5
            t.record_task_end(float(10 + i * 5))  # 10, 15, 20, 25, 30
            t.total_tokens = 1000 * (i + 1)  # 1000, 2000, 3000, 4000, 5000
            trackers.append(t)

        stats = compute_latency_stats(trackers)

        assert stats["task_count"] == 5

        ttftc = stats["time_to_first_tool_call"]
        assert ttftc is not None
        assert ttftc["mean"] == pytest.approx(3.0)
        assert ttftc["p50"] == pytest.approx(3.0)

        duration = stats["total_task_duration"]
        assert duration is not None
        assert duration["mean"] == pytest.approx(20.0)

    def test_trackers_with_missing_data(self) -> None:
        """Test stats when some trackers lack certain timestamps."""
        t1 = LatencyTracker(task_id="complete")
        t1.record_task_start(0.0)
        t1.record_first_tool_call(1.0)
        t1.record_task_end(10.0)
        t1.total_tokens = 5000

        t2 = LatencyTracker(task_id="no-tool-call")
        t2.record_task_start(0.0)
        t2.record_task_end(8.0)
        # No first_tool_call or tokens

        t3 = LatencyTracker(task_id="incomplete")
        t3.record_task_start(0.0)
        # No end, no tool call

        stats = compute_latency_stats([t1, t2, t3])

        assert stats["task_count"] == 3

        # Only t1 has time_to_first_tool_call
        ttftc = stats["time_to_first_tool_call"]
        assert ttftc is not None
        assert ttftc["mean"] == pytest.approx(1.0)

        # t1 and t2 have total_task_duration
        duration = stats["total_task_duration"]
        assert duration is not None
        assert duration["mean"] == pytest.approx(9.0)

        # Only t1 has tokens_per_second
        tps = stats["tokens_per_second"]
        assert tps is not None
        assert tps["mean"] == pytest.approx(500.0)

    def test_trackers_with_no_relevant_data(self) -> None:
        """Test stats when trackers have no usable timing data."""
        t1 = LatencyTracker(task_id="empty-1")
        t2 = LatencyTracker(task_id="empty-2")

        stats = compute_latency_stats([t1, t2])

        assert stats["task_count"] == 2
        assert stats["time_to_first_tool_call"] is None
        assert stats["total_task_duration"] is None
        assert stats["tool_call_latency"] is None
        assert stats["tokens_per_second"] is None

    def test_tool_call_latencies_aggregated(self) -> None:
        """Test that tool call latencies are aggregated across all trackers."""
        t1 = LatencyTracker(task_id="t1")
        t1.record_tool_call_latency(1.0)
        t1.record_tool_call_latency(2.0)

        t2 = LatencyTracker(task_id="t2")
        t2.record_tool_call_latency(3.0)

        stats = compute_latency_stats([t1, t2])

        tool_lat = stats["tool_call_latency"]
        assert tool_lat is not None
        # Mean of [1.0, 2.0, 3.0] = 2.0
        assert tool_lat["mean"] == pytest.approx(2.0)
        assert tool_lat["p50"] == pytest.approx(2.0)

    def test_stats_output_keys(self) -> None:
        """Test that stats output always contains the expected keys."""
        stats = compute_latency_stats([])
        expected_keys = {
            "time_to_first_tool_call",
            "total_task_duration",
            "tool_call_latency",
            "tokens_per_second",
            "task_count",
        }
        assert set(stats.keys()) == expected_keys

    def test_distribution_keys(self) -> None:
        """Test that each distribution contains p50, p95, p99, mean."""
        tracker = LatencyTracker(task_id="keys-test")
        tracker.record_task_start(0.0)
        tracker.record_first_tool_call(1.0)
        tracker.record_tool_call_latency(0.5)
        tracker.record_task_end(10.0)
        tracker.total_tokens = 1000

        stats = compute_latency_stats([tracker])

        for key in [
            "time_to_first_tool_call",
            "total_task_duration",
            "tool_call_latency",
            "tokens_per_second",
        ]:
            dist = stats[key]
            assert dist is not None
            assert "p50" in dist
            assert "p95" in dist
            assert "p99" in dist
            assert "mean" in dist


class TestFormatLatencyReport:
    """Tests for format_latency_report function."""

    def test_report_with_all_data(self) -> None:
        """Test report formatting with full stats."""
        trackers = []
        for i in range(3):
            t = LatencyTracker(task_id=f"task-{i}")
            t.record_task_start(0.0)
            t.record_first_tool_call(float(i + 1))
            t.record_tool_call_latency(0.5 + i * 0.1)
            t.record_task_end(float(20 + i * 5))
            t.total_tokens = 3000 + i * 1000
            trackers.append(t)

        stats = compute_latency_stats(trackers)
        report = format_latency_report(stats)

        assert "Latency & Performance Report" in report
        assert "Tasks analyzed: 3" in report
        assert "Time to First Tool Call" in report
        assert "Total Task Duration" in report
        assert "Tool Call Latency" in report
        assert "Throughput" in report
        assert "Mean:" in report
        assert "p50:" in report
        assert "p95:" in report
        assert "p99:" in report
        assert "tok/s" in report

    def test_report_with_no_data(self) -> None:
        """Test report formatting with empty stats."""
        stats = compute_latency_stats([])
        report = format_latency_report(stats)

        assert "Latency & Performance Report" in report
        assert "Tasks analyzed: 0" in report
        assert "No latency data available." in report

    def test_report_with_partial_data(self) -> None:
        """Test report formatting when only some metrics are available."""
        tracker = LatencyTracker(task_id="partial")
        tracker.record_task_start(0.0)
        tracker.record_task_end(5.0)
        # No tool calls, no tokens

        stats = compute_latency_stats([tracker])
        report = format_latency_report(stats)

        assert "Tasks analyzed: 1" in report
        assert "Total Task Duration" in report
        # Should NOT have tool call or throughput sections
        assert "Time to First Tool Call" not in report
        assert "Tool Call Latency" not in report
        assert "Throughput" not in report

    def test_report_contains_separator_lines(self) -> None:
        """Test that report has separator lines for readability."""
        stats = compute_latency_stats([])
        report = format_latency_report(stats)

        lines = report.split("\n")
        assert lines[0] == "=" * 50
        assert lines[-1] == "=" * 50

    def test_report_is_string(self) -> None:
        """Test that report returns a string type."""
        stats = compute_latency_stats([])
        report = format_latency_report(stats)
        assert isinstance(report, str)

    def test_report_numeric_formatting(self) -> None:
        """Test that numeric values are formatted with 3 decimal places."""
        tracker = LatencyTracker(task_id="fmt")
        tracker.record_task_start(0.0)
        tracker.record_first_tool_call(1.23456789)
        tracker.record_task_end(10.0)

        stats = compute_latency_stats([tracker])
        report = format_latency_report(stats)

        # Values should be formatted to 3 decimal places
        assert "1.235" in report  # 1.23456789 rounded to 3 decimals
