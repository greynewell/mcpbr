"""Latency and performance benchmarking metrics for evaluation runs.

This module complements the PerformanceProfiler in profiler.py by providing
aggregate latency statistics across multiple evaluation tasks. While the profiler
tracks per-task performance, this module computes cross-task percentile distributions
and throughput metrics suitable for benchmarking reports.

Key capabilities:
- Per-task event timestamp tracking (start, first tool call, first response, end)
- Per-tool-call latency recording within each task
- Aggregate percentile statistics (p50, p95, p99, mean) across tasks
- Tokens-per-second throughput calculation
- Human-readable latency report formatting
"""

import statistics
from dataclasses import dataclass, field
from typing import Any


def percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of a list of values.

    Uses linear interpolation between closest ranks for accurate percentile
    estimation, falling back to boundary values for edge cases.

    Args:
        data: List of numeric values. Must not be empty.
        p: Percentile to compute, in range [0, 100].

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If data is empty or p is outside [0, 100].
    """
    if not data:
        raise ValueError("Cannot compute percentile of empty data")
    if p < 0 or p > 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {p}")

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n == 1:
        return sorted_data[0]

    # Compute the rank using the C = 1 interpolation method (same as Excel PERCENTILE.INC)
    rank = (p / 100) * (n - 1)
    lower_index = int(rank)
    upper_index = lower_index + 1
    fraction = rank - lower_index

    if upper_index >= n:
        return sorted_data[-1]

    return sorted_data[lower_index] + fraction * (
        sorted_data[upper_index] - sorted_data[lower_index]
    )


@dataclass
class LatencyTracker:
    """Records timestamps for key events during a single evaluation task.

    Tracks the lifecycle of a task from start to end, including when the first
    tool call and first response occur. Also records individual tool call latencies
    for fine-grained analysis.

    Attributes:
        task_id: Identifier for the task being tracked.
        task_start: Timestamp (seconds since epoch) when the task began.
        first_tool_call: Timestamp when the first tool call was initiated.
        first_response: Timestamp when the first response was received.
        task_end: Timestamp when the task completed.
        tool_call_latencies: List of individual tool call durations in seconds.
        total_tokens: Total tokens (input + output) consumed during the task.
    """

    task_id: str = ""
    task_start: float | None = None
    first_tool_call: float | None = None
    first_response: float | None = None
    task_end: float | None = None
    tool_call_latencies: list[float] = field(default_factory=list)
    total_tokens: int = 0

    def record_task_start(self, timestamp: float) -> None:
        """Record the task start timestamp.

        Args:
            timestamp: Time in seconds (e.g., from time.time()).
        """
        self.task_start = timestamp

    def record_first_tool_call(self, timestamp: float) -> None:
        """Record the first tool call timestamp.

        Only records the first occurrence; subsequent calls are ignored.

        Args:
            timestamp: Time in seconds.
        """
        if self.first_tool_call is None:
            self.first_tool_call = timestamp

    def record_first_response(self, timestamp: float) -> None:
        """Record the first response timestamp.

        Only records the first occurrence; subsequent calls are ignored.

        Args:
            timestamp: Time in seconds.
        """
        if self.first_response is None:
            self.first_response = timestamp

    def record_task_end(self, timestamp: float) -> None:
        """Record the task end timestamp.

        Args:
            timestamp: Time in seconds.
        """
        self.task_end = timestamp

    def record_tool_call_latency(self, duration_seconds: float) -> None:
        """Record the latency of an individual tool call.

        Args:
            duration_seconds: Duration of the tool call in seconds.
        """
        self.tool_call_latencies.append(duration_seconds)

    @property
    def time_to_first_tool_call(self) -> float | None:
        """Calculate time from task start to first tool call.

        Returns:
            Duration in seconds, or None if either timestamp is missing.
        """
        if self.task_start is not None and self.first_tool_call is not None:
            return self.first_tool_call - self.task_start
        return None

    @property
    def total_task_duration(self) -> float | None:
        """Calculate total task duration from start to end.

        Returns:
            Duration in seconds, or None if either timestamp is missing.
        """
        if self.task_start is not None and self.task_end is not None:
            return self.task_end - self.task_start
        return None

    @property
    def tokens_per_second(self) -> float | None:
        """Calculate throughput in tokens per second.

        Returns:
            Tokens per second, or None if duration is zero or unavailable.
        """
        duration = self.total_task_duration
        if duration is not None and duration > 0 and self.total_tokens > 0:
            return self.total_tokens / duration
        return None


def _compute_distribution(values: list[float]) -> dict[str, float]:
    """Compute percentile distribution and mean for a list of values.

    Args:
        values: List of numeric values. Must not be empty.

    Returns:
        Dictionary with keys: p50, p95, p99, mean.
    """
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "mean": statistics.mean(values),
    }


def compute_latency_stats(trackers: list["LatencyTracker"]) -> dict[str, Any]:
    """Compute aggregate latency statistics across multiple task trackers.

    Collects timing data from all trackers and produces percentile distributions
    for key metrics: time to first tool call, total task duration, individual
    tool call latency, and tokens-per-second throughput.

    Args:
        trackers: List of LatencyTracker instances with recorded data.

    Returns:
        Dictionary containing:
        - time_to_first_tool_call: {p50, p95, p99, mean} or None
        - total_task_duration: {p50, p95, p99, mean} or None
        - tool_call_latency: {p50, p95, p99, mean} or None
        - tokens_per_second: {p50, p95, p99, mean} or None
        - task_count: number of trackers analyzed
    """
    if not trackers:
        return {
            "time_to_first_tool_call": None,
            "total_task_duration": None,
            "tool_call_latency": None,
            "tokens_per_second": None,
            "task_count": 0,
        }

    # Collect values from all trackers
    ttftc_values: list[float] = []
    duration_values: list[float] = []
    tool_latency_values: list[float] = []
    tps_values: list[float] = []

    for tracker in trackers:
        ttftc = tracker.time_to_first_tool_call
        if ttftc is not None:
            ttftc_values.append(ttftc)

        duration = tracker.total_task_duration
        if duration is not None:
            duration_values.append(duration)

        tool_latency_values.extend(tracker.tool_call_latencies)

        tps = tracker.tokens_per_second
        if tps is not None:
            tps_values.append(tps)

    return {
        "time_to_first_tool_call": _compute_distribution(ttftc_values) if ttftc_values else None,
        "total_task_duration": _compute_distribution(duration_values) if duration_values else None,
        "tool_call_latency": (
            _compute_distribution(tool_latency_values) if tool_latency_values else None
        ),
        "tokens_per_second": _compute_distribution(tps_values) if tps_values else None,
        "task_count": len(trackers),
    }


def _format_distribution(label: str, dist: dict[str, float], unit: str = "s") -> str:
    """Format a single distribution as a human-readable line.

    Args:
        label: Name of the metric.
        dist: Distribution dict with p50, p95, p99, mean.
        unit: Unit suffix to append to values.

    Returns:
        Formatted string line.
    """
    return (
        f"  {label}:\n"
        f"    Mean: {dist['mean']:.3f}{unit}\n"
        f"    p50:  {dist['p50']:.3f}{unit}\n"
        f"    p95:  {dist['p95']:.3f}{unit}\n"
        f"    p99:  {dist['p99']:.3f}{unit}"
    )


def format_latency_report(stats: dict[str, Any]) -> str:
    """Format latency statistics into a human-readable report.

    Produces a multi-line text report suitable for console output or inclusion
    in benchmark result files.

    Args:
        stats: Statistics dictionary as returned by compute_latency_stats().

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("Latency & Performance Report")
    lines.append("=" * 50)
    lines.append(f"Tasks analyzed: {stats.get('task_count', 0)}")
    lines.append("")

    ttftc = stats.get("time_to_first_tool_call")
    if ttftc is not None:
        lines.append(_format_distribution("Time to First Tool Call", ttftc))
        lines.append("")

    duration = stats.get("total_task_duration")
    if duration is not None:
        lines.append(_format_distribution("Total Task Duration", duration))
        lines.append("")

    tool_latency = stats.get("tool_call_latency")
    if tool_latency is not None:
        lines.append(_format_distribution("Tool Call Latency", tool_latency))
        lines.append("")

    tps = stats.get("tokens_per_second")
    if tps is not None:
        lines.append(_format_distribution("Throughput", tps, unit=" tok/s"))
        lines.append("")

    if all(
        stats.get(key) is None
        for key in [
            "time_to_first_tool_call",
            "total_task_duration",
            "tool_call_latency",
            "tokens_per_second",
        ]
    ):
        lines.append("  No latency data available.")
        lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)
