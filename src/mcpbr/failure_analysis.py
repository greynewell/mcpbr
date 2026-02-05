"""Failure analysis module for evaluation results.

Provides detailed analysis of evaluation failures to understand common failure
modes, categorize errors, extract patterns, and generate actionable reports.
"""

from collections import Counter
from enum import StrEnum


class FailureCategory(StrEnum):
    """Categories for classifying evaluation failures."""

    TIMEOUT = "timeout"
    NO_PATCH = "no_patch"
    WRONG_ANSWER = "wrong_answer"
    RUNTIME_ERROR = "runtime_error"
    COMPILATION_ERROR = "compilation_error"
    TEST_FAILURE = "test_failure"
    TOOL_ERROR = "tool_error"
    ENVIRONMENT_ERROR = "environment_error"
    UNKNOWN = "unknown"


def categorize_failure(result: dict) -> str:
    """Categorize a failed result into a failure category.

    Examines the result dictionary to determine the most likely failure
    category based on status, error messages, patch presence, and other
    signals.

    Args:
        result: A result dictionary with keys like resolved, error, patch,
            status, tool_calls, tool_usage, etc.

    Returns:
        The failure category string from FailureCategory.
    """
    # Resolved results are not failures
    if result.get("resolved"):
        return FailureCategory.UNKNOWN

    status = str(result.get("status", "")).lower()
    error = str(result.get("error", "")).lower()

    # Check for timeout
    if status == "timeout" or "timeout" in error or "timed out" in error:
        return FailureCategory.TIMEOUT

    # Check for environment errors (docker, container, permission, network)
    if any(
        keyword in error
        for keyword in [
            "docker",
            "container",
            "permission denied",
            "access denied",
            "network",
            "connection refused",
            "connection reset",
            "environment",
            "no space left",
            "disk quota",
        ]
    ):
        return FailureCategory.ENVIRONMENT_ERROR

    # Check for tool errors (MCP tool failures)
    if any(
        keyword in error
        for keyword in [
            "mcp",
            "tool call",
            "tool error",
            "tool failed",
            "server error",
        ]
    ):
        return FailureCategory.TOOL_ERROR

    # Check tool failure rate as a signal for tool errors
    tool_failures = result.get("tool_failures", {})
    tool_usage = result.get("tool_usage", {})
    if tool_failures:
        total_failures = sum(tool_failures.values())
        total_calls = sum(tool_usage.values()) if tool_usage else 0
        if total_calls > 0 and total_failures / total_calls > 0.5:
            return FailureCategory.TOOL_ERROR

    # Check for compilation errors
    if any(
        keyword in error
        for keyword in [
            "compilation",
            "compile error",
            "syntax error",
            "syntaxerror",
            "indentation",
            "importerror",
            "modulenotfounderror",
            "nameerror",
        ]
    ):
        return FailureCategory.COMPILATION_ERROR

    # Check for runtime errors
    if any(
        keyword in error
        for keyword in [
            "runtime",
            "traceback",
            "exception",
            "attributeerror",
            "typeerror",
            "valueerror",
            "keyerror",
            "indexerror",
            "zerodivisionerror",
            "recursionerror",
            "memoryerror",
            "overflowerror",
        ]
    ):
        return FailureCategory.RUNTIME_ERROR

    # Check for no patch generated
    patch = result.get("patch")
    if not patch or (isinstance(patch, str) and not patch.strip()):
        return FailureCategory.NO_PATCH

    # Check for test failure (patch exists but tests failed)
    if any(
        keyword in error
        for keyword in [
            "test failed",
            "test failure",
            "assertion",
            "assertionerror",
            "fail_to_pass",
            "pass_to_pass",
        ]
    ):
        return FailureCategory.TEST_FAILURE

    # If there is a patch but the result is not resolved and no specific error
    # was detected, it is most likely a wrong answer or test failure
    if patch and isinstance(patch, str) and patch.strip():
        return FailureCategory.WRONG_ANSWER

    return FailureCategory.UNKNOWN


def extract_failure_patterns(results: list[dict]) -> dict:
    """Find common patterns across failed results.

    Analyzes all failed results to identify recurring error messages,
    tool-related patterns, and other commonalities.

    Args:
        results: List of result dictionaries.

    Returns:
        Dictionary with pattern analysis including:
        - common_errors: Most frequent error message prefixes
        - tool_failure_patterns: Tools that fail most frequently
        - failure_by_tool_count: Distribution of failures by number of tool calls
        - high_token_failures: Failures with above-average token usage
        - zero_tool_failures: Failures where no tools were called
    """
    failed_results = [r for r in results if not r.get("resolved")]

    if not failed_results:
        return {
            "common_errors": [],
            "tool_failure_patterns": {},
            "failure_by_tool_count": {},
            "high_token_failures": 0,
            "zero_tool_failures": 0,
        }

    # Collect common error message prefixes (first 80 chars)
    error_counter: Counter[str] = Counter()
    for r in failed_results:
        error = r.get("error", "")
        if error:
            # Normalize: take first 80 chars as a prefix fingerprint
            prefix = error[:80].strip()
            error_counter[prefix] += 1

    # Tool failure patterns
    tool_failure_counter: Counter[str] = Counter()
    for r in failed_results:
        tool_failures = r.get("tool_failures", {})
        for tool_name, count in tool_failures.items():
            tool_failure_counter[tool_name] += count

    # Failure distribution by tool call count
    tool_count_dist: Counter[int] = Counter()
    zero_tool_failures = 0
    for r in failed_results:
        tool_calls = r.get("tool_calls", 0)
        # Also count from tool_usage if tool_calls is not set
        if not tool_calls and r.get("tool_usage"):
            tool_calls = sum(r["tool_usage"].values())
        tool_count_dist[tool_calls] += 1
        if tool_calls == 0:
            zero_tool_failures += 1

    # High token usage failures
    all_tokens = []
    for r in results:
        tokens_in = r.get("tokens_input", 0)
        tokens_out = r.get("tokens_output", 0)
        # Also try nested tokens dict
        tokens_dict = r.get("tokens", {})
        if isinstance(tokens_dict, dict):
            tokens_in = tokens_in or tokens_dict.get("input", 0)
            tokens_out = tokens_out or tokens_dict.get("output", 0)
        all_tokens.append(tokens_in + tokens_out)

    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0

    high_token_failures = 0
    for r in failed_results:
        tokens_in = r.get("tokens_input", 0)
        tokens_out = r.get("tokens_output", 0)
        tokens_dict = r.get("tokens", {})
        if isinstance(tokens_dict, dict):
            tokens_in = tokens_in or tokens_dict.get("input", 0)
            tokens_out = tokens_out or tokens_dict.get("output", 0)
        total = tokens_in + tokens_out
        if avg_tokens > 0 and total > avg_tokens:
            high_token_failures += 1

    return {
        "common_errors": error_counter.most_common(10),
        "tool_failure_patterns": dict(tool_failure_counter.most_common(10)),
        "failure_by_tool_count": dict(sorted(tool_count_dist.items())),
        "high_token_failures": high_token_failures,
        "zero_tool_failures": zero_tool_failures,
    }


def generate_failure_report(results: list[dict]) -> dict:
    """Generate a comprehensive failure analysis report.

    Combines failure categorization, pattern extraction, and per-benchmark
    analysis into a single report dictionary.

    Args:
        results: List of result dictionaries.

    Returns:
        Dictionary with:
        - total_results: Total number of results analyzed
        - total_failures: Number of failed results
        - failure_rate: Overall failure rate (0.0 to 1.0)
        - category_distribution: Count of failures per category
        - category_percentages: Percentage of failures per category
        - common_error_messages: Most frequent error messages
        - failure_by_benchmark: Failure rates grouped by benchmark/instance prefix
        - tool_failure_breakdown: Tool-related failure details
        - patterns: Output from extract_failure_patterns
        - recommendations: List of actionable recommendations
    """
    total = len(results)
    failed = [r for r in results if not r.get("resolved")]
    total_failures = len(failed)
    failure_rate = total_failures / total if total > 0 else 0.0

    # Category distribution
    category_counter: Counter[str] = Counter()
    for r in failed:
        category = categorize_failure(r)
        category_counter[category] += 1

    category_percentages = {}
    for category, count in category_counter.items():
        category_percentages[category] = count / total_failures if total_failures > 0 else 0.0

    # Common error messages (full text, deduplicated)
    error_counter: Counter[str] = Counter()
    for r in failed:
        error = r.get("error", "")
        if error:
            error_counter[error] += 1

    # Failure by benchmark/task prefix
    benchmark_failures: dict[str, dict[str, int]] = {}
    for r in results:
        instance_id = r.get("instance_id", "")
        # Extract benchmark prefix (e.g., "django/django" from "django/django-12345")
        parts = instance_id.rsplit("-", 1)
        prefix = parts[0] if len(parts) > 1 else instance_id

        if prefix not in benchmark_failures:
            benchmark_failures[prefix] = {"total": 0, "failures": 0}
        benchmark_failures[prefix]["total"] += 1
        if not r.get("resolved"):
            benchmark_failures[prefix]["failures"] += 1

    # Calculate failure rates per benchmark
    failure_by_benchmark = {}
    for prefix, counts in benchmark_failures.items():
        failure_by_benchmark[prefix] = {
            "total": counts["total"],
            "failures": counts["failures"],
            "failure_rate": counts["failures"] / counts["total"] if counts["total"] > 0 else 0.0,
        }

    # Sort by failure rate descending
    failure_by_benchmark = dict(
        sorted(failure_by_benchmark.items(), key=lambda x: x[1]["failure_rate"], reverse=True)
    )

    # Tool failure breakdown
    tool_total_calls: Counter[str] = Counter()
    tool_total_failures: Counter[str] = Counter()
    for r in failed:
        tool_usage = r.get("tool_usage", {})
        for tool_name, count in tool_usage.items():
            tool_total_calls[tool_name] += count
        tool_failures = r.get("tool_failures", {})
        for tool_name, count in tool_failures.items():
            tool_total_failures[tool_name] += count

    tool_failure_breakdown = {}
    for tool_name in set(tool_total_calls.keys()) | set(tool_total_failures.keys()):
        calls = tool_total_calls.get(tool_name, 0)
        failures = tool_total_failures.get(tool_name, 0)
        tool_failure_breakdown[tool_name] = {
            "calls": calls,
            "failures": failures,
            "failure_rate": failures / calls if calls > 0 else 0.0,
        }

    # Sort by failure count descending
    tool_failure_breakdown = dict(
        sorted(tool_failure_breakdown.items(), key=lambda x: x[1]["failures"], reverse=True)
    )

    # Extract patterns
    patterns = extract_failure_patterns(results)

    # Generate recommendations
    recommendations = _generate_recommendations(
        category_counter=category_counter,
        total_failures=total_failures,
        patterns=patterns,
        tool_failure_breakdown=tool_failure_breakdown,
    )

    return {
        "total_results": total,
        "total_failures": total_failures,
        "failure_rate": failure_rate,
        "category_distribution": dict(category_counter.most_common()),
        "category_percentages": category_percentages,
        "common_error_messages": error_counter.most_common(10),
        "failure_by_benchmark": failure_by_benchmark,
        "tool_failure_breakdown": tool_failure_breakdown,
        "patterns": patterns,
        "recommendations": recommendations,
    }


def _generate_recommendations(
    category_counter: Counter,
    total_failures: int,
    patterns: dict,
    tool_failure_breakdown: dict,
) -> list[str]:
    """Generate actionable recommendations based on failure analysis.

    Args:
        category_counter: Counter of failure categories.
        total_failures: Total number of failures.
        patterns: Extracted failure patterns.
        tool_failure_breakdown: Tool failure details.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if total_failures == 0:
        return ["All tasks resolved successfully. No improvements needed."]

    # Timeout recommendations
    timeout_count = category_counter.get(FailureCategory.TIMEOUT, 0)
    if timeout_count > 0:
        pct = timeout_count / total_failures * 100
        recommendations.append(
            f"Timeouts account for {pct:.0f}% of failures ({timeout_count} tasks). "
            "Consider increasing the timeout limit or optimizing agent efficiency."
        )

    # No patch recommendations
    no_patch_count = category_counter.get(FailureCategory.NO_PATCH, 0)
    if no_patch_count > 0:
        pct = no_patch_count / total_failures * 100
        recommendations.append(
            f"No patch generated for {pct:.0f}% of failures ({no_patch_count} tasks). "
            "The agent may need better prompting or more context about the codebase."
        )

    # Tool error recommendations
    tool_error_count = category_counter.get(FailureCategory.TOOL_ERROR, 0)
    if tool_error_count > 0:
        pct = tool_error_count / total_failures * 100
        recommendations.append(
            f"Tool errors account for {pct:.0f}% of failures ({tool_error_count} tasks). "
            "Check MCP server stability and tool reliability."
        )

    # Environment error recommendations
    env_error_count = category_counter.get(FailureCategory.ENVIRONMENT_ERROR, 0)
    if env_error_count > 0:
        pct = env_error_count / total_failures * 100
        recommendations.append(
            f"Environment errors account for {pct:.0f}% of failures ({env_error_count} tasks). "
            "Review Docker configuration and resource allocation."
        )

    # Wrong answer recommendations
    wrong_answer_count = category_counter.get(FailureCategory.WRONG_ANSWER, 0)
    if wrong_answer_count > 0:
        pct = wrong_answer_count / total_failures * 100
        recommendations.append(
            f"Wrong answers account for {pct:.0f}% of failures ({wrong_answer_count} tasks). "
            "The agent generates patches but they do not pass tests. "
            "Consider improving the agent's debugging and test validation capabilities."
        )

    # Compilation error recommendations
    compile_count = category_counter.get(FailureCategory.COMPILATION_ERROR, 0)
    if compile_count > 0:
        pct = compile_count / total_failures * 100
        recommendations.append(
            f"Compilation errors account for {pct:.0f}% of failures ({compile_count} tasks). "
            "The agent is generating syntactically invalid code. "
            "Consider adding a syntax check step before submitting patches."
        )

    # Zero tool usage pattern
    zero_tool_count = patterns.get("zero_tool_failures", 0)
    if zero_tool_count > 0:
        recommendations.append(
            f"{zero_tool_count} failed tasks had zero tool calls. "
            "The agent may not be discovering or using available tools."
        )

    # High-failure tools
    for tool_name, stats in tool_failure_breakdown.items():
        if stats["failure_rate"] > 0.3 and stats["calls"] >= 5:
            recommendations.append(
                f"Tool '{tool_name}' has a {stats['failure_rate']:.0%} failure rate "
                f"across {stats['calls']} calls in failed tasks. "
                "Investigate this tool's reliability."
            )

    return recommendations


def format_failure_report(report: dict) -> str:
    """Format a failure analysis report as human-readable text.

    Args:
        report: Report dictionary from generate_failure_report.

    Returns:
        Formatted multi-line string suitable for console output or logging.
    """
    lines = []

    lines.append("=" * 60)
    lines.append("FAILURE ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total Results:  {report['total_results']}")
    lines.append(f"Total Failures: {report['total_failures']}")
    lines.append(f"Failure Rate:   {report['failure_rate']:.1%}")
    lines.append("")

    # Category distribution
    category_dist = report.get("category_distribution", {})
    if category_dist:
        lines.append("FAILURE CATEGORIES")
        lines.append("-" * 40)
        for category, count in category_dist.items():
            pct = report["category_percentages"].get(category, 0.0)
            lines.append(f"  {category:<25} {count:>4}  ({pct:.1%})")
        lines.append("")

    # Common error messages
    common_errors = report.get("common_error_messages", [])
    if common_errors:
        lines.append("COMMON ERROR MESSAGES")
        lines.append("-" * 40)
        for error_msg, count in common_errors[:5]:
            # Truncate long messages
            display_msg = error_msg[:70] + "..." if len(error_msg) > 70 else error_msg
            lines.append(f"  [{count}x] {display_msg}")
        lines.append("")

    # Failure by benchmark
    failure_by_benchmark = report.get("failure_by_benchmark", {})
    if failure_by_benchmark:
        lines.append("FAILURE RATE BY BENCHMARK")
        lines.append("-" * 40)
        for prefix, stats in list(failure_by_benchmark.items())[:10]:
            lines.append(
                f"  {prefix:<35} {stats['failures']}/{stats['total']} ({stats['failure_rate']:.1%})"
            )
        lines.append("")

    # Tool failure breakdown
    tool_breakdown = report.get("tool_failure_breakdown", {})
    failing_tools = {k: v for k, v in tool_breakdown.items() if v["failures"] > 0}
    if failing_tools:
        lines.append("TOOL FAILURES (in failed tasks)")
        lines.append("-" * 40)
        for tool_name, stats in list(failing_tools.items())[:10]:
            lines.append(
                f"  {tool_name:<25} {stats['failures']}/{stats['calls']} failed "
                f"({stats['failure_rate']:.1%})"
            )
        lines.append("")

    # Patterns
    patterns = report.get("patterns", {})
    zero_tool = patterns.get("zero_tool_failures", 0)
    high_token = patterns.get("high_token_failures", 0)
    if zero_tool > 0 or high_token > 0:
        lines.append("PATTERNS")
        lines.append("-" * 40)
        if zero_tool > 0:
            lines.append(f"  Tasks with zero tool calls:       {zero_tool}")
        if high_token > 0:
            lines.append(f"  Failures with high token usage:   {high_token}")
        lines.append("")

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
