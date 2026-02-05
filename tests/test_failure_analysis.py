"""Tests for the failure analysis module."""

import pytest

from mcpbr.failure_analysis import (
    FailureCategory,
    categorize_failure,
    extract_failure_patterns,
    format_failure_report,
    generate_failure_report,
)

# ---------------------------------------------------------------------------
# Helper: build mock result dicts
# ---------------------------------------------------------------------------


def _make_result(
    resolved: bool = False,
    error: str = "",
    patch: str = "",
    status: str = "",
    tool_calls: int = 0,
    tool_usage: dict | None = None,
    tool_failures: dict | None = None,
    tokens_input: int = 0,
    tokens_output: int = 0,
    instance_id: str = "test-instance-1",
) -> dict:
    """Create a mock result dict with sensible defaults."""
    result: dict = {
        "resolved": resolved,
        "error": error,
        "patch": patch,
        "status": status,
        "tool_calls": tool_calls,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "instance_id": instance_id,
    }
    if tool_usage is not None:
        result["tool_usage"] = tool_usage
    if tool_failures is not None:
        result["tool_failures"] = tool_failures
    return result


# ===========================================================================
# categorize_failure tests
# ===========================================================================


class TestCategorizeFailure:
    """Tests for categorize_failure function."""

    def test_resolved_returns_unknown(self) -> None:
        """Resolved results should return UNKNOWN (not a failure)."""
        result = _make_result(resolved=True, patch="--- a/foo\n+++ b/foo\n")
        assert categorize_failure(result) == FailureCategory.UNKNOWN

    def test_timeout_by_status(self) -> None:
        """Status 'timeout' should be categorized as TIMEOUT."""
        result = _make_result(status="timeout")
        assert categorize_failure(result) == FailureCategory.TIMEOUT

    def test_timeout_by_error_message(self) -> None:
        """Error containing 'timed out' should be TIMEOUT."""
        result = _make_result(error="Task timed out after 300s")
        assert categorize_failure(result) == FailureCategory.TIMEOUT

    def test_timeout_keyword_in_error(self) -> None:
        """Error containing 'timeout' should be TIMEOUT."""
        result = _make_result(error="Timeout exceeded waiting for response")
        assert categorize_failure(result) == FailureCategory.TIMEOUT

    def test_no_patch_empty_string(self) -> None:
        """Empty patch string should be categorized as NO_PATCH."""
        result = _make_result(patch="")
        assert categorize_failure(result) == FailureCategory.NO_PATCH

    def test_no_patch_none(self) -> None:
        """None patch should be categorized as NO_PATCH."""
        result = _make_result()
        result["patch"] = None
        assert categorize_failure(result) == FailureCategory.NO_PATCH

    def test_no_patch_whitespace_only(self) -> None:
        """Whitespace-only patch should be categorized as NO_PATCH."""
        result = _make_result(patch="   \n  ")
        assert categorize_failure(result) == FailureCategory.NO_PATCH

    def test_wrong_answer_with_patch(self) -> None:
        """A patch that does not resolve should be WRONG_ANSWER."""
        result = _make_result(patch="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n")
        assert categorize_failure(result) == FailureCategory.WRONG_ANSWER

    def test_runtime_error(self) -> None:
        """Error with traceback keyword should be RUNTIME_ERROR."""
        result = _make_result(error="Traceback (most recent call last): ...")
        assert categorize_failure(result) == FailureCategory.RUNTIME_ERROR

    def test_runtime_error_typeerror(self) -> None:
        """TypeError in error message should be RUNTIME_ERROR."""
        result = _make_result(error="TypeError: unsupported operand type(s)")
        assert categorize_failure(result) == FailureCategory.RUNTIME_ERROR

    def test_runtime_error_attributeerror(self) -> None:
        """AttributeError in error message should be RUNTIME_ERROR."""
        result = _make_result(error="AttributeError: 'NoneType' has no attribute 'foo'")
        assert categorize_failure(result) == FailureCategory.RUNTIME_ERROR

    def test_compilation_error_syntax(self) -> None:
        """SyntaxError in error message should be COMPILATION_ERROR."""
        result = _make_result(error="SyntaxError: invalid syntax at line 42")
        assert categorize_failure(result) == FailureCategory.COMPILATION_ERROR

    def test_compilation_error_import(self) -> None:
        """ImportError should be COMPILATION_ERROR."""
        result = _make_result(error="ImportError: cannot import name 'Foo'")
        assert categorize_failure(result) == FailureCategory.COMPILATION_ERROR

    def test_compilation_error_indentation(self) -> None:
        """Indentation error should be COMPILATION_ERROR."""
        result = _make_result(error="IndentationError: unexpected indent")
        assert categorize_failure(result) == FailureCategory.COMPILATION_ERROR

    def test_compilation_error_module_not_found(self) -> None:
        """ModuleNotFoundError should be COMPILATION_ERROR."""
        result = _make_result(error="ModuleNotFoundError: No module named 'bar'")
        assert categorize_failure(result) == FailureCategory.COMPILATION_ERROR

    def test_test_failure(self) -> None:
        """Test failure error should be TEST_FAILURE."""
        result = _make_result(
            error="AssertionError: expected 42 got 0",
            patch="--- a/fix.py\n+++ b/fix.py\n",
        )
        assert categorize_failure(result) == FailureCategory.TEST_FAILURE

    def test_test_failure_fail_to_pass(self) -> None:
        """fail_to_pass error should be TEST_FAILURE."""
        result = _make_result(
            error="fail_to_pass tests did not pass",
            patch="--- a/fix.py\n+++ b/fix.py\n",
        )
        assert categorize_failure(result) == FailureCategory.TEST_FAILURE

    def test_tool_error_mcp(self) -> None:
        """MCP-related error should be TOOL_ERROR."""
        result = _make_result(error="MCP server connection failed")
        assert categorize_failure(result) == FailureCategory.TOOL_ERROR

    def test_tool_error_by_high_failure_rate(self) -> None:
        """High tool failure rate should be categorized as TOOL_ERROR."""
        result = _make_result(
            tool_usage={"Read": 10, "Write": 10},
            tool_failures={"Read": 8, "Write": 6},
        )
        assert categorize_failure(result) == FailureCategory.TOOL_ERROR

    def test_environment_error_docker(self) -> None:
        """Docker error should be ENVIRONMENT_ERROR."""
        result = _make_result(error="Docker container failed to start")
        assert categorize_failure(result) == FailureCategory.ENVIRONMENT_ERROR

    def test_environment_error_permission(self) -> None:
        """Permission denied should be ENVIRONMENT_ERROR."""
        result = _make_result(error="Permission denied: /testbed/file.py")
        assert categorize_failure(result) == FailureCategory.ENVIRONMENT_ERROR

    def test_environment_error_network(self) -> None:
        """Network error should be ENVIRONMENT_ERROR."""
        result = _make_result(error="Network unreachable: connection refused")
        assert categorize_failure(result) == FailureCategory.ENVIRONMENT_ERROR

    def test_environment_error_disk(self) -> None:
        """Disk space error should be ENVIRONMENT_ERROR."""
        result = _make_result(error="No space left on device")
        assert categorize_failure(result) == FailureCategory.ENVIRONMENT_ERROR

    def test_unknown_for_generic_error_no_patch(self) -> None:
        """Generic error without patch should be NO_PATCH."""
        result = _make_result(error="Something unexpected happened")
        assert categorize_failure(result) == FailureCategory.NO_PATCH

    def test_unknown_for_generic_error_with_patch(self) -> None:
        """Generic error with a valid patch should be WRONG_ANSWER."""
        result = _make_result(
            error="Something unexpected happened",
            patch="--- a/f.py\n+++ b/f.py\n",
        )
        assert categorize_failure(result) == FailureCategory.WRONG_ANSWER


# ===========================================================================
# extract_failure_patterns tests
# ===========================================================================


class TestExtractFailurePatterns:
    """Tests for extract_failure_patterns function."""

    def test_no_failures(self) -> None:
        """All resolved results should return empty patterns."""
        results = [
            _make_result(resolved=True, instance_id="t-1"),
            _make_result(resolved=True, instance_id="t-2"),
        ]
        patterns = extract_failure_patterns(results)
        assert patterns["common_errors"] == []
        assert patterns["tool_failure_patterns"] == {}
        assert patterns["high_token_failures"] == 0
        assert patterns["zero_tool_failures"] == 0

    def test_common_errors_ranked(self) -> None:
        """Common errors should be ranked by frequency."""
        results = [
            _make_result(error="Timeout exceeded", instance_id="t-1"),
            _make_result(error="Timeout exceeded", instance_id="t-2"),
            _make_result(error="File not found", instance_id="t-3"),
        ]
        patterns = extract_failure_patterns(results)
        assert len(patterns["common_errors"]) == 2
        assert patterns["common_errors"][0][0] == "Timeout exceeded"
        assert patterns["common_errors"][0][1] == 2
        assert patterns["common_errors"][1][0] == "File not found"
        assert patterns["common_errors"][1][1] == 1

    def test_tool_failure_patterns(self) -> None:
        """Tool failure patterns should aggregate across results."""
        results = [
            _make_result(
                tool_usage={"Read": 5},
                tool_failures={"Read": 2},
                instance_id="t-1",
            ),
            _make_result(
                tool_usage={"Read": 3, "Write": 4},
                tool_failures={"Read": 1, "Write": 3},
                instance_id="t-2",
            ),
        ]
        patterns = extract_failure_patterns(results)
        assert patterns["tool_failure_patterns"]["Read"] == 3
        assert patterns["tool_failure_patterns"]["Write"] == 3

    def test_zero_tool_failures_count(self) -> None:
        """Results with zero tool calls should be counted."""
        results = [
            _make_result(tool_calls=0, instance_id="t-1"),
            _make_result(tool_calls=5, instance_id="t-2"),
            _make_result(tool_calls=0, instance_id="t-3"),
        ]
        patterns = extract_failure_patterns(results)
        assert patterns["zero_tool_failures"] == 2

    def test_high_token_failures(self) -> None:
        """Failures with above-average token usage should be counted."""
        results = [
            _make_result(resolved=True, tokens_input=100, tokens_output=100, instance_id="t-1"),
            _make_result(tokens_input=500, tokens_output=500, instance_id="t-2"),
            _make_result(tokens_input=50, tokens_output=50, instance_id="t-3"),
        ]
        # avg = (200 + 1000 + 100) / 3 = 433.3
        # failed: t-2 has 1000 > 433.3, t-3 has 100 < 433.3
        patterns = extract_failure_patterns(results)
        assert patterns["high_token_failures"] == 1

    def test_failure_by_tool_count_distribution(self) -> None:
        """Tool count distribution should be tracked."""
        results = [
            _make_result(tool_calls=0, instance_id="t-1"),
            _make_result(tool_calls=5, instance_id="t-2"),
            _make_result(tool_calls=5, instance_id="t-3"),
            _make_result(tool_calls=10, instance_id="t-4"),
        ]
        patterns = extract_failure_patterns(results)
        assert patterns["failure_by_tool_count"][0] == 1
        assert patterns["failure_by_tool_count"][5] == 2
        assert patterns["failure_by_tool_count"][10] == 1

    def test_tool_usage_fallback_for_tool_calls(self) -> None:
        """Tool calls should be computed from tool_usage when tool_calls is 0."""
        results = [
            _make_result(
                tool_calls=0,
                tool_usage={"Read": 3, "Write": 2},
                instance_id="t-1",
            ),
        ]
        patterns = extract_failure_patterns(results)
        # tool_calls was 0 but tool_usage sums to 5
        assert 5 in patterns["failure_by_tool_count"]


# ===========================================================================
# generate_failure_report tests
# ===========================================================================


class TestGenerateFailureReport:
    """Tests for generate_failure_report function."""

    def test_empty_results(self) -> None:
        """Empty results should produce a report with zero counts."""
        report = generate_failure_report([])
        assert report["total_results"] == 0
        assert report["total_failures"] == 0
        assert report["failure_rate"] == 0.0
        assert report["category_distribution"] == {}
        # With zero failures, the "all resolved" recommendation is generated
        assert report["recommendations"] == [
            "All tasks resolved successfully. No improvements needed."
        ]

    def test_all_resolved(self) -> None:
        """All resolved results should produce zero failures."""
        results = [
            _make_result(resolved=True, instance_id="t-1"),
            _make_result(resolved=True, instance_id="t-2"),
        ]
        report = generate_failure_report(results)
        assert report["total_results"] == 2
        assert report["total_failures"] == 0
        assert report["failure_rate"] == 0.0
        assert report["recommendations"] == [
            "All tasks resolved successfully. No improvements needed."
        ]

    def test_mixed_results_failure_rate(self) -> None:
        """Failure rate should be computed correctly."""
        results = [
            _make_result(resolved=True, instance_id="t-1"),
            _make_result(resolved=False, error="Timeout", status="timeout", instance_id="t-2"),
            _make_result(resolved=False, instance_id="t-3"),
        ]
        report = generate_failure_report(results)
        assert report["total_results"] == 3
        assert report["total_failures"] == 2
        assert report["failure_rate"] == pytest.approx(2 / 3, rel=1e-4)

    def test_category_distribution(self) -> None:
        """Category distribution should reflect all failure types."""
        results = [
            _make_result(status="timeout", instance_id="t-1"),
            _make_result(status="timeout", instance_id="t-2"),
            _make_result(error="Docker container crashed", instance_id="t-3"),
            _make_result(
                patch="--- a/f.py\n+++ b/f.py\n",
                instance_id="t-4",
            ),
        ]
        report = generate_failure_report(results)
        dist = report["category_distribution"]
        assert dist[FailureCategory.TIMEOUT] == 2
        assert dist[FailureCategory.ENVIRONMENT_ERROR] == 1
        assert dist[FailureCategory.WRONG_ANSWER] == 1

    def test_category_percentages(self) -> None:
        """Category percentages should sum to 1.0."""
        results = [
            _make_result(status="timeout", instance_id="t-1"),
            _make_result(error="Docker container crashed", instance_id="t-2"),
        ]
        report = generate_failure_report(results)
        pct_sum = sum(report["category_percentages"].values())
        assert pct_sum == pytest.approx(1.0, rel=1e-4)

    def test_common_error_messages(self) -> None:
        """Common error messages should be tracked."""
        results = [
            _make_result(error="Timeout exceeded", instance_id="t-1"),
            _make_result(error="Timeout exceeded", instance_id="t-2"),
            _make_result(error="File not found", instance_id="t-3"),
        ]
        report = generate_failure_report(results)
        errors = report["common_error_messages"]
        assert len(errors) >= 2
        assert errors[0] == ("Timeout exceeded", 2)

    def test_failure_by_benchmark(self) -> None:
        """Failures should be grouped by benchmark prefix."""
        results = [
            _make_result(resolved=True, instance_id="django/django-1001"),
            _make_result(resolved=False, instance_id="django/django-1002"),
            _make_result(resolved=False, instance_id="sympy/sympy-2001"),
            _make_result(resolved=False, instance_id="sympy/sympy-2002"),
        ]
        report = generate_failure_report(results)
        by_bench = report["failure_by_benchmark"]
        assert "django/django" in by_bench
        assert by_bench["django/django"]["total"] == 2
        assert by_bench["django/django"]["failures"] == 1
        assert by_bench["django/django"]["failure_rate"] == pytest.approx(0.5)
        assert "sympy/sympy" in by_bench
        assert by_bench["sympy/sympy"]["failure_rate"] == pytest.approx(1.0)

    def test_tool_failure_breakdown(self) -> None:
        """Tool failure breakdown should aggregate across failed tasks."""
        results = [
            _make_result(
                tool_usage={"Read": 10, "Write": 5},
                tool_failures={"Write": 2},
                instance_id="t-1",
            ),
            _make_result(
                tool_usage={"Read": 8},
                tool_failures={"Read": 3},
                instance_id="t-2",
            ),
        ]
        report = generate_failure_report(results)
        breakdown = report["tool_failure_breakdown"]
        assert "Read" in breakdown
        assert breakdown["Read"]["failures"] == 3
        assert "Write" in breakdown
        assert breakdown["Write"]["failures"] == 2

    def test_recommendations_timeout(self) -> None:
        """Timeout failures should generate a recommendation."""
        results = [
            _make_result(status="timeout", instance_id="t-1"),
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("timeout" in r.lower() for r in recs)

    def test_recommendations_no_patch(self) -> None:
        """No-patch failures should generate a recommendation."""
        results = [
            _make_result(instance_id="t-1"),  # no patch, no error
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("no patch" in r.lower() for r in recs)

    def test_recommendations_tool_error(self) -> None:
        """Tool errors should generate a recommendation."""
        results = [
            _make_result(error="MCP server connection lost", instance_id="t-1"),
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("tool" in r.lower() for r in recs)

    def test_recommendations_wrong_answer(self) -> None:
        """Wrong answer failures should generate a recommendation."""
        results = [
            _make_result(
                patch="--- a/f.py\n+++ b/f.py\n",
                instance_id="t-1",
            ),
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("wrong answer" in r.lower() for r in recs)

    def test_recommendations_environment_error(self) -> None:
        """Environment errors should generate a recommendation."""
        results = [
            _make_result(error="Docker container crashed", instance_id="t-1"),
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("environment" in r.lower() for r in recs)

    def test_recommendations_compilation_error(self) -> None:
        """Compilation errors should generate a recommendation."""
        results = [
            _make_result(error="SyntaxError: invalid syntax", instance_id="t-1"),
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("compilation" in r.lower() for r in recs)

    def test_patterns_included(self) -> None:
        """Patterns should be included in the report."""
        results = [
            _make_result(error="Some error", instance_id="t-1"),
        ]
        report = generate_failure_report(results)
        assert "patterns" in report
        assert "common_errors" in report["patterns"]

    def test_large_result_set(self) -> None:
        """Report should handle large result sets without errors."""
        results = []
        for i in range(100):
            if i < 30:
                results.append(_make_result(resolved=True, instance_id=f"proj/repo-{i}"))
            elif i < 50:
                results.append(_make_result(status="timeout", instance_id=f"proj/repo-{i}"))
            elif i < 70:
                results.append(
                    _make_result(
                        patch="--- a/f.py\n+++ b/f.py\n",
                        instance_id=f"proj/repo-{i}",
                    )
                )
            else:
                results.append(
                    _make_result(
                        error="Docker container failed",
                        instance_id=f"proj/repo-{i}",
                    )
                )

        report = generate_failure_report(results)
        assert report["total_results"] == 100
        assert report["total_failures"] == 70
        assert report["failure_rate"] == pytest.approx(0.7, rel=1e-4)
        assert len(report["recommendations"]) > 0

    def test_recommendations_high_failure_rate_tool(self) -> None:
        """Tools with high failure rates should generate recommendations."""
        results = [
            _make_result(
                tool_usage={"BadTool": 10},
                tool_failures={"BadTool": 8},
                instance_id=f"t-{i}",
            )
            for i in range(3)
        ]
        report = generate_failure_report(results)
        recs = report["recommendations"]
        assert any("badtool" in r.lower() for r in recs)


# ===========================================================================
# format_failure_report tests
# ===========================================================================


class TestFormatFailureReport:
    """Tests for format_failure_report function."""

    def test_basic_formatting(self) -> None:
        """Report should contain key section headers."""
        results = [
            _make_result(status="timeout", instance_id="django/django-1001"),
            _make_result(error="Docker container crashed", instance_id="sympy/sympy-2001"),
        ]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        assert "FAILURE ANALYSIS REPORT" in text
        assert "OVERVIEW" in text
        assert "FAILURE CATEGORIES" in text
        assert "RECOMMENDATIONS" in text

    def test_empty_report_formatting(self) -> None:
        """Empty report should still produce valid output."""
        report = generate_failure_report([])
        text = format_failure_report(report)

        assert "FAILURE ANALYSIS REPORT" in text
        assert "Total Results:  0" in text
        assert "Total Failures: 0" in text

    def test_all_resolved_formatting(self) -> None:
        """All-resolved report should show 0% failure rate."""
        results = [_make_result(resolved=True, instance_id="t-1")]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        assert "Failure Rate:   0.0%" in text

    def test_error_messages_in_output(self) -> None:
        """Common error messages should appear in formatted output."""
        results = [
            _make_result(error="Timeout exceeded", instance_id="t-1"),
            _make_result(error="Timeout exceeded", instance_id="t-2"),
        ]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        assert "COMMON ERROR MESSAGES" in text
        assert "Timeout exceeded" in text

    def test_benchmark_section_in_output(self) -> None:
        """Benchmark failure section should appear when data is present."""
        results = [
            _make_result(instance_id="django/django-1001"),
            _make_result(instance_id="django/django-1002"),
        ]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        assert "FAILURE RATE BY BENCHMARK" in text
        assert "django/django" in text

    def test_tool_failures_in_output(self) -> None:
        """Tool failures should appear in formatted output."""
        results = [
            _make_result(
                tool_usage={"Write": 10},
                tool_failures={"Write": 4},
                instance_id="t-1",
            ),
        ]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        assert "TOOL FAILURES" in text
        assert "Write" in text

    def test_long_error_truncation(self) -> None:
        """Very long error messages should be truncated in output."""
        long_error = "A" * 200
        results = [_make_result(error=long_error, instance_id="t-1")]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        # The formatted output should not contain the full 200-char message
        assert "A" * 200 not in text
        assert "..." in text

    def test_patterns_section_shown(self) -> None:
        """Patterns section should appear when there are zero-tool failures."""
        results = [
            _make_result(tool_calls=0, instance_id="t-1"),
        ]
        report = generate_failure_report(results)
        text = format_failure_report(report)

        assert "PATTERNS" in text
        assert "zero tool calls" in text


# ===========================================================================
# FailureCategory enum tests
# ===========================================================================


class TestFailureCategory:
    """Tests for the FailureCategory enum."""

    def test_all_categories_are_strings(self) -> None:
        """All categories should be valid strings."""
        for category in FailureCategory:
            assert isinstance(category, str)
            assert len(category) > 0

    def test_expected_categories_exist(self) -> None:
        """All expected categories should exist."""
        expected = [
            "timeout",
            "no_patch",
            "wrong_answer",
            "runtime_error",
            "compilation_error",
            "test_failure",
            "tool_error",
            "environment_error",
            "unknown",
        ]
        actual = [c.value for c in FailureCategory]
        for exp in expected:
            assert exp in actual, f"Missing expected category: {exp}"

    def test_category_count(self) -> None:
        """There should be exactly 9 categories."""
        assert len(FailureCategory) == 9
