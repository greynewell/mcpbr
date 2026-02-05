"""Tests for the custom metrics framework."""

import pytest

from mcpbr.custom_metrics import (
    BUILTIN_METRICS,
    MetricDefinition,
    MetricRegistry,
    _aggregate,
    _compute_accuracy,
    _compute_avg_cost,
    _compute_avg_time,
    _compute_avg_tokens,
    _compute_failure_rate,
    _compute_pass_rate,
    _compute_tool_call_rate,
    compute_metrics,
    create_default_registry,
    validate_metric,
)

# ---------------------------------------------------------------------------
# Sample result data used across tests
# ---------------------------------------------------------------------------

SAMPLE_RESULTS: list[dict] = [
    {
        "resolved": True,
        "tokens": {"input": 100, "output": 400},
        "cost": 0.01,
        "runtime_seconds": 30.0,
        "tool_usage": {"Read": 3, "Write": 1},
    },
    {
        "resolved": False,
        "tokens": {"input": 200, "output": 600},
        "cost": 0.02,
        "runtime_seconds": 60.0,
        "error": "Timeout exceeded",
    },
    {
        "resolved": True,
        "tokens": {"input": 150, "output": 350},
        "cost": 0.015,
        "runtime_seconds": 45.0,
        "tool_usage": {"Read": 2},
    },
]


# ===================================================================
# MetricDefinition
# ===================================================================


class TestMetricDefinition:
    """Tests for the MetricDefinition dataclass."""

    def test_create_with_callable(self) -> None:
        """Test creating a metric with a callable compute_fn."""
        metric = MetricDefinition(
            name="my_metric",
            description="A test metric",
            compute_fn=lambda results: 1.0,
        )
        assert metric.name == "my_metric"
        assert metric.description == "A test metric"
        assert callable(metric.compute_fn)
        assert metric.aggregation == "mean"
        assert metric.higher_is_better is True

    def test_create_with_expression(self) -> None:
        """Test creating a composite metric with a string expression."""
        metric = MetricDefinition(
            name="cost_efficiency",
            description="Pass rate divided by average cost",
            compute_fn="pass_rate / avg_cost",
            aggregation="mean",
            higher_is_better=True,
        )
        assert isinstance(metric.compute_fn, str)
        assert metric.compute_fn == "pass_rate / avg_cost"

    def test_defaults(self) -> None:
        """Test default values for optional fields."""
        metric = MetricDefinition(
            name="m",
            description="d",
            compute_fn=lambda r: 0.0,
        )
        assert metric.aggregation == "mean"
        assert metric.higher_is_better is True

    def test_custom_aggregation(self) -> None:
        """Test setting a non-default aggregation method."""
        metric = MetricDefinition(
            name="m",
            description="d",
            compute_fn=lambda r: 0.0,
            aggregation="max",
            higher_is_better=False,
        )
        assert metric.aggregation == "max"
        assert metric.higher_is_better is False


# ===================================================================
# MetricRegistry
# ===================================================================


class TestMetricRegistry:
    """Tests for the MetricRegistry class."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a metric."""
        registry = MetricRegistry()
        metric = MetricDefinition(
            name="my_metric",
            description="test",
            compute_fn=lambda r: 0.0,
        )
        registry.register(metric)
        assert registry.get("my_metric") is metric

    def test_get_missing_returns_none(self) -> None:
        """Test that looking up a non-existent metric returns None."""
        registry = MetricRegistry()
        assert registry.get("nonexistent") is None

    def test_register_duplicate_raises(self) -> None:
        """Test that registering a duplicate name raises ValueError."""
        registry = MetricRegistry()
        metric = MetricDefinition(name="dup", description="d", compute_fn=lambda r: 0.0)
        registry.register(metric)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(metric)

    def test_list_metrics(self) -> None:
        """Test listing registered metric names in sorted order."""
        registry = MetricRegistry()
        registry.register(MetricDefinition(name="zz", description="d", compute_fn=lambda r: 0.0))
        registry.register(MetricDefinition(name="aa", description="d", compute_fn=lambda r: 0.0))
        registry.register(MetricDefinition(name="mm", description="d", compute_fn=lambda r: 0.0))
        assert registry.list_metrics() == ["aa", "mm", "zz"]

    def test_unregister_existing(self) -> None:
        """Test removing an existing metric returns True."""
        registry = MetricRegistry()
        registry.register(MetricDefinition(name="m", description="d", compute_fn=lambda r: 0.0))
        assert registry.unregister("m") is True
        assert registry.get("m") is None

    def test_unregister_missing(self) -> None:
        """Test removing a non-existent metric returns False."""
        registry = MetricRegistry()
        assert registry.unregister("nope") is False

    def test_contains(self) -> None:
        """Test the 'in' operator on the registry."""
        registry = MetricRegistry()
        registry.register(MetricDefinition(name="m", description="d", compute_fn=lambda r: 0.0))
        assert "m" in registry
        assert "other" not in registry

    def test_len(self) -> None:
        """Test len() on the registry."""
        registry = MetricRegistry()
        assert len(registry) == 0
        registry.register(MetricDefinition(name="a", description="d", compute_fn=lambda r: 0.0))
        registry.register(MetricDefinition(name="b", description="d", compute_fn=lambda r: 0.0))
        assert len(registry) == 2


# ===================================================================
# Built-in compute functions
# ===================================================================


class TestBuiltinComputeFunctions:
    """Tests for individual built-in metric compute functions."""

    def test_accuracy_basic(self) -> None:
        """Test accuracy computation with mixed results."""
        assert _compute_accuracy(SAMPLE_RESULTS) == pytest.approx(2 / 3)

    def test_accuracy_empty(self) -> None:
        """Test accuracy returns 0.0 for empty input."""
        assert _compute_accuracy([]) == 0.0

    def test_accuracy_all_resolved(self) -> None:
        """Test accuracy is 1.0 when all tasks resolved."""
        results = [{"resolved": True}, {"resolved": True}]
        assert _compute_accuracy(results) == 1.0

    def test_accuracy_none_resolved(self) -> None:
        """Test accuracy is 0.0 when no tasks resolved."""
        results = [{"resolved": False}, {"resolved": False}]
        assert _compute_accuracy(results) == 0.0

    def test_pass_rate_matches_accuracy(self) -> None:
        """Test that pass_rate is an alias for accuracy."""
        assert _compute_pass_rate(SAMPLE_RESULTS) == _compute_accuracy(SAMPLE_RESULTS)

    def test_avg_tokens_basic(self) -> None:
        """Test average token computation."""
        # Tokens: 500, 800, 500 -> mean = 600
        assert _compute_avg_tokens(SAMPLE_RESULTS) == pytest.approx(600.0)

    def test_avg_tokens_empty(self) -> None:
        """Test avg_tokens returns 0.0 for empty input."""
        assert _compute_avg_tokens([]) == 0.0

    def test_avg_tokens_missing_tokens(self) -> None:
        """Test avg_tokens handles results without tokens key."""
        results = [{"resolved": True}]
        assert _compute_avg_tokens(results) == 0.0

    def test_avg_cost_basic(self) -> None:
        """Test average cost computation."""
        # Costs: 0.01, 0.02, 0.015 -> mean = 0.015
        assert _compute_avg_cost(SAMPLE_RESULTS) == pytest.approx(0.015)

    def test_avg_cost_empty(self) -> None:
        """Test avg_cost returns 0.0 for empty input."""
        assert _compute_avg_cost([]) == 0.0

    def test_avg_cost_missing_cost(self) -> None:
        """Test avg_cost handles results without cost key."""
        results = [{"resolved": True}, {"resolved": False}]
        assert _compute_avg_cost(results) == 0.0

    def test_avg_time_basic(self) -> None:
        """Test average runtime computation."""
        # Runtimes: 30, 60, 45 -> mean = 45
        assert _compute_avg_time(SAMPLE_RESULTS) == pytest.approx(45.0)

    def test_avg_time_empty(self) -> None:
        """Test avg_time returns 0.0 for empty input."""
        assert _compute_avg_time([]) == 0.0

    def test_avg_time_missing_runtime(self) -> None:
        """Test avg_time handles results without runtime_seconds key."""
        results = [{"resolved": True}]
        assert _compute_avg_time(results) == 0.0

    def test_tool_call_rate_basic(self) -> None:
        """Test tool call rate computation."""
        # 2 out of 3 results have tool_usage
        assert _compute_tool_call_rate(SAMPLE_RESULTS) == pytest.approx(2 / 3)

    def test_tool_call_rate_empty(self) -> None:
        """Test tool_call_rate returns 0.0 for empty input."""
        assert _compute_tool_call_rate([]) == 0.0

    def test_tool_call_rate_none_with_tools(self) -> None:
        """Test tool_call_rate when no results have tool usage."""
        results = [{"resolved": True}, {"resolved": False}]
        assert _compute_tool_call_rate(results) == 0.0

    def test_tool_call_rate_all_with_tools(self) -> None:
        """Test tool_call_rate when all results have tool usage."""
        results = [
            {"tool_usage": {"Read": 1}},
            {"tool_usage": {"Write": 2}},
        ]
        assert _compute_tool_call_rate(results) == 1.0

    def test_failure_rate_basic(self) -> None:
        """Test failure rate computation."""
        # 1 out of 3 has an error
        assert _compute_failure_rate(SAMPLE_RESULTS) == pytest.approx(1 / 3)

    def test_failure_rate_empty(self) -> None:
        """Test failure_rate returns 0.0 for empty input."""
        assert _compute_failure_rate([]) == 0.0

    def test_failure_rate_no_errors(self) -> None:
        """Test failure_rate is 0.0 when no errors are present."""
        results = [{"resolved": True}, {"resolved": True}]
        assert _compute_failure_rate(results) == 0.0

    def test_failure_rate_all_errors(self) -> None:
        """Test failure_rate is 1.0 when every result has an error."""
        results = [{"error": "err1"}, {"error": "err2"}]
        assert _compute_failure_rate(results) == 1.0


# ===================================================================
# Built-in metrics list
# ===================================================================


class TestBuiltinMetrics:
    """Tests for the BUILTIN_METRICS list and default registry."""

    def test_builtin_metrics_count(self) -> None:
        """Test that the expected number of built-in metrics exist."""
        assert len(BUILTIN_METRICS) == 7

    def test_builtin_metric_names(self) -> None:
        """Test that all expected built-in metric names are present."""
        names = {m.name for m in BUILTIN_METRICS}
        expected = {
            "accuracy",
            "pass_rate",
            "avg_tokens",
            "avg_cost",
            "avg_time",
            "tool_call_rate",
            "failure_rate",
        }
        assert names == expected

    def test_default_registry_contains_builtins(self) -> None:
        """Test that create_default_registry includes all built-in metrics."""
        registry = create_default_registry()
        for metric in BUILTIN_METRICS:
            assert metric.name in registry

    def test_default_registry_length(self) -> None:
        """Test the length of the default registry matches builtin count."""
        registry = create_default_registry()
        assert len(registry) == len(BUILTIN_METRICS)


# ===================================================================
# Aggregation helper
# ===================================================================


class TestAggregate:
    """Tests for the _aggregate helper function."""

    def test_mean(self) -> None:
        """Test mean aggregation."""
        assert _aggregate([1.0, 2.0, 3.0], "mean") == pytest.approx(2.0)

    def test_sum(self) -> None:
        """Test sum aggregation."""
        assert _aggregate([1.0, 2.0, 3.0], "sum") == pytest.approx(6.0)

    def test_min(self) -> None:
        """Test min aggregation."""
        assert _aggregate([3.0, 1.0, 2.0], "min") == pytest.approx(1.0)

    def test_max(self) -> None:
        """Test max aggregation."""
        assert _aggregate([1.0, 3.0, 2.0], "max") == pytest.approx(3.0)

    def test_median(self) -> None:
        """Test median aggregation."""
        assert _aggregate([1.0, 3.0, 2.0], "median") == pytest.approx(2.0)

    def test_empty_list(self) -> None:
        """Test aggregation on an empty list returns 0.0."""
        assert _aggregate([], "mean") == 0.0

    def test_unknown_method_raises(self) -> None:
        """Test that an unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            _aggregate([1.0], "unknown")


# ===================================================================
# compute_metrics
# ===================================================================


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_single_builtin_metric(self) -> None:
        """Test computing a single built-in metric."""
        result = compute_metrics(SAMPLE_RESULTS, ["accuracy"])
        assert "accuracy" in result
        assert result["accuracy"] == pytest.approx(2 / 3)

    def test_multiple_builtin_metrics(self) -> None:
        """Test computing multiple built-in metrics at once."""
        result = compute_metrics(
            SAMPLE_RESULTS,
            ["accuracy", "avg_cost", "failure_rate"],
        )
        assert len(result) == 3
        assert result["accuracy"] == pytest.approx(2 / 3)
        assert result["avg_cost"] == pytest.approx(0.015)
        assert result["failure_rate"] == pytest.approx(1 / 3)

    def test_all_builtin_metrics(self) -> None:
        """Test computing all built-in metrics."""
        names = [m.name for m in BUILTIN_METRICS]
        result = compute_metrics(SAMPLE_RESULTS, names)
        assert len(result) == len(BUILTIN_METRICS)
        for name in names:
            assert name in result
            assert isinstance(result[name], float)

    def test_unknown_metric_raises(self) -> None:
        """Test that requesting an unknown metric raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            compute_metrics(SAMPLE_RESULTS, ["nonexistent_metric"])

    def test_empty_results(self) -> None:
        """Test computing metrics on empty results."""
        result = compute_metrics([], ["accuracy", "avg_cost", "failure_rate"])
        assert result["accuracy"] == 0.0
        assert result["avg_cost"] == 0.0
        assert result["failure_rate"] == 0.0

    def test_custom_registry(self) -> None:
        """Test computing metrics with a custom registry."""
        registry = MetricRegistry()
        registry.register(
            MetricDefinition(
                name="always_one",
                description="Always returns 1.0",
                compute_fn=lambda r: 1.0,
            )
        )
        result = compute_metrics(SAMPLE_RESULTS, ["always_one"], registry=registry)
        assert result["always_one"] == 1.0

    def test_composite_metric(self) -> None:
        """Test computing a composite metric from an expression."""
        registry = create_default_registry()
        registry.register(
            MetricDefinition(
                name="cost_efficiency",
                description="Pass rate divided by average cost",
                compute_fn="pass_rate / avg_cost",
                higher_is_better=True,
            )
        )
        result = compute_metrics(
            SAMPLE_RESULTS,
            ["pass_rate", "avg_cost", "cost_efficiency"],
            registry=registry,
        )
        expected_efficiency = result["pass_rate"] / result["avg_cost"]
        assert result["cost_efficiency"] == pytest.approx(expected_efficiency)

    def test_composite_metric_division_by_zero(self) -> None:
        """Test that composite metrics handle division by zero gracefully."""
        registry = MetricRegistry()
        registry.register(
            MetricDefinition(
                name="numerator",
                description="Always 1.0",
                compute_fn=lambda r: 1.0,
            )
        )
        registry.register(
            MetricDefinition(
                name="denominator",
                description="Always 0.0",
                compute_fn=lambda r: 0.0,
            )
        )
        registry.register(
            MetricDefinition(
                name="ratio",
                description="numerator / denominator",
                compute_fn="numerator / denominator",
            )
        )
        result = compute_metrics(
            SAMPLE_RESULTS,
            ["numerator", "denominator", "ratio"],
            registry=registry,
        )
        assert result["ratio"] == 0.0

    def test_composite_metric_invalid_expression(self) -> None:
        """Test that an invalid composite expression raises ValueError."""
        registry = MetricRegistry()
        registry.register(
            MetricDefinition(
                name="bad_expr",
                description="Invalid expression",
                compute_fn="undefined_var + 1",
            )
        )
        with pytest.raises(ValueError, match="Failed to evaluate"):
            compute_metrics(SAMPLE_RESULTS, ["bad_expr"], registry=registry)

    def test_composite_metric_arithmetic(self) -> None:
        """Test composite metric with more complex arithmetic."""
        registry = create_default_registry()
        registry.register(
            MetricDefinition(
                name="weighted_score",
                description="Weighted combination of accuracy and failure rate",
                compute_fn="accuracy * 0.8 + (1 - failure_rate) * 0.2",
                higher_is_better=True,
            )
        )
        result = compute_metrics(
            SAMPLE_RESULTS,
            ["accuracy", "failure_rate", "weighted_score"],
            registry=registry,
        )
        expected = result["accuracy"] * 0.8 + (1 - result["failure_rate"]) * 0.2
        assert result["weighted_score"] == pytest.approx(expected)


# ===================================================================
# validate_metric
# ===================================================================


class TestValidateMetric:
    """Tests for the validate_metric function."""

    def test_valid_callable_metric(self) -> None:
        """Test validation of a valid metric with callable compute_fn."""
        assert (
            validate_metric(
                {
                    "name": "my_metric",
                    "description": "A metric",
                    "compute_fn": lambda r: 0.0,
                }
            )
            is True
        )

    def test_valid_expression_metric(self) -> None:
        """Test validation of a valid metric with string expression compute_fn."""
        assert (
            validate_metric(
                {
                    "name": "composite",
                    "description": "Composite metric",
                    "compute_fn": "accuracy / avg_cost",
                }
            )
            is True
        )

    def test_valid_with_all_fields(self) -> None:
        """Test validation with all optional fields specified."""
        assert (
            validate_metric(
                {
                    "name": "full",
                    "description": "Full definition",
                    "compute_fn": lambda r: 0.0,
                    "aggregation": "max",
                    "higher_is_better": False,
                }
            )
            is True
        )

    def test_missing_name(self) -> None:
        """Test validation fails when name is missing."""
        assert (
            validate_metric(
                {
                    "description": "No name",
                    "compute_fn": lambda r: 0.0,
                }
            )
            is False
        )

    def test_empty_name(self) -> None:
        """Test validation fails when name is an empty string."""
        assert (
            validate_metric(
                {
                    "name": "",
                    "description": "Empty name",
                    "compute_fn": lambda r: 0.0,
                }
            )
            is False
        )

    def test_whitespace_only_name(self) -> None:
        """Test validation fails when name is whitespace-only."""
        assert (
            validate_metric(
                {
                    "name": "   ",
                    "description": "Whitespace name",
                    "compute_fn": lambda r: 0.0,
                }
            )
            is False
        )

    def test_missing_description(self) -> None:
        """Test validation fails when description is missing."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "compute_fn": lambda r: 0.0,
                }
            )
            is False
        )

    def test_non_string_description(self) -> None:
        """Test validation fails when description is not a string."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": 123,
                    "compute_fn": lambda r: 0.0,
                }
            )
            is False
        )

    def test_missing_compute_fn(self) -> None:
        """Test validation fails when compute_fn is missing."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": "d",
                }
            )
            is False
        )

    def test_invalid_compute_fn_type(self) -> None:
        """Test validation fails when compute_fn is neither callable nor string."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": "d",
                    "compute_fn": 42,
                }
            )
            is False
        )

    def test_empty_expression_compute_fn(self) -> None:
        """Test validation fails when compute_fn is an empty string."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": "d",
                    "compute_fn": "",
                }
            )
            is False
        )

    def test_whitespace_expression_compute_fn(self) -> None:
        """Test validation fails when compute_fn is whitespace-only string."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": "d",
                    "compute_fn": "   ",
                }
            )
            is False
        )

    def test_invalid_aggregation(self) -> None:
        """Test validation fails for an unsupported aggregation method."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": "d",
                    "compute_fn": lambda r: 0.0,
                    "aggregation": "invalid",
                }
            )
            is False
        )

    def test_valid_aggregation_methods(self) -> None:
        """Test validation passes for all valid aggregation methods."""
        for method in ("mean", "sum", "min", "max", "median"):
            assert (
                validate_metric(
                    {
                        "name": "m",
                        "description": "d",
                        "compute_fn": lambda r: 0.0,
                        "aggregation": method,
                    }
                )
                is True
            )

    def test_invalid_higher_is_better_type(self) -> None:
        """Test validation fails when higher_is_better is not a bool."""
        assert (
            validate_metric(
                {
                    "name": "m",
                    "description": "d",
                    "compute_fn": lambda r: 0.0,
                    "higher_is_better": "yes",
                }
            )
            is False
        )

    def test_name_not_string(self) -> None:
        """Test validation fails when name is not a string."""
        assert (
            validate_metric(
                {
                    "name": 123,
                    "description": "d",
                    "compute_fn": lambda r: 0.0,
                }
            )
            is False
        )


# ===================================================================
# Integration / end-to-end scenarios
# ===================================================================


class TestIntegrationScenarios:
    """End-to-end integration tests for the custom metrics framework."""

    def test_register_custom_then_compute(self) -> None:
        """Test the full workflow: create registry, register custom, compute."""
        registry = create_default_registry()
        registry.register(
            MetricDefinition(
                name="tokens_per_dollar",
                description="Average tokens per dollar spent",
                compute_fn="avg_tokens / avg_cost",
                higher_is_better=True,
            )
        )

        result = compute_metrics(
            SAMPLE_RESULTS,
            ["avg_tokens", "avg_cost", "tokens_per_dollar"],
            registry=registry,
        )

        assert result["tokens_per_dollar"] == pytest.approx(
            result["avg_tokens"] / result["avg_cost"]
        )

    def test_overwrite_after_unregister(self) -> None:
        """Test unregistering and re-registering a metric with different logic."""
        registry = create_default_registry()
        registry.unregister("accuracy")
        registry.register(
            MetricDefinition(
                name="accuracy",
                description="Custom accuracy that always returns 0.42",
                compute_fn=lambda r: 0.42,
            )
        )

        result = compute_metrics(SAMPLE_RESULTS, ["accuracy"], registry=registry)
        assert result["accuracy"] == pytest.approx(0.42)

    def test_multiple_composite_metrics(self) -> None:
        """Test multiple composite metrics in a single compute call."""
        registry = create_default_registry()
        registry.register(
            MetricDefinition(
                name="cost_efficiency",
                description="pass_rate / avg_cost",
                compute_fn="pass_rate / avg_cost",
            )
        )
        registry.register(
            MetricDefinition(
                name="time_efficiency",
                description="pass_rate / avg_time",
                compute_fn="pass_rate / avg_time",
            )
        )

        result = compute_metrics(
            SAMPLE_RESULTS,
            ["pass_rate", "avg_cost", "avg_time", "cost_efficiency", "time_efficiency"],
            registry=registry,
        )

        assert result["cost_efficiency"] == pytest.approx(result["pass_rate"] / result["avg_cost"])
        assert result["time_efficiency"] == pytest.approx(result["pass_rate"] / result["avg_time"])

    def test_single_result(self) -> None:
        """Test computing metrics with a single result dict."""
        results = [{"resolved": True, "tokens": {"input": 50, "output": 50}, "cost": 0.005}]
        output = compute_metrics(results, ["accuracy", "avg_tokens", "avg_cost"])
        assert output["accuracy"] == 1.0
        assert output["avg_tokens"] == 100.0
        assert output["avg_cost"] == 0.005

    def test_results_without_optional_fields(self) -> None:
        """Test computing metrics when results lack optional fields."""
        results = [{"resolved": True}, {"resolved": False}]
        output = compute_metrics(
            results,
            ["accuracy", "avg_tokens", "avg_cost", "avg_time", "tool_call_rate", "failure_rate"],
        )
        assert output["accuracy"] == 0.5
        assert output["avg_tokens"] == 0.0
        assert output["avg_cost"] == 0.0
        assert output["avg_time"] == 0.0
        assert output["tool_call_rate"] == 0.0
        assert output["failure_rate"] == 0.0

    def test_default_registry_used_when_none(self) -> None:
        """Test that compute_metrics uses the default registry when None is passed."""
        result = compute_metrics(SAMPLE_RESULTS, ["accuracy"])
        assert result["accuracy"] == pytest.approx(2 / 3)
