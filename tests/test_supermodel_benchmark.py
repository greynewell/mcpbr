"""Unit tests for the Supermodel benchmark."""

import json
import tempfile
from pathlib import Path

import pytest

from mcpbr.benchmarks.supermodel.endpoints import (
    ENDPOINT_REGISTRY,
    CircularDepsPlugin,
    DeadCodePlugin,
    ImpactAnalysisPlugin,
    TestCoveragePlugin,
    get_endpoint,
)
from mcpbr.benchmarks.supermodel.endpoints.dead_code import _parse_diff
from mcpbr.benchmarks.supermodel.evaluation import (
    build_comparison_set,
    compute_prf1,
    normalize_name,
    normalize_path,
)

# ---------------------------------------------------------------------------
# Evaluation module tests
# ---------------------------------------------------------------------------


class TestNormalizePath:
    def test_strip_leading_dot_slash(self) -> None:
        assert normalize_path("./src/foo.ts") == "src/foo.ts"

    def test_strip_multiple_leading_dot_slash(self) -> None:
        assert normalize_path("././src/foo.ts") == "src/foo.ts"

    def test_strip_leading_slash(self) -> None:
        assert normalize_path("/src/foo.ts") == "src/foo.ts"

    def test_backslash_to_forward(self) -> None:
        assert normalize_path("src\\utils\\helper.ts") == "src/utils/helper.ts"

    def test_already_normalized(self) -> None:
        assert normalize_path("src/foo.ts") == "src/foo.ts"


class TestNormalizeName:
    def test_strip_whitespace(self) -> None:
        assert normalize_name("  unusedFunc  ") == "unusedFunc"

    def test_already_clean(self) -> None:
        assert normalize_name("myFunction") == "myFunction"


class TestBuildComparisonSet:
    def test_basic_set(self) -> None:
        items = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/b.ts", "name": "bar"},
        ]
        result = build_comparison_set(items)
        assert result == {("src/a.ts", "foo"), ("src/b.ts", "bar")}

    def test_deduplication(self) -> None:
        items = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/a.ts", "name": "foo"},
        ]
        result = build_comparison_set(items)
        assert len(result) == 1

    def test_normalizes_paths(self) -> None:
        items = [
            {"file": "./src/a.ts", "name": "foo"},
            {"file": "src/a.ts", "name": "foo"},
        ]
        result = build_comparison_set(items)
        assert len(result) == 1

    def test_skips_empty_fields(self) -> None:
        items = [
            {"file": "", "name": "foo"},
            {"file": "src/a.ts", "name": ""},
            {"file": "src/b.ts", "name": "bar"},
        ]
        result = build_comparison_set(items)
        assert len(result) == 1
        assert ("src/b.ts", "bar") in result

    def test_custom_key_fields(self) -> None:
        items = [
            {"module_a": "auth", "module_b": "db"},
        ]
        result = build_comparison_set(items, key_fields=("module_a", "module_b"))
        assert ("auth", "db") in result


class TestComputePrf1:
    def test_perfect_match(self) -> None:
        gt = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/b.ts", "name": "bar"},
        ]
        metrics = compute_prf1(gt, gt)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_no_overlap(self) -> None:
        pred = [{"file": "src/a.ts", "name": "foo"}]
        gt = [{"file": "src/b.ts", "name": "bar"}]
        metrics = compute_prf1(pred, gt)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1

    def test_partial_overlap(self) -> None:
        pred = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/c.ts", "name": "baz"},
        ]
        gt = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/b.ts", "name": "bar"},
        ]
        metrics = compute_prf1(pred, gt)
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5

    def test_empty_predictions(self) -> None:
        gt = [{"file": "src/a.ts", "name": "foo"}]
        metrics = compute_prf1([], gt)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["found"] == 0
        assert metrics["expected"] == 1

    def test_empty_ground_truth(self) -> None:
        pred = [{"file": "src/a.ts", "name": "foo"}]
        metrics = compute_prf1(pred, [])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_both_empty(self) -> None:
        metrics = compute_prf1([], [])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_high_precision_low_recall(self) -> None:
        pred = [{"file": "src/a.ts", "name": "foo"}]
        gt = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/b.ts", "name": "bar"},
            {"file": "src/c.ts", "name": "baz"},
        ]
        metrics = compute_prf1(pred, gt)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == round(1 / 3, 3)

    def test_found_and_expected_counts(self) -> None:
        pred = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/c.ts", "name": "baz"},
            {"file": "src/d.ts", "name": "qux"},
        ]
        gt = [
            {"file": "src/a.ts", "name": "foo"},
            {"file": "src/b.ts", "name": "bar"},
        ]
        metrics = compute_prf1(pred, gt)
        assert metrics["found"] == 3
        assert metrics["expected"] == 2


# ---------------------------------------------------------------------------
# Endpoint plugin tests
# ---------------------------------------------------------------------------


class TestEndpointRegistry:
    def test_all_endpoints_registered(self) -> None:
        assert "dead-code" in ENDPOINT_REGISTRY
        assert "impact" in ENDPOINT_REGISTRY
        assert "test-coverage" in ENDPOINT_REGISTRY
        assert "circular-deps" in ENDPOINT_REGISTRY

    def test_get_endpoint_returns_instance(self) -> None:
        ep = get_endpoint("dead-code")
        assert isinstance(ep, DeadCodePlugin)

    def test_get_endpoint_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown endpoint"):
            get_endpoint("nonexistent")


class TestDeadCodePlugin:
    def test_name(self) -> None:
        plugin = DeadCodePlugin()
        assert plugin.name == "dead_code"

    def test_api_path(self) -> None:
        plugin = DeadCodePlugin()
        assert plugin.api_path == "/v1/analysis/dead-code"

    def test_key_fields(self) -> None:
        plugin = DeadCodePlugin()
        assert plugin.key_fields == ("file", "name")

    def test_analysis_filename(self) -> None:
        plugin = DeadCodePlugin()
        assert plugin.analysis_filename == "supermodel_dead_code_analysis.json"

    def test_parse_api_response_filters_framework_names(self) -> None:
        plugin = DeadCodePlugin()
        response = {
            "deadCodeCandidates": [
                {"file": "src/a.ts", "name": "execute", "type": "function"},
                {"file": "src/a.ts", "name": "reallyDead", "type": "function"},
            ]
        }
        result = plugin.parse_api_response(response)
        names = [c["name"] for c in result["deadCodeCandidates"]]
        assert "reallyDead" in names
        assert "execute" not in names

    def test_parse_api_response_filters_test_files(self) -> None:
        plugin = DeadCodePlugin()
        response = {
            "deadCodeCandidates": [
                {"file": "src/a.test.ts", "name": "helper", "type": "function"},
                {"file": "src/b.ts", "name": "actual", "type": "function"},
            ]
        }
        result = plugin.parse_api_response(response)
        names = [c["name"] for c in result["deadCodeCandidates"]]
        assert "actual" in names
        assert "helper" not in names

    def test_parse_api_response_metadata(self) -> None:
        plugin = DeadCodePlugin()
        response = {
            "deadCodeCandidates": [
                {"file": "src/a.ts", "name": "execute", "type": "function"},
                {"file": "src/b.ts", "name": "realFunc", "type": "function"},
            ]
        }
        result = plugin.parse_api_response(response)
        assert result["metadata"]["rawCount"] == 2
        assert result["metadata"]["filteredCount"] == 1


class TestParseDiff:
    """Test the diff parser used for ground truth extraction."""

    def test_typescript_function(self) -> None:
        diff = (
            "diff --git a/src/utils.ts b/src/utils.ts\n"
            "--- a/src/utils.ts\n"
            "+++ b/src/utils.ts\n"
            "-export function unusedHelper() {\n"
        )
        result = _parse_diff(diff, "typescript")
        assert len(result) == 1
        assert result[0].file == "src/utils.ts"
        assert result[0].name == "unusedHelper"
        assert result[0].type == "function"

    def test_typescript_async_function(self) -> None:
        diff = "diff --git a/src/api.ts b/src/api.ts\n-export async function fetchData() {\n"
        result = _parse_diff(diff, "typescript")
        assert len(result) == 1
        assert result[0].name == "fetchData"
        assert result[0].type == "function"

    def test_typescript_class(self) -> None:
        diff = "diff --git a/src/models.ts b/src/models.ts\n-export class OldModel {\n"
        result = _parse_diff(diff, "typescript")
        assert len(result) == 1
        assert result[0].name == "OldModel"
        assert result[0].type == "class"

    def test_typescript_const(self) -> None:
        diff = "diff --git a/src/config.ts b/src/config.ts\n-export const OLD_VALUE = 42;\n"
        result = _parse_diff(diff, "typescript")
        assert len(result) == 1
        assert result[0].name == "OLD_VALUE"
        assert result[0].type == "const"

    def test_python_function(self) -> None:
        diff = "diff --git a/src/utils.py b/src/utils.py\n-def old_helper(x, y):\n"
        result = _parse_diff(diff, "python")
        assert len(result) == 1
        assert result[0].name == "old_helper"
        assert result[0].type == "function"

    def test_python_class(self) -> None:
        diff = "diff --git a/src/models.py b/src/models.py\n-class LegacyModel:\n"
        result = _parse_diff(diff, "python")
        assert len(result) == 1
        assert result[0].name == "LegacyModel"
        assert result[0].type == "class"

    def test_skips_test_files(self) -> None:
        diff = (
            "diff --git a/src/utils.test.ts b/src/utils.test.ts\n-export function testHelper() {\n"
        )
        result = _parse_diff(diff, "typescript")
        assert len(result) == 0

    def test_skips_reserved_names(self) -> None:
        diff = "diff --git a/src/index.ts b/src/index.ts\n-export function default() {\n"
        result = _parse_diff(diff, "typescript")
        assert len(result) == 0

    def test_deduplication(self) -> None:
        diff = (
            "diff --git a/src/a.ts b/src/a.ts\n-export function foo() {\n-export function foo() {\n"
        )
        result = _parse_diff(diff, "typescript")
        assert len(result) == 1

    def test_multiple_files(self) -> None:
        diff = (
            "diff --git a/src/a.ts b/src/a.ts\n"
            "-export function funcA() {\n"
            "diff --git a/src/b.ts b/src/b.ts\n"
            "-export class ClassB {\n"
        )
        result = _parse_diff(diff, "typescript")
        assert len(result) == 2
        files = {d.file for d in result}
        assert files == {"src/a.ts", "src/b.ts"}

    def test_ignores_added_lines(self) -> None:
        diff = "diff --git a/src/a.ts b/src/a.ts\n+export function newFunc() {\n"
        result = _parse_diff(diff, "typescript")
        assert len(result) == 0

    def test_ignores_context_lines(self) -> None:
        diff = "diff --git a/src/a.ts b/src/a.ts\n export function existingFunc() {\n"
        result = _parse_diff(diff, "typescript")
        assert len(result) == 0


class TestImpactAnalysisPlugin:
    def test_name(self) -> None:
        plugin = ImpactAnalysisPlugin()
        assert plugin.name == "impact_analysis"

    def test_key_fields(self) -> None:
        plugin = ImpactAnalysisPlugin()
        assert plugin.key_fields == ("file", "name")


class TestTestCoveragePlugin:
    def test_name(self) -> None:
        plugin = TestCoveragePlugin()
        assert plugin.name == "test_coverage"


class TestCircularDepsPlugin:
    def test_name(self) -> None:
        plugin = CircularDepsPlugin()
        assert plugin.name == "circular_deps"

    def test_key_fields(self) -> None:
        plugin = CircularDepsPlugin()
        assert plugin.key_fields == ("module_a", "module_b")


# ---------------------------------------------------------------------------
# SupermodelBenchmark class tests
# ---------------------------------------------------------------------------


class TestSupermodelBenchmark:
    """Test the SupermodelBenchmark class (no network/Docker required)."""

    def test_import(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        assert SupermodelBenchmark is not None

    def test_init_defaults(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        assert bm.analysis_type == "dead-code"
        assert bm.api_base == "https://staging.api.supermodeltools.com"
        assert bm.resolved_threshold == 0.8
        assert bm.name == "supermodel"
        assert bm.evaluate_without_patch is True

    def test_init_custom_params(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark(
            analysis_type="impact",
            supermodel_api_base="https://example.com",
            resolved_threshold=0.5,
            supermodel_api_timeout=300,
        )
        assert bm.analysis_type == "impact"
        assert bm.api_base == "https://example.com"
        assert bm.resolved_threshold == 0.5
        assert bm.api_timeout == 300

    def test_load_tasks_empty(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark(tasks=[])
        tasks = bm.load_tasks()
        assert tasks == []

    def test_load_tasks_corpus_mode(self) -> None:
        """Test loading tasks from a corpus-mode GT file."""
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        gt_data = [
            {"file": "src/a.ts", "name": "foo", "type": "function"},
            {"file": "src/b.ts", "name": "bar", "type": "class"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_data, f)
            gt_path = f.name

        try:
            bm = SupermodelBenchmark(
                tasks=[
                    {
                        "id": "test_task",
                        "repo": "test/repo",
                        "language": "typescript",
                        "ground_truth_file": gt_path,
                        "description": "Test task",
                    }
                ]
            )
            tasks = bm.load_tasks()
            assert len(tasks) == 1
            assert tasks[0]["instance_id"] == "test_task"
            assert len(tasks[0]["ground_truth"]) == 2
            assert "problem_statement" in tasks[0]
            assert "problem_statement_enhanced" in tasks[0]
            assert "problem_statement_baseline" in tasks[0]
        finally:
            Path(gt_path).unlink()

    def test_load_tasks_filter_by_id(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        gt_data = [{"file": "a.ts", "name": "x", "type": "function"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_data, f)
            gt_path = f.name

        try:
            bm = SupermodelBenchmark(
                tasks=[
                    {
                        "id": "task_a",
                        "repo": "r",
                        "language": "typescript",
                        "ground_truth_file": gt_path,
                    },
                    {
                        "id": "task_b",
                        "repo": "r",
                        "language": "typescript",
                        "ground_truth_file": gt_path,
                    },
                ]
            )
            tasks = bm.load_tasks(task_ids=["task_b"])
            assert len(tasks) == 1
            assert tasks[0]["instance_id"] == "task_b"
        finally:
            Path(gt_path).unlink()

    def test_load_tasks_sample_size(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        gt_data = [{"file": "a.ts", "name": "x", "type": "function"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_data, f)
            gt_path = f.name

        try:
            bm = SupermodelBenchmark(
                tasks=[
                    {
                        "id": f"task_{i}",
                        "repo": "r",
                        "language": "typescript",
                        "ground_truth_file": gt_path,
                    }
                    for i in range(5)
                ]
            )
            tasks = bm.load_tasks(sample_size=2)
            assert len(tasks) == 2
        finally:
            Path(gt_path).unlink()

    def test_normalize_task(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        task = {
            "instance_id": "test_123",
            "problem_statement": "Find dead code",
            "repo": "test/repo",
            "merge_commit": "abc123",
            "language": "typescript",
            "ground_truth": [{"file": "a.ts", "name": "x"}],
        }
        bt = bm.normalize_task(task)
        assert bt.task_id == "test_123"
        assert bt.repo == "test/repo"
        assert bt.commit == "abc123"
        assert bt.metadata["language"] == "typescript"
        assert bt.metadata["ground_truth_count"] == 1

    def test_extract_findings_from_text(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        text = json.dumps(
            {
                "dead_code": [
                    {"file": "src/a.ts", "name": "foo", "type": "function"},
                ],
                "analysis_complete": True,
            }
        )
        findings = bm._extract_findings_from_text(text)
        assert len(findings) == 1
        assert findings[0]["name"] == "foo"

    def test_extract_findings_from_text_no_match(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        findings = bm._extract_findings_from_text("no json here")
        assert findings == []

    def test_get_prebuilt_image(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        assert bm.get_prebuilt_image({}) is None

    def test_get_prompt_template(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        assert bm.get_prompt_template() == "{problem_statement}"

    def test_enhanced_prompt_contains_analysis_file(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        prompt = bm._generate_enhanced_problem_statement({"language": "typescript"})
        assert "supermodel_dead_code_analysis.json" in prompt
        assert "REPORT.json" in prompt

    def test_baseline_prompt_contains_methodology(self) -> None:
        from mcpbr.benchmarks.supermodel import SupermodelBenchmark

        bm = SupermodelBenchmark()
        prompt = bm._generate_baseline_problem_statement({"language": "typescript"})
        assert "REPORT.json" in prompt
        assert "dead code" in prompt.lower()


# ---------------------------------------------------------------------------
# EndpointPlugin base tests
# ---------------------------------------------------------------------------


class TestEndpointPluginBase:
    def test_should_skip_file_test_file(self) -> None:
        from mcpbr.benchmarks.supermodel.endpoints.base import EndpointPlugin

        assert EndpointPlugin.should_skip_file("src/a.test.ts", [r"\.test\."])

    def test_should_skip_file_no_match(self) -> None:
        from mcpbr.benchmarks.supermodel.endpoints.base import EndpointPlugin

        assert not EndpointPlugin.should_skip_file("src/a.ts", [r"\.test\."])

    def test_scope_prompt_with_prefix(self) -> None:
        plugin = DeadCodePlugin()
        result = plugin.scope_prompt("Analyze code", "packages")
        assert "packages" in result

    def test_scope_prompt_without_prefix(self) -> None:
        plugin = DeadCodePlugin()
        result = plugin.scope_prompt("Analyze code", None)
        assert result == "Analyze code"
