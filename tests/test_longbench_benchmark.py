"""Tests for LongBench benchmark implementation."""

from unittest.mock import MagicMock, patch

import pytest

from mcpbr.benchmarks.base import Benchmark
from mcpbr.benchmarks.longbench import (
    ACCURACY_SUBSETS,
    ALL_SUBSETS,
    EDIT_SIM_SUBSETS,
    F1_SUBSETS,
    ROUGE_SUBSETS,
    SUBSET_TO_CATEGORY,
    LongBenchBenchmark,
)


class TestLongBenchInitialization:
    """Tests for LongBench benchmark initialization."""

    def test_default_initialization(self) -> None:
        """Test default LongBench initialization."""
        benchmark = LongBenchBenchmark()
        assert benchmark.name == "longbench"
        assert benchmark.dataset == "THUDM/LongBench"
        assert benchmark.subset == "hotpotqa"

    def test_custom_dataset(self) -> None:
        """Test LongBench with custom dataset."""
        benchmark = LongBenchBenchmark(dataset="custom/dataset")
        assert benchmark.dataset == "custom/dataset"

    def test_custom_subset(self) -> None:
        """Test LongBench with custom subset."""
        benchmark = LongBenchBenchmark(subset="narrativeqa")
        assert benchmark.subset == "narrativeqa"

    def test_implements_protocol(self) -> None:
        """Test that LongBenchBenchmark implements the Benchmark protocol."""
        benchmark = LongBenchBenchmark()
        assert isinstance(benchmark, Benchmark)
        assert hasattr(benchmark, "name")
        assert hasattr(benchmark, "load_tasks")
        assert hasattr(benchmark, "normalize_task")
        assert hasattr(benchmark, "create_environment")
        assert hasattr(benchmark, "evaluate")
        assert hasattr(benchmark, "get_prebuilt_image")
        assert hasattr(benchmark, "get_prompt_template")


class TestSubsetMapping:
    """Tests for subset-to-category mapping."""

    def test_all_subsets_have_categories(self) -> None:
        """Test that all subsets are mapped to a category."""
        for subset in ALL_SUBSETS:
            assert subset in SUBSET_TO_CATEGORY, f"Subset {subset} missing category mapping"

    def test_single_doc_qa_subsets(self) -> None:
        """Test single_doc_qa category subsets."""
        single_doc = [s for s, c in SUBSET_TO_CATEGORY.items() if c == "single_doc_qa"]
        assert "narrativeqa" in single_doc
        assert "qasper" in single_doc
        assert "multifieldqa_en" in single_doc
        assert "multifieldqa_zh" in single_doc

    def test_multi_doc_qa_subsets(self) -> None:
        """Test multi_doc_qa category subsets."""
        multi_doc = [s for s, c in SUBSET_TO_CATEGORY.items() if c == "multi_doc_qa"]
        assert "hotpotqa" in multi_doc
        assert "2wikimqa" in multi_doc
        assert "musique" in multi_doc
        assert "dureader" in multi_doc

    def test_summarization_subsets(self) -> None:
        """Test summarization category subsets."""
        summ = [s for s, c in SUBSET_TO_CATEGORY.items() if c == "summarization"]
        assert "gov_report" in summ
        assert "qmsum" in summ
        assert "multi_news" in summ
        assert "vcsum" in summ
        assert "samsum" in summ

    def test_few_shot_subsets(self) -> None:
        """Test few_shot category subsets."""
        few = [s for s, c in SUBSET_TO_CATEGORY.items() if c == "few_shot"]
        assert "triviaqa" in few
        assert "trec" in few
        assert "lsht" in few

    def test_synthetic_subsets(self) -> None:
        """Test synthetic category subsets."""
        syn = [s for s, c in SUBSET_TO_CATEGORY.items() if c == "synthetic"]
        assert "passage_count" in syn
        assert "passage_retrieval_en" in syn
        assert "passage_retrieval_zh" in syn

    def test_code_subsets(self) -> None:
        """Test code category subsets."""
        code = [s for s, c in SUBSET_TO_CATEGORY.items() if c == "code"]
        assert "lcc" in code
        assert "repobench-p" in code

    def test_metric_subsets_cover_all(self) -> None:
        """Test that metric subset sets cover all LongBench subsets."""
        all_metric_subsets = F1_SUBSETS | ROUGE_SUBSETS | ACCURACY_SUBSETS | EDIT_SIM_SUBSETS
        for subset in ALL_SUBSETS:
            assert subset in all_metric_subsets, f"Subset {subset} not assigned to any metric set"


class TestResolveSubsets:
    """Tests for subset resolution logic."""

    def test_no_filter_uses_configured_subset(self) -> None:
        """Test that no filter returns the configured subset."""
        benchmark = LongBenchBenchmark(subset="narrativeqa")
        result = benchmark._resolve_subsets(None)
        assert result == ["narrativeqa"]

    def test_filter_single_category(self) -> None:
        """Test filtering by a single category."""
        benchmark = LongBenchBenchmark()
        result = benchmark._resolve_subsets(["single_doc_qa"])
        assert "narrativeqa" in result
        assert "qasper" in result
        assert "multifieldqa_en" in result
        assert "multifieldqa_zh" in result
        assert "hotpotqa" not in result

    def test_filter_multiple_categories(self) -> None:
        """Test filtering by multiple categories."""
        benchmark = LongBenchBenchmark()
        result = benchmark._resolve_subsets(["code", "synthetic"])
        assert "lcc" in result
        assert "repobench-p" in result
        assert "passage_count" in result
        assert "narrativeqa" not in result

    def test_filter_unknown_category_falls_back(self) -> None:
        """Test that unknown category falls back to configured subset."""
        benchmark = LongBenchBenchmark(subset="qasper")
        result = benchmark._resolve_subsets(["nonexistent_category"])
        assert result == ["qasper"]

    def test_filter_case_insensitive(self) -> None:
        """Test that category filter is case-insensitive."""
        benchmark = LongBenchBenchmark()
        result = benchmark._resolve_subsets(["SINGLE_DOC_QA"])
        assert "narrativeqa" in result


class TestLoadTasks:
    """Tests for task loading with mock data."""

    def _make_mock_dataset(self, items: list[dict]) -> MagicMock:
        """Create a mock dataset that supports iteration and indexing."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(items))
        mock_ds.__len__ = MagicMock(return_value=len(items))
        return mock_ds

    @patch("mcpbr.benchmarks.longbench.load_dataset")
    def test_load_tasks_basic(self, mock_load: MagicMock) -> None:
        """Test basic task loading."""
        mock_items = [
            {
                "input": "What is the main theme?",
                "context": "A long document about AI...",
                "answers": ["artificial intelligence"],
                "length": 5000,
                "dataset": "hotpotqa",
                "language": "en",
                "all_classes": None,
                "_id": "abc123",
            },
            {
                "input": "Summarize the key findings.",
                "context": "Research paper text...",
                "answers": ["key finding summary"],
                "length": 8000,
                "dataset": "hotpotqa",
                "language": "en",
                "all_classes": None,
                "_id": "def456",
            },
        ]
        mock_load.return_value = self._make_mock_dataset(mock_items)

        benchmark = LongBenchBenchmark(subset="hotpotqa")
        tasks = benchmark.load_tasks()

        assert len(tasks) == 2
        assert tasks[0]["instance_id"] == "longbench_hotpotqa_0"
        assert tasks[1]["instance_id"] == "longbench_hotpotqa_1"
        assert "problem_statement" in tasks[0]
        assert tasks[0]["ground_truth_answers"] == ["artificial intelligence"]
        assert tasks[0]["task_category"] == "multi_doc_qa"

    @patch("mcpbr.benchmarks.longbench.load_dataset")
    def test_load_tasks_with_sample_size(self, mock_load: MagicMock) -> None:
        """Test task loading with sample size limit."""
        mock_items = [
            {
                "input": f"Question {i}",
                "context": f"Context {i}",
                "answers": [f"Answer {i}"],
                "length": 1000,
                "dataset": "hotpotqa",
                "language": "en",
                "all_classes": None,
                "_id": f"id_{i}",
            }
            for i in range(10)
        ]
        mock_load.return_value = self._make_mock_dataset(mock_items)

        benchmark = LongBenchBenchmark(subset="hotpotqa")
        tasks = benchmark.load_tasks(sample_size=3)

        assert len(tasks) == 3

    @patch("mcpbr.benchmarks.longbench.load_dataset")
    def test_load_tasks_with_task_ids(self, mock_load: MagicMock) -> None:
        """Test task loading with specific task IDs."""
        mock_items = [
            {
                "input": f"Question {i}",
                "context": f"Context {i}",
                "answers": [f"Answer {i}"],
                "length": 1000,
                "dataset": "hotpotqa",
                "language": "en",
                "all_classes": None,
                "_id": f"id_{i}",
            }
            for i in range(5)
        ]
        mock_load.return_value = self._make_mock_dataset(mock_items)

        benchmark = LongBenchBenchmark(subset="hotpotqa")
        tasks = benchmark.load_tasks(task_ids=["longbench_hotpotqa_1", "longbench_hotpotqa_3"])

        assert len(tasks) == 2
        assert tasks[0]["instance_id"] == "longbench_hotpotqa_1"
        assert tasks[1]["instance_id"] == "longbench_hotpotqa_3"

    @patch("mcpbr.benchmarks.longbench.load_dataset")
    def test_load_tasks_with_category_filter(self, mock_load: MagicMock) -> None:
        """Test task loading with category filter loads multiple subsets."""
        mock_items = [
            {
                "input": "Question",
                "context": "Context",
                "answers": ["Answer"],
                "length": 1000,
                "dataset": "lcc",
                "language": "en",
                "all_classes": None,
                "_id": "id_0",
            }
        ]
        mock_load.return_value = self._make_mock_dataset(mock_items)

        benchmark = LongBenchBenchmark()
        benchmark.load_tasks(filter_category=["code"])

        # Should attempt to load both code subsets: lcc and repobench-p
        assert mock_load.call_count == 2

    @patch("mcpbr.benchmarks.longbench.load_dataset")
    def test_load_tasks_handles_load_failure(self, mock_load: MagicMock) -> None:
        """Test that load_tasks gracefully handles dataset load failures."""
        mock_load.side_effect = Exception("Dataset not found")

        benchmark = LongBenchBenchmark(subset="hotpotqa")
        tasks = benchmark.load_tasks()

        assert len(tasks) == 0


class TestNormalizeTask:
    """Tests for task normalization."""

    def test_normalize_task_basic(self) -> None:
        """Test basic task normalization."""
        benchmark = LongBenchBenchmark()
        task = {
            "instance_id": "longbench_hotpotqa_0",
            "input": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris.",
            "answers": ["Paris"],
            "length": 100,
            "dataset": "hotpotqa",
            "language": "en",
            "all_classes": None,
            "_subset": "hotpotqa",
        }

        normalized = benchmark.normalize_task(task)
        assert normalized.task_id == "longbench_hotpotqa_0"
        assert "capital of France" in normalized.problem_statement
        assert "Paris" in normalized.problem_statement  # context included
        assert normalized.repo == "longbench/context"
        assert normalized.commit == "HEAD"
        assert normalized.metadata["language"] == "en"
        assert normalized.metadata["answers"] == ["Paris"]
        assert normalized.metadata["category"] == "multi_doc_qa"

    def test_normalize_task_missing_instance_id(self) -> None:
        """Test normalizing task without instance_id raises error."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "Some question",
            "context": "Some context",
            "answers": ["answer"],
        }

        with pytest.raises(ValueError, match="missing required 'instance_id' field"):
            benchmark.normalize_task(task)

    def test_normalize_task_missing_input_and_context(self) -> None:
        """Test normalizing task without input or context raises error."""
        benchmark = LongBenchBenchmark()
        task = {
            "instance_id": "longbench_test_0",
            "input": "",
            "context": "",
            "answers": ["answer"],
        }

        with pytest.raises(ValueError, match="missing required 'input' or 'context' field"):
            benchmark.normalize_task(task)

    def test_normalize_task_metadata_fields(self) -> None:
        """Test that metadata contains expected fields."""
        benchmark = LongBenchBenchmark()
        task = {
            "instance_id": "longbench_qasper_0",
            "input": "What method was proposed?",
            "context": "We propose a new method...",
            "answers": ["new method"],
            "length": 3000,
            "dataset": "qasper",
            "language": "en",
            "all_classes": None,
            "_subset": "qasper",
        }

        normalized = benchmark.normalize_task(task)
        assert normalized.metadata["context_length"] == 3000
        assert normalized.metadata["dataset"] == "qasper"
        assert normalized.metadata["subset"] == "qasper"
        assert normalized.metadata["category"] == "single_doc_qa"


class TestGenerateProblemStatement:
    """Tests for problem statement generation."""

    def test_qa_task_statement(self) -> None:
        """Test problem statement for QA tasks."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "What happened in the story?",
            "context": "Once upon a time, a hero saved the village.",
            "_subset": "narrativeqa",
        }

        statement = benchmark._generate_problem_statement(task)
        assert "Read the following document(s) carefully" in statement
        assert "answer the question" in statement
        assert "CONTEXT" in statement
        assert "hero saved the village" in statement
        assert "What happened in the story?" in statement

    def test_summarization_task_statement(self) -> None:
        """Test problem statement for summarization tasks."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "Summarize the report.",
            "context": "Government report content here...",
            "_subset": "gov_report",
        }

        statement = benchmark._generate_problem_statement(task)
        assert "concise summary" in statement
        assert "Government report content" in statement

    def test_few_shot_task_statement(self) -> None:
        """Test problem statement for few-shot tasks."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "What is the answer?",
            "context": "Example 1: Q: ... A: ...\nExample 2: Q: ... A: ...",
            "_subset": "triviaqa",
        }

        statement = benchmark._generate_problem_statement(task)
        assert "examples carefully" in statement

    def test_synthetic_task_statement(self) -> None:
        """Test problem statement for synthetic tasks."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "How many passages discuss AI?",
            "context": "Passage 1: ...\nPassage 2: ...",
            "_subset": "passage_count",
        }

        statement = benchmark._generate_problem_statement(task)
        assert "Analyze the provided passages" in statement

    def test_code_task_statement(self) -> None:
        """Test problem statement for code tasks."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "Complete the function.",
            "context": "def foo():\n    pass",
            "_subset": "lcc",
        }

        statement = benchmark._generate_problem_statement(task)
        assert "complete the code" in statement.lower()

    def test_no_context_statement(self) -> None:
        """Test problem statement when no context is provided."""
        benchmark = LongBenchBenchmark()
        task = {
            "input": "What is the answer?",
            "context": "",
            "_subset": "hotpotqa",
        }

        statement = benchmark._generate_problem_statement(task)
        assert "CONTEXT" not in statement
        assert "What is the answer?" in statement


class TestF1Scoring:
    """Tests for F1 score computation."""

    def test_exact_match(self) -> None:
        """Test F1 with exact match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("Paris", "Paris")
        assert score == 1.0

    def test_partial_overlap(self) -> None:
        """Test F1 with partial token overlap."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("the big red fox", "the small red fox")
        # Common: "the", "red", "fox" (3 tokens)
        # Pred: 4 tokens, Ref: 4 tokens
        # Precision = 3/4, Recall = 3/4
        # F1 = 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        # Note: "the" is removed as article, so:
        # Pred: "big red fox" (3), Ref: "small red fox" (3)
        # Common: "red", "fox" (2)
        # P = 2/3, R = 2/3, F1 = 2/3
        assert abs(score - 2 / 3) < 0.01

    def test_no_overlap(self) -> None:
        """Test F1 with no token overlap."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("apple orange", "banana grape")
        assert score == 0.0

    def test_empty_prediction(self) -> None:
        """Test F1 with empty prediction."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("", "some reference")
        assert score == 0.0

    def test_both_empty(self) -> None:
        """Test F1 with both empty strings."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("", "")
        assert score == 1.0

    def test_f1_max_multiple_references(self) -> None:
        """Test maximum F1 across multiple references."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1_max(
            "Paris",
            ["Paris", "London", "Berlin"],
        )
        assert score == 1.0

    def test_f1_max_no_references(self) -> None:
        """Test F1 max with empty reference list."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1_max("Paris", [])
        assert score == 0.0

    def test_normalization_removes_articles(self) -> None:
        """Test that text normalization removes articles."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("the answer is Paris", "answer is Paris")
        assert score == 1.0

    def test_normalization_case_insensitive(self) -> None:
        """Test that text normalization is case-insensitive."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("PARIS", "paris")
        assert score == 1.0

    def test_normalization_removes_punctuation(self) -> None:
        """Test that text normalization removes punctuation."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_f1("Paris, France.", "Paris France")
        assert score == 1.0


class TestRougeLScoring:
    """Tests for ROUGE-L score computation."""

    def test_exact_match(self) -> None:
        """Test ROUGE-L with exact match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_rouge_l(
            "machine learning is great",
            "machine learning is great",
        )
        assert score == 1.0

    def test_partial_subsequence(self) -> None:
        """Test ROUGE-L with partial subsequence match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_rouge_l(
            "machine learning is very great",
            "machine learning is great",
        )
        # After normalization: pred="machine learning is very great" (5 tokens)
        # ref="machine learning is great" (4 tokens)
        # LCS = "machine learning is great" (4)
        # P = 4/5, R = 4/4 = 1.0
        # F = 2 * 0.8 * 1.0 / (0.8 + 1.0) = 1.6/1.8 = 0.889
        assert score > 0.8

    def test_no_overlap(self) -> None:
        """Test ROUGE-L with no overlap."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_rouge_l("apple orange", "banana grape")
        assert score == 0.0

    def test_empty_strings(self) -> None:
        """Test ROUGE-L with empty strings."""
        benchmark = LongBenchBenchmark()
        assert benchmark._compute_rouge_l("", "") == 1.0
        assert benchmark._compute_rouge_l("word", "") == 0.0
        assert benchmark._compute_rouge_l("", "word") == 0.0

    def test_rouge_l_max(self) -> None:
        """Test maximum ROUGE-L across multiple references."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_rouge_l_max(
            "machine learning is great",
            ["deep learning is great", "machine learning is great"],
        )
        assert score == 1.0


class TestLCSLength:
    """Tests for LCS length computation."""

    def test_identical_sequences(self) -> None:
        """Test LCS with identical sequences."""
        benchmark = LongBenchBenchmark()
        assert benchmark._lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_no_common(self) -> None:
        """Test LCS with no common elements."""
        benchmark = LongBenchBenchmark()
        assert benchmark._lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial_common(self) -> None:
        """Test LCS with partial common subsequence."""
        benchmark = LongBenchBenchmark()
        assert benchmark._lcs_length(["a", "b", "c", "d"], ["a", "c", "d"]) == 3

    def test_empty_sequences(self) -> None:
        """Test LCS with empty sequences."""
        benchmark = LongBenchBenchmark()
        assert benchmark._lcs_length([], ["a", "b"]) == 0
        assert benchmark._lcs_length(["a"], []) == 0
        assert benchmark._lcs_length([], []) == 0


class TestClassificationAccuracy:
    """Tests for classification accuracy computation."""

    def test_exact_match(self) -> None:
        """Test classification with exact match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_classification_accuracy("5", ["5"])
        assert score == 1.0

    def test_substring_match(self) -> None:
        """Test classification with substring match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_classification_accuracy(
            "The answer is 5.",
            ["5"],
        )
        assert score == 1.0

    def test_no_match(self) -> None:
        """Test classification with no match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_classification_accuracy("3", ["5"])
        assert score == 0.0

    def test_case_insensitive(self) -> None:
        """Test classification is case-insensitive."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_classification_accuracy("PARIS", ["paris"])
        assert score == 1.0

    def test_multiple_references(self) -> None:
        """Test classification with multiple valid answers."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_classification_accuracy("5", ["3", "5", "7"])
        assert score == 1.0


class TestEditSimilarity:
    """Tests for edit similarity computation."""

    def test_identical_strings(self) -> None:
        """Test edit similarity with identical strings."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_edit_similarity("hello", "hello")
        assert score == 1.0

    def test_completely_different(self) -> None:
        """Test edit similarity with completely different strings."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_edit_similarity("abc", "xyz")
        assert score == 0.0

    def test_partial_similarity(self) -> None:
        """Test edit similarity with partial match."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_edit_similarity("hello", "hallo")
        # Edit distance = 1 (substitute 'e' -> 'a'), max_length = 5
        # Similarity = 1 - 1/5 = 0.8
        assert abs(score - 0.8) < 0.01

    def test_empty_strings(self) -> None:
        """Test edit similarity with empty strings."""
        benchmark = LongBenchBenchmark()
        assert benchmark._compute_edit_similarity("", "") == 1.0
        assert benchmark._compute_edit_similarity("abc", "") == 0.0
        assert benchmark._compute_edit_similarity("", "abc") == 0.0

    def test_edit_similarity_max(self) -> None:
        """Test maximum edit similarity across references."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_edit_similarity_max(
            "hello",
            ["world", "hello", "xyz"],
        )
        assert score == 1.0

    def test_edit_similarity_max_no_references(self) -> None:
        """Test edit similarity max with empty list."""
        benchmark = LongBenchBenchmark()
        score = benchmark._compute_edit_similarity_max("hello", [])
        assert score == 0.0


class TestEvaluate:
    """Tests for evaluation logic."""

    @pytest.mark.asyncio
    async def test_evaluate_f1_task_correct(self) -> None:
        """Test evaluation of F1-based task with correct answer."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "hotpotqa",
            "ground_truth_answers": ["Paris"],
            "all_classes": None,
        }
        result = await benchmark.evaluate(None, task, "Paris")
        assert result["resolved"] is True
        assert result["score"] == 1.0
        assert result["metric"] == "f1"

    @pytest.mark.asyncio
    async def test_evaluate_f1_task_wrong(self) -> None:
        """Test evaluation of F1-based task with wrong answer."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "hotpotqa",
            "ground_truth_answers": ["Paris"],
            "all_classes": None,
        }
        result = await benchmark.evaluate(None, task, "London")
        assert result["resolved"] is False
        assert result["score"] == 0.0
        assert result["metric"] == "f1"

    @pytest.mark.asyncio
    async def test_evaluate_rouge_task(self) -> None:
        """Test evaluation of ROUGE-L based task."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "gov_report",
            "ground_truth_answers": ["government report summarizes key policy findings"],
            "all_classes": None,
        }
        result = await benchmark.evaluate(
            None, task, "government report summarizes key policy findings"
        )
        assert result["resolved"] is True
        assert result["score"] == 1.0
        assert result["metric"] == "rouge_l"

    @pytest.mark.asyncio
    async def test_evaluate_accuracy_task_correct(self) -> None:
        """Test evaluation of accuracy-based task with correct answer."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "trec",
            "ground_truth_answers": ["5"],
            "all_classes": ["1", "2", "3", "4", "5"],
        }
        result = await benchmark.evaluate(None, task, "5")
        assert result["resolved"] is True
        assert result["score"] == 1.0
        assert result["metric"] == "accuracy"

    @pytest.mark.asyncio
    async def test_evaluate_accuracy_task_wrong(self) -> None:
        """Test evaluation of accuracy-based task with wrong answer."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "trec",
            "ground_truth_answers": ["5"],
            "all_classes": ["1", "2", "3", "4", "5"],
        }
        result = await benchmark.evaluate(None, task, "3")
        assert result["resolved"] is False
        assert result["score"] == 0.0
        assert result["metric"] == "accuracy"

    @pytest.mark.asyncio
    async def test_evaluate_edit_sim_task(self) -> None:
        """Test evaluation of edit-similarity based task."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "lcc",
            "ground_truth_answers": ["return x + y"],
            "all_classes": None,
        }
        result = await benchmark.evaluate(None, task, "return x + y")
        assert result["resolved"] is True
        assert result["score"] == 1.0
        assert result["metric"] == "edit_similarity"

    @pytest.mark.asyncio
    async def test_evaluate_no_answers(self) -> None:
        """Test evaluation with no ground truth answers."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "hotpotqa",
            "ground_truth_answers": [],
            "all_classes": None,
        }
        result = await benchmark.evaluate(None, task, "some answer")
        assert result["resolved"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evaluate_returns_subset_info(self) -> None:
        """Test that evaluation returns subset and category information."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "qasper",
            "ground_truth_answers": ["answer"],
            "all_classes": None,
        }
        result = await benchmark.evaluate(None, task, "answer")
        assert result["subset"] == "qasper"
        assert result["category"] == "single_doc_qa"

    @pytest.mark.asyncio
    async def test_evaluate_unknown_subset_defaults_to_f1(self) -> None:
        """Test that unknown subsets default to F1 scoring."""
        benchmark = LongBenchBenchmark()
        task = {
            "_subset": "unknown_subset",
            "ground_truth_answers": ["answer"],
            "all_classes": None,
        }
        result = await benchmark.evaluate(None, task, "answer")
        assert result["metric"] == "f1"


class TestTextNormalization:
    """Tests for text normalization helper."""

    def test_lowercase(self) -> None:
        """Test lowercasing."""
        benchmark = LongBenchBenchmark()
        assert benchmark._normalize_text("HELLO") == "hello"

    def test_remove_articles(self) -> None:
        """Test article removal."""
        benchmark = LongBenchBenchmark()
        result = benchmark._normalize_text("a big the red an apple")
        assert "a " not in f" {result} ".replace("apple", "")
        assert "the " not in result
        assert "an " not in f" {result} ".replace("an", "").replace("apple", "")

    def test_remove_punctuation(self) -> None:
        """Test punctuation removal."""
        benchmark = LongBenchBenchmark()
        assert benchmark._normalize_text("hello, world!") == "hello world"

    def test_collapse_whitespace(self) -> None:
        """Test whitespace collapsing."""
        benchmark = LongBenchBenchmark()
        assert benchmark._normalize_text("hello   world") == "hello world"

    def test_strip(self) -> None:
        """Test stripping leading/trailing whitespace."""
        benchmark = LongBenchBenchmark()
        assert benchmark._normalize_text("  hello  ") == "hello"


class TestGetPrebuiltImage:
    """Tests for pre-built image retrieval."""

    def test_returns_none(self) -> None:
        """Test that LongBench has no pre-built images."""
        benchmark = LongBenchBenchmark()
        assert benchmark.get_prebuilt_image({}) is None
        assert benchmark.get_prebuilt_image({"instance_id": "test"}) is None


class TestGetPromptTemplate:
    """Tests for prompt template."""

    def test_contains_placeholder(self) -> None:
        """Test that prompt template contains problem_statement placeholder."""
        benchmark = LongBenchBenchmark()
        prompt = benchmark.get_prompt_template()
        assert "{problem_statement}" in prompt

    def test_contains_instructions(self) -> None:
        """Test that prompt template contains useful instructions."""
        benchmark = LongBenchBenchmark()
        prompt = benchmark.get_prompt_template()
        assert "context" in prompt.lower()
        assert "answer" in prompt.lower()

    def test_mentions_task_types(self) -> None:
        """Test that prompt template mentions different task types."""
        benchmark = LongBenchBenchmark()
        prompt = benchmark.get_prompt_template()
        assert "QA" in prompt
        assert "summarization" in prompt.lower()
        assert "code" in prompt.lower()
