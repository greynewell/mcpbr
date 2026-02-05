"""Tests for few-shot learning module."""

import pytest

from mcpbr.few_shot import (
    FewShotConfig,
    compute_learning_curve,
    format_few_shot_prompt,
    select_examples,
)

# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

EXAMPLE_POOL = [
    {
        "id": "ex1",
        "category": "math",
        "tags": ["algebra", "linear"],
        "language": "python",
        "difficulty": "easy",
        "input": "Solve x + 2 = 5",
        "output": "x = 3",
    },
    {
        "id": "ex2",
        "category": "math",
        "tags": ["calculus", "integration"],
        "language": "python",
        "difficulty": "hard",
        "input": "Integrate x^2 dx",
        "output": "x^3/3 + C",
    },
    {
        "id": "ex3",
        "category": "code",
        "tags": ["sorting", "algorithms"],
        "language": "python",
        "difficulty": "medium",
        "input": "Implement quicksort",
        "output": "def quicksort(arr): ...",
    },
    {
        "id": "ex4",
        "category": "code",
        "tags": ["search", "algorithms"],
        "language": "javascript",
        "difficulty": "easy",
        "input": "Implement binary search",
        "output": "function binarySearch(arr, target) { ... }",
    },
    {
        "id": "ex5",
        "category": "writing",
        "tags": ["essay", "creative"],
        "language": "english",
        "difficulty": "medium",
        "input": "Write a haiku about code",
        "output": "Lines of silent code / ...",
    },
    {
        "id": "ex6",
        "category": "math",
        "tags": ["algebra", "quadratic"],
        "language": "python",
        "difficulty": "medium",
        "input": "Solve x^2 - 4 = 0",
        "output": "x = 2 or x = -2",
    },
]


QUERY = {
    "id": "q1",
    "category": "math",
    "tags": ["algebra", "linear"],
    "language": "python",
    "difficulty": "easy",
    "input": "Solve 3x - 1 = 8",
}


# ---------------------------------------------------------------------------
# FewShotConfig tests
# ---------------------------------------------------------------------------


class TestFewShotConfig:
    """Tests for FewShotConfig dataclass validation."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FewShotConfig()
        assert config.num_shots == 0
        assert config.selection_strategy == "random"
        assert config.seed is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = FewShotConfig(num_shots=3, selection_strategy="similar", seed=42)
        assert config.num_shots == 3
        assert config.selection_strategy == "similar"
        assert config.seed == 42

    def test_negative_num_shots_raises(self) -> None:
        """Test that negative num_shots raises ValueError."""
        with pytest.raises(ValueError, match="num_shots must be non-negative"):
            FewShotConfig(num_shots=-1)

    def test_invalid_strategy_raises(self) -> None:
        """Test that an invalid selection strategy raises ValueError."""
        with pytest.raises(ValueError, match="selection_strategy must be one of"):
            FewShotConfig(selection_strategy="invalid")

    def test_zero_shots_valid(self) -> None:
        """Test that zero shots is a valid configuration."""
        config = FewShotConfig(num_shots=0)
        assert config.num_shots == 0

    def test_all_strategies_valid(self) -> None:
        """Test that all documented strategies are accepted."""
        for strategy in ("random", "similar", "diverse"):
            config = FewShotConfig(selection_strategy=strategy)
            assert config.selection_strategy == strategy


# ---------------------------------------------------------------------------
# Selection strategy tests
# ---------------------------------------------------------------------------


class TestSelectRandom:
    """Tests for random selection strategy."""

    def test_basic_random_selection(self) -> None:
        """Test that random selection returns the correct number of examples."""
        config = FewShotConfig(num_shots=3, selection_strategy="random", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        assert len(result) == 3

    def test_random_reproducibility_with_seed(self) -> None:
        """Test that the same seed produces the same results."""
        config = FewShotConfig(num_shots=3, selection_strategy="random", seed=123)
        result1 = select_examples(EXAMPLE_POOL, QUERY, config)
        result2 = select_examples(EXAMPLE_POOL, QUERY, config)
        assert result1 == result2

    def test_random_different_seeds_differ(self) -> None:
        """Test that different seeds typically produce different results."""
        config1 = FewShotConfig(num_shots=3, selection_strategy="random", seed=1)
        config2 = FewShotConfig(num_shots=3, selection_strategy="random", seed=999)
        result1 = select_examples(EXAMPLE_POOL, QUERY, config1)
        result2 = select_examples(EXAMPLE_POOL, QUERY, config2)
        # With 6 items choosing 3, different seeds will almost certainly differ
        assert result1 != result2

    def test_random_returns_unique_items(self) -> None:
        """Test that random selection does not return duplicate items."""
        config = FewShotConfig(num_shots=4, selection_strategy="random", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        ids = [ex["id"] for ex in result]
        assert len(ids) == len(set(ids))

    def test_random_num_shots_exceeds_pool(self) -> None:
        """Test that requesting more shots than pool size returns all items."""
        config = FewShotConfig(num_shots=100, selection_strategy="random", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        assert len(result) == len(EXAMPLE_POOL)


class TestSelectSimilar:
    """Tests for similar selection strategy."""

    def test_similar_prefers_matching_category(self) -> None:
        """Test that similar strategy favours items with matching category."""
        config = FewShotConfig(num_shots=3, selection_strategy="similar", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        # QUERY is category=math, so we expect math examples to dominate
        math_count = sum(1 for ex in result if ex.get("category") == "math")
        assert math_count >= 2, f"Expected >= 2 math examples, got {math_count}"

    def test_similar_prefers_matching_tags(self) -> None:
        """Test that similar strategy favours items with matching tags."""
        config = FewShotConfig(num_shots=1, selection_strategy="similar", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        # ex1 shares category=math, tags=algebra+linear, language=python, difficulty=easy
        # It should score highest
        assert result[0]["id"] == "ex1"

    def test_similar_reproducibility(self) -> None:
        """Test that similar strategy is reproducible with a seed."""
        config = FewShotConfig(num_shots=3, selection_strategy="similar", seed=42)
        result1 = select_examples(EXAMPLE_POOL, QUERY, config)
        result2 = select_examples(EXAMPLE_POOL, QUERY, config)
        assert result1 == result2

    def test_similar_with_no_metadata(self) -> None:
        """Test similar strategy with examples lacking metadata fields."""
        plain_pool = [
            {"id": "a", "text": "hello"},
            {"id": "b", "text": "world"},
            {"id": "c", "text": "foo"},
        ]
        plain_query = {"id": "q", "text": "bar"}
        config = FewShotConfig(num_shots=2, selection_strategy="similar", seed=42)
        result = select_examples(plain_pool, plain_query, config)
        # All have score 0 so we just need the right count
        assert len(result) == 2


class TestSelectDiverse:
    """Tests for diverse selection strategy."""

    def test_diverse_covers_categories(self) -> None:
        """Test that diverse strategy selects from different categories."""
        config = FewShotConfig(num_shots=3, selection_strategy="diverse", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        categories = {ex["category"] for ex in result}
        # Pool has 3 categories: math, code, writing. With 3 shots we should
        # get one from each.
        assert len(categories) == 3

    def test_diverse_round_robin(self) -> None:
        """Test round-robin allocation across categories."""
        config = FewShotConfig(num_shots=5, selection_strategy="diverse", seed=42)
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        assert len(result) == 5
        categories = [ex["category"] for ex in result]
        # Each of the 3 categories should have at least 1
        unique_cats = set(categories)
        assert len(unique_cats) == 3

    def test_diverse_reproducibility(self) -> None:
        """Test that diverse strategy is reproducible with a seed."""
        config = FewShotConfig(num_shots=4, selection_strategy="diverse", seed=42)
        result1 = select_examples(EXAMPLE_POOL, QUERY, config)
        result2 = select_examples(EXAMPLE_POOL, QUERY, config)
        assert result1 == result2

    def test_diverse_single_category(self) -> None:
        """Test diverse strategy when pool has only one category."""
        single_cat_pool = [
            {"id": "a", "category": "math"},
            {"id": "b", "category": "math"},
            {"id": "c", "category": "math"},
        ]
        config = FewShotConfig(num_shots=2, selection_strategy="diverse", seed=42)
        result = select_examples(single_cat_pool, QUERY, config)
        assert len(result) == 2

    def test_diverse_no_category_key(self) -> None:
        """Test diverse strategy when examples lack category key."""
        no_cat_pool = [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ]
        config = FewShotConfig(num_shots=2, selection_strategy="diverse", seed=42)
        result = select_examples(no_cat_pool, QUERY, config)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Zero-shot (0-shot) tests
# ---------------------------------------------------------------------------


class TestZeroShot:
    """Tests for zero-shot configurations."""

    def test_zero_shot_returns_empty(self) -> None:
        """Test that 0-shot config returns no examples."""
        config = FewShotConfig(num_shots=0, selection_strategy="random")
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        assert result == []

    def test_zero_shot_with_similar(self) -> None:
        """Test that 0-shot works with similar strategy."""
        config = FewShotConfig(num_shots=0, selection_strategy="similar")
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        assert result == []

    def test_zero_shot_with_diverse(self) -> None:
        """Test that 0-shot works with diverse strategy."""
        config = FewShotConfig(num_shots=0, selection_strategy="diverse")
        result = select_examples(EXAMPLE_POOL, QUERY, config)
        assert result == []

    def test_empty_pool_returns_empty(self) -> None:
        """Test that an empty pool always returns no examples."""
        config = FewShotConfig(num_shots=5, selection_strategy="random", seed=42)
        result = select_examples([], QUERY, config)
        assert result == []


# ---------------------------------------------------------------------------
# Prompt formatting tests
# ---------------------------------------------------------------------------


class TestFormatFewShotPrompt:
    """Tests for prompt formatting."""

    def test_basic_formatting(self) -> None:
        """Test basic prompt formatting with examples."""
        examples = [
            {"input": "2 + 2", "output": "4"},
            {"input": "3 + 3", "output": "6"},
        ]
        query = {"input": "5 + 5"}
        template = "Examples:\n{examples}\n\nQuery:\n{query}"

        result = format_few_shot_prompt(examples, query, template)

        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "2 + 2" in result
        assert "5 + 5" in result

    def test_formatting_with_no_examples(self) -> None:
        """Test formatting when there are no examples (zero-shot)."""
        query = {"input": "5 + 5"}
        template = "Examples:\n{examples}\n\nQuery:\n{query}"

        result = format_few_shot_prompt([], query, template)

        assert "No examples provided." in result
        assert "5 + 5" in result

    def test_formatting_preserves_template(self) -> None:
        """Test that non-placeholder parts of the template are preserved."""
        examples = [{"input": "hello"}]
        query = {"input": "world"}
        template = "PREFIX {examples} MIDDLE {query} SUFFIX"

        result = format_few_shot_prompt(examples, query, template)

        assert result.startswith("PREFIX ")
        assert " MIDDLE " in result
        assert result.endswith(" SUFFIX")

    def test_formatting_single_example(self) -> None:
        """Test formatting with a single example."""
        examples = [{"question": "What is 1+1?", "answer": "2"}]
        query = {"question": "What is 2+2?"}
        template = "{examples}\n---\n{query}"

        result = format_few_shot_prompt(examples, query, template)

        assert "Example 1:" in result
        assert "Example 2:" not in result
        assert "What is 1+1?" in result
        assert "What is 2+2?" in result


# ---------------------------------------------------------------------------
# Learning curve tests
# ---------------------------------------------------------------------------


class TestComputeLearningCurve:
    """Tests for learning curve computation."""

    def test_basic_learning_curve(self) -> None:
        """Test learning curve with increasing accuracy by shot count."""
        results_by_shots = {
            0: [
                {"resolved": False, "cost": 0.01, "tokens": {"input": 100, "output": 50}},
                {"resolved": False, "cost": 0.01, "tokens": {"input": 100, "output": 50}},
            ],
            1: [
                {"resolved": True, "cost": 0.02, "tokens": {"input": 200, "output": 100}},
                {"resolved": False, "cost": 0.02, "tokens": {"input": 200, "output": 100}},
            ],
            3: [
                {"resolved": True, "cost": 0.05, "tokens": {"input": 500, "output": 200}},
                {"resolved": True, "cost": 0.05, "tokens": {"input": 500, "output": 200}},
            ],
        }

        curve = compute_learning_curve(results_by_shots)

        assert curve["shot_counts"] == [0, 1, 3]
        assert curve["accuracy"] == [0.0, 0.5, 1.0]
        assert curve["num_samples"] == [2, 2, 2]
        assert len(curve["avg_cost"]) == 3
        assert len(curve["avg_tokens"]) == 3

    def test_empty_results(self) -> None:
        """Test learning curve with no results."""
        curve = compute_learning_curve({})

        assert curve["shot_counts"] == []
        assert curve["accuracy"] == []
        assert curve["avg_cost"] == []
        assert curve["avg_tokens"] == []
        assert curve["num_samples"] == []

    def test_single_shot_count(self) -> None:
        """Test learning curve with a single shot count."""
        results_by_shots = {
            5: [
                {"resolved": True, "cost": 0.10, "tokens": {"input": 1000, "output": 500}},
            ],
        }

        curve = compute_learning_curve(results_by_shots)

        assert curve["shot_counts"] == [5]
        assert curve["accuracy"] == [1.0]
        assert curve["avg_cost"] == [0.10]
        assert curve["avg_tokens"] == [1500.0]
        assert curve["num_samples"] == [1]

    def test_zero_shot_only(self) -> None:
        """Test learning curve with only zero-shot results."""
        results_by_shots = {
            0: [
                {"resolved": True, "cost": 0.01, "tokens": {"input": 50, "output": 20}},
                {"resolved": False, "cost": 0.01, "tokens": {"input": 50, "output": 20}},
            ],
        }

        curve = compute_learning_curve(results_by_shots)

        assert curve["shot_counts"] == [0]
        assert curve["accuracy"] == [0.5]

    def test_missing_fields_default_to_zero(self) -> None:
        """Test that missing cost and token fields default to zero."""
        results_by_shots = {
            1: [
                {"resolved": True},
                {"resolved": False},
            ],
        }

        curve = compute_learning_curve(results_by_shots)

        assert curve["accuracy"] == [0.5]
        assert curve["avg_cost"] == [0.0]
        assert curve["avg_tokens"] == [0.0]

    def test_empty_results_for_shot_count(self) -> None:
        """Test learning curve handles an empty result list for a shot count."""
        results_by_shots = {
            0: [],
            1: [{"resolved": True, "cost": 0.01, "tokens": {"input": 10, "output": 5}}],
        }

        curve = compute_learning_curve(results_by_shots)

        assert curve["shot_counts"] == [0, 1]
        assert curve["accuracy"] == [0.0, 1.0]
        assert curve["avg_cost"] == [0.0, 0.01]
        assert curve["num_samples"] == [0, 1]

    def test_shot_counts_are_sorted(self) -> None:
        """Test that shot counts in the result are sorted ascending."""
        results_by_shots = {
            10: [{"resolved": True}],
            0: [{"resolved": False}],
            5: [{"resolved": True}],
            2: [{"resolved": False}],
        }

        curve = compute_learning_curve(results_by_shots)

        assert curve["shot_counts"] == [0, 2, 5, 10]


# ---------------------------------------------------------------------------
# Integration-style tests combining multiple components
# ---------------------------------------------------------------------------


class TestFewShotIntegration:
    """Integration tests combining selection and formatting."""

    def test_select_then_format(self) -> None:
        """Test selecting examples then formatting a prompt."""
        config = FewShotConfig(num_shots=2, selection_strategy="random", seed=42)
        examples = select_examples(EXAMPLE_POOL, QUERY, config)

        template = "Given these examples:\n{examples}\n\nSolve:\n{query}"
        prompt = format_few_shot_prompt(examples, QUERY, template)

        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert QUERY["input"] in prompt

    def test_zero_shot_then_format(self) -> None:
        """Test zero-shot selection then formatting."""
        config = FewShotConfig(num_shots=0)
        examples = select_examples(EXAMPLE_POOL, QUERY, config)

        template = "Examples:\n{examples}\n\nSolve:\n{query}"
        prompt = format_few_shot_prompt(examples, QUERY, template)

        assert "No examples provided." in prompt
        assert QUERY["input"] in prompt
