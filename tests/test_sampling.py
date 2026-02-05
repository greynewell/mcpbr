"""Tests for sampling strategies module."""

import pytest

from mcpbr.sampling import SamplingStrategy, sample_tasks


def _make_tasks(n: int, category_cycle: list[str] | None = None) -> list[dict]:
    """Create a list of test task dictionaries.

    Args:
        n: Number of tasks to create.
        category_cycle: If provided, cycle through these categories assigning them
            to the 'category' field.

    Returns:
        List of task dictionaries with 'instance_id' and optionally 'category'.
    """
    tasks = []
    for i in range(n):
        task: dict = {"instance_id": f"task_{i}", "index": i}
        if category_cycle:
            task["category"] = category_cycle[i % len(category_cycle)]
        tasks.append(task)
    return tasks


class TestSamplingStrategy:
    """Tests for the SamplingStrategy enum."""

    def test_enum_values(self) -> None:
        """Test that all expected strategies exist with correct string values."""
        assert SamplingStrategy.SEQUENTIAL.value == "sequential"
        assert SamplingStrategy.RANDOM.value == "random"
        assert SamplingStrategy.STRATIFIED.value == "stratified"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string values."""
        assert SamplingStrategy("sequential") == SamplingStrategy.SEQUENTIAL
        assert SamplingStrategy("random") == SamplingStrategy.RANDOM
        assert SamplingStrategy("stratified") == SamplingStrategy.STRATIFIED

    def test_invalid_strategy_raises(self) -> None:
        """Test that invalid strategy string raises ValueError."""
        with pytest.raises(ValueError):
            SamplingStrategy("invalid")


class TestSequentialSampling:
    """Tests for SEQUENTIAL sampling strategy (backward compatible)."""

    def test_basic_sequential(self) -> None:
        """Test that sequential returns first N tasks."""
        tasks = _make_tasks(10)
        result = sample_tasks(tasks, sample_size=5, strategy=SamplingStrategy.SEQUENTIAL)
        assert len(result) == 5
        assert [t["instance_id"] for t in result] == [
            "task_0",
            "task_1",
            "task_2",
            "task_3",
            "task_4",
        ]

    def test_sequential_is_default(self) -> None:
        """Test that SEQUENTIAL is the default strategy."""
        tasks = _make_tasks(10)
        result = sample_tasks(tasks, sample_size=3)
        assert len(result) == 3
        assert result[0]["instance_id"] == "task_0"
        assert result[1]["instance_id"] == "task_1"
        assert result[2]["instance_id"] == "task_2"

    def test_sequential_full_dataset(self) -> None:
        """Test sequential with sample_size equal to dataset size."""
        tasks = _make_tasks(5)
        result = sample_tasks(tasks, sample_size=5, strategy=SamplingStrategy.SEQUENTIAL)
        assert len(result) == 5

    def test_sequential_preserves_order(self) -> None:
        """Test that sequential preserves original order."""
        tasks = _make_tasks(20)
        result = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.SEQUENTIAL)
        for i, task in enumerate(result):
            assert task["index"] == i


class TestRandomSampling:
    """Tests for RANDOM sampling strategy."""

    def test_basic_random(self) -> None:
        """Test that random returns correct number of tasks."""
        tasks = _make_tasks(100)
        result = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        assert len(result) == 10

    def test_random_no_duplicates(self) -> None:
        """Test that random sampling does not produce duplicates."""
        tasks = _make_tasks(100)
        result = sample_tasks(tasks, sample_size=50, strategy=SamplingStrategy.RANDOM, seed=42)
        ids = [t["instance_id"] for t in result]
        assert len(ids) == len(set(ids))

    def test_random_with_seed_is_reproducible(self) -> None:
        """Test that same seed produces same results."""
        tasks = _make_tasks(100)
        result1 = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        result2 = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        assert [t["instance_id"] for t in result1] == [t["instance_id"] for t in result2]

    def test_random_different_seeds_differ(self) -> None:
        """Test that different seeds produce different results."""
        tasks = _make_tasks(100)
        result1 = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        result2 = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=123)
        ids1 = [t["instance_id"] for t in result1]
        ids2 = [t["instance_id"] for t in result2]
        # Extremely unlikely to be identical with different seeds on 100 items
        assert ids1 != ids2

    def test_random_without_seed_works(self) -> None:
        """Test that random sampling without a seed does not raise."""
        tasks = _make_tasks(50)
        result = sample_tasks(tasks, sample_size=5, strategy=SamplingStrategy.RANDOM, seed=None)
        assert len(result) == 5

    def test_random_not_sequential(self) -> None:
        """Test that random sampling is not just taking first N (with known seed)."""
        tasks = _make_tasks(100)
        result = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        sequential = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.SEQUENTIAL)
        # With seed=42 and 100 tasks, random should differ from sequential
        assert [t["instance_id"] for t in result] != [t["instance_id"] for t in sequential]

    def test_random_all_tasks_are_from_original(self) -> None:
        """Test that all randomly sampled tasks exist in the original list."""
        tasks = _make_tasks(50)
        original_ids = {t["instance_id"] for t in tasks}
        result = sample_tasks(tasks, sample_size=20, strategy=SamplingStrategy.RANDOM, seed=7)
        for task in result:
            assert task["instance_id"] in original_ids


class TestStratifiedSampling:
    """Tests for STRATIFIED sampling strategy."""

    def test_basic_stratified(self) -> None:
        """Test basic stratified sampling with uniform groups."""
        # 4 groups of 25 each = 100 tasks
        tasks = _make_tasks(100, category_cycle=["a", "b", "c", "d"])
        result = sample_tasks(
            tasks,
            sample_size=20,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert len(result) == 20

    def test_stratified_proportional_representation(self) -> None:
        """Test that stratified sampling maintains group proportions."""
        # 60 tasks in group "a", 40 in group "b"
        tasks = []
        for i in range(60):
            tasks.append({"instance_id": f"a_{i}", "category": "a", "index": i})
        for i in range(40):
            tasks.append({"instance_id": f"b_{i}", "category": "b", "index": i})

        result = sample_tasks(
            tasks,
            sample_size=10,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert len(result) == 10

        # Count by category
        category_counts: dict[str, int] = {}
        for task in result:
            cat = task["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # 60% of 10 = 6 from "a", 40% of 10 = 4 from "b"
        assert category_counts["a"] == 6
        assert category_counts["b"] == 4

    def test_stratified_with_seed_is_reproducible(self) -> None:
        """Test that same seed produces same stratified results."""
        tasks = _make_tasks(100, category_cycle=["x", "y", "z"])
        result1 = sample_tasks(
            tasks,
            sample_size=15,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        result2 = sample_tasks(
            tasks,
            sample_size=15,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert [t["instance_id"] for t in result1] == [t["instance_id"] for t in result2]

    def test_stratified_different_seeds_differ(self) -> None:
        """Test that different seeds produce different stratified results."""
        tasks = _make_tasks(100, category_cycle=["x", "y", "z"])
        result1 = sample_tasks(
            tasks,
            sample_size=15,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        result2 = sample_tasks(
            tasks,
            sample_size=15,
            strategy=SamplingStrategy.STRATIFIED,
            seed=99,
            stratify_field="category",
        )
        ids1 = [t["instance_id"] for t in result1]
        ids2 = [t["instance_id"] for t in result2]
        assert ids1 != ids2

    def test_stratified_requires_stratify_field(self) -> None:
        """Test that STRATIFIED raises if stratify_field is None."""
        tasks = _make_tasks(10, category_cycle=["a", "b"])
        with pytest.raises(ValueError, match="stratify_field is required"):
            sample_tasks(
                tasks,
                sample_size=5,
                strategy=SamplingStrategy.STRATIFIED,
                seed=42,
                stratify_field=None,
            )

    def test_stratified_missing_field_raises(self) -> None:
        """Test that STRATIFIED raises if stratify_field is not in any task."""
        tasks = _make_tasks(10)  # No 'difficulty' field
        with pytest.raises(ValueError, match="not found in any task"):
            sample_tasks(
                tasks,
                sample_size=5,
                strategy=SamplingStrategy.STRATIFIED,
                seed=42,
                stratify_field="difficulty",
            )

    def test_stratified_single_group(self) -> None:
        """Test stratified with all tasks in same group."""
        tasks = _make_tasks(20, category_cycle=["only_group"])
        result = sample_tasks(
            tasks,
            sample_size=5,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert len(result) == 5
        for task in result:
            assert task["category"] == "only_group"

    def test_stratified_many_small_groups(self) -> None:
        """Test stratified with many groups smaller than allocation would suggest."""
        # 10 groups of 1 task each
        tasks = []
        for i in range(10):
            tasks.append({"instance_id": f"task_{i}", "category": f"group_{i}"})

        result = sample_tasks(
            tasks,
            sample_size=5,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert len(result) == 5

    def test_stratified_uneven_groups(self) -> None:
        """Test stratified with very uneven group sizes."""
        tasks = []
        # 90 in group "large", 10 in group "small"
        for i in range(90):
            tasks.append({"instance_id": f"large_{i}", "category": "large"})
        for i in range(10):
            tasks.append({"instance_id": f"small_{i}", "category": "small"})

        result = sample_tasks(
            tasks,
            sample_size=10,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert len(result) == 10

        category_counts: dict[str, int] = {}
        for task in result:
            cat = task["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # 90% of 10 = 9, 10% of 10 = 1
        assert category_counts["large"] == 9
        assert category_counts["small"] == 1


class TestEdgeCases:
    """Tests for edge cases across all sampling strategies."""

    def test_empty_tasks(self) -> None:
        """Test sampling from an empty task list."""
        for strategy in SamplingStrategy:
            kwargs: dict = {"tasks": [], "sample_size": 5, "strategy": strategy}
            if strategy == SamplingStrategy.STRATIFIED:
                kwargs["stratify_field"] = "category"
            result = sample_tasks(**kwargs)
            assert result == []

    def test_sample_size_none_returns_all(self) -> None:
        """Test that sample_size=None returns all tasks."""
        tasks = _make_tasks(10)
        for strategy in [SamplingStrategy.SEQUENTIAL, SamplingStrategy.RANDOM]:
            result = sample_tasks(tasks, sample_size=None, strategy=strategy)
            assert len(result) == 10

    def test_sample_size_larger_than_tasks(self) -> None:
        """Test that sample_size > len(tasks) returns all tasks."""
        tasks = _make_tasks(5)
        for strategy in [SamplingStrategy.SEQUENTIAL, SamplingStrategy.RANDOM]:
            result = sample_tasks(tasks, sample_size=100, strategy=strategy)
            assert len(result) == 5

    def test_sample_size_zero(self) -> None:
        """Test that sample_size=0 returns empty list."""
        tasks = _make_tasks(10)
        result = sample_tasks(tasks, sample_size=0, strategy=SamplingStrategy.SEQUENTIAL)
        assert result == []

    def test_sample_size_negative(self) -> None:
        """Test that negative sample_size returns empty list."""
        tasks = _make_tasks(10)
        result = sample_tasks(tasks, sample_size=-5, strategy=SamplingStrategy.SEQUENTIAL)
        assert result == []

    def test_sample_size_one(self) -> None:
        """Test sampling exactly one task."""
        tasks = _make_tasks(10)
        result = sample_tasks(tasks, sample_size=1, strategy=SamplingStrategy.SEQUENTIAL)
        assert len(result) == 1
        assert result[0]["instance_id"] == "task_0"

        result = sample_tasks(tasks, sample_size=1, strategy=SamplingStrategy.RANDOM, seed=42)
        assert len(result) == 1

    def test_single_task_list(self) -> None:
        """Test sampling from a list with a single task."""
        tasks = _make_tasks(1)
        result = sample_tasks(tasks, sample_size=1, strategy=SamplingStrategy.RANDOM, seed=42)
        assert len(result) == 1
        assert result[0]["instance_id"] == "task_0"

    def test_sample_size_equals_tasks(self) -> None:
        """Test that sample_size == len(tasks) returns all tasks."""
        tasks = _make_tasks(10)
        result = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        assert len(result) == 10

    def test_does_not_mutate_input(self) -> None:
        """Test that sampling does not mutate the input task list."""
        tasks = _make_tasks(20)
        original_ids = [t["instance_id"] for t in tasks]
        _ = sample_tasks(tasks, sample_size=5, strategy=SamplingStrategy.RANDOM, seed=42)
        assert [t["instance_id"] for t in tasks] == original_ids

    def test_stratified_sample_size_larger_returns_all(self) -> None:
        """Test that stratified with sample_size > len(tasks) returns all."""
        tasks = _make_tasks(5, category_cycle=["a", "b"])
        result = sample_tasks(
            tasks,
            sample_size=100,
            strategy=SamplingStrategy.STRATIFIED,
            seed=42,
            stratify_field="category",
        )
        assert len(result) == 5

    def test_stratified_sample_size_none_returns_all(self) -> None:
        """Test that stratified with sample_size=None returns all."""
        tasks = _make_tasks(10, category_cycle=["a", "b"])
        result = sample_tasks(
            tasks,
            sample_size=None,
            strategy=SamplingStrategy.STRATIFIED,
            stratify_field="category",
        )
        assert len(result) == 10


class TestReproducibility:
    """Focused tests on reproducibility guarantees."""

    def test_random_reproducibility_multiple_calls(self) -> None:
        """Test that multiple calls with same seed produce identical results."""
        tasks = _make_tasks(200)
        results = []
        for _ in range(5):
            result = sample_tasks(
                tasks, sample_size=50, strategy=SamplingStrategy.RANDOM, seed=12345
            )
            results.append([t["instance_id"] for t in result])

        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_stratified_reproducibility_multiple_calls(self) -> None:
        """Test that stratified sampling is reproducible across calls."""
        tasks = _make_tasks(200, category_cycle=["easy", "medium", "hard"])
        results = []
        for _ in range(5):
            result = sample_tasks(
                tasks,
                sample_size=30,
                strategy=SamplingStrategy.STRATIFIED,
                seed=12345,
                stratify_field="category",
            )
            results.append([t["instance_id"] for t in result])

        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_seed_isolation(self) -> None:
        """Test that sampling with a seed does not affect global random state."""
        import random

        random.seed(999)
        before = random.random()

        random.seed(999)
        tasks = _make_tasks(50)
        _ = sample_tasks(tasks, sample_size=10, strategy=SamplingStrategy.RANDOM, seed=42)
        after = random.random()

        assert before == after
