"""HumanEval benchmark implementation."""

from typing import Any

from datasets import load_dataset

from ..docker_env import DockerEnvironmentManager, TaskEnvironment
from .base import BenchmarkTask


class HumanEvalBenchmark:
    """HumanEval benchmark implementation.

    HumanEval is a benchmark for evaluating code generation models on Python
    programming problems. Each task requires completing a function given its
    signature and docstring.

    Tasks involve generating function implementations that pass unit tests.
    Evaluation runs the provided test cases against the generated code.
    """

    name = "humaneval"

    def __init__(self, dataset: str = "openai_humaneval"):
        """Initialize HumanEval benchmark.

        Args:
            dataset: HuggingFace dataset identifier.
        """
        self.dataset = dataset

    def load_tasks(
        self,
        sample_size: int | None = None,
        task_ids: list[str] | None = None,
        level: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load tasks from HumanEval dataset.

        Args:
            sample_size: Maximum number of tasks to load (None for all).
            task_ids: Specific task IDs to load (None for all).
            level: Unused for HumanEval (no difficulty levels).

        Returns:
            List of HumanEval task dictionaries.
        """
        dataset = load_dataset(self.dataset, split="test")

        if task_ids:
            # Use set for O(1) lookup performance
            task_id_set = set(task_ids)
            tasks = []
            for item in dataset:
                if item["task_id"] in task_id_set:
                    tasks.append(item)
        else:
            tasks = list(dataset)

        if sample_size and len(tasks) > sample_size:
            tasks = tasks[:sample_size]

        # Augment tasks with instance_id for compatibility with harness
        augmented_tasks = []
        for task in tasks:
            augmented = dict(task)
            # Use task_id as instance_id (e.g., "HumanEval/0")
            # Replace slash with underscore for Docker-safe naming
            augmented["instance_id"] = task["task_id"].replace("/", "_")
            # Generate problem_statement for the harness
            augmented["problem_statement"] = self._generate_problem_statement(augmented)
            augmented_tasks.append(augmented)

        return augmented_tasks

    def normalize_task(self, task: dict[str, Any]) -> BenchmarkTask:
        """Convert HumanEval task to normalized format.

        Args:
            task: HumanEval task dictionary.

        Returns:
            Normalized BenchmarkTask.

        Raises:
            ValueError: If task_id is missing from task.
        """
        task_id = task.get("task_id")
        if not task_id:
            # Fallback to instance_id if available
            task_id = task.get("instance_id")
            if not task_id:
                msg = f"Task missing required 'task_id' field: {task.keys()}"
                raise ValueError(msg)

        problem_statement = self._generate_problem_statement(task)

        return BenchmarkTask(
            task_id=task_id,
            problem_statement=problem_statement,
            repo="openai/humaneval",
            commit="HEAD",
            metadata={
                "prompt": task.get("prompt", ""),
                "canonical_solution": task.get("canonical_solution", ""),
                "test": task.get("test", ""),
                "entry_point": task.get("entry_point", ""),
            },
        )

    def _generate_problem_statement(self, task: dict[str, Any]) -> str:
        """Generate problem statement from task.

        Args:
            task: HumanEval task dictionary.

        Returns:
            Problem statement for the agent.
        """
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        entry_point = task.get("entry_point", "")

        statement = (
            f"Complete the following Python function ({task_id}):\n\n"
            f"```python\n{prompt}\n```\n\n"
            f"INSTRUCTIONS:\n"
            f"- Implement the function '{entry_point}' according to the docstring\n"
            f"- The function signature is already provided\n"
            f"- Write only the function implementation\n"
            f"- Ensure your code passes all test cases\n"
            f"- Do NOT modify the function signature\n"
            f"- Save your implementation to a file named 'solution.py'"
        )

        return statement

    async def create_environment(
        self,
        task: dict[str, Any],
        docker_manager: DockerEnvironmentManager,
    ) -> TaskEnvironment:
        """Create environment for HumanEval task.

        Creates a lightweight Python environment for code execution.

        Args:
            task: HumanEval task dictionary.
            docker_manager: Docker environment manager.

        Returns:
            TaskEnvironment for the task.
        """
        # Get instance_id with fallback to task_id (sanitized)
        instance_id = task.get("instance_id")
        if not instance_id:
            # Fallback to task_id with slash replaced by underscore
            task_id = task.get("task_id", "unknown")
            instance_id = task_id.replace("/", "_")

        # Create minimal Python environment
        temp_task = {
            "instance_id": instance_id,
            "repo": "openai/humaneval",
            "base_commit": "HEAD",
        }

        env = await docker_manager.create_environment(temp_task)

        # Ensure Python 3 is available
        await self._setup_python_environment(env)

        return env

    async def _setup_python_environment(self, env: TaskEnvironment) -> None:
        """Setup Python environment with necessary packages.

        Args:
            env: Task environment.
        """
        # Install Python if not available (most Docker images should have it)
        install_cmd = (
            "apt-get update -qq && apt-get install -y -qq python3 python3-pip 2>&1 || true"
        )
        await env.exec_command(install_cmd, timeout=300)

    async def evaluate(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        solution: str,
    ) -> dict[str, Any]:
        """Evaluate a solution for HumanEval task.

        Runs the unit tests against the generated code.

        Args:
            env: Task environment.
            task: HumanEval task dictionary.
            solution: Solution code to evaluate (function implementation).

        Returns:
            Dictionary with evaluation results including 'resolved' boolean.
        """
        # Extract test code from task
        test_code = task.get("test", "")
        entry_point = task.get("entry_point", "")

        if not test_code:
            return {
                "resolved": False,
                "error": "No test code provided in task",
            }

        # Try to find the solution file created by the agent
        solution_file = await self._find_solution_file(env)

        if not solution_file:
            # Agent might have included the solution directly in the response
            # Try to extract code from the solution string
            solution_code = self._extract_code_from_solution(solution)
            if not solution_code:
                return {
                    "resolved": False,
                    "error": "No solution file found and could not extract code from solution",
                }

            # Write the solution to a file
            solution_file = "solution.py"
            exit_code, stdout, stderr = await env.exec_command(
                f"cat > {solution_file} << 'SOLUTION_EOF'\n{solution_code}\nSOLUTION_EOF",
                timeout=10,
            )
            if exit_code != 0:
                return {
                    "resolved": False,
                    "error": f"Failed to write solution file: {stderr}",
                }

        # Read the solution code
        exit_code, solution_content, stderr = await env.exec_command(
            f"cat {solution_file}",
            timeout=10,
        )
        if exit_code != 0:
            return {
                "resolved": False,
                "error": f"Failed to read solution file: {stderr}",
            }

        # Create a test file combining the solution and test code
        test_file_content = f"{solution_content}\n\n{test_code}\n\ncheck({entry_point})\n"

        # Write test file
        test_file = "test_solution.py"
        exit_code, stdout, stderr = await env.exec_command(
            f"cat > {test_file} << 'TEST_EOF'\n{test_file_content}\nTEST_EOF",
            timeout=10,
        )
        if exit_code != 0:
            return {
                "resolved": False,
                "error": f"Failed to write test file: {stderr}",
            }

        # Run the test
        exit_code, stdout, stderr = await env.exec_command(
            f"python3 {test_file}",
            timeout=30,
        )

        # Test passes if exit code is 0 and no assertion errors
        passed = exit_code == 0 and "AssertionError" not in stderr

        result = {
            "resolved": passed,
            "exit_code": exit_code,
            "stdout": stdout[:1000] if stdout else "",  # Limit output size
            "stderr": stderr[:1000] if stderr else "",
        }

        if not passed:
            if "AssertionError" in stderr:
                result["error"] = "Test assertions failed"
            else:
                result["error"] = f"Test execution failed with exit code {exit_code}"

        return result

    async def _find_solution_file(self, env: TaskEnvironment) -> str | None:
        """Find the solution file created by the agent.

        Args:
            env: Task environment.

        Returns:
            Path to solution file or None if not found.
        """
        # Common solution filenames
        candidates = [
            "solution.py",
            "answer.py",
            "code.py",
            "implementation.py",
            "main.py",
        ]

        for filename in candidates:
            exit_code, _, _ = await env.exec_command(
                f"test -f {filename}",
                timeout=5,
            )
            if exit_code == 0:
                return filename

        return None

    def _extract_code_from_solution(self, solution: str) -> str | None:
        """Extract Python code from solution string.

        Handles various formats like markdown code blocks, plain text, etc.

        Args:
            solution: Solution string from agent.

        Returns:
            Extracted code or None if no code found.
        """
        # Try to extract from markdown code block
        if "```python" in solution:
            start = solution.find("```python") + len("```python")
            end = solution.find("```", start)
            if end != -1:
                return solution[start:end].strip()

        # Try generic code block
        if "```" in solution:
            start = solution.find("```") + 3
            end = solution.find("```", start)
            if end != -1:
                code = solution[start:end].strip()
                # Check if it looks like Python code
                if "def " in code or "return" in code:
                    return code

        # If solution looks like code directly (contains def keyword)
        if "def " in solution:
            # Try to extract just the function definition
            lines = solution.split("\n")
            code_lines = []
            in_function = False
            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)
                    # Stop at next function or class definition
                    if (
                        code_lines
                        and len(code_lines) > 1
                        and (line.strip().startswith("def ") or line.strip().startswith("class "))
                        and not line == code_lines[0]
                    ):
                        # Remove the last line (next function/class)
                        code_lines = code_lines[:-1]
                        break

            if code_lines:
                return "\n".join(code_lines)

        return None

    def get_prebuilt_image(self, task: dict[str, Any]) -> str | None:
        """Get pre-built Docker image name for HumanEval task.

        HumanEval doesn't use pre-built images - uses minimal Python environments.

        Args:
            task: HumanEval task dictionary.

        Returns:
            None (no pre-built images available).
        """
        return None

    def get_prompt_template(self) -> str:
        """Get HumanEval prompt template.

        Returns:
            Prompt template for code generation tasks.
        """
        return (
            "Complete the following Python function:\n\n"
            "{problem_statement}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Implement the function according to its docstring\n"
            "- The function signature is already provided - do NOT change it\n"
            "- Write clean, correct Python code\n"
            "- Ensure your implementation passes all test cases\n"
            "- Save your implementation to a file named 'solution.py'\n"
            "- Include ONLY the function implementation in the file"
        )
