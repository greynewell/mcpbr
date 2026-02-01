---
description: "HumanEval benchmark for mcpbr - 164 Python programming problems from OpenAI for evaluating code generation capabilities."
benchmark_howto:
  name: "HumanEval"
  description: "Evaluate MCP server-assisted code generation on 164 Python function completion tasks from OpenAI's HumanEval dataset."
  benchmark_id: "humaneval"
faq:
  - q: "What is HumanEval and how does mcpbr use it?"
    a: "HumanEval is a benchmark of 164 Python programming problems created by OpenAI. Each task requires completing a function given its signature and docstring. mcpbr uses it to evaluate how well MCP servers assist language models in generating correct Python code."
  - q: "How are HumanEval tasks evaluated in mcpbr?"
    a: "The agent's solution is saved to solution.py, combined with the task's unit tests, and executed with Python. A task is resolved if all unit tests pass without assertion errors."
  - q: "Do I need Docker pre-built images for HumanEval?"
    a: "No. HumanEval uses lightweight Python environments created on the fly. No git repository cloning or pre-built Docker images are needed, making it fast to set up and run."
---

# HumanEval

| Property | Value |
|----------|-------|
| **Benchmark ID** | `humaneval` |
| **Dataset** | [openai_humaneval](https://huggingface.co/datasets/openai_humaneval) |
| **Tasks** | 164 Python programming problems |
| **Evaluation** | Unit tests combined with solution |
| **Output Type** | Test pass/fail |
| **Timeout** | 60-180s |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark humaneval -n 20
    ```

## Overview

[HumanEval](https://github.com/openai/human-eval) is a code generation benchmark created by OpenAI consisting of 164 hand-written Python programming problems. Each task provides a function signature with a descriptive docstring, and the agent must produce a correct implementation that passes all accompanying unit tests.

HumanEval is widely used as a baseline measure of code generation capability. Problems range from simple string manipulation and list operations to moderate algorithmic challenges. Because tasks are self-contained single functions with no external dependencies, HumanEval is an excellent choice for quick evaluations and smoke testing before running more expensive benchmarks like SWE-bench.

In mcpbr, HumanEval evaluates how effectively an MCP server assists the language model in generating correct, test-passing Python code within lightweight Docker containers.

## Task Structure

Each HumanEval task contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Unique identifier such as `HumanEval/0`, `HumanEval/1`, etc. |
| **prompt** | Function signature with docstring describing the requirements and examples |
| **entry_point** | Name of the function to implement (e.g., `has_close_elements`) |
| **canonical_solution** | Reference implementation (not shown to the agent) |
| **test** | Unit tests to verify correctness |

**Example task prompt:**

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

The agent receives the function signature and docstring, implements the function body, and saves the result to `solution.py`. Task IDs are converted to Docker-safe names by replacing slashes with underscores (e.g., `HumanEval/0` becomes `HumanEval_0`).

## Running the Benchmark

=== "CLI"

    ```bash
    # Run HumanEval with default settings
    mcpbr run -c config.yaml --benchmark humaneval

    # Run a small sample for quick testing
    mcpbr run -c config.yaml --benchmark humaneval -n 20

    # Run specific tasks by ID
    mcpbr run -c config.yaml --benchmark humaneval -t HumanEval/0 -t HumanEval/1

    # Run with verbose output
    mcpbr run -c config.yaml --benchmark humaneval -n 10 -v

    # Save results to JSON
    mcpbr run -c config.yaml --benchmark humaneval -n 20 -o results.json
    ```

=== "YAML Configuration"

    ```yaml
    benchmark: "humaneval"
    sample_size: 10
    timeout_seconds: 180
    max_iterations: 15

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"
    ```

## Evaluation Methodology

HumanEval evaluation follows a straightforward test-execution pipeline:

1. **Solution discovery**: The evaluator looks for the agent's solution file. It checks several candidate filenames in order: `solution.py`, `answer.py`, `code.py`, `implementation.py`, `main.py`. If no file is found, it attempts to extract code from the agent's raw response (markdown code blocks or inline function definitions).

2. **Test assembly**: The solution code is combined with the task's unit test code and a `check()` call that invokes the tests against the implemented entry point function.

3. **Execution**: The combined file is executed with `python3` inside the Docker container with a 30-second timeout.

4. **Verdict**: The task is marked as **resolved** if the Python process exits with code 0 and no `AssertionError` appears in stderr.

The evaluation does not require any external packages. All HumanEval tasks use only the Python standard library.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "exit_code": 0,
  "stdout": "",
  "stderr": ""
}
```

**Failed resolution (assertion error):**

```json
{
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "Traceback (most recent call last):\n  File \"test_solution.py\", line 12, in <module>\n    check(has_close_elements)\n  File \"test_solution.py\", line 8, in check\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nAssertionError",
  "error": "Test assertions failed"
}
```

**Failed resolution (no solution found):**

```json
{
  "resolved": false,
  "error": "No solution file found and could not extract code from solution"
}
```

## Troubleshooting

**Agent does not create `solution.py`**

The agent may produce a correct implementation but fail to save it to a file. Ensure your agent prompt explicitly instructs saving to `solution.py`. The evaluator will attempt to extract code from the agent's raw response as a fallback, but file-based solutions are more reliable.

**Tests fail with import errors**

HumanEval tasks use only the Python standard library. If the agent imports third-party packages (e.g., `numpy`), the tests will fail. Verify that the agent prompt discourages external dependencies.

**Timeout during evaluation**

Individual test execution has a 30-second timeout. If the agent's implementation contains infinite loops or extremely slow algorithms, the test will time out. Set `timeout_seconds` in your configuration to control the overall task timeout (recommended: 60-180s).

**Function signature is modified**

The agent must preserve the original function signature exactly. If the function name, parameter names, or type hints are altered, the test harness will fail because it calls the original `entry_point` function name.

## Best Practices

- **Start with a small sample** (10-20 tasks) to verify your setup before running all 164 tasks.
- **Use shorter timeouts** (60-180s) since HumanEval tasks are quick and well-defined.
- **Run HumanEval first** before expensive benchmarks like SWE-bench as a smoke test for your MCP server integration.
- **Leverage low resource usage** by running higher concurrency (`max_concurrent: 8` or higher) since each task uses a lightweight Python container.
- **Monitor `solution.py` creation** to verify the agent is saving code to the expected filename.
- **Use standard library only** -- ensure the agent prompt discourages external dependencies.
- **Set `max_iterations` lower** (10-15) since tasks are simpler than SWE-bench and require fewer agent turns.

## Related Links

- [HumanEval Repository](https://github.com/openai/human-eval)
- [HumanEval Paper (Evaluating Large Language Models Trained on Code)](https://arxiv.org/abs/2107.03374)
- [HumanEval Dataset on HuggingFace](https://huggingface.co/datasets/openai_humaneval)
- [Benchmarks Overview](index.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
