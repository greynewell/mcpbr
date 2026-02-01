---
description: "MBPP benchmark for mcpbr - ~1,000 crowd-sourced Python programming problems designed for entry-level programmers."
benchmark_howto:
  name: "MBPP"
  description: "Evaluate MCP server-assisted code generation on crowd-sourced Python programming problems from Google Research's MBPP dataset."
  benchmark_id: "mbpp"
faq:
  - q: "What is MBPP and how does it differ from HumanEval?"
    a: "MBPP (Mostly Basic Python Problems) is a dataset of ~1,000 crowd-sourced Python problems designed for entry-level programmers. Unlike HumanEval which provides function signatures with docstrings, MBPP provides natural language descriptions with example test cases, testing the agent's ability to interpret requirements and write functions from scratch."
  - q: "How are MBPP solutions evaluated?"
    a: "The agent's solution code is combined with the task's test cases and executed. If all test assertions pass and 'ALL_TESTS_PASSED' is printed, the task is marked as resolved."
  - q: "What dataset subset does mcpbr use for MBPP?"
    a: "By default, mcpbr loads the 'full' subset of the google-research-datasets/mbpp dataset and uses the 'test' split for evaluation."
---

# MBPP

| Property | Value |
|----------|-------|
| **Benchmark ID** | `mbpp` |
| **Dataset** | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| **Tasks** | ~1,000 crowd-sourced Python problems |
| **Evaluation** | Runs test cases with `ALL_TESTS_PASSED` marker |
| **Output Type** | Test pass/fail |
| **Timeout** | 60-180s |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark mbpp -n 20
    ```

## Overview

[MBPP (Mostly Basic Python Problems)](https://github.com/google-research/google-research/tree/master/mbpp) is a benchmark of approximately 1,000 crowd-sourced Python programming problems created by Google Research. The problems are designed to be solvable by entry-level programmers and cover fundamental programming concepts such as string manipulation, list operations, mathematical computations, and basic data structure usage.

Unlike HumanEval, which provides a function signature with a detailed docstring, MBPP tasks present a natural language problem description along with example test cases. The agent must interpret the requirements, design an appropriate function, and implement it correctly. This tests a broader set of skills including requirement comprehension, function design, and code correctness.

In mcpbr, MBPP evaluates how well an MCP server helps the language model understand problem descriptions and generate working Python solutions that pass all provided test assertions.

## Task Structure

Each MBPP task contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Numeric identifier for the task (e.g., `1`, `2`, `601`) |
| **text** | Natural language description of the problem |
| **code** | Canonical solution (reference implementation, not shown to agent) |
| **test_list** | List of assertion-based test cases |

**Example task:**

```text
text: "Write a function to find the minimum cost path to reach (m, n) from (0, 0)
       for the given cost matrix."

test_list:
  - "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8"
  - "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12"
  - "assert min_cost([[20, 30, 40], [50, 90, 30], [20, 60, 40]], 2, 2) == 120"
```

Instance IDs are generated in the format `mbpp_{task_id}` (e.g., `mbpp_601`). The problem statement shown to the agent includes the text description and up to 3 example test cases.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run MBPP with default settings
    mcpbr run -c config.yaml --benchmark mbpp

    # Run a small sample for quick testing
    mcpbr run -c config.yaml --benchmark mbpp -n 20

    # Run specific tasks by ID
    mcpbr run -c config.yaml --benchmark mbpp -t mbpp_601 -t mbpp_602

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark mbpp -n 50 -v -o results.json

    # Run MCP-only evaluation (skip baseline)
    mcpbr run -c config.yaml --benchmark mbpp -n 20 -M
    ```

=== "YAML Configuration"

    ```yaml
    benchmark: "mbpp"
    sample_size: 10
    timeout_seconds: 180
    max_iterations: 15

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"
    ```

## Evaluation Methodology

MBPP evaluation uses a test-execution pipeline with an explicit pass marker:

1. **Solution extraction**: The agent's solution code (either from the agent response or from a saved `solution.py` file) is combined with the task's test cases.

2. **Test assembly**: A test file is constructed by concatenating the solution code, all test assertions from `test_list`, and a final `print('ALL_TESTS_PASSED')` statement.

3. **Execution**: The assembled file is base64-encoded, written to `test_solution.py`, and executed with `python3` inside the Docker container with a 30-second timeout.

4. **Verdict**: The task is marked as **resolved** if:
    - The Python process exits with code 0, AND
    - The string `ALL_TESTS_PASSED` appears in stdout

This two-condition check ensures that the code not only runs without errors but also successfully executes past all assertion statements to reach the final print statement.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "exit_code": 0,
  "stdout": "ALL_TESTS_PASSED\n",
  "stderr": ""
}
```

**Failed resolution (assertion error):**

```json
{
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "Traceback (most recent call last):\n  File \"test_solution.py\", line 5, in <module>\n    assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8\nAssertionError"
}
```

**Failed resolution (no test cases):**

```json
{
  "resolved": false,
  "error": "No test cases provided"
}
```

## Troubleshooting

**Agent output does not contain a function definition**

MBPP tasks require the agent to design a function from a natural language description. If the agent produces only an explanation or pseudocode, the tests will fail. Ensure your agent prompt explicitly instructs the agent to write executable Python code and save it to `solution.py`.

**Tests fail with `NameError` for the function name**

MBPP test cases reference specific function names (e.g., `min_cost`, `find_max`). The agent must name its function to match what the test cases call. Providing the test cases in the prompt (which mcpbr does by default with up to 3 examples) helps the agent infer the correct function name.

**Timeout during test execution**

Some MBPP problems involve recursive solutions or large inputs that can cause slow execution. If you see frequent timeouts, consider increasing `timeout_seconds` to 180s or higher. The default per-test execution timeout is 30 seconds.

**Import errors for standard library modules**

While MBPP tasks are designed to use only the Python standard library, some problems may benefit from modules like `math`, `itertools`, or `collections`. These are available by default in the Docker environment. If the agent imports third-party packages, execution will fail.

## Best Practices

- **Start with a small sample** (10-20 tasks) to verify your setup before scaling to the full dataset.
- **Include test cases in the prompt** -- mcpbr does this by default, showing up to 3 example assertions so the agent can infer function names and expected behavior.
- **Use shorter timeouts** (60-180s) since MBPP tasks are entry-level problems that should solve quickly.
- **Set `max_iterations` to 10-15** since MBPP tasks are simpler than SWE-bench and require fewer agent turns.
- **Run MBPP alongside HumanEval** to get complementary views on code generation: HumanEval tests function completion from signatures, while MBPP tests function creation from descriptions.
- **Leverage concurrency** -- MBPP tasks are lightweight and can run at higher parallelism (`max_concurrent: 8` or more).
- **Monitor function naming** -- a common failure mode is the agent choosing a different function name than what the tests expect.

## Related Links

- [MBPP Paper (Program Synthesis with Large Language Models)](https://arxiv.org/abs/2108.07732)
- [MBPP Dataset on HuggingFace](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [Google Research MBPP Repository](https://github.com/google-research/google-research/tree/master/mbpp)
- [Benchmarks Overview](index.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
