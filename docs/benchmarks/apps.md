---
description: "APPS evaluates AI agents on 10,000 coding problems collected from open-access programming platforms, spanning introductory, interview, and competition difficulty levels."
benchmark_howto:
  name: "APPS"
  description: "10,000 coding problems from open-access platforms with stdin/stdout test case evaluation across three difficulty levels."
  benchmark_id: "apps"
faq:
  - q: "What difficulty levels does APPS support?"
    a: "APPS has three difficulty levels: introductory (basic programming concepts), interview (typical coding interview questions), and competition (competitive programming difficulty). Use filter_difficulty to select specific levels."
  - q: "How are APPS solutions evaluated?"
    a: "Solutions are evaluated by running the generated code with provided test case inputs via stdin and comparing the stdout output against expected outputs. All test cases must pass for a task to be marked as resolved."
  - q: "What format should APPS solutions use?"
    a: "Solutions must be Python programs that read from stdin and write to stdout. The agent should save its solution to a file named 'solution.py' in the working directory."
---

# APPS

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `apps` |
| **Dataset** | [metr-evals/apps](https://huggingface.co/datasets/metr-evals/apps) |
| **Tasks** | 10,000 coding problems |
| **Evaluation** | stdin/stdout test case comparison |
| **Output Type** | Test pass rate |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark apps
    ```

## Overview

APPS (Automated Programming Progress Standard) is a large-scale coding benchmark containing 10,000 problems collected from open-access coding websites such as Codeforces, Kattis, and other competitive programming platforms. The benchmark tests a broad range of programming skills, from basic string manipulation and arithmetic to advanced algorithmic problem solving involving dynamic programming, graph theory, and complex data structures.

Each problem provides a natural language description along with input/output specifications and test cases. The agent must generate a Python program that reads from standard input, processes the data according to the problem requirements, and writes the correct output to standard output.

APPS problems are categorized into three difficulty levels:

- **Introductory**: Basic programming concepts, simple loops and conditionals, and straightforward I/O handling. Suitable for testing fundamental code generation capabilities.
- **Interview**: Problems at the level of typical software engineering interviews, requiring knowledge of standard data structures and algorithms.
- **Competition**: Competitive programming problems that demand advanced algorithmic techniques, optimization, and careful handling of edge cases.

## Task Structure

Each APPS task includes the following components:

- **Question**: The full problem description in natural language, including input format, output format, constraints, and examples.
- **Difficulty**: One of `introductory`, `interview`, or `competition`, indicating the problem complexity.
- **Solutions**: Reference solutions (not shown to the agent) used for verification purposes.
- **Input/Output**: Structured test cases with input strings and expected output strings for evaluation.
- **Instance ID**: An auto-generated identifier in the format `apps_{index}` (e.g., `apps_0`, `apps_42`).

The agent receives the problem statement with difficulty information and must produce a self-contained Python program that handles all specified test cases correctly.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run APPS with default settings
    mcpbr run -c config.yaml --benchmark apps

    # Run a sample of 20 problems
    mcpbr run -c config.yaml --benchmark apps -n 20

    # Run a specific task
    mcpbr run -c config.yaml --benchmark apps -t apps_42

    # Filter by difficulty level
    mcpbr run -c config.yaml --benchmark apps --filter-difficulty introductory

    # Filter for interview and competition problems only
    mcpbr run -c config.yaml --benchmark apps \
      --filter-difficulty interview --filter-difficulty competition
    ```

=== "YAML"

    ```yaml
    benchmark: "apps"
    sample_size: 10
    timeout_seconds: 300

    # Optional: Filter by difficulty
    filter_difficulty:
      - "introductory"
    ```

    Configuration for harder problems:

    ```yaml
    benchmark: "apps"
    sample_size: 20
    timeout_seconds: 300

    filter_difficulty:
      - "interview"
      - "competition"
    ```

## Evaluation Methodology

APPS evaluation is based on direct input/output comparison:

1. **Solution Writing**: The agent's generated code is written to `solution.py` inside the Docker container.
2. **Test Case Execution**: For each test case, the input string is piped to the solution via stdin.
3. **Output Comparison**: The program's stdout is captured and compared (after stripping whitespace) against the expected output string.
4. **Pass Rate Calculation**: The evaluation counts the number of test cases passed out of the total. A task is marked as **resolved** only if all test cases pass.
5. **Result Reporting**: The result includes the number of passed tests, total tests, and the overall pass rate.

Each test case execution has an individual timeout of 10 seconds to prevent infinite loops or excessively slow solutions from blocking the evaluation.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "apps_42",
  "resolved": true,
  "passed": 5,
  "total": 5,
  "pass_rate": 1.0
}
```

### Partial Pass

```json
{
  "instance_id": "apps_157",
  "resolved": false,
  "passed": 3,
  "total": 5,
  "pass_rate": 0.6
}
```

### No Test Cases

```json
{
  "instance_id": "apps_999",
  "resolved": false,
  "error": "No test cases provided"
}
```

## Troubleshooting

**Solution produces wrong output format**
APPS problems require exact output matching. Ensure the agent does not include extra whitespace, trailing newlines, or debug print statements. The comparison strips leading and trailing whitespace, but any differences in the body of the output will cause a failure.

**Solution times out on individual test cases**
Each test case has a 10-second execution limit. Competition-level problems with large inputs may require optimized algorithms. If the agent produces a brute-force solution, it may pass small test cases but time out on larger ones. Encourage the agent to analyze time complexity constraints.

**Test case parsing fails**
If the test cases in the dataset cannot be parsed as JSON, the evaluation returns an error. This is rare but can occur with malformed entries. Use `--task` to skip problematic tasks and report the issue.

**Solution reads input incorrectly**
APPS problems expect input to be read from stdin. The agent must use `input()` or `sys.stdin` in Python. Solutions that hardcode test values or read from files will fail. Verify that the prompt template instructs the agent to use stdin/stdout.

## Best Practices

- **Start with introductory problems** (`--filter-difficulty introductory`) to validate your setup before tackling harder problems.
- **Use a moderate sample size** (`-n 20` to `-n 50`) for initial evaluation, as the full 10,000 problems take significant time and resources.
- **Set timeouts appropriately**: 180 seconds is sufficient for introductory problems, but competition-level problems may benefit from 300 seconds to allow the agent more reasoning time.
- **Monitor pass rates**: Unlike binary pass/fail benchmarks, APPS reports partial progress. A pass rate of 0.8 on competition problems may indicate strong performance even if not all tasks are fully resolved.
- **Encourage efficient solutions**: Include guidance in your agent prompt about time complexity, especially for competition-level problems with large input sizes.
- **Compare across difficulty levels**: Running separate evaluations for each difficulty level provides more granular insight into your agent's capabilities.

## Related Links

- [APPS Paper (arXiv)](https://arxiv.org/abs/2105.09938)
- [APPS Dataset on HuggingFace](https://huggingface.co/datasets/metr-evals/apps)
- [Benchmarks Overview](index.md)
- [CodeContests](codecontests.md) | [LeetCode](leetcode.md) | [BigCodeBench](bigcodebench.md)
