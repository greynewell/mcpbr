---
description: "CodeContests evaluates AI agents on competitive programming problems from Codeforces, CodeChef, and other platforms, featuring public and private test cases with time and memory constraints."
benchmark_howto:
  name: "CodeContests"
  description: "Competitive programming problems from DeepMind sourced from Codeforces, CodeChef, and other platforms with multi-level test case evaluation."
  benchmark_id: "codecontests"
faq:
  - q: "What platforms do CodeContests problems come from?"
    a: "CodeContests problems are sourced from Codeforces, CodeChef, HackerEarth, AtCoder, and other competitive programming platforms. You can filter by source platform using the filter_category option."
  - q: "How does CodeContests differ from APPS?"
    a: "CodeContests includes problems from more diverse platforms and features both public and private test cases. It also includes per-task time_limit and memory_limit_bytes constraints. The problems tend to be more algorithmically challenging than APPS introductory problems."
  - q: "Are there per-problem time and memory limits?"
    a: "Yes. Each CodeContests problem may include time_limit and memory_limit_bytes fields that specify the execution constraints from the original platform. These are stored in task metadata for reference, though the evaluation uses a fixed per-test-case timeout."
---

# CodeContests

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `codecontests` |
| **Dataset** | [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) |
| **Tasks** | Varies (competitive programming problems) |
| **Evaluation** | Run code against public + private test cases, compare stdout |
| **Output Type** | Test pass rate |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark codecontests
    ```

## Overview

CodeContests is a competitive programming benchmark created by DeepMind, containing problems sourced from Codeforces, CodeChef, HackerEarth, AtCoder, and other competitive programming platforms. The benchmark is designed to evaluate code generation capabilities on problems that require deep algorithmic thinking, mathematical reasoning, and efficient implementation.

Each problem includes a natural language description, input/output format specifications, sample test cases, and hidden (private) test cases. The problems cover a wide range of algorithmic topics including dynamic programming, graph algorithms, number theory, greedy strategies, binary search, segment trees, and more.

A key feature of CodeContests is its multi-level test case structure:

- **Public tests**: Sample test cases visible in the problem statement, provided to the agent as examples.
- **Private tests**: Hidden test cases used for evaluation, designed to cover edge cases and large inputs.
- **Generated tests**: Additional automatically generated test cases that further validate solution correctness.

The agent must produce a Python program that reads from stdin and writes to stdout, handling all test cases correctly within the specified time and memory constraints.

## Task Structure

Each CodeContests task includes the following components:

- **Name**: A unique problem identifier from the source platform.
- **Description**: The full problem statement in natural language, including input format, output format, constraints, and examples.
- **Difficulty**: A numeric difficulty rating from the source platform.
- **Source**: The platform the problem was sourced from (e.g., `codeforces`, `codechef`).
- **Public Tests**: Sample input/output pairs shown in the problem description.
- **Private Tests**: Hidden input/output pairs used for final evaluation.
- **Generated Tests**: Additional generated input/output pairs for broader coverage.
- **Time Limit**: Per-test-case time limit from the source platform (stored in metadata).
- **Memory Limit**: Memory limit in bytes from the source platform (stored in metadata).
- **Instance ID**: An auto-generated identifier in the format `codecontests_{name}`.

The agent receives the problem description along with sample test cases and must produce a solution that handles both the visible and hidden test cases.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run CodeContests with default settings
    mcpbr run -c config.yaml --benchmark codecontests

    # Run a sample of 20 problems
    mcpbr run -c config.yaml --benchmark codecontests -n 20

    # Run a specific task by name
    mcpbr run -c config.yaml --benchmark codecontests -t codecontests_problem_name

    # Filter by source platform
    mcpbr run -c config.yaml --benchmark codecontests --filter-category codeforces

    # Filter by difficulty level
    mcpbr run -c config.yaml --benchmark codecontests --filter-difficulty 1

    # Combine filters
    mcpbr run -c config.yaml --benchmark codecontests \
      --filter-category codeforces --filter-difficulty 2
    ```

=== "YAML"

    ```yaml
    benchmark: "codecontests"
    sample_size: 10
    timeout_seconds: 300

    # Optional: Filter by source platform
    filter_category:
      - "codeforces"
    ```

    Configuration with difficulty filtering:

    ```yaml
    benchmark: "codecontests"
    sample_size: 20
    timeout_seconds: 300

    filter_difficulty:
      - "1"
      - "2"
    filter_category:
      - "codeforces"
      - "codechef"
    ```

## Evaluation Methodology

CodeContests evaluation runs the solution against all available test cases:

1. **Solution Writing**: The agent's generated code is written to `solution.py` inside the Docker container.
2. **Test Case Collection**: Test cases are gathered from three sources: `public_tests`, `private_tests`, and `generated_tests`. All input/output pairs are combined into a single evaluation set.
3. **Per-Test Execution**: For each test case, the input is piped to the solution via stdin. Each individual test case execution has a 10-second timeout (with a 15-second outer timeout for the Docker command).
4. **Output Comparison**: The program's stdout is stripped of whitespace and compared exactly against the expected output string.
5. **Pass Rate Calculation**: The evaluation counts passed tests out of the total. A task is **resolved** only when all test cases pass.
6. **Result Reporting**: Results include the number of passed tests, total tests, and the overall pass rate.

The use of both public and private test cases ensures that solutions are genuinely correct rather than overfitted to the visible examples.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "codecontests_watermelon",
  "resolved": true,
  "passed": 12,
  "total": 12,
  "pass_rate": 1.0
}
```

### Partial Pass

```json
{
  "instance_id": "codecontests_theatre_square",
  "resolved": false,
  "passed": 8,
  "total": 12,
  "pass_rate": 0.667
}
```

### No Test Cases Available

```json
{
  "instance_id": "codecontests_unknown_problem",
  "resolved": false,
  "error": "No test cases available"
}
```

## Troubleshooting

**Solution passes public tests but fails private tests**
Private test cases often include edge cases with large inputs, boundary values, or special conditions not covered by the sample tests. The agent should analyze constraints carefully and consider edge cases. Encourage the agent to generate its own test cases based on the constraint descriptions.

**Solution times out on large inputs**
Each test case has a 10-second execution limit. Competitive programming problems often require O(n log n) or better algorithms. If the agent produces an O(n^2) solution, it may work on small inputs but fail on large ones. Include algorithmic complexity guidance in your prompt.

**Test case format parsing errors**
CodeContests test cases may be stored as JSON strings or dictionaries. The evaluation handles both formats automatically. If parsing fails for a specific problem, it typically indicates a malformed dataset entry. Use `-t` to skip and report the issue.

**Incorrect output formatting**
Some problems require specific output formatting (e.g., floating point precision, spacing between values). The comparison is exact after whitespace stripping, so the agent must match the expected format precisely. Pay attention to output format specifications in the problem description.

## Best Practices

- **Start with a small sample** (`-n 10`) to validate your configuration before running larger evaluations.
- **Filter by platform** using `filter_category` to focus on problems from a specific source like Codeforces, which has well-structured problem descriptions.
- **Use longer timeouts** (300 seconds) to give the agent sufficient reasoning time for complex algorithmic problems.
- **Include algorithm hints in prompts**: For competitive programming, prompting the agent to analyze time complexity and choose appropriate algorithms improves performance significantly.
- **Compare public vs. total pass rates**: If the agent consistently passes public tests but fails private ones, it may be overfitting to examples rather than generalizing to the full problem specification.
- **Track per-problem difficulty**: Running evaluations segmented by difficulty level helps identify the algorithmic complexity threshold where your agent's performance drops off.

## Related Links

- [CodeContests Paper (Science)](https://www.science.org/doi/10.1126/science.abq1158)
- [CodeContests Dataset on HuggingFace](https://huggingface.co/datasets/deepmind/code_contests)
- [Codeforces](https://codeforces.com/)
- [CodeChef](https://www.codechef.com/)
- [Benchmarks Overview](index.md)
- [APPS](apps.md) | [LeetCode](leetcode.md) | [BigCodeBench](bigcodebench.md)
