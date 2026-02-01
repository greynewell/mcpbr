---
description: "LeetCode evaluates AI agents on algorithmic coding problems across easy, medium, and hard difficulty levels, covering data structures, algorithms, and common interview topics."
benchmark_howto:
  name: "LeetCode"
  description: "Algorithmic coding problems spanning easy, medium, and hard difficulty levels with topic-based filtering for data structures, dynamic programming, graphs, and more."
  benchmark_id: "leetcode"
faq:
  - q: "How are LeetCode solutions evaluated?"
    a: "Solutions are evaluated by executing the generated Python code to check for syntax errors and successful execution. The evaluation appends a verification marker and confirms the code runs without errors. For best results, the agent should include its own test assertions."
  - q: "Can I filter LeetCode problems by topic?"
    a: "Yes. Use filter_tags to select problems by topic tags (e.g., '--filter-tags array --filter-tags dynamic-programming'). Use filter_category for broader category filtering and filter_difficulty for easy, medium, or hard problems."
  - q: "Does mcpbr support all LeetCode problems?"
    a: "mcpbr uses the greengerong/leetcode dataset from HuggingFace, which contains a large collection of LeetCode problems. The dataset may not include every problem on the LeetCode platform, but it provides comprehensive coverage of common algorithmic topics."
---

# LeetCode

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `leetcode` |
| **Dataset** | [greengerong/leetcode](https://huggingface.co/datasets/greengerong/leetcode) |
| **Tasks** | Algorithmic coding problems (varies) |
| **Evaluation** | Execute code, check for syntax errors and test execution |
| **Output Type** | Code execution result |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark leetcode
    ```

## Overview

LeetCode is a widely recognized benchmark for evaluating algorithmic problem-solving skills. The mcpbr LeetCode benchmark draws from a HuggingFace dataset of LeetCode problems, covering the full spectrum of difficulty levels and algorithmic topics commonly encountered in software engineering interviews and competitive programming.

Problems span a broad range of topics:

- **Data Structures**: Arrays, linked lists, trees, graphs, heaps, hash tables, stacks, and queues.
- **Algorithms**: Sorting, searching, binary search, two pointers, sliding window, and greedy algorithms.
- **Dynamic Programming**: Memoization, tabulation, knapsack variants, and sequence optimization.
- **Graph Algorithms**: BFS, DFS, shortest paths, topological sort, union-find, and minimum spanning trees.
- **Math and Bit Manipulation**: Number theory, modular arithmetic, bitwise operations.
- **String Processing**: Pattern matching, parsing, and string transformation.

Each problem includes a title, content description, difficulty rating (easy, medium, or hard), and topic tags that enable fine-grained filtering for targeted evaluations.

## Task Structure

Each LeetCode task includes the following components:

- **ID**: A numeric identifier corresponding to the LeetCode problem number.
- **Title**: The problem name (e.g., "Two Sum", "Longest Substring Without Repeating Characters").
- **Title Slug**: A URL-friendly version of the title used for identification.
- **Content**: The full problem description in HTML/text format, including examples, constraints, and follow-up challenges.
- **Difficulty**: One of `Easy`, `Medium`, or `Hard`.
- **Tags**: Topic tags indicating the algorithmic concepts involved (e.g., `array`, `hash-table`, `dynamic-programming`).
- **Category**: The broader problem category.
- **Instance ID**: An auto-generated identifier in the format `leetcode_{id}` (e.g., `leetcode_1`).

The agent receives the problem title, difficulty, and full description, and must produce a Python solution saved to `solution.py`.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run LeetCode with default settings
    mcpbr run -c config.yaml --benchmark leetcode

    # Run a sample of 20 problems
    mcpbr run -c config.yaml --benchmark leetcode -n 20

    # Run a specific problem by ID or slug
    mcpbr run -c config.yaml --benchmark leetcode -t 1

    # Filter by difficulty
    mcpbr run -c config.yaml --benchmark leetcode --filter-difficulty easy

    # Filter for medium and hard problems
    mcpbr run -c config.yaml --benchmark leetcode \
      --filter-difficulty medium --filter-difficulty hard

    # Filter by topic tag
    mcpbr run -c config.yaml --benchmark leetcode --filter-tags dynamic-programming

    # Combine difficulty and topic filters
    mcpbr run -c config.yaml --benchmark leetcode \
      --filter-difficulty medium --filter-tags array --filter-tags two-pointers
    ```

=== "YAML"

    ```yaml
    benchmark: "leetcode"
    sample_size: 10
    timeout_seconds: 180
    ```

    Configuration filtered by difficulty:

    ```yaml
    benchmark: "leetcode"
    sample_size: 20
    timeout_seconds: 180

    filter_difficulty:
      - "easy"
      - "medium"
    ```

    Configuration filtered by topic:

    ```yaml
    benchmark: "leetcode"
    sample_size: 15
    timeout_seconds: 300

    filter_difficulty:
      - "hard"
    filter_tags:
      - "dynamic-programming"
      - "graph"
    ```

## Evaluation Methodology

LeetCode evaluation in mcpbr checks for correct code execution:

1. **Solution Writing**: The agent's generated code is written to `solution.py` inside the Docker container.
2. **Test Script Assembly**: A test script is created by appending `print('SOLUTION_EXECUTED')` to the solution code, ensuring the evaluation can detect successful execution.
3. **Execution**: The test script is run using Python 3 with a 30-second timeout.
4. **Verification**: A task is marked as **resolved** if the exit code is 0 and the output contains the `SOLUTION_EXECUTED` marker, confirming the code runs without syntax errors, import errors, or runtime exceptions.
5. **Result Reporting**: Results include the resolution status, exit code, and captured stdout/stderr.

Since the LeetCode dataset does not always include structured test cases in a machine-readable format, the evaluation primarily verifies that the generated code is syntactically correct and executes without errors. For more rigorous evaluation, agents are encouraged to include their own test assertions within the solution.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "leetcode_1",
  "resolved": true,
  "exit_code": 0,
  "stdout": "SOLUTION_EXECUTED",
  "stderr": ""
}
```

### Syntax Error

```json
{
  "instance_id": "leetcode_42",
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "SyntaxError: unexpected EOF while parsing"
}
```

### Runtime Error

```json
{
  "instance_id": "leetcode_100",
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "IndexError: list index out of range"
}
```

## Troubleshooting

**Solution has syntax errors**
The most common failure mode is syntax errors in generated code. Check the stderr output for the specific error. Agents should be prompted to validate their code mentally before saving, and to use standard Python constructs.

**Solution imports unavailable modules**
The Docker environment provides a standard Python installation. If the agent's solution imports non-standard libraries (e.g., `sortedcontainers`, `numpy`), the import will fail. Encourage the agent to use only the Python standard library unless the problem explicitly requires specific packages.

**Execution times out**
The evaluation has a 30-second timeout for code execution. Solutions with infinite loops or extremely inefficient algorithms will timeout. If this happens frequently, check whether the agent is implementing brute-force solutions for problems that require optimized approaches.

**Agent does not save solution to correct file**
The evaluation expects the solution to be in `solution.py`. If the agent saves to a different filename or does not save a file at all, the evaluation will fail. Ensure your prompt template clearly instructs the agent to save to `solution.py`.

## Best Practices

- **Start with easy problems** (`--filter-difficulty easy`) to establish a baseline before testing harder problems.
- **Use topic tags for focused evaluation**: Filter by specific algorithmic topics like `--filter-tags dynamic-programming` to evaluate your agent on particular skill areas.
- **Encourage self-testing**: Since the evaluation primarily checks for successful execution, prompting the agent to include its own test cases within the solution improves the meaningfulness of results.
- **Set appropriate timeouts**: 180 seconds is sufficient for most problems. Hard problems with complex implementations may benefit from 300 seconds.
- **Combine filters for targeted assessment**: Use both `filter_difficulty` and `filter_tags` together to create focused evaluation suites (e.g., medium-difficulty graph problems).
- **Track per-difficulty pass rates**: Separate evaluations by difficulty level to understand the complexity threshold where your agent's performance begins to decline.

## Related Links

- [LeetCode Dataset on HuggingFace](https://huggingface.co/datasets/greengerong/leetcode)
- [LeetCode Platform](https://leetcode.com/)
- [Benchmarks Overview](index.md)
- [APPS](apps.md) | [CodeContests](codecontests.md) | [BigCodeBench](bigcodebench.md)
