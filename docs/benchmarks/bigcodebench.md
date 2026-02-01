---
description: "BigCodeBench evaluates AI agents on 1,140 practical coding tasks requiring function composition from 139 libraries across 7 domains, testing real-world library usage skills."
benchmark_howto:
  name: "BigCodeBench"
  description: "1,140 practical coding tasks requiring function composition from 139 libraries across 7 domains including data science, web development, and system programming."
  benchmark_id: "bigcodebench"
faq:
  - q: "What makes BigCodeBench different from other coding benchmarks?"
    a: "BigCodeBench focuses on practical coding tasks that require composing multiple function calls from real-world libraries (NumPy, Pandas, Flask, etc.) rather than pure algorithmic problems. It tests whether agents can effectively use library APIs in realistic scenarios."
  - q: "What domains and libraries does BigCodeBench cover?"
    a: "BigCodeBench covers 7 domains with 139 libraries including data analysis (Pandas, NumPy), machine learning (scikit-learn, TensorFlow), web development (Flask, Django), data visualization (Matplotlib, Seaborn), file processing, networking, and system programming."
  - q: "Can I filter tasks by specific library?"
    a: "Yes. Use filter_tags to select tasks that require specific libraries (e.g., '--filter-tags pandas --filter-tags numpy'). Use filter_category to filter by broader domain categories. Tags require all specified libraries to be present in the task."
---

# BigCodeBench

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `bigcodebench` |
| **Dataset** | [bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench) |
| **Tasks** | 1,140 coding tasks |
| **Evaluation** | Run provided test cases against generated code |
| **Output Type** | Test pass/fail |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark bigcodebench
    ```

## Overview

BigCodeBench is a benchmark designed to evaluate code generation capabilities on practical, real-world programming tasks. Unlike algorithmic benchmarks that focus on standalone functions, BigCodeBench tasks require composing multiple function calls from 139 different libraries across 7 major domains. This makes it an excellent test of whether an AI agent can effectively leverage library APIs in realistic software development scenarios.

The benchmark includes 1,140 tasks that span a diverse range of practical programming activities:

- **Data Analysis**: Tasks using Pandas, NumPy, SciPy for data manipulation and statistical analysis.
- **Machine Learning**: Tasks involving scikit-learn, TensorFlow, PyTorch for model building and evaluation.
- **Web Development**: Tasks using Flask, Django, requests for building and interacting with web services.
- **Data Visualization**: Tasks requiring Matplotlib, Seaborn, Plotly for creating charts and plots.
- **File Processing**: Tasks involving CSV, JSON, XML parsing and file system operations.
- **Networking**: Tasks using socket, HTTP libraries, and API interactions.
- **System Programming**: Tasks involving OS operations, subprocess management, and system utilities.

Each task provides either an instruction prompt (describing what to implement) or a completion prompt (providing partial code to complete), along with test cases that validate the implementation.

## Task Structure

Each BigCodeBench task includes the following components:

- **Task ID**: A unique identifier for the task (e.g., `BigCodeBench/0`).
- **Instruct Prompt**: A natural language description of the function to implement, including its purpose, parameters, and expected behavior.
- **Complete Prompt**: An alternative prompt format providing the function signature and partial implementation for completion.
- **Test Code**: Python test cases that validate the generated implementation against expected behavior.
- **Libraries**: A list of required libraries that the solution must use (e.g., `["pandas", "numpy", "matplotlib"]`).
- **Domain**: The broad domain category the task belongs to.
- **Instance ID**: An auto-generated identifier in the format `bigcodebench_{task_id}`.

The agent receives the instruction or completion prompt along with the list of required libraries and must produce a complete implementation that passes all test cases.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run BigCodeBench with default settings
    mcpbr run -c config.yaml --benchmark bigcodebench

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark bigcodebench -n 20

    # Run a specific task
    mcpbr run -c config.yaml --benchmark bigcodebench -t BigCodeBench/0

    # Filter by domain
    mcpbr run -c config.yaml --benchmark bigcodebench --filter-category "data analysis"

    # Filter by required library
    mcpbr run -c config.yaml --benchmark bigcodebench --filter-tags pandas

    # Filter tasks requiring both pandas and numpy
    mcpbr run -c config.yaml --benchmark bigcodebench \
      --filter-tags pandas --filter-tags numpy
    ```

=== "YAML"

    ```yaml
    benchmark: "bigcodebench"
    sample_size: 10
    timeout_seconds: 300
    ```

    Configuration filtered by domain:

    ```yaml
    benchmark: "bigcodebench"
    sample_size: 20
    timeout_seconds: 300

    # Filter by domain
    filter_category:
      - "data analysis"
      - "machine learning"
    ```

    Configuration filtered by libraries:

    ```yaml
    benchmark: "bigcodebench"
    sample_size: 15
    timeout_seconds: 300

    # Only tasks requiring these specific libraries
    filter_tags:
      - "pandas"
      - "matplotlib"
    ```

## Evaluation Methodology

BigCodeBench evaluation combines the generated solution with provided test code:

1. **Solution Generation**: The agent produces a Python implementation based on the task prompt.
2. **Test Assembly**: The generated solution is concatenated with the task's test code to create a single executable test file (`test_solution.py`).
3. **Test Execution**: The combined file is executed using Python 3 inside the Docker container with a 60-second timeout.
4. **Pass/Fail Determination**: The task is marked as **resolved** if the test execution completes with exit code 0 (all assertions pass). Any assertion error, import error, or runtime exception results in a failure.
5. **Result Reporting**: Results include the resolution status, exit code, and captured stdout/stderr for debugging.

Since BigCodeBench tasks require specific libraries, the Docker environment must have the relevant Python packages installed. The evaluation captures both stdout and stderr (truncated to 1,000 characters) to help diagnose failures.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "bigcodebench_BigCodeBench/42",
  "resolved": true,
  "exit_code": 0,
  "stdout": "...",
  "stderr": ""
}
```

### Failed Resolution

```json
{
  "instance_id": "bigcodebench_BigCodeBench/99",
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "AssertionError: Expected DataFrame with 3 columns, got 2"
}
```

### Missing Test Code

```json
{
  "instance_id": "bigcodebench_BigCodeBench/500",
  "resolved": false,
  "error": "No test code provided"
}
```

## Troubleshooting

**ImportError for required libraries**
BigCodeBench tasks require specific Python libraries (e.g., pandas, numpy, scikit-learn). If the Docker environment does not have these packages installed, tests will fail with ImportError. Ensure your Docker image includes the common data science Python stack, or configure a base image that pre-installs these dependencies.

**Test assertions fail despite correct logic**
Some BigCodeBench tests check specific output formats, column names, or data types. The agent must follow the prompt specifications exactly. For example, a task may require returning a Pandas DataFrame with specific column names, and returning a dictionary instead will fail the assertion.

**Timeout during test execution**
The test execution has a 60-second timeout. Tasks involving large dataset generation, model training, or complex plotting may approach this limit. If you see frequent timeouts, consider increasing the per-test timeout in the evaluation configuration.

**Agent does not use required libraries**
BigCodeBench tasks explicitly list required libraries. If the agent implements the solution using different libraries or pure Python, it may technically work but fail specific test assertions that check for library-specific behavior (e.g., verifying the return type is a Pandas DataFrame rather than a list).

## Best Practices

- **Start with a small sample** (`-n 10`) to verify that the Docker environment has the required libraries installed.
- **Filter by domain** using `filter_category` to focus evaluations on specific areas like data analysis or web development.
- **Use filter_tags for library-specific evaluation**: Test your MCP server's effectiveness with specific library ecosystems (e.g., `--filter-tags pandas` for data manipulation tasks).
- **Ensure library availability**: The Docker environment should include common Python packages. Consider using a data science base image or installing dependencies in advance.
- **Review stderr on failures**: The captured stderr output often contains the exact assertion error, making it straightforward to diagnose why a solution failed.
- **Compare instruct vs. complete modes**: BigCodeBench provides both instruction-style and completion-style prompts. Evaluating with both can reveal different aspects of your agent's capabilities.

## Related Links

- [BigCodeBench Paper (arXiv)](https://arxiv.org/abs/2406.15877)
- [BigCodeBench Dataset on HuggingFace](https://huggingface.co/datasets/bigcode/bigcodebench)
- [BigCodeBench GitHub](https://github.com/bigcode-project/bigcodebench)
- [Benchmarks Overview](index.md)
- [APPS](apps.md) | [CodeContests](codecontests.md) | [CoderEval](codereval.md)
