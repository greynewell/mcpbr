---
description: "SWE-bench evaluates AI agents on real-world GitHub bug fixes, testing their ability to generate patches that resolve actual software issues from popular Python repositories."
benchmark_howto:
  name: "SWE-bench"
  description: "Real GitHub bug fixes from popular Python repositories with three variants: Verified (500 tasks), Lite (300 tasks), and Full (2,294 tasks)."
  benchmark_id: "swe-bench-verified"
faq:
  - q: "What is the difference between SWE-bench Verified, Lite, and Full?"
    a: "SWE-bench Verified (500 tasks) contains manually validated test cases for the most reliable evaluation. Lite (300 tasks) is a curated subset ideal for quick testing. Full (2,294 tasks) is the complete dataset for comprehensive research evaluations."
  - q: "Do I need pre-built Docker images for SWE-bench?"
    a: "Pre-built images from Epoch AI are recommended but not required. They include the repository at the correct commit with all dependencies pre-installed, which provides faster and more reliable evaluation. Without them, mcpbr will build environments from scratch."
  - q: "How do I filter SWE-bench tasks by repository?"
    a: "Use the filter_category option with repository name patterns. For example, '--filter-category django' will select only Django-related tasks. You can specify multiple categories to include tasks from several repositories."
---

# SWE-bench

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `swe-bench-verified`, `swe-bench-lite`, `swe-bench-full` |
| **Dataset** | [SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified), [SWE-bench/SWE-bench_Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite), [SWE-bench/SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench) |
| **Tasks** | 500 (Verified), 300 (Lite), 2,294 (Full) |
| **Evaluation** | Apply unified diff patch, run FAIL_TO_PASS and PASS_TO_PASS test suites |
| **Output Type** | Patch (unified diff) |
| **Timeout** | 300-600s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark swe-bench-verified
    ```

## Overview

SWE-bench is the gold-standard benchmark for evaluating AI agents on real-world software engineering tasks. Each task is a genuine GitHub issue from a popular open-source Python repository, and the agent must produce a unified diff patch that resolves the bug. The evaluation verifies the fix by running the repository's test suite, checking that previously failing tests now pass while existing passing tests remain unbroken.

SWE-bench is widely used by the research community and industry to measure progress in automated software engineering. mcpbr supports all three official variants:

- **swe-bench-verified** (default): 500 tasks that have been manually validated by human annotators to confirm test correctness. This is the recommended variant for accurate benchmarking.
- **swe-bench-lite**: 300 curated tasks from popular repositories, suitable for quick testing and iteration.
- **swe-bench-full**: The complete dataset of 2,294 tasks for comprehensive evaluation and research purposes.

Pre-built Docker images from [Epoch AI](https://github.com/orgs/Epoch-Research/packages) are available for most tasks. These images include the repository checked out at the correct commit with all dependencies pre-installed and validated, providing faster and more reproducible evaluations.

## Task Structure

Each SWE-bench task includes the following components:

- **Instance ID**: A unique identifier combining the repository and issue number (e.g., `django__django-11099`).
- **Problem Statement**: The original bug description from the GitHub issue, including reproduction steps and expected behavior.
- **Repository**: The GitHub repository name (e.g., `django/django`, `scikit-learn/scikit-learn`).
- **Base Commit**: The specific commit hash where the bug exists, ensuring reproducible evaluation.
- **Test Patch**: Additional test code that verifies the fix, applied alongside the agent's patch.
- **FAIL_TO_PASS**: A list of test cases that should fail before the fix and pass after it is applied.
- **PASS_TO_PASS**: A list of test cases that must continue passing after the fix, ensuring no regressions.

The agent receives the problem statement and access to the repository at the base commit. It must analyze the codebase, identify the root cause, and generate a minimal patch that resolves the issue.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run SWE-bench Verified (default, manually validated tests)
    mcpbr run -c config.yaml --benchmark swe-bench-verified

    # Run SWE-bench Lite (300 tasks, quick testing)
    mcpbr run -c config.yaml --benchmark swe-bench-lite

    # Run SWE-bench Full (2,294 tasks, comprehensive)
    mcpbr run -c config.yaml --benchmark swe-bench-full

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark swe-bench-verified -n 20

    # Run specific tasks by instance ID
    mcpbr run -c config.yaml --benchmark swe-bench-verified -t django__django-11099

    # Filter by repository
    mcpbr run -c config.yaml --benchmark swe-bench-verified --filter-category django

    # Filter by multiple repositories
    mcpbr run -c config.yaml --benchmark swe-bench-verified \
      --filter-category django --filter-category scikit-learn
    ```

=== "YAML"

    ```yaml
    # SWE-bench Verified (recommended)
    benchmark: "swe-bench-verified"
    sample_size: 10
    timeout_seconds: 300
    use_prebuilt_images: true

    # Optional: Filter by repository
    filter_category:
      - "django"
      - "scikit-learn"
    ```

    Alternative variant configurations:

    ```yaml
    # SWE-bench Lite for quick testing
    benchmark: "swe-bench-lite"
    sample_size: 20
    timeout_seconds: 300
    ```

    ```yaml
    # SWE-bench Full for comprehensive evaluation
    benchmark: "swe-bench-full"
    sample_size: 50
    timeout_seconds: 600
    ```

## Evaluation Methodology

SWE-bench evaluation follows a rigorous multi-step process:

1. **Patch Generation**: The agent analyzes the repository and produces a unified diff patch targeting the bug described in the problem statement.
2. **Patch Application**: The generated patch is applied to the repository at the base commit using standard `git apply` or `patch` utilities.
3. **Test Patch Application**: If the task includes a test patch (additional tests that verify the fix), it is applied on top of the agent's changes.
4. **FAIL_TO_PASS Verification**: The tests listed in FAIL_TO_PASS are executed. All of these tests must now pass, confirming the bug has been fixed.
5. **PASS_TO_PASS Verification**: The tests listed in PASS_TO_PASS are executed. All of these tests must continue to pass, confirming no regressions were introduced.
6. **Resolution**: A task is marked as **resolved** only if the patch applies cleanly, all FAIL_TO_PASS tests pass, and all PASS_TO_PASS tests remain passing.

The evaluation uses pre-built Docker images when available (`use_prebuilt_images: true`), which include the repository at the correct commit with all Python dependencies installed and validated. This eliminates environment setup variability and produces more reliable results.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "django__django-11099",
  "resolved": true,
  "patch_applied": true,
  "fail_to_pass": {
    "passed": 3,
    "total": 3
  },
  "pass_to_pass": {
    "passed": 47,
    "total": 47
  }
}
```

### Failed Resolution

```json
{
  "instance_id": "scikit-learn__scikit-learn-13779",
  "resolved": false,
  "patch_applied": true,
  "fail_to_pass": {
    "passed": 1,
    "total": 2
  },
  "pass_to_pass": {
    "passed": 45,
    "total": 47
  }
}
```

### Patch Application Failure

```json
{
  "instance_id": "sympy__sympy-18199",
  "resolved": false,
  "patch_applied": false,
  "eval_error": "Patch failed to apply: hunks FAILED -- saving rejects to file"
}
```

## Troubleshooting

**Patch fails to apply cleanly**
The agent's patch may target incorrect line numbers or file paths. Ensure the agent is working with the correct version of the repository. Pre-built images guarantee the repository is at the exact base commit. If building from scratch, verify the checkout succeeded.

**PASS_TO_PASS tests fail after patch**
The agent introduced a regression. This often happens when the fix is too broad or modifies shared utility functions. Encourage the agent to make minimal, targeted changes by using a focused prompt template.

**Evaluation times out**
SWE-bench tasks involving large repositories or complex test suites may need longer timeouts. Increase `timeout_seconds` to 600 or higher for repositories like Django or Matplotlib. Tasks from smaller repositories like Flask typically complete within 300 seconds.

**Docker image pull fails**
Pre-built images from Epoch AI may not be available for all tasks. If an image pull fails, mcpbr falls back to building the environment from scratch. Set `use_prebuilt_images: false` to always build from scratch, though this is slower and less reliable.

## Best Practices

- **Start with swe-bench-verified** for the most reliable evaluation results, as all tasks have been manually validated.
- **Use pre-built images** (`use_prebuilt_images: true`) for faster and more consistent evaluation environments.
- **Test with a small sample first** (`-n 5` or `-n 10`) before running the full benchmark to verify your configuration works.
- **Filter by repository** using `filter_category` to focus on specific projects relevant to your MCP server's capabilities.
- **Set appropriate timeouts**: 300 seconds works for most tasks, but complex repositories like Django may need 600 seconds.
- **Monitor token usage**: Bug-fixing tasks can require extensive code exploration, so track costs carefully with large sample sizes.
- **Use swe-bench-lite for iteration**: When developing or tuning your MCP server, the Lite variant offers a good balance of coverage and speed.

## Related Links

- [SWE-bench Official Site](https://www.swebench.com/)
- [SWE-bench Paper (arXiv)](https://arxiv.org/abs/2310.06770)
- [SWE-bench Verified Dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)
- [SWE-bench Lite Dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)
- [SWE-bench Full Dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench)
- [Epoch AI Pre-built Images](https://github.com/orgs/Epoch-Research/packages)
- [Benchmarks Overview](index.md)
- [APPS](apps.md) | [CodeContests](codecontests.md) | [BigCodeBench](bigcodebench.md)
