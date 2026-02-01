---
description: "MLAgentBench evaluates AI agents on real ML research tasks based on Kaggle competitions, testing their ability to train models, improve performance metrics, and debug ML pipelines."
benchmark_howto:
  name: "MLAgentBench"
  description: "ML research benchmark based on real Kaggle competitions where agents must train models, improve performance metrics, and debug pipelines, evaluated by comparing achieved scores against baselines."
  benchmark_id: "mlagentbench"
faq:
  - q: "How does MLAgentBench determine if a task is resolved?"
    a: "The evaluation runs an eval script (default: python3 evaluate.py) and extracts a numeric score from the output. The score is compared against a baseline value. For accuracy-style metrics, the agent must exceed the baseline. For loss-style metrics (loss, RMSE, MAE), the agent must achieve a lower value. Metric direction is automatically detected."
  - q: "What ML domains does MLAgentBench cover?"
    a: "MLAgentBench covers multiple ML domains including NLP (natural language processing), CV (computer vision), and tabular data. Use --filter-category to select tasks from a specific domain."
  - q: "Does MLAgentBench detect whether higher or lower scores are better?"
    a: "Yes. MLAgentBench automatically detects metric direction based on the metric name. Metrics containing 'loss', 'rmse', 'mae', 'mse', 'error', or 'perplexity' are treated as lower-is-better. All other metrics (accuracy, score, f1, etc.) are treated as higher-is-better."
---

# MLAgentBench

| Property | Value |
|----------|-------|
| **Benchmark ID** | `mlagentbench` |
| **Dataset** | [MLAgentBench/MLAgentBench](https://huggingface.co/datasets/MLAgentBench/MLAgentBench) |
| **Tasks** | ML research tasks based on real Kaggle competitions and research challenges |
| **Evaluation** | Runs eval script, extracts numeric score, compares against baseline with automatic metric direction detection |
| **Output Type** | Numeric metric (accuracy, loss, F1, etc.) |
| **Timeout** | 300-900s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark mlagentbench
    ```

## Overview

[MLAgentBench](https://github.com/snap-stanford/MLAgentBench) evaluates AI agents on their ability to perform real-world machine learning research tasks. Each task is based on an actual Kaggle competition or ML research challenge, requiring agents to analyze datasets, design and implement model architectures, train models, tune hyperparameters, debug ML pipelines, and ultimately improve performance metrics beyond a given baseline.

Unlike code generation benchmarks that test isolated function implementations, MLAgentBench tests end-to-end ML engineering competency. Agents must:

- Understand the research problem and target metric
- Explore and analyze provided datasets
- Implement or modify ML training pipelines
- Train models and evaluate results
- Iterate on their approach to improve performance

The evaluation is automated: after the agent completes its work, an evaluation script runs in the environment and produces a numeric score. This score is compared against a known baseline to determine whether the agent improved performance. The system automatically detects whether the metric is "higher is better" (e.g., accuracy, F1 score) or "lower is better" (e.g., loss, RMSE) based on the metric name.

MLAgentBench is particularly useful for evaluating MCP servers that provide data analysis, ML framework integration, or computational notebook capabilities.

## Task Structure

Each MLAgentBench task contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Unique identifier for the task |
| **research_problem** | Detailed description of the ML research challenge |
| **domain** | ML domain: `nlp`, `cv`, `tabular`, or other specializations |
| **metric** | Target metric name (e.g., `accuracy`, `loss`, `rmse`, `f1`) |
| **baseline_score** | Known baseline performance to improve upon |
| **eval_command** | Command to run the evaluation script (default: `python3 evaluate.py`) |
| **repo** | Repository with starter code, data, and evaluation scripts |

**Example task:**

```text
Complete the following ML research task:

Improve the text classification model on the IMDB sentiment analysis dataset.
The current model achieves 85% accuracy using a basic logistic regression approach.
Implement a more effective model architecture and training procedure.

Target metric: accuracy
Baseline score: 0.85

Improve upon the baseline and save your results.
```

The agent must analyze the provided code, implement improvements, train a model, and ensure the evaluation script reports a score above 0.85.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run MLAgentBench with default settings
    mcpbr run -c config.yaml --benchmark mlagentbench

    # Run a sample of 5 tasks (ML tasks are resource-intensive)
    mcpbr run -c config.yaml --benchmark mlagentbench -n 5

    # Filter by ML domain
    mcpbr run -c config.yaml --benchmark mlagentbench --filter-category nlp

    # Filter by multiple domains
    mcpbr run -c config.yaml --benchmark mlagentbench \
      --filter-category cv --filter-category tabular

    # Run with extended timeout for training
    mcpbr run -c config.yaml --benchmark mlagentbench -n 3 --timeout 900

    # Run with verbose output
    mcpbr run -c config.yaml --benchmark mlagentbench -n 5 -v

    # Save results to JSON
    mcpbr run -c config.yaml --benchmark mlagentbench -n 10 -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "mlagentbench"
    sample_size: 5
    timeout_seconds: 600

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter by domain
    filter_category:
      - "nlp"
    ```

    Configuration for compute-intensive CV tasks:

    ```yaml
    benchmark: "mlagentbench"
    sample_size: 3
    timeout_seconds: 900
    max_iterations: 40
    max_concurrent: 2

    filter_category:
      - "cv"

    model: "opus"
    ```

    Configuration for quick tabular tasks:

    ```yaml
    benchmark: "mlagentbench"
    sample_size: 10
    timeout_seconds: 300
    max_iterations: 25

    filter_category:
      - "tabular"

    model: "sonnet"
    ```

## Evaluation Methodology

MLAgentBench evaluation measures performance improvement over a baseline through the following process:

1. **Agent Execution**: The agent receives the research problem, target metric, and baseline score. It works within the provided repository to analyze data, modify code, train models, and save results.

2. **Evaluation Script Execution**: After the agent completes its work, the evaluation command (default: `python3 evaluate.py`) is executed in the environment with a 300-second timeout. This script loads the agent's trained model or predictions and computes the target metric.

3. **Score Extraction**: The evaluation parses stdout for a line matching the pattern `score|accuracy|loss|metric = <number>` (case-insensitive). The extracted numeric value is the agent's achieved score.

4. **Metric Direction Detection**: The system automatically determines whether higher or lower values indicate improvement:
   - **Higher is better**: Metrics not containing loss-related keywords (accuracy, score, f1, precision, recall, etc.)
   - **Lower is better**: Metrics containing `loss`, `rmse`, `mae`, `mse`, `error`, or `perplexity`

5. **Baseline Comparison**: The agent's score is compared against the task's baseline:
   - For higher-is-better metrics: resolved if `score > baseline`
   - For lower-is-better metrics: resolved if `score < baseline` (and baseline > 0)

6. **Resolution**: The task is marked as **resolved** if the agent's score improves upon the baseline in the correct direction. If the evaluation script fails (non-zero exit code) or no score can be extracted, the task is marked as unresolved.

## Example Output

**Successful resolution (higher is better):**

```json
{
  "resolved": true,
  "score": 0.912,
  "baseline": 0.85,
  "metric_direction": "higher_is_better"
}
```

**Successful resolution (lower is better):**

```json
{
  "resolved": true,
  "score": 0.234,
  "baseline": 0.312,
  "metric_direction": "lower_is_better"
}
```

**Failed resolution (did not beat baseline):**

```json
{
  "resolved": false,
  "score": 0.83,
  "baseline": 0.85,
  "metric_direction": "higher_is_better"
}
```

**Failed resolution (evaluation script error):**

```json
{
  "resolved": false,
  "error": "Evaluation script failed: ModuleNotFoundError: No module named 'sklearn'"
}
```

## Troubleshooting

**Evaluation script fails with import errors**

ML tasks often require specific Python packages (scikit-learn, torch, tensorflow, pandas, etc.) that may not be in the base Docker image. The agent should install required packages as part of its workflow, or the task environment may need custom setup. Check stderr output for specific missing modules.

**Score not extracted from evaluation output**

The evaluation looks for patterns like `accuracy = 0.92` or `loss: 0.234` in stdout. If the evaluation script uses a different format, the score extraction will fail. Ensure the evaluation script outputs scores in a recognized format: `metric_name = numeric_value` or `metric_name: numeric_value`.

**Training times out**

ML training can be computationally expensive, especially for CV tasks with large datasets or complex models. Increase `timeout_seconds` to 900 for GPU-intensive tasks. Consider reducing `max_concurrent` to 1-2 for resource-constrained environments, and use `--filter-category tabular` for faster tasks during initial testing.

**Agent does not improve baseline**

The agent may struggle with complex ML tasks. Ensure the agent prompt encourages iterative experimentation and mentions the baseline score as a target to beat. Providing more context through `max_iterations: 30-40` gives the agent more attempts to refine its approach.

## Best Practices

- **Start with a very small sample** (`-n 2` or `-n 3`) since ML tasks are computationally expensive and time-consuming.
- **Use extended timeouts** (600-900s) to account for model training time, especially for CV and NLP tasks.
- **Reduce concurrency** (`max_concurrent: 1-2`) for ML tasks that are memory and CPU intensive.
- **Filter by domain** to focus on task types relevant to your evaluation goals. Tabular tasks tend to be fastest; CV tasks tend to be slowest.
- **Increase `max_iterations`** to 30-40 to give the agent sufficient turns to explore the data, implement solutions, train models, and iterate on improvements.
- **Monitor metric direction** in results to verify the system correctly identified whether higher or lower values are better for each task's metric.
- **Start with tabular tasks** (`--filter-category tabular`) for initial testing, as they typically have shorter training times and lower resource requirements.
- **Track costs carefully** since ML tasks require many agent turns and long execution times, which increases API usage.

## Related Links

- [MLAgentBench Repository](https://github.com/snap-stanford/MLAgentBench)
- [MLAgentBench Dataset on HuggingFace](https://huggingface.co/datasets/MLAgentBench/MLAgentBench)
- [MLAgentBench Paper (arXiv)](https://arxiv.org/abs/2310.08036)
- [Benchmarks Overview](index.md)
- [HumanEval](humaneval.md) | [SWE-bench](swe-bench.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
