---
description: "GAIA benchmark for evaluating general AI assistant capabilities including reasoning, web browsing, and tool use."
benchmark_howto:
  name: "GAIA"
  description: "Tests real-world AI assistant capabilities with questions requiring reasoning, multi-modality, web browsing, and tool use across three difficulty levels."
  benchmark_id: "gaia"
faq:
  - q: "What makes GAIA different from other benchmarks?"
    a: "GAIA tests real-world assistant capabilities that require combining multiple skills: reasoning, multi-modality, web browsing, and tool use. Unlike academic benchmarks, GAIA questions are designed to be easy for humans but hard for AI, with unambiguous, fact-based answers."
  - q: "How are GAIA difficulty levels structured?"
    a: "Level 1 questions require basic reasoning or a single tool. Level 2 questions need multi-step reasoning or combining multiple tools. Level 3 questions demand complex planning, multi-hop reasoning, and sophisticated tool orchestration."
  - q: "How does GAIA evaluation work?"
    a: "GAIA uses exact match evaluation. The model's answer is normalized (lowercased, stripped) and compared against the ground truth. The answer matches if the normalized ground truth equals or is a substring of the normalized response."
---

# GAIA

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `gaia` |
| **Dataset** | [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) |
| **Tasks** | ~460 validation questions |
| **Evaluation** | Exact match on final answer (case-insensitive) |
| **Output Type** | Free-form answer |
| **Timeout** | 180-600 seconds |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark gaia -n 10
    ```

## Overview

GAIA (General AI Assistants) is a benchmark designed to evaluate AI systems on real-world questions that require fundamental assistant capabilities. Unlike many academic benchmarks that test narrow skills, GAIA questions are intentionally designed to be:

- **Easy for humans**: A human with access to standard tools (web browser, calculator, etc.) can answer most questions in minutes
- **Hard for AI**: Questions require combining multiple capabilities that current AI systems struggle to integrate

GAIA covers a broad range of tasks including:

- **Factual reasoning**: Questions that require multi-step logical deduction
- **Multi-modality**: Tasks involving images, audio, or documents
- **Web browsing**: Questions requiring information retrieval from the internet
- **Tool use**: Tasks that need calculator, code execution, or API access
- **File handling**: Questions that involve processing attached files

Each question has an unambiguous, fact-based answer that can be verified without subjective judgment.

### Difficulty Levels

GAIA organizes questions into three difficulty levels:

| Level | Description | Typical Skills Required |
|-------|-------------|------------------------|
| **Level 1** | Simple questions | Single tool use, basic reasoning, direct factual lookup |
| **Level 2** | Moderate questions | Multi-step reasoning, combining 2-3 tools, information synthesis |
| **Level 3** | Complex questions | Multi-hop reasoning, complex planning, sophisticated tool orchestration |

## Task Structure

Each GAIA task contains the following fields:

- **Question**: The question to answer (may reference attached files)
- **Level**: Difficulty level (1, 2, or 3)
- **Final answer**: The ground truth answer for evaluation
- **task_id**: Unique identifier for the task
- **Annotator Metadata**: Additional context from human annotators (steps required, tools needed)

The agent receives the question with its difficulty level and must provide a concise, factual answer.

### Example Task

```text
Difficulty Level: 1

Question: What is the population of the capital city of the country where
the 2024 Summer Olympics were held? Give your answer to the nearest million.

Expected Answer: 2 million
```

### Example Task (Level 3)

```text
Difficulty Level: 3

Question: Download the CSV file at [URL]. Calculate the median value in the
'revenue' column for entries where the 'region' field is 'North America',
then convert this value to EUR using the exchange rate on January 15, 2024.
Round to the nearest hundred.

Expected Answer: 45200
```

## Running the Benchmark

=== "CLI"

    ```bash
    # Run GAIA with default settings
    mcpbr run -c config.yaml --benchmark gaia

    # Run a small sample
    mcpbr run -c config.yaml --benchmark gaia -n 10

    # Filter by difficulty level using the level parameter
    mcpbr run -c config.yaml --benchmark gaia --level 1

    # Filter by difficulty using filter-difficulty
    mcpbr run -c config.yaml --benchmark gaia --filter-difficulty 1

    # Run only Level 3 (hardest) questions
    mcpbr run -c config.yaml --benchmark gaia --filter-difficulty 3

    # Run with extended timeout for complex tasks
    mcpbr run -c config.yaml --benchmark gaia -n 10 -v -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "gaia"
    sample_size: 10
    timeout_seconds: 300

    # Optional: filter by difficulty level
    filter_difficulty:
      - "1"
      - "2"
    ```

### Level Filtering

GAIA supports two methods for filtering by difficulty:

1. **`--level` flag**: Directly sets the difficulty level (1, 2, or 3). This is applied first during task loading.

2. **`--filter-difficulty` flag**: Accepts difficulty level strings. Applied after level filtering, providing additional refinement.

Both methods are case-sensitive on the numeric value. Valid values are `1`, `2`, and `3`.

## Evaluation Methodology

GAIA uses exact match evaluation with normalization:

1. **Answer normalization**: Both the ground truth and the model's response are stripped of leading/trailing whitespace and lowercased.

2. **Exact match check**: The normalized ground truth is compared to the normalized response. A match occurs if:
   - The ground truth exactly equals the response, OR
   - The ground truth is a substring of the response

3. **Result determination**: The task is resolved if and only if the normalized ground truth matches or is contained within the normalized response.

### Scoring

```
resolved = (gt_normalized == solution_normalized) OR (gt_normalized in solution_normalized)
```

Where:
- `gt_normalized`: Ground truth answer, stripped and lowercased
- `solution_normalized`: Model's response, stripped and lowercased

### Answer Format

GAIA expects concise, factual answers. The substring matching allows for some flexibility:

- `"Paris"` matches ground truth `"paris"` (exact match after normalization)
- `"The answer is 42"` matches ground truth `"42"` (substring match)
- `"Based on my research, the population is approximately 2 million"` matches ground truth `"2 million"` (substring match)

However, overly verbose responses risk matching unintended substrings. Keep answers concise.

## Example Output

### Successful Evaluation

```json
{
  "resolved": true,
  "agent_answer": "The population of Paris is approximately 2 million.",
  "ground_truth": "2 million"
}
```

### Failed Evaluation (Wrong Answer)

```json
{
  "resolved": false,
  "agent_answer": "The population is about 11 million in the metropolitan area.",
  "ground_truth": "2 million"
}
```

### Failed Evaluation (No Ground Truth)

```json
{
  "resolved": false,
  "error": "No ground truth answer available"
}
```

## Troubleshooting

### Model gives correct reasoning but wrong final answer format

GAIA expects answers in a specific format matching the ground truth exactly. If the ground truth is "42" but the model responds "42.0" or "forty-two", the evaluation will fail. Instruct the model to provide concise, precise answers:

```yaml
agent_prompt: |
  {problem_statement}

  Provide your final answer as concisely as possible. Match the expected format exactly.
  If the answer is a number, provide just the number. If it is a name, provide just the name.
```

### Timeout issues on Level 3 questions

Level 3 questions often require multi-step reasoning, web browsing, and tool use, which can take significant time. Increase the timeout for GAIA evaluations:

```yaml
timeout_seconds: 600  # 10 minutes for complex GAIA tasks
```

### Model cannot access referenced files or URLs

Some GAIA tasks reference external files or URLs that the model needs to access. Ensure your MCP server configuration provides the necessary tools (web browsing, file access) for the model to complete these tasks.

### Low scores on Level 1 questions

If the model struggles even with Level 1 questions, verify that:

1. The model has access to web browsing tools for factual lookups
2. The timeout is sufficient (at least 180 seconds)
3. The prompt instructs the model to provide concise answers matching the expected format

## Best Practices

- **Start with Level 1**: Begin evaluation with Level 1 questions to establish a baseline. These questions test fundamental capabilities and help verify your configuration.
- **Use generous timeouts**: GAIA tasks often require multi-step reasoning and tool use. Set `timeout_seconds` to at least 300 for Level 2 and 600 for Level 3.
- **Provide tool access**: GAIA is designed to test tool use. Ensure your MCP server provides web browsing, file access, and computation tools for the best results.
- **Keep answers concise**: The exact-match evaluation penalizes verbose responses. Instruct the model to provide only the final answer without extensive explanation.
- **Evaluate per-level**: Track accuracy separately for each difficulty level. This provides a more nuanced view of model capabilities than aggregate accuracy.
- **Compare with human baseline**: GAIA reports human accuracy per level. Use these baselines to contextualize your model's performance.
- **Monitor the agent_answer field**: The evaluation truncates stored agent answers to 500 characters. For debugging, use verbose output to see full responses.

## Related Links

- [Benchmarks Overview](index.md)
- [AgentBench](agentbench.md) - Multi-environment agent benchmark
- [MCPToolBench++](mcptoolbench.md) - MCP-specific tool use benchmark
- [ToolBench](toolbench.md) - Real-world API tool use benchmark
- [GAIA Dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [GAIA Paper](https://arxiv.org/abs/2311.12983)
- [GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
