---
description: "GSM8K benchmark for mcpbr - 1,319 grade-school math word problems testing mathematical reasoning and chain-of-thought capabilities."
benchmark_howto:
  name: "GSM8K"
  description: "Evaluate MCP server-assisted mathematical reasoning on grade-school math word problems from OpenAI's GSM8K dataset."
  benchmark_id: "gsm8k"
faq:
  - q: "What is GSM8K and what does it test?"
    a: "GSM8K (Grade School Math 8K) is a dataset of 8,500 linguistically diverse grade-school math word problems created by OpenAI. The test split contains 1,319 problems requiring 2-8 steps of arithmetic and basic reasoning. mcpbr uses it to evaluate mathematical reasoning and chain-of-thought capabilities."
  - q: "How does mcpbr extract and compare numeric answers?"
    a: "mcpbr supports multiple answer formats: GSM8K format (#### 42), LaTeX boxed (\\boxed{42}), sentence format ('The answer is 42'), dollar amounts ($1,234.56), and numbers with commas (1,234). Answers are compared with both relative tolerance (0.1%) and absolute tolerance (0.001)."
  - q: "Can the agent use Python for calculations in GSM8K?"
    a: "Yes. The Docker environment includes Python 3 with numpy, scipy, and sympy pre-installed. The agent can write and execute Python scripts for complex arithmetic, which is encouraged for multi-step calculations."
---

# GSM8K

| Property | Value |
|----------|-------|
| **Benchmark ID** | `gsm8k` |
| **Dataset** | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| **Tasks** | 1,319 test problems |
| **Evaluation** | Numeric answer extraction with tolerance (rtol=0.001, atol=0.001) |
| **Output Type** | Numeric answer |
| **Timeout** | 60-180s |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark gsm8k -n 20
    ```

## Overview

[GSM8K (Grade School Math 8K)](https://github.com/openai/grade-school-math) is a dataset of 8,500 linguistically diverse grade-school math word problems created by OpenAI. The test split used by mcpbr contains 1,319 problems that each require 2 to 8 steps of mathematical reasoning to solve. Problems involve arithmetic, basic algebra, and real-world reasoning about quantities such as money, time, distances, and rates.

GSM8K is one of the most widely used benchmarks for evaluating chain-of-thought reasoning in language models. Rather than testing code generation, it tests whether the model can break down word problems into logical steps and arrive at the correct numeric answer.

In mcpbr, GSM8K evaluates how effectively an MCP server assists the language model in mathematical reasoning tasks. The environment provides Python with math libraries (numpy, scipy, sympy) so the agent can optionally use computation tools for verification.

## Task Structure

Each GSM8K task contains the following fields:

| Field | Description |
|-------|-------------|
| **question** | A natural language math word problem |
| **answer** | Chain-of-thought solution ending with the numeric answer in `#### N` format |

**Example task:**

```text
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four. She sells
every duck egg at the farmers' market daily for $2. How much in dollars does
she make every day at the farmers' market?

Answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = <<9*2=18>>$18 every day at the farmer's market.
#### 18
```

Instance IDs are generated in the format `gsm8k_{index}` where the index corresponds to the position in the test split (e.g., `gsm8k_0`, `gsm8k_1`).

## Running the Benchmark

=== "CLI"

    ```bash
    # Run GSM8K with default settings
    mcpbr run -c config.yaml --benchmark gsm8k

    # Run a small sample for quick testing
    mcpbr run -c config.yaml --benchmark gsm8k -n 20

    # Run specific tasks by index
    mcpbr run -c config.yaml --benchmark gsm8k -t 0 -t 1 -t 2

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark gsm8k -n 50 -v -o results.json

    # Save both JSON results and Markdown report
    mcpbr run -c config.yaml --benchmark gsm8k -n 100 -o results.json -r report.md
    ```

=== "YAML Configuration"

    ```yaml
    benchmark: "gsm8k"
    sample_size: 10
    timeout_seconds: 180
    max_iterations: 15

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Custom prompt encouraging chain-of-thought
    agent_prompt: |
      Solve this math problem step-by-step:

      {problem_statement}

      Show your work clearly. Use Python if needed for calculations.
      End with: "The answer is: [number]"
    ```

## Evaluation Methodology

GSM8K evaluation focuses on extracting and comparing numeric answers rather than executing code:

1. **Ground truth extraction**: The expected numeric answer is extracted from the task's `answer` field, which uses the GSM8K `#### N` format.

2. **Agent answer extraction**: The evaluator attempts to extract a numeric answer from the agent's response using multiple pattern-matching strategies, tried in this order:

    | Priority | Pattern | Example |
    |----------|---------|---------|
    | 1 | GSM8K format: `#### N` | `#### 42` |
    | 2 | LaTeX boxed: `\boxed{N}` | `\boxed{42}` |
    | 3 | Sentence format | `The answer is 42` or `Final answer: 42` |
    | 4 | Dollar amounts | `$1,234.56` |
    | 5 | Last number in text (fallback) | Any numeric value |

    Numbers with commas (e.g., `1,234`), dollar signs, and percentage symbols are automatically cleaned during parsing.

3. **Comparison**: The extracted numeric values are compared using both relative and absolute tolerance:
    - **Relative tolerance (rtol)**: 0.001 (0.1%) -- handles large numbers where small absolute differences are acceptable
    - **Absolute tolerance (atol)**: 0.001 -- handles small numbers and rounding differences
    - A match is declared if the absolute difference is within `atol` OR the relative difference is within `rtol`

4. **Verdict**: The task is marked as **resolved** if the agent's extracted answer matches the ground truth within tolerance.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "agent_answer": 18.0,
  "ground_truth_answer": 18.0,
  "answer_match": true
}
```

**Failed resolution (wrong answer):**

```json
{
  "resolved": false,
  "agent_answer": 16.0,
  "ground_truth_answer": 18.0,
  "answer_match": false
}
```

**Failed resolution (no answer extracted):**

```json
{
  "resolved": false,
  "error": "Could not extract numeric answer from agent's solution",
  "agent_solution": "Janet has 16 eggs. She eats 3 and bakes 4. She sells the rest at $2 each..."
}
```

**Failed resolution (ground truth parse error):**

```json
{
  "resolved": false,
  "error": "Could not parse ground truth answer: [malformed answer string]"
}
```

## Troubleshooting

**Agent does not provide a clear numeric answer**

If the agent provides extensive reasoning but does not clearly state the final answer, the evaluator falls back to extracting the last number in the response. For more reliable extraction, instruct the agent to end with a specific format like `The answer is: [number]` or `#### [number]` in your prompt.

**Answer is correct but marked as wrong**

Check for unit mismatches (e.g., the agent says "18 dollars" but the ground truth is just "18"). The evaluator strips dollar signs and commas, but unusual formatting may cause issues. Also verify that the tolerance is sufficient for your problem set -- the default 0.1% relative tolerance should handle most rounding differences.

**Environment setup is slow**

GSM8K environments install Python 3 along with numpy, scipy, and sympy. This initial setup can take 1-2 minutes per container. For faster iteration, consider running fewer concurrent tasks so containers are reused, or increase `timeout_seconds` to account for setup time.

**Agent produces code instead of a numeric answer**

Some agents may write Python scripts to solve the problem. This is fine as long as the agent's final response includes the numeric answer. The evaluator extracts the answer from the agent's text response, not from executed code output. If the agent relies entirely on code execution, ensure it prints or states the result.

## Best Practices

- **Start with a small sample** (10-20 problems) to verify answer extraction works correctly with your model and prompt.
- **Enable chain-of-thought** in the agent prompt for better reasoning. GSM8K was specifically designed to benefit from step-by-step reasoning.
- **Check answer format** by reviewing a few results to ensure the agent clearly states its final numeric answer.
- **Use Python for calculations** -- the environment includes numpy, scipy, and sympy. Encourage the agent to use Python for multi-step arithmetic to reduce calculation errors.
- **Low resource usage** -- GSM8K tasks require minimal Docker environments, so you can run higher concurrency (`max_concurrent: 8` or more).
- **Quick evaluation** -- problems typically solve in under 3 minutes. Set `timeout_seconds: 180` for comfortable margins.
- **Monitor token usage** -- chain-of-thought reasoning increases input/output tokens. Consider using `thinking_budget` in your YAML configuration for extended thinking mode if your model supports it.
- **Pair with MATH benchmark** -- GSM8K tests basic reasoning while MATH tests competition-level mathematics, giving a complete picture of math capabilities.

## Related Links

- [GSM8K Repository](https://github.com/openai/grade-school-math)
- [GSM8K Paper (Training Verifiers to Solve Math Word Problems)](https://arxiv.org/abs/2110.14168)
- [GSM8K Dataset on HuggingFace](https://huggingface.co/datasets/openai/gsm8k)
- [MATH Benchmark](math.md) -- competition-level mathematics
- [Benchmarks Overview](index.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
