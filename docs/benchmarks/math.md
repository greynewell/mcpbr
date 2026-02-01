---
description: "MATH benchmark for mcpbr - 12,500 competition mathematics problems from AMC, AIME, and other competitions across 7 subjects and 5 difficulty levels."
benchmark_howto:
  name: "MATH"
  description: "Evaluate MCP server-assisted mathematical reasoning on competition-level math problems spanning algebra, geometry, number theory, and more."
  benchmark_id: "math"
faq:
  - q: "What is the MATH benchmark and what subjects does it cover?"
    a: "The MATH dataset contains 12,500 competition mathematics problems from AMC, AIME, and other competitions. It covers 7 subjects: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, and precalculus. Each problem is rated at one of 5 difficulty levels."
  - q: "How are MATH answers evaluated?"
    a: "The evaluator extracts the answer from \\boxed{answer} format in both the ground truth solution and the agent's response. Answers are normalized by removing LaTeX formatting (\\left, \\right, \\,) and whitespace, then compared as strings for exact match."
  - q: "Can I filter MATH tasks by subject or difficulty?"
    a: "Yes. Use filter_category to select specific subjects (e.g., algebra, geometry) and the level parameter (1-5) or filter_difficulty to filter by difficulty. These can be combined to target specific problem types."
---

# MATH

| Property | Value |
|----------|-------|
| **Benchmark ID** | `math` |
| **Dataset** | [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) |
| **Tasks** | 12,500 competition math problems |
| **Evaluation** | Extracts `\boxed{answer}` with normalization |
| **Output Type** | Mathematical expression (LaTeX or numeric) |
| **Timeout** | 120-300s |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark math -n 20
    ```

## Overview

[MATH](https://github.com/hendrycks/math) is a dataset of 12,500 competition mathematics problems drawn from AMC (American Mathematics Competitions), AIME (American Invitational Mathematics Examination), and other math competitions. The dataset was created to evaluate the mathematical problem-solving capabilities of language models on problems that require deep reasoning, multi-step solutions, and knowledge of advanced mathematical concepts.

Problems span 7 mathematical subjects and are rated across 5 difficulty levels, providing fine-grained evaluation of a model's mathematical capabilities. Unlike GSM8K, which tests grade-school arithmetic, MATH problems require knowledge of algebra, geometry, combinatorics, number theory, and calculus at a competition level.

In mcpbr, the MATH benchmark evaluates how effectively an MCP server assists the language model in solving challenging mathematics problems. The agent is expected to show its work step-by-step and provide a final answer in `\boxed{answer}` format.

## Task Structure

Each MATH task contains the following fields:

| Field | Description |
|-------|-------------|
| **problem** | The competition math problem statement |
| **solution** | Detailed step-by-step solution with the final answer in `\boxed{}` |
| **level** | Difficulty level: `Level 1` through `Level 5` |
| **type** | Mathematical subject category |

**Subjects (7 total):**

| Subject | Description |
|---------|-------------|
| `algebra` | Equations, inequalities, polynomials, functions |
| `counting_and_probability` | Combinatorics, permutations, probability |
| `geometry` | Euclidean geometry, coordinate geometry, trigonometry |
| `intermediate_algebra` | Advanced algebra, sequences, series, complex numbers |
| `number_theory` | Divisibility, primes, modular arithmetic, Diophantine equations |
| `prealgebra` | Arithmetic, fractions, ratios, basic problem solving |
| `precalculus` | Limits, trigonometric identities, conic sections, vectors |

**Difficulty levels (5 total):**

| Level | Description |
|-------|-------------|
| Level 1 | Easiest competition problems, accessible with strong high school math |
| Level 2 | Moderate difficulty, requiring solid mathematical reasoning |
| Level 3 | Challenging, typical of AMC 10/12 mid-range problems |
| Level 4 | Difficult, approaching AIME-level complexity |
| Level 5 | Hardest problems, requiring creative insight and deep knowledge |

**Example task:**

```text
Problem: What is the largest value of $x$ such that the expression
         $\dfrac{\sqrt{x-2}}{x^2+x-6}$ is defined?

Level: Level 3
Type: algebra

Solution: For the expression to be defined, we need $x-2 \ge 0$ and
          $x^2+x-6 \ne 0$. The first condition gives $x \ge 2$.
          Factoring the denominator: $x^2+x-6=(x+3)(x-2)$, so $x \ne -3$
          and $x \ne 2$. Combined: $x > 2$. There is no largest such
          value, but in context the answer is $\boxed{2}$.
```

Instance IDs are generated in the format `math_{index}` based on the position in the filtered/sampled task list.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run MATH with default settings
    mcpbr run -c config.yaml --benchmark math

    # Run a small sample for quick testing
    mcpbr run -c config.yaml --benchmark math -n 20

    # Filter by subject
    mcpbr run -c config.yaml --benchmark math --filter-category algebra

    # Filter by difficulty level (Level 1 = easiest, Level 5 = hardest)
    mcpbr run -c config.yaml --benchmark math --level 3

    # Combine subject and difficulty filters
    mcpbr run -c config.yaml --benchmark math \
      --filter-category geometry --level 2 -n 50

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark math -n 100 -v -o results.json -r report.md
    ```

=== "YAML Configuration"

    ```yaml
    benchmark: "math"
    sample_size: 10
    timeout_seconds: 300
    max_iterations: 20

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter by subject
    filter_category:
      - "algebra"
      - "number_theory"

    # Optional: Enable extended thinking for complex problems
    thinking_budget: 10000
    ```

## Evaluation Methodology

MATH evaluation extracts and compares mathematical answers with normalization:

1. **Ground truth extraction**: The expected answer is extracted from the task's `solution` field by finding the `\boxed{answer}` pattern. The parser handles nested braces up to 2 levels deep to support complex LaTeX expressions like `\boxed{\frac{1}{2}}`.

2. **Agent answer extraction**: The same `\boxed{answer}` extraction is applied to the agent's response. As a fallback, the evaluator looks for "The answer is" or "Final answer:" patterns.

3. **Normalization**: Both answers are normalized before comparison:
    - All whitespace is removed
    - LaTeX formatting commands are stripped: `\left`, `\right`, `\,`
    - The result is a compact string representation

4. **Comparison**: Normalized answers are compared as exact string matches. This means `\frac{1}{2}` and `0.5` would NOT match -- the agent should use the same mathematical notation as the ground truth.

5. **Verdict**: The task is marked as **resolved** if the normalized agent answer exactly matches the normalized ground truth answer.

!!! note "Evaluation is offline"
    Unlike code generation benchmarks, MATH evaluation does not execute code in the Docker container. The comparison is performed entirely on the extracted text answers. The Docker environment is still created (with Python/SymPy available) so the agent can use computation tools during problem solving.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "agent_answer": "\\frac{1}{2}",
  "ground_truth_answer": "\\frac{1}{2}"
}
```

**Failed resolution (wrong answer):**

```json
{
  "resolved": false,
  "agent_answer": "\\frac{1}{3}",
  "ground_truth_answer": "\\frac{1}{2}"
}
```

**Failed resolution (answer not extracted):**

```json
{
  "resolved": false,
  "error": "Could not extract answer from solution"
}
```

**Failed resolution (ground truth parse error):**

```json
{
  "resolved": false,
  "error": "Could not parse ground truth answer"
}
```

## Troubleshooting

**Agent provides correct answer but in different format**

The MATH evaluator uses exact string matching after normalization. If the ground truth is `\frac{1}{2}` but the agent responds with `0.5` or `1/2`, the match will fail. Instruct the agent to use `\boxed{}` format with proper LaTeX notation.

**Agent does not use `\boxed{}` format**

If the agent omits the `\boxed{}` wrapper, the evaluator falls back to "The answer is" pattern matching. For best results, ensure your prompt explicitly requests `\boxed{answer}` format. The default mcpbr prompt includes this instruction.

**Complex LaTeX expressions fail to extract**

The `\boxed{}` parser handles up to 2 levels of nested braces. Deeply nested expressions like `\boxed{\sqrt{\frac{a+\sqrt{b}}{c}}}` should extract correctly. If extraction fails, check for mismatched braces in the agent's output.

**Filtering returns no tasks**

Subject names must match exactly (case-insensitive): `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`. Level filtering matches on the string "Level N" in the task's level field. Verify your filter values against these lists.

## Best Practices

- **Start with easier levels** (Level 1-2) to verify your setup before attempting harder problems. Level 5 problems are extremely challenging even for frontier models.
- **Use longer timeouts** (120-300s) since competition math problems require more reasoning time than code generation tasks.
- **Enable extended thinking** with `thinking_budget: 10000` or higher for Level 4-5 problems where deep reasoning is needed.
- **Filter by subject** to evaluate specific mathematical capabilities. Running `algebra` and `prealgebra` together gives a good baseline before trying harder subjects like `geometry` or `number_theory`.
- **Instruct LaTeX format** -- ensure the agent prompt explicitly requests `\boxed{answer}` format. The default prompt does this, but custom prompts should include the same instruction.
- **Use Python/SymPy for verification** -- the Docker environment includes SymPy. Encourage the agent to verify its symbolic computations programmatically.
- **Pair with GSM8K** -- run GSM8K for basic math reasoning and MATH for advanced reasoning to get a complete picture of mathematical capabilities.
- **Set `max_iterations` to 15-20** since competition problems may require multiple rounds of reasoning and verification.

## Related Links

- [MATH Dataset Repository](https://github.com/hendrycks/math)
- [MATH Paper (Measuring Mathematical Problem Solving)](https://arxiv.org/abs/2103.03874)
- [MATH-lighteval Dataset on HuggingFace](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)
- [GSM8K Benchmark](gsm8k.md) -- grade-school math reasoning
- [Benchmarks Overview](index.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
