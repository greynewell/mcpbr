---
description: "BigBench-Hard benchmark for mcpbr - 27 challenging reasoning tasks from BIG-Bench where language models score below average human performance."
benchmark_howto:
  name: "BigBench-Hard"
  description: "Evaluate MCP server-assisted reasoning on 27 challenging tasks from BIG-Bench where prior language models fell below human-level performance."
  benchmark_id: "bigbench-hard"
faq:
  - q: "What is BigBench-Hard and why is it significant?"
    a: "BigBench-Hard (BBH) is a curated subset of 27 tasks from the BIG-Bench collaborative benchmark where prior language model evaluations scored below average human performance. These tasks test diverse reasoning capabilities including logical deduction, causal reasoning, date understanding, and object tracking."
  - q: "How are BigBench-Hard answers evaluated?"
    a: "Evaluation uses exact match (case-insensitive) on the last non-empty line of the agent's response compared to the target answer. The agent must provide a clear, definitive final answer as the last line of its output."
  - q: "Can I run only specific BBH subtasks?"
    a: "Yes. Use filter_category to select specific subtask names such as 'boolean_expressions', 'date_understanding', or 'logical_deduction_five_objects'. Multiple subtasks can be specified."
---

# BigBench-Hard

| Property | Value |
|----------|-------|
| **Benchmark ID** | `bigbench-hard` |
| **Dataset** | [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh) |
| **Tasks** | 27 challenging subtasks (varying number of examples per subtask) |
| **Evaluation** | Exact match on last line of solution (case-insensitive) |
| **Output Type** | Text answer (exact match) |
| **Timeout** | 60-180s |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark bigbench-hard -n 20
    ```

## Overview

[BigBench-Hard (BBH)](https://github.com/suzgunmirac/BIG-Bench-Hard) is a curated collection of 27 tasks from the [BIG-Bench](https://github.com/google/BIG-bench) collaborative benchmark. These tasks were specifically selected because prior language model evaluations (including PaLM 540B) fell below average human-rater performance. BBH tasks span a diverse range of reasoning capabilities including logical deduction, temporal reasoning, boolean logic, causal judgment, natural language understanding, and algorithmic thinking.

BBH is widely used to evaluate whether language models can perform complex multi-step reasoning when given appropriate prompting strategies such as chain-of-thought. The tasks are designed to be challenging for models but solvable by humans with careful thinking.

In mcpbr, BigBench-Hard evaluates how effectively an MCP server assists the language model in reasoning tasks that require careful step-by-step thinking and precise answers.

## Task Structure

Each BBH task contains the following fields:

| Field | Description |
|-------|-------------|
| **input** | The task prompt with the question or problem to solve |
| **target** | The expected answer (ground truth) |
| **subtask** | The name of the BBH subtask category |

**All 27 subtasks:**

| Subtask | Description |
|---------|-------------|
| `boolean_expressions` | Evaluate nested boolean expressions |
| `causal_judgement` | Determine causal relationships in scenarios |
| `date_understanding` | Reason about dates and temporal relationships |
| `disambiguation_qa` | Resolve ambiguous pronoun references |
| `dyck_languages` | Complete sequences of balanced parentheses |
| `formal_fallacies` | Identify logical fallacies in arguments |
| `geometric_shapes` | Reason about geometric shapes from SVG paths |
| `hyperbaton` | Identify correct adjective ordering in English |
| `logical_deduction_five_objects` | Deduce orderings from clues (5 objects) |
| `logical_deduction_seven_objects` | Deduce orderings from clues (7 objects) |
| `logical_deduction_three_objects` | Deduce orderings from clues (3 objects) |
| `movie_recommendation` | Recommend movies based on preferences |
| `multistep_arithmetic_two` | Solve multi-step arithmetic expressions |
| `navigate` | Determine final position after navigation instructions |
| `object_counting` | Count objects described in text |
| `penguins_in_a_table` | Answer questions about tabular penguin data |
| `reasoning_about_colored_objects` | Reason about object colors and positions |
| `ruin_names` | Identify humorous edits to artist/movie names |
| `salient_translation_error_detection` | Find errors in translations |
| `snarks` | Identify sarcastic statements |
| `sports_understanding` | Reason about plausibility of sports statements |
| `temporal_sequences` | Reason about temporal ordering of events |
| `tracking_shuffled_objects_five_objects` | Track object positions through shuffles (5) |
| `tracking_shuffled_objects_seven_objects` | Track object positions through shuffles (7) |
| `tracking_shuffled_objects_three_objects` | Track object positions through shuffles (3) |
| `web_of_lies` | Determine truth values through chains of assertions |
| `word_sorting` | Sort words alphabetically |

Instance IDs are generated in the format `bbh_{subtask}_{index}` (e.g., `bbh_boolean_expressions_0`, `bbh_date_understanding_14`).

**Example task (boolean_expressions):**

```text
Input: not ( ( not not True ) ) is

Target: False
```

**Example task (date_understanding):**

```text
Input: Today is Christmas Eve of 1937. What is the date 10 days ago
       in MM/DD/YYYY?
       Options:
       (A) 12/14/2026
       (B) 12/14/1937
       (C) 12/14/1938
       (D) 12/14/1924

Target: (B)
```

## Running the Benchmark

=== "CLI"

    ```bash
    # Run BigBench-Hard with default settings (all 27 subtasks)
    mcpbr run -c config.yaml --benchmark bigbench-hard

    # Run a small sample for quick testing
    mcpbr run -c config.yaml --benchmark bigbench-hard -n 20

    # Filter to specific subtasks
    mcpbr run -c config.yaml --benchmark bigbench-hard \
      --filter-category boolean_expressions \
      --filter-category date_understanding

    # Run only logical deduction tasks
    mcpbr run -c config.yaml --benchmark bigbench-hard \
      --filter-category logical_deduction_three_objects \
      --filter-category logical_deduction_five_objects \
      --filter-category logical_deduction_seven_objects

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark bigbench-hard -n 100 -v -o results.json

    # Run specific tasks by ID
    mcpbr run -c config.yaml --benchmark bigbench-hard \
      -t bbh_boolean_expressions_0 -t bbh_date_understanding_14
    ```

=== "YAML Configuration"

    ```yaml
    benchmark: "bigbench-hard"
    sample_size: 10
    timeout_seconds: 180
    max_iterations: 15

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter to specific subtasks
    filter_category:
      - "boolean_expressions"
      - "logical_deduction_five_objects"
      - "tracking_shuffled_objects_three_objects"
    ```

## Evaluation Methodology

BigBench-Hard uses a simple but strict exact-match evaluation:

1. **Target extraction**: The expected answer is taken from the task's `target` field and normalized by stripping whitespace and converting to lowercase.

2. **Agent answer extraction**: The agent's response is processed by:
    - Splitting the response into lines
    - Removing empty lines
    - Taking the **last non-empty line** as the agent's final answer
    - Stripping whitespace and converting to lowercase

3. **Comparison**: The normalized agent answer is compared to the normalized target for exact string equality (case-insensitive).

4. **Verdict**: The task is marked as **resolved** if the agent's last non-empty line exactly matches the target answer.

!!! warning "Last-line extraction"
    The evaluator uses the **last non-empty line** of the agent's response as the answer. This means the agent must place its final answer on the last line. If the agent adds commentary or explanations after the answer, the evaluation will likely fail.

!!! note "Evaluation is offline"
    BBH evaluation does not execute code in the Docker container. The comparison is performed entirely on text. The Docker environment is created so the agent has access to tools for any computation it wants to perform during reasoning, but the final evaluation is text-based.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "agent_answer": "not ( ( not not True ) ) is\n\nLet me work through this step by step:\n1. Start from the innermost: not True = False\n2. not False = True  \n3. not not True = True\n4. ( not not True ) = True\n5. ( True ) = True\n6. not ( True ) = False\n\nFalse",
  "target": "False"
}
```

**Failed resolution (wrong answer):**

```json
{
  "resolved": false,
  "agent_answer": "Let me think about this...\n\nnot ( ( not not True ) )\n= not ( ( True ) )\n= not ( True )\n= True",
  "target": "False"
}
```

**Failed resolution (no target available):**

```json
{
  "resolved": false,
  "error": "No target answer available"
}
```

## Troubleshooting

**Agent provides correct reasoning but wrong final line**

The evaluator strictly uses the last non-empty line. If the agent writes "The answer is False" but then adds "Let me know if you need more explanation" on the next line, the evaluation will fail. Instruct the agent to place only the answer on the final line.

**Subtask fails to load from the dataset**

BBH loads each subtask separately from the HuggingFace dataset. If a specific subtask cannot be loaded (network issues, dataset changes), it is skipped with a warning and the remaining subtasks are still evaluated. Check the logs for `Failed to load BBH subtask` warnings.

**Case sensitivity issues**

The evaluation is case-insensitive -- `True`, `true`, `TRUE`, and `tRuE` all match. However, the comparison is otherwise strict: `(B)` and `B` would NOT match. Ensure the agent includes the full answer format expected by the task (including parentheses for multiple-choice answers).

**No tasks loaded after filtering**

If `filter_category` values do not match any of the 27 official subtask names, no tasks will be loaded. Subtask names use underscores (e.g., `boolean_expressions`, not `boolean-expressions`). Check the subtask table above for exact names.

## Best Practices

- **Start with specific subtasks** rather than all 27 at once. Boolean expressions, word sorting, and object counting are good starting points for verifying your setup.
- **Use chain-of-thought prompting** -- BBH tasks were originally studied to show that chain-of-thought significantly improves performance over direct answering.
- **Instruct clear final answers** -- ensure your prompt tells the agent to put only the final answer on the last line. The default mcpbr prompt includes "Provide a clear, definitive final answer."
- **Group related subtasks** for focused evaluation. For example, run all three `logical_deduction` variants together to evaluate deductive reasoning, or all three `tracking_shuffled_objects` variants for state-tracking ability.
- **Use shorter timeouts** (60-180s) since BBH tasks are reasoning problems that do not require code compilation or complex setup.
- **Set `max_iterations` to 10-15** since BBH tasks typically require a single reasoning pass rather than iterative refinement.
- **Compare across subtasks** -- different subtasks test different capabilities. A model might excel at `boolean_expressions` but struggle with `causal_judgement`. Use subtask-level results to identify specific reasoning weaknesses.
- **Monitor answer format** for multiple-choice tasks (like `date_understanding`) to ensure the agent includes the option letter with parentheses, e.g., `(B)` rather than just `B`.

## Related Links

- [BIG-Bench Hard Repository](https://github.com/suzgunmirac/BIG-Bench-Hard)
- [BIG-Bench Hard Paper (Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them)](https://arxiv.org/abs/2210.09261)
- [BBH Dataset on HuggingFace](https://huggingface.co/datasets/lukaemon/bbh)
- [BIG-Bench Project](https://github.com/google/BIG-bench)
- [Benchmarks Overview](index.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
