---
description: "AgentBench benchmark for evaluating LLMs as autonomous agents across diverse environments including OS, databases, and web."
benchmark_howto:
  name: "AgentBench"
  description: "Multi-dimensional benchmark evaluating LLMs as agents across operating systems, databases, knowledge graphs, web shopping, digital card games, and lateral thinking environments."
  benchmark_id: "agentbench"
faq:
  - q: "What environments does AgentBench cover?"
    a: "AgentBench spans multiple environments: operating system (OS) interaction, database (DB) queries, knowledge graph (KG) navigation, web shopping, digital card games, lateral thinking puzzles, and house-holding tasks. Each environment tests different agent capabilities."
  - q: "How does AgentBench evaluation work?"
    a: "AgentBench uses string matching evaluation. The model's response is compared to the expected output using case-insensitive substring matching -- the task is resolved if the expected output appears as a substring within the agent's response."
  - q: "Can I filter AgentBench tasks by environment type?"
    a: "Yes, use the filter_category parameter with environment names such as 'os', 'db', 'kg', 'web', etc. This allows you to evaluate agent performance on specific environment types independently."
---

# AgentBench

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `agentbench` |
| **Dataset** | [THUDM/AgentBench](https://huggingface.co/datasets/THUDM/AgentBench) |
| **Tasks** | Varies by environment |
| **Evaluation** | String matching on expected output (case-insensitive substring) |
| **Output Type** | Completion verification |
| **Timeout** | 180-600 seconds |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark agentbench -n 10
    ```

## Overview

AgentBench is a multi-dimensional benchmark designed to evaluate LLMs as autonomous agents across diverse, interactive environments. Unlike benchmarks that test a single capability, AgentBench provides a comprehensive evaluation of an LLM's ability to understand instructions, interact with environments, and produce correct outcomes across fundamentally different domains.

The benchmark spans the following environment types:

| Environment | Code | Description |
|-------------|------|-------------|
| **Operating System** | `os` | Execute shell commands, navigate file systems, manage processes |
| **Database** | `db` | Write and execute SQL queries, manage data |
| **Knowledge Graph** | `kg` | Navigate and query structured knowledge bases |
| **Web Shopping** | `web` | Browse web stores, find products, complete purchases |
| **Digital Card Game** | `card` | Play strategy card games with defined rules |
| **Lateral Thinking** | `lateral` | Solve puzzles requiring creative, non-obvious reasoning |
| **House-holding** | `house` | Complete household tasks in simulated environments |

AgentBench is particularly valuable for evaluating:

- **Environment adaptation**: Can the agent adjust its approach across different domains?
- **Instruction following**: Does the agent correctly interpret and execute complex instructions?
- **Multi-step planning**: Can the agent break down complex tasks into executable steps?
- **Tool interaction**: How effectively does the agent use environment-specific tools and commands?
- **Error recovery**: Can the agent handle unexpected outcomes and adjust its strategy?

## Task Structure

Each AgentBench task contains the following fields:

- **task_id**: Unique identifier for the task
- **environment**: The environment type (e.g., "os", "db", "kg", "web")
- **instruction** (or **description**): The task description telling the agent what to accomplish
- **expected_output**: The expected result for evaluation

The agent receives the instruction along with the environment context and must interact with the environment to produce the expected output.

### Example Task (OS Environment)

```text
Environment: os

Instruction: Find all Python files in the /home/user/projects directory that
contain the string "import pandas" and count how many there are.

Expected Output: 7
```

### Example Task (Database Environment)

```text
Environment: db

Instruction: Write a SQL query to find the top 5 customers by total order
amount from the orders table. Return their names and total amounts.

Expected Output: SELECT c.name, SUM(o.amount) as total FROM customers c
JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY total DESC LIMIT 5
```

### Example Task (Web Shopping)

```text
Environment: web

Instruction: Find a pair of wireless noise-cancelling headphones under $100
on the shopping website. Add the cheapest option to the cart.

Expected Output: Added to cart
```

## Running the Benchmark

=== "CLI"

    ```bash
    # Run AgentBench with default settings
    mcpbr run -c config.yaml --benchmark agentbench

    # Run a small sample
    mcpbr run -c config.yaml --benchmark agentbench -n 10

    # Filter by environment type
    mcpbr run -c config.yaml --benchmark agentbench --filter-category os
    mcpbr run -c config.yaml --benchmark agentbench --filter-category db

    # Run multiple environment types
    mcpbr run -c config.yaml --benchmark agentbench \
      --filter-category os --filter-category db

    # Run with extended timeout and verbose output
    mcpbr run -c config.yaml --benchmark agentbench -n 20 -v -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "agentbench"
    sample_size: 10
    timeout_seconds: 300

    # Optional: filter to specific environments
    filter_category:
      - "os"
      - "db"
    ```

### Environment Filtering

AgentBench supports filtering by environment type using `filter_category`. The environment names are matched case-insensitively against the `environment` field in each task.

Common environment codes:

| Code | Environment |
|------|-------------|
| `os` | Operating system interaction |
| `db` | Database queries |
| `kg` | Knowledge graph navigation |
| `web` | Web shopping and browsing |
| `card` | Digital card games |
| `lateral` | Lateral thinking puzzles |
| `house` | House-holding tasks |

You can combine multiple environment types in a single evaluation by specifying multiple `filter_category` values.

## Evaluation Methodology

AgentBench uses a straightforward string matching evaluation:

1. **Expected output retrieval**: The `expected_output` field from the task is used as the ground truth.

2. **Case-insensitive substring matching**: Both the expected output and the agent's solution are stripped of whitespace and lowercased. The task is resolved if the normalized expected output appears as a substring within the normalized solution.

3. **Result truncation**: The agent's output is truncated to 500 characters when stored in results for display purposes, but the full output is used for evaluation.

### Scoring

```
resolved = expected_output.strip().lower() in solution.strip().lower()
```

Where:
- `expected_output`: The ground truth answer from the task
- `solution`: The agent's full response text

### Evaluation Characteristics

- **Lenient matching**: Since substring matching is used, the agent can include the expected output within a longer explanation and still pass.
- **Case insensitive**: "SELECT" and "select" are treated as equivalent.
- **Whitespace normalized**: Leading and trailing whitespace is stripped before comparison.
- **No partial credit**: The task is either resolved or not -- there is no scoring gradient.

## Example Output

### Successful Evaluation

```json
{
  "resolved": true,
  "agent_output": "After searching the directory, I found 7 Python files containing 'import pandas'.",
  "expected_output": "7"
}
```

### Failed Evaluation (Wrong Answer)

```json
{
  "resolved": false,
  "agent_output": "I found 5 Python files with the import statement.",
  "expected_output": "7"
}
```

### Failed Evaluation (No Expected Output)

```json
{
  "resolved": false,
  "error": "No expected output available"
}
```

### Successful Evaluation (Substring Match)

```json
{
  "resolved": true,
  "agent_output": "The item has been added to cart successfully. Your cart now contains 1 item.",
  "expected_output": "Added to cart"
}
```

The substring "added to cart" (lowercased) is found within the agent's response.

## Troubleshooting

### Agent provides correct reasoning but wrong output format

Since evaluation uses substring matching, the agent's response must contain the expected output string. If the expected output is "7" and the agent responds "seven", the evaluation will fail. Instruct the agent to include explicit, direct answers:

```yaml
agent_prompt: |
  {problem_statement}

  After completing the task, state your final answer clearly.
  If the answer is a number, include the numeric value.
  If the answer is a command result, include the exact output.
```

### Timeout on complex environment tasks

Some AgentBench tasks, especially those in the OS and web environments, require multiple interaction steps. Increase the timeout accordingly:

```yaml
timeout_seconds: 600  # 10 minutes for complex agent tasks
```

### Environment-specific capabilities are missing

Different AgentBench environments require different agent capabilities:

- **OS tasks**: Need shell access and file system navigation
- **DB tasks**: Need SQL query execution
- **Web tasks**: Need web browsing capabilities
- **KG tasks**: Need knowledge graph query tools

Ensure your MCP server provides the necessary tools for the environment types you are evaluating.

### Low scores across all environments

If the agent performs poorly across all environments, check:

1. The agent has access to the necessary interaction tools
2. The timeout is long enough for multi-step tasks
3. The agent prompt encourages environment interaction rather than just reasoning
4. The model is capable of following complex, multi-step instructions

## Best Practices

- **Evaluate per-environment**: Track performance separately for each environment type. Aggregating across environments can mask environment-specific weaknesses.
- **Start with OS and DB environments**: These are typically the most straightforward and help verify basic agent interaction capabilities before testing more complex environments.
- **Use generous timeouts**: Agent tasks often require multiple rounds of interaction. Set `timeout_seconds` to at least 300, and 600 for web and lateral thinking tasks.
- **Provide appropriate tools**: Match your MCP server capabilities to the environments you are evaluating. An agent without web browsing tools will fail all web shopping tasks regardless of model capability.
- **Monitor agent_output**: The stored output is truncated to 500 characters. Use verbose logging (`-v` or `-vv`) to capture the full agent interaction for debugging.
- **Use for comprehensive evaluation**: AgentBench's multi-environment design makes it ideal for understanding an agent's overall capability profile. Run it alongside domain-specific benchmarks for a complete picture.
- **Compare across models**: AgentBench provides a standardized multi-dimensional score that is useful for comparing different models' agent capabilities.

## Related Links

- [Benchmarks Overview](index.md)
- [GAIA](gaia.md) - General AI assistant benchmark
- [MCPToolBench++](mcptoolbench.md) - MCP-specific tool use benchmark
- [ToolBench](toolbench.md) - Real-world API tool use benchmark
- [AgentBench Dataset](https://huggingface.co/datasets/THUDM/AgentBench)
- [AgentBench Paper](https://arxiv.org/abs/2308.03688)
- [AgentBench Project](https://llmbench.ai/agent)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
