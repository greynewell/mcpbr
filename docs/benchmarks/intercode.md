---
description: "InterCode evaluates agents on interactive coding tasks requiring multi-turn interaction with Bash, SQL, and Python interpreters through observation-action loops."
benchmark_howto:
  name: "InterCode"
  description: "Interactive code environment benchmark testing multi-turn interaction with Bash, SQL, and Python interpreters. Agents must iteratively write and debug code to achieve target outcomes."
  benchmark_id: "intercode"
faq:
  - q: "What environments does InterCode support?"
    a: "InterCode supports three interactive environments: Bash (shell commands and scripting), SQL (database queries using SQLite), and Python (general-purpose scripting). Use filter_category to select tasks for a specific environment."
  - q: "How does InterCode evaluate solutions?"
    a: "InterCode runs the gold (reference) solution in the environment and captures its output. It then compares this output with the agent's output (read from output.txt). The task is resolved if both outputs match exactly after whitespace trimming."
  - q: "Does InterCode install environment-specific tools automatically?"
    a: "Yes. For SQL tasks, sqlite3 is automatically installed in the Docker container during environment setup. Bash and Python environments use the tools available in the base Docker image."
---

# InterCode

| Property | Value |
|----------|-------|
| **Benchmark ID** | `intercode` |
| **Dataset** | [intercode-benchmark/intercode](https://huggingface.co/datasets/intercode-benchmark/intercode) |
| **Tasks** | Interactive coding tasks across Bash, SQL, and Python environments |
| **Evaluation** | Compares gold solution output with agent output (exact match after trimming) |
| **Output Type** | Code execution results (stdout) |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark intercode
    ```

## Overview

[InterCode](https://intercode-benchmark.github.io/) is a framework for evaluating agents in interactive code environments. Unlike benchmarks that test single-shot code generation, InterCode requires agents to engage in multi-turn interactions with code interpreters -- writing commands, observing output, diagnosing errors, and iterating until they reach the correct solution.

InterCode provides three distinct execution environments:

- **Bash**: Shell command tasks including file processing, text manipulation, system queries, and pipeline construction.
- **SQL**: Database query tasks using SQLite, requiring agents to explore schemas, construct queries, and extract specific data.
- **Python**: General-purpose programming tasks executed through the Python interpreter.

In each environment, the agent must interactively explore, execute, and debug code. The evaluation compares the output of the agent's solution against the output of a gold (reference) solution. This tests not just code correctness but the agent's ability to use feedback loops effectively -- a critical skill for real-world development workflows.

InterCode is particularly well-suited for evaluating MCP servers that provide code execution, database access, or interactive shell capabilities.

## Task Structure

Each InterCode task contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Unique identifier for the task |
| **query** | Natural language description of the task to complete |
| **environment** | Target environment: `bash`, `sql`, or `python` |
| **gold_solution** | Reference solution code (not shown to the agent, used for evaluation) |

**Example Bash task:**

```text
Complete the following task in a bash environment:

Count the number of unique IP addresses in /var/log/access.log
and save the result to output.txt.

Use the bash interpreter to solve this interactively.
```

**Example SQL task:**

```text
Complete the following task in a sql environment:

Find the top 5 customers by total order amount from the orders table
and save the result to output.txt.

Use the sql interpreter to solve this interactively.
```

**Example Python task:**

```text
Complete the following task in a python environment:

Write a function that finds all prime numbers up to 1000 and
save the count to output.txt.

Use the python interpreter to solve this interactively.
```

In all cases, the agent must save its final output to `output.txt` in the working directory.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run InterCode with default settings
    mcpbr run -c config.yaml --benchmark intercode

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark intercode -n 20

    # Filter by environment type
    mcpbr run -c config.yaml --benchmark intercode --filter-category bash

    # Run only SQL tasks
    mcpbr run -c config.yaml --benchmark intercode --filter-category sql

    # Run only Python tasks
    mcpbr run -c config.yaml --benchmark intercode --filter-category python

    # Run specific tasks by ID
    mcpbr run -c config.yaml --benchmark intercode -t 42 -t 43

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark intercode -n 10 -v -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "intercode"
    sample_size: 10
    timeout_seconds: 180

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter to specific environment
    filter_category:
      - "bash"
    ```

    Configuration for SQL tasks with longer timeout:

    ```yaml
    benchmark: "intercode"
    sample_size: 10
    timeout_seconds: 300
    max_iterations: 25

    filter_category:
      - "sql"

    model: "sonnet"
    ```

    Configuration for all environments:

    ```yaml
    benchmark: "intercode"
    sample_size: 30
    timeout_seconds: 240

    filter_category:
      - "bash"
      - "sql"
      - "python"

    model: "sonnet"
    ```

## Evaluation Methodology

InterCode evaluation compares the agent's output against a gold solution through the following process:

1. **Environment Preparation**: A Docker container is created for the task. For SQL tasks, `sqlite3` is automatically installed. Bash and Python environments use the default tools in the base image.

2. **Agent Execution**: The agent receives the task query and environment type as a problem statement. It interacts with the environment, iteratively writing and debugging code. The agent must save its final output to `output.txt`.

3. **Gold Solution Execution**: The gold (reference) solution is written to a temporary file in the container and executed in the appropriate interpreter:
   - **Bash tasks**: Executed via `bash /tmp/gold_solution.sh`
   - **SQL tasks**: Executed via `sqlite3 database.db < /tmp/gold_solution.sql`
   - **Python tasks**: Executed via `python3 /tmp/gold_solution.py`

4. **Output Comparison**: The stdout from the gold solution is compared with the contents of the agent's `output.txt` file. Both outputs are trimmed of leading and trailing whitespace.

5. **Resolution**: The task is marked as **resolved** if the gold solution output exactly matches the agent's output after trimming. Even minor formatting differences (extra spaces, different newline patterns) will cause a mismatch.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "gold_output": "247",
  "agent_output": "247"
}
```

**Failed resolution (output mismatch):**

```json
{
  "resolved": false,
  "gold_output": "247",
  "agent_output": "There are 247 unique IP addresses"
}
```

**Failed resolution (no gold solution):**

```json
{
  "resolved": false,
  "error": "No gold solution available"
}
```

## Troubleshooting

**Agent output does not match gold solution format**

InterCode uses exact string matching (after whitespace trimming) between the gold solution output and the agent's `output.txt`. The agent must produce output in the same format as the reference solution. Instruct the agent to output only the raw result without additional text, headers, or formatting.

**SQL tasks fail with "sqlite3: command not found"**

The evaluation automatically installs `sqlite3` during environment setup. If installation fails (e.g., due to network issues in the Docker container), SQL tasks will not work. Verify that `apt-get` has network access inside your Docker configuration.

**Agent does not create `output.txt`**

The evaluation reads the agent's output from `output.txt` in the working directory. If this file does not exist, the agent output will be empty, causing a mismatch. Ensure the agent prompt clearly instructs saving output to this file. The evaluation falls back to an empty string if the file is missing.

**Gold solution execution times out**

Both the gold solution execution and the agent output reading have 30-second and 10-second timeouts respectively. Complex gold solutions that require significant computation may time out. This is rare but can happen with large dataset processing tasks.

## Best Practices

- **Filter by environment type** to focus on the interaction style that matches your MCP server's capabilities (e.g., `bash` for filesystem servers, `sql` for database servers).
- **Instruct raw output only** in the agent prompt to avoid formatting mismatches with the gold solution. The agent should write only the computed result to `output.txt`.
- **Start with Bash tasks** as they typically have the simplest output formats and are easiest to debug.
- **Use 180-300 second timeouts** since multi-turn interactive coding requires time for iteration and debugging.
- **Provide code execution tools** through your MCP server. InterCode tasks require the agent to actually run and observe code, not just generate it.
- **Set `max_iterations` to 20-25** to allow sufficient turns for the agent to explore, make mistakes, and correct its approach.
- **Monitor `output.txt` contents** by running with `-vv` to see exactly what the agent produces and compare it with the gold output in the results.

## Related Links

- [InterCode Project](https://intercode-benchmark.github.io/)
- [InterCode Dataset on HuggingFace](https://huggingface.co/datasets/intercode-benchmark/intercode)
- [InterCode Paper (arXiv)](https://arxiv.org/abs/2306.14898)
- [Benchmarks Overview](index.md)
- [TerminalBench](terminalbench.md) | [RepoQA](repoqa.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
