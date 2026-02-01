---
description: "TerminalBench evaluates AI agents on practical terminal and shell tasks including file manipulation, system administration, scripting, and command-line tool usage."
benchmark_howto:
  name: "TerminalBench"
  description: "Terminal/shell task completion benchmark testing file manipulation, system administration, scripting, and tool usage with validation-command-based evaluation."
  benchmark_id: "terminalbench"
faq:
  - q: "How does TerminalBench evaluate task completion?"
    a: "After the agent completes its work, a validation command is executed in the Docker environment. If the validation command exits with code 0, the task is marked as resolved. This approach checks the actual terminal state rather than the agent's textual output."
  - q: "Can I filter TerminalBench tasks by difficulty or category?"
    a: "Yes. Use --filter-difficulty to select tasks by difficulty level (e.g., easy, medium, hard) and --filter-category to select tasks by category (e.g., file-manipulation, system-admin, scripting). Both filters can be combined."
  - q: "Does TerminalBench support setup commands?"
    a: "Yes. Tasks can include a setup_command field that runs before the agent begins its work. This prepares the environment with necessary files, directories, or configurations that the task requires."
---

# TerminalBench

| Property | Value |
|----------|-------|
| **Benchmark ID** | `terminalbench` |
| **Dataset** | [ia03/terminal-bench](https://huggingface.co/datasets/ia03/terminal-bench) |
| **Tasks** | Terminal/shell tasks across file manipulation, system administration, scripting, and tool usage |
| **Evaluation** | Executes validation command, checks exit code (0 = success) |
| **Output Type** | Shell command result (environment state verification) |
| **Timeout** | 120-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark terminalbench
    ```

## Overview

[TerminalBench](https://huggingface.co/datasets/ia03/terminal-bench) is a benchmark that evaluates AI agents' ability to complete practical tasks in a terminal/shell environment. Tasks cover a wide range of command-line competencies -- from basic file manipulation and text processing to system administration, shell scripting, and effective use of Unix tools.

Unlike benchmarks that evaluate code generation in isolation, TerminalBench tests whether an agent can interact with a real shell environment to achieve concrete outcomes. The evaluation does not inspect the agent's textual response; instead, it runs a validation command that checks the actual state of the environment after the agent has finished working. This means the agent must execute real commands that produce lasting changes, not just describe what should be done.

TerminalBench is well-suited for evaluating MCP servers that provide shell access, filesystem operations, or system administration capabilities. It tests practical command-line competency rather than abstract code generation.

## Task Structure

Each TerminalBench task contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Unique identifier for the task |
| **instruction** | Natural language description of the terminal task to complete |
| **category** | Task category (e.g., file-manipulation, system-admin, scripting, tool-usage) |
| **difficulty** | Difficulty level of the task |
| **validation_command** | Shell command that verifies task completion (exit code 0 = success) |
| **setup_command** | Optional command to prepare the environment before the agent starts |

**Example task:**

The agent receives a problem statement like:

```text
Complete the following terminal task (file-manipulation):

Create a directory called 'backup' in /workspace, then copy all .log files
from /var/log into it, preserving file permissions.
```

After the agent executes its commands, the validation command (e.g., `test -d /workspace/backup && ls /workspace/backup/*.log > /dev/null 2>&1`) checks whether the task was completed correctly.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run TerminalBench with default settings
    mcpbr run -c config.yaml --benchmark terminalbench

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark terminalbench -n 20

    # Filter by difficulty
    mcpbr run -c config.yaml --benchmark terminalbench --filter-difficulty easy

    # Filter by category
    mcpbr run -c config.yaml --benchmark terminalbench --filter-category scripting

    # Combine difficulty and category filters
    mcpbr run -c config.yaml --benchmark terminalbench \
      --filter-difficulty medium --filter-category file-manipulation

    # Run with verbose output
    mcpbr run -c config.yaml --benchmark terminalbench -n 10 -v

    # Save results to JSON
    mcpbr run -c config.yaml --benchmark terminalbench -n 20 -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "terminalbench"
    sample_size: 10
    timeout_seconds: 120

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter by difficulty and category
    filter_difficulty:
      - "easy"
      - "medium"
    filter_category:
      - "file-manipulation"
    ```

    Configuration for advanced system administration tasks:

    ```yaml
    benchmark: "terminalbench"
    sample_size: 10
    timeout_seconds: 300
    max_iterations: 25

    filter_category:
      - "system-admin"
      - "scripting"

    model: "sonnet"
    ```

## Evaluation Methodology

TerminalBench evaluation focuses on the actual state of the environment rather than the agent's textual output:

1. **Environment Setup**: If the task includes a `setup_command`, it is executed first to prepare the environment (e.g., creating test files, configuring services). The setup command must succeed (exit code 0) or the task preparation fails.

2. **Agent Execution**: The agent receives the task instruction as a problem statement and interacts with the terminal environment using available shell tools. The agent's textual response is not directly evaluated.

3. **Validation**: After the agent completes its work, the task's `validation_command` is executed in the same environment with a 30-second timeout. This command inspects the environment state to verify the task was completed correctly.

4. **Resolution**: The task is marked as **resolved** if the validation command exits with code 0. Any non-zero exit code means the task was not completed successfully. Both stdout and stderr from the validation command are captured in the results for debugging.

Tasks without a validation command are marked as unresolved since there is no way to verify completion.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "exit_code": 0,
  "stdout": "backup directory exists with 5 log files",
  "stderr": ""
}
```

**Failed resolution (validation check failed):**

```json
{
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "/workspace/backup: No such file or directory"
}
```

**Failed resolution (no validation command):**

```json
{
  "resolved": false,
  "error": "No validation command provided"
}
```

## Troubleshooting

**Setup command fails**

If a task's `setup_command` fails, the task will raise a `RuntimeError` before the agent starts. Check that the Docker environment has the necessary base tools installed. Some setup commands may require packages not present in the base image.

**Validation command returns non-zero but task seems correct**

The validation command may check for very specific conditions (exact file permissions, specific file contents, precise directory structure). Review the validation command logic by running with `-vv` to see exactly what is being checked. The agent may have completed the task in a slightly different way than expected.

**Agent does not execute shell commands**

TerminalBench requires the agent to actually execute commands in the terminal, not just describe them. Ensure your MCP server provides shell execution tools and that the agent prompt instructs the agent to run commands rather than output code snippets.

**Timeout during task execution**

System administration tasks that involve package installation, service configuration, or large file operations may need extended timeouts. Increase `timeout_seconds` to 300 for complex tasks. The validation command itself has a separate 30-second timeout.

## Best Practices

- **Provide shell execution tools** through your MCP server, as TerminalBench fundamentally requires running commands in a real terminal environment.
- **Start with easy tasks** (`--filter-difficulty easy`) to verify your setup before progressing to harder difficulty levels.
- **Use shorter timeouts** (120s) for file manipulation tasks and longer timeouts (300s) for system administration tasks.
- **Filter by category** to focus on task types relevant to your MCP server's capabilities (e.g., `file-manipulation` for filesystem servers).
- **Inspect validation commands** to understand exactly what constitutes success for each task. This helps debug unexpected failures.
- **Run with higher concurrency** (`max_concurrent: 8`) since terminal tasks use lightweight environments and typically complete quickly.
- **Set `max_iterations` appropriately**: Simple file operations need only 5-10 iterations, while scripting tasks may require 15-20.

## Related Links

- [TerminalBench Dataset on HuggingFace](https://huggingface.co/datasets/ia03/terminal-bench)
- [Benchmarks Overview](index.md)
- [InterCode](intercode.md) | [CyberGym](cybergym.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
