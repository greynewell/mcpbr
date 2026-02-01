---
description: "RepoQA evaluates long-context code understanding by testing whether agents can find and identify specific functions within large repository codebases."
benchmark_howto:
  name: "RepoQA"
  description: "Long-context code understanding benchmark where agents must find and describe specific 'needle' functions within large repository codebases, tested across multiple programming languages."
  benchmark_id: "repoqa"
faq:
  - q: "What is the 'Searching Needle Function' task in RepoQA?"
    a: "The Searching Needle Function task gives the agent a natural language description of a function and the full source code of a repository. The agent must search through the codebase to find and identify the specific function that matches the description, then provide its exact name."
  - q: "How does RepoQA evaluate answers?"
    a: "Evaluation checks whether the target function name appears anywhere in the agent's response using a case-insensitive substring match. The agent must mention the correct function name to receive credit, regardless of additional explanation provided."
  - q: "Can I filter RepoQA tasks by programming language?"
    a: "Yes. Use --filter-category to select tasks for specific programming languages. For example, '--filter-category python' will select only Python repository tasks. The filter matches against the language field in each task."
---

# RepoQA

| Property | Value |
|----------|-------|
| **Benchmark ID** | `repoqa` |
| **Dataset** | [evaluating/RepoQA](https://huggingface.co/datasets/evaluating/RepoQA) |
| **Tasks** | Long-context code understanding (Searching Needle Function) |
| **Evaluation** | Case-insensitive substring match for target function name in agent response |
| **Output Type** | Boolean (function found / not found) |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark repoqa
    ```

## Overview

[RepoQA](https://github.com/evalplus/repoqa) is a benchmark designed to test long-context code understanding. The core task -- called "Searching Needle Function" -- presents an agent with a natural language description of a function and the source code of a large repository. The agent must search through the entire codebase to find the specific function that matches the description and return its exact name.

This benchmark is fundamentally different from code generation tasks. Rather than writing new code, the agent must demonstrate deep comprehension of existing codebases by:

- Processing large volumes of source code across multiple files
- Understanding function semantics from implementation details
- Matching natural language descriptions to code behavior
- Identifying the correct function among many candidates

RepoQA covers repositories in multiple programming languages, making it useful for evaluating how well an agent (with MCP server assistance) can navigate and understand diverse codebases. It tests the kind of code comprehension skills needed for real-world software maintenance, code review, and documentation tasks.

RepoQA is particularly well-suited for evaluating MCP servers that provide code search, code navigation, or semantic code analysis capabilities.

## Task Structure

Each RepoQA task contains the following fields:

| Field | Description |
|-------|-------------|
| **instance_id** | Auto-generated identifier (e.g., `repoqa_0`, `repoqa_1`) |
| **repo** | Repository name or path containing the target function |
| **language** | Programming language of the repository (e.g., Python, Java, TypeScript) |
| **function_name** | The exact name of the target function (used for evaluation, not shown to agent) |
| **description** | Natural language description of what the target function does |

**Example task:**

```text
Find and describe the function in owner/repo-name (Python) that matches
the following description:

This function takes a list of file paths and a regular expression pattern,
searches each file for lines matching the pattern, and returns a dictionary
mapping file paths to lists of matching line numbers. It handles file encoding
errors gracefully by skipping unreadable files.

Identify the function name and explain what it does.
```

The agent must search through the repository, identify the function matching this description (e.g., `search_files_for_pattern`), and include its name in the response.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run RepoQA with default settings
    mcpbr run -c config.yaml --benchmark repoqa

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark repoqa -n 20

    # Filter by programming language
    mcpbr run -c config.yaml --benchmark repoqa --filter-category python

    # Filter by multiple languages
    mcpbr run -c config.yaml --benchmark repoqa \
      --filter-category python --filter-category java

    # Run with verbose output
    mcpbr run -c config.yaml --benchmark repoqa -n 10 -v

    # Save results to JSON
    mcpbr run -c config.yaml --benchmark repoqa -n 20 -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "repoqa"
    sample_size: 10
    timeout_seconds: 180

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter by language
    filter_category:
      - "python"
    ```

    Configuration for multi-language evaluation:

    ```yaml
    benchmark: "repoqa"
    sample_size: 20
    timeout_seconds: 300
    max_iterations: 20

    filter_category:
      - "python"
      - "java"
      - "typescript"

    model: "opus"
    ```

## Evaluation Methodology

RepoQA evaluation uses a straightforward function name matching approach:

1. **Problem Presentation**: The agent receives the repository name, programming language, and a natural language description of the target function. It does not receive the function name.

2. **Agent Search**: The agent explores the repository code using available tools (file listing, code search, file reading, etc.) to find the function matching the description.

3. **Response Collection**: The agent's full textual response is captured as the solution.

4. **Name Matching**: The evaluation checks whether the target function name (from the `function_name` field) appears anywhere in the agent's response. The match is **case-insensitive** and uses **substring matching**, so the function name can appear within a sentence, code block, or any other context.

5. **Resolution**: The task is marked as **resolved** if the target function name is found in the response. Tasks without a specified target function are marked as unresolved.

The evaluation is performed entirely offline against the agent's textual output. No code execution or environment state inspection is involved.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "target_function": "search_files_for_pattern",
  "function_found": true
}
```

**Failed resolution (wrong function identified):**

```json
{
  "resolved": false,
  "target_function": "search_files_for_pattern",
  "function_found": false
}
```

**Failed resolution (no target function specified):**

```json
{
  "resolved": false,
  "error": "No target function specified"
}
```

## Troubleshooting

**Agent identifies the correct function but uses a different name format**

The evaluation uses case-insensitive substring matching, so `SearchFilesForPattern` and `search_files_for_pattern` will both match `search_files_for_pattern`. However, if the agent describes the function without ever mentioning its name (e.g., "the function in utils.py on line 45"), the evaluation will fail. Ensure the agent prompt instructs the agent to provide the exact function name.

**Agent cannot navigate large repositories**

RepoQA tasks involve searching through entire repositories, which can contain hundreds of files. Ensure your MCP server provides efficient file listing and code search tools. Without these, the agent may time out before finding the target function. Consider using `timeout_seconds: 300` for large repositories.

**False positives from common function names**

If the target function has a very common name (e.g., `get`, `run`, `process`), the agent might mention it incidentally while discussing other functions. This would result in a false positive. This is a known limitation of substring matching evaluation.

**Tasks with unfamiliar programming languages**

RepoQA covers multiple programming languages. The agent's ability to understand code varies by language. Use `filter_category` to focus on languages where your agent performs best, or to specifically test cross-language understanding.

## Best Practices

- **Provide code search and navigation tools** through your MCP server, as RepoQA fundamentally requires exploring large codebases. File listing, content search (grep-like), and file reading tools are essential.
- **Instruct the agent to name the function explicitly** in the prompt. The evaluation requires the exact function name to appear in the response.
- **Start with a small sample** (`-n 5` or `-n 10`) to gauge how well your agent handles repository navigation before scaling up.
- **Use `filter_category` for specific languages** to focus on repositories where your agent has the strongest code understanding.
- **Set appropriate timeouts** (180-300s) since exploring large repositories takes time, especially when reading multiple files.
- **Enable extended thinking** (`thinking_budget: 10000`) for complex repositories where the agent needs to reason about function semantics across many candidates.
- **Run with higher concurrency** (`max_concurrent: 8`) since RepoQA tasks use lightweight evaluation (no code execution) and primarily consume API tokens.

## Related Links

- [RepoQA Project](https://github.com/evalplus/repoqa)
- [RepoQA Dataset on HuggingFace](https://huggingface.co/datasets/evaluating/RepoQA)
- [RepoQA Paper (arXiv)](https://arxiv.org/abs/2406.06025)
- [Benchmarks Overview](index.md)
- [InterCode](intercode.md) | [SWE-bench](swe-bench.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
