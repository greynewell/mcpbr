---
description: "ToolBench benchmark for evaluating real-world API tool selection and invocation with proper parameters."
benchmark_howto:
  name: "ToolBench"
  description: "Evaluates language models' ability to select and invoke correct real-world API tools with proper parameters to fulfill user requests."
  benchmark_id: "toolbench"
faq:
  - q: "What does ToolBench evaluate?"
    a: "ToolBench evaluates a model's ability to select the correct API tools from a set of available options and invoke them with proper parameters. It uses real-world APIs and compares the agent's tool call sequence against ground truth."
  - q: "How does ToolBench differ from MCPToolBench++?"
    a: "ToolBench focuses on general API tool use with real-world REST APIs, while MCPToolBench++ is specifically designed for MCP protocol tool evaluation. ToolBench also supports filtering by tags, difficulty, and category, and expects tool calls in JSON format."
  - q: "What output format does ToolBench expect?"
    a: "ToolBench expects tool calls as a JSON array where each element has 'name' and 'parameters' fields. The evaluation can extract JSON from direct responses, markdown code blocks, or structured objects."
---

# ToolBench

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `toolbench` |
| **Dataset** | [tuandunghcmut/toolbench-v1](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1) |
| **Tasks** | Varies |
| **Evaluation** | Tool call sequence comparison against ground truth |
| **Output Type** | Tool selection accuracy (JSON tool calls) |
| **Timeout** | 180-300 seconds |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark toolbench -n 20
    ```

## Overview

ToolBench evaluates language models on real-world API tool use. Each task presents a user query along with a set of available API tools, and the model must select and invoke the correct tools with proper parameters to fulfill the request.

Key characteristics of ToolBench:

- **Real-world APIs**: Tasks use actual REST API tools spanning diverse categories
- **Tool selection**: Models must choose from multiple available tools
- **Parameter accuracy**: Correct tool names alone are not sufficient -- parameters must be accurate
- **Sequence matching**: For multi-step tasks, the order of tool calls matters
- **JSON output**: Tool calls are expected in structured JSON format

ToolBench is useful for evaluating:

- **API comprehension**: Understanding tool descriptions and selecting appropriate endpoints
- **Parameter inference**: Deducing correct parameter values from user queries
- **Multi-tool orchestration**: Sequencing multiple API calls to complete complex requests
- **Schema adherence**: Following tool schemas for parameter naming and types

## Task Structure

Each ToolBench task contains the following fields:

- **query**: The user's natural language request
- **tools**: A list of available API tools with names and descriptions
- **category**: The API category (e.g., "weather", "finance", "social_media")
- **ground_truth**: The expected sequence of tool calls with parameters
- **difficulty**: Task difficulty level (when available)

The agent receives the query and available tools, then must produce the correct tool call sequence.

### Example Task

```text
Complete the following task using the available tools:

Get the current weather forecast for London, UK in metric units.

Available tools:
  - get_weather: Get current weather conditions for a location
  - get_forecast: Get multi-day weather forecast
  - convert_units: Convert between measurement units
  - get_timezone: Get timezone information for a location

Expected Tool Calls:
[
  {
    "name": "get_weather",
    "parameters": {
      "location": "London, UK",
      "units": "metric"
    }
  }
]
```

### Multi-Tool Example

```text
Query: Find restaurants near Central Park in New York, then check the
weather to decide if outdoor dining is feasible.

Expected Tool Calls:
[
  {
    "name": "search_restaurants",
    "parameters": {
      "location": "Central Park, New York",
      "type": "restaurant"
    }
  },
  {
    "name": "get_weather",
    "parameters": {
      "location": "New York",
      "units": "imperial"
    }
  }
]
```

## Running the Benchmark

=== "CLI"

    ```bash
    # Run ToolBench with default settings
    mcpbr run -c config.yaml --benchmark toolbench

    # Run a small sample
    mcpbr run -c config.yaml --benchmark toolbench -n 20

    # Filter by difficulty
    mcpbr run -c config.yaml --benchmark toolbench --filter-difficulty easy

    # Filter by API category
    mcpbr run -c config.yaml --benchmark toolbench --filter-category weather

    # Filter by tool tags
    mcpbr run -c config.yaml --benchmark toolbench --filter-tags "api" --filter-tags "rest"

    # Combine all filters
    mcpbr run -c config.yaml --benchmark toolbench \
      --filter-difficulty easy \
      --filter-category finance \
      --filter-tags "stock"

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark toolbench -n 50 -v -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "toolbench"
    sample_size: 10
    timeout_seconds: 300

    # Optional: apply filters
    filter_difficulty:
      - "easy"
    filter_category:
      - "weather"
      - "finance"
    filter_tags:
      - "rest"
      - "api"
    ```

### Filter Options

ToolBench supports three types of filtering:

| Filter | Field | Behavior |
|--------|-------|----------|
| `filter_difficulty` | `difficulty` | Exact match (case-insensitive) |
| `filter_category` | `category` | Substring match (case-insensitive) -- matches if any filter value is contained in the category |
| `filter_tags` | `tools` | All tags must match (AND logic) -- checks if each tag appears in the tool descriptions |

**Important**: `filter_tags` uses AND logic, meaning all specified tags must be present in a task's tool descriptions for the task to be included.

## Evaluation Methodology

ToolBench evaluation compares the agent's tool call sequence against the ground truth:

### Tool Call Extraction

The evaluation extracts tool calls from the agent's response using a multi-strategy approach:

1. **Direct JSON parsing**: If the entire response is valid JSON (a list or a single object), it is used directly.
2. **Markdown code block extraction**: The evaluation searches for JSON within markdown code blocks (` ```json ... ``` ` or ` ``` ... ``` `).
3. **Fallback**: If no valid JSON is found, the evaluation returns no tool calls and the task fails.

### Comparison Method

1. **Tool name extraction**: Tool names are extracted from both the ground truth and the agent's calls.
2. **Exact sequence match**: The primary metric checks if the agent's tool names match the ground truth in exact order.
3. **Tool overlap metric**: A secondary metric calculates the overlap between expected and actual tool sets.

### Scoring

```
resolved = (ground_truth_tool_names == agent_tool_names)
tool_selection_accuracy = |expected_tools INTERSECT agent_tools| / |expected_tools|
```

The task is **resolved** only when the exact sequence of tool names matches. The `tool_selection_accuracy` provides a softer metric even when the exact sequence does not match.

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `resolved` | boolean | Whether tool call sequence exactly matches ground truth |
| `tool_selection_accuracy` | float | Proportion of expected tools that were called (0.0 to 1.0) |
| `expected_tools` | list | Ground truth tool names in order |
| `agent_tools` | list | Agent's tool names in order |

## Example Output

### Successful Evaluation

```json
{
  "resolved": true,
  "tool_selection_accuracy": 1.0,
  "expected_tools": ["get_weather"],
  "agent_tools": ["get_weather"]
}
```

### Failed Evaluation (Wrong Tool Order)

```json
{
  "resolved": false,
  "tool_selection_accuracy": 1.0,
  "expected_tools": ["search_restaurants", "get_weather"],
  "agent_tools": ["get_weather", "search_restaurants"]
}
```

In this case, both tools were selected correctly, but the order is reversed. The `tool_selection_accuracy` is 1.0 but `resolved` is false because sequence matching failed.

### Failed Evaluation (Missing Tool)

```json
{
  "resolved": false,
  "tool_selection_accuracy": 0.5,
  "expected_tools": ["search_restaurants", "get_weather"],
  "agent_tools": ["search_restaurants"]
}
```

### Failed Evaluation (No Tool Calls Extracted)

```json
{
  "resolved": false,
  "error": "Could not extract tool calls from solution"
}
```

## Troubleshooting

### Agent response is not parseable as JSON

ToolBench requires tool calls in JSON format. If the agent describes tool usage in natural language, the extraction will fail. Use a prompt that explicitly requests JSON output:

```yaml
agent_prompt: |
  {problem_statement}

  IMPORTANT: Output your answer as a JSON array of tool calls.
  Each tool call should have "name" and "parameters" fields.

  Example format:
  ```json
  [
    {"name": "tool_name", "parameters": {"key": "value"}}
  ]
  ```
```

### Sequence match fails despite correct tools

ToolBench uses strict sequence matching. If the agent calls the right tools but in a different order, `resolved` will be false. Consider whether the task truly requires a specific order, and instruct the agent to follow the logical sequence of operations.

### Category filter is too broad or too narrow

`filter_category` uses substring matching, so filtering by "finance" will also match "personal_finance" or "finance_api". Use more specific terms if you need precise filtering, or check the available categories in the dataset:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('tuandunghcmut/toolbench-v1', split='train')
cats = sorted(set(item.get('category', '') for item in ds))
for cat in cats[:30]:
    print(cat)
"
```

### Tag filtering returns no results

The `filter_tags` parameter requires ALL specified tags to match (AND logic). Each tag is checked as a case-insensitive substring of the stringified tools field. If you specify too many tags, the intersection may be empty. Start with a single tag and add more progressively.

## Best Practices

- **Use JSON output prompts**: Always instruct the agent to output tool calls as structured JSON. This is critical for successful evaluation.
- **Start with simple tasks**: Begin with single-tool tasks to verify JSON extraction and tool name matching before progressing to multi-tool sequences.
- **Monitor tool_selection_accuracy**: Even when `resolved` is false (sequence mismatch), the `tool_selection_accuracy` metric shows how close the agent was. Use this for progressive improvement.
- **Category-specific evaluation**: Different API categories require different domain knowledge. Evaluate categories separately to identify strengths and weaknesses.
- **Use filter_tags strategically**: Tags provide fine-grained filtering beyond difficulty and category. Use them to focus on specific API types (e.g., REST, GraphQL).
- **Check parameter formatting**: Even when tools are correct, parameter mismatches (different naming conventions, types, or formats) can prevent resolution in downstream evaluation. Review failed cases to identify patterns.
- **Set appropriate timeouts**: Single-tool tasks typically complete in under 120 seconds, but multi-tool sequences may need 300 seconds or more.

## Related Links

- [Benchmarks Overview](index.md)
- [MCPToolBench++](mcptoolbench.md) - MCP-specific tool use benchmark
- [GAIA](gaia.md) - General AI assistant benchmark with tool use
- [AgentBench](agentbench.md) - Multi-environment agent benchmark
- [ToolBench Dataset](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
