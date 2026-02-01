---
description: "MCPToolBench++ benchmark for evaluating MCP tool discovery, selection, invocation, and result interpretation."
benchmark_howto:
  name: "MCPToolBench++"
  description: "Evaluates AI agents' ability to effectively use MCP tools across four dimensions: discovery, selection, invocation, and result interpretation, with 45+ task categories."
  benchmark_id: "mcptoolbench"
faq:
  - q: "What does MCPToolBench++ evaluate?"
    a: "MCPToolBench++ evaluates four key MCP tool-use capabilities: tool discovery (understanding available tools), tool selection (choosing the right tool), tool invocation (calling tools with correct parameters), and result interpretation (using tool outputs correctly)."
  - q: "How is MCPToolBench++ different from ToolBench?"
    a: "MCPToolBench++ is specifically designed for MCP (Model Context Protocol) tool use, testing the full lifecycle of MCP tool interaction. ToolBench evaluates general API tool use. MCPToolBench++ uses MCP-specific tool schemas, categories, and evaluation metrics."
  - q: "What accuracy thresholds determine a passing score?"
    a: "A task is considered resolved when tool selection accuracy is 0.8 or higher (80% of correct tools selected) AND parameter accuracy is 0.7 or higher (70% of parameters correct), AND the agent does not make more than 1.5x the expected number of tool calls."
---

# MCPToolBench++

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `mcptoolbench` |
| **Dataset** | [MCPToolBench/MCPToolBenchPP](https://huggingface.co/datasets/MCPToolBench/MCPToolBenchPP) |
| **Tasks** | Varies (45+ categories) |
| **Evaluation** | Tool selection accuracy (>=0.8) and parameter accuracy (>=0.7) |
| **Output Type** | Tool call accuracy metrics |
| **Timeout** | 180-300 seconds |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark mcptoolbench -n 20
    ```

## Overview

MCPToolBench++ is a benchmark specifically designed to evaluate how well AI agents use MCP (Model Context Protocol) tools. It tests the complete tool-use lifecycle across four key dimensions:

1. **Tool Discovery**: Understanding what MCP tools are available and their capabilities
2. **Tool Selection**: Choosing the appropriate tool(s) for a given task
3. **Tool Invocation**: Calling tools with correct parameters matching their schemas
4. **Result Interpretation**: Understanding and correctly using tool outputs

The benchmark covers 45+ categories spanning diverse tool-use scenarios:

- **Browser**: Web browsing and page interaction
- **Finance**: Financial data retrieval and calculations
- **Code Analysis**: Source code inspection and manipulation
- **Database**: Query and data management operations
- **File Management**: File system operations
- **Communication**: Email, messaging, and notification tools
- **Weather**: Weather data retrieval
- **Search**: Information retrieval and search operations
- And many more

Each task provides a query, a set of available MCP tools with their schemas, and a ground truth sequence of tool calls that correctly completes the task.

## Task Structure

Each MCPToolBench++ task contains the following fields:

- **uuid**: Unique task identifier
- **query**: The natural language task description
- **category**: Task category (e.g., "browser", "finance", "code_analysis")
- **call_type**: Task complexity -- "single" (one tool call) or "multi" (multiple sequential calls)
- **tools**: List of available tool names
- **mcp_tools_dict**: Full MCP tool definitions including schemas and descriptions
- **function_call_label**: Ground truth sequence of tool calls with parameters

The agent receives the query, task metadata, and available tools, then must select and invoke the correct tools with proper parameters.

### Example Task

```text
Category: finance
Task Type: single-step tool call
Available Tools: get_stock_price, get_exchange_rate, calculate_tax

Task:
What is the current stock price of Apple Inc. (AAPL)?

Expected Tool Call:
  - name: get_stock_price
    parameters:
      symbol: "AAPL"
```

### Multi-Step Example

```text
Category: finance
Task Type: multi-step tool call
Available Tools: get_stock_price, get_exchange_rate, calculate_tax

Task:
Get the current stock price of AAPL in USD, then convert it to EUR.

Expected Tool Calls:
  1. name: get_stock_price
     parameters: { symbol: "AAPL" }
  2. name: get_exchange_rate
     parameters: { from: "USD", to: "EUR" }
```

## Running the Benchmark

=== "CLI"

    ```bash
    # Run MCPToolBench++ with default settings
    mcpbr run -c config.yaml --benchmark mcptoolbench

    # Run a small sample
    mcpbr run -c config.yaml --benchmark mcptoolbench -n 20

    # Filter by difficulty (single vs multi-step)
    mcpbr run -c config.yaml --benchmark mcptoolbench --filter-difficulty easy
    mcpbr run -c config.yaml --benchmark mcptoolbench --filter-difficulty hard

    # Filter by category
    mcpbr run -c config.yaml --benchmark mcptoolbench --filter-category browser
    mcpbr run -c config.yaml --benchmark mcptoolbench --filter-category finance

    # Combine filters
    mcpbr run -c config.yaml --benchmark mcptoolbench \
      --filter-difficulty easy --filter-category browser

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark mcptoolbench -n 50 -v -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "mcptoolbench"
    sample_size: 10
    timeout_seconds: 300

    # Optional: filter by difficulty and category
    filter_difficulty:
      - "easy"    # single-step tasks
    filter_category:
      - "browser"
      - "finance"
    ```

### Difficulty Filtering

MCPToolBench++ maps difficulty labels to the `call_type` field:

| Filter Value | Maps To | Description |
|--------------|---------|-------------|
| `easy` or `single` | `single` | Single tool call tasks |
| `hard`, `multi`, or `medium` | `multi` | Multi-step tool call sequences |

### Category Filtering

Filter by any of the 45+ task categories. Category matching is case-insensitive. Common categories include:

- `browser` - Web browsing tasks
- `finance` - Financial operations
- `code_analysis` - Code inspection and analysis
- `database` - Database operations
- `file` - File management
- `weather` - Weather data
- `search` - Information retrieval
- `communication` - Messaging and notification

## Evaluation Methodology

MCPToolBench++ evaluation compares the agent's tool calls against the ground truth using three metrics:

### 1. Tool Selection Accuracy

Measures how many of the expected tools the agent correctly selected:

```
tool_selection_accuracy = correct_tools / total_expected_tools
```

A tool is considered correctly selected if the agent made at least one call to a tool with the same name as an expected tool. Each expected tool is counted at most once.

### 2. Parameter Accuracy

Measures how correctly the agent filled in tool parameters:

```
parameter_accuracy = correct_parameters / total_expected_parameters
```

For each correctly selected tool, the agent's parameters are compared against the ground truth. A parameter is correct if both the name and value match exactly.

### 3. Sequence Match

A boolean indicating whether the agent called the exact same tools in the exact same order as the ground truth.

### Overall Resolution

A task is **resolved** when all three conditions are met:

```
resolved = (tool_selection_accuracy >= 0.8)
       AND (parameter_accuracy >= 0.7)
       AND (agent_call_count <= expected_call_count * 1.5)
```

The thresholds allow for minor variations:

- **0.8 tool selection**: Allows missing up to 20% of expected tools (e.g., 4 out of 5 correct)
- **0.7 parameter accuracy**: Allows up to 30% parameter errors (e.g., minor formatting differences)
- **1.5x call limit**: Allows some extra exploratory tool calls but prevents excessive flailing

### Tool Call Extraction

The evaluation attempts to extract tool calls from the agent's response in two ways:

1. **JSON parsing**: If the response is valid JSON (a list of tool calls or an object with a `tool_calls` key), the calls are extracted directly.
2. **Text parsing**: If JSON parsing fails, the evaluation falls back to pattern matching in the response text.

## Example Output

### Successful Evaluation

```json
{
  "resolved": true,
  "tool_selection_accuracy": 1.0,
  "parameter_accuracy": 0.85,
  "sequence_match": true,
  "details": "Tool selection: 100.0%, Parameter accuracy: 85.0%, Sequence match: True"
}
```

### Failed Evaluation (Low Tool Selection)

```json
{
  "resolved": false,
  "tool_selection_accuracy": 0.5,
  "parameter_accuracy": 0.9,
  "sequence_match": false,
  "details": "Tool selection: 50.0%, Parameter accuracy: 90.0%, Sequence match: False"
}
```

### Failed Evaluation (No Tool Calls Extracted)

```json
{
  "resolved": false,
  "tool_selection_accuracy": 0.0,
  "parameter_accuracy": 0.0,
  "sequence_match": false,
  "details": "Agent made no tool calls"
}
```

### Failed Evaluation (No Ground Truth)

```json
{
  "resolved": false,
  "error": "No ground truth function calls provided for evaluation"
}
```

## Troubleshooting

### Agent does not produce structured tool calls

MCPToolBench++ expects the agent's response to contain tool calls in a parseable format. If the agent describes tool calls in natural language instead of structured output, the extraction will fail. Configure your prompt to request structured output:

```yaml
agent_prompt: |
  {problem_statement}

  Output your tool calls as a JSON array. Each element should have:
  - "name": the tool name
  - "parameters": an object with parameter key-value pairs

  Example:
  [{"name": "tool_name", "parameters": {"param1": "value1"}}]
```

### Tool selection is high but parameter accuracy is low

This indicates the agent is selecting the right tools but providing incorrect parameter values. Common causes include:

- Parameter name mismatches (e.g., "stock_symbol" vs "symbol")
- Incorrect value types (e.g., string instead of number)
- Missing required parameters

Review the MCP tool schemas to ensure parameter names and types match exactly.

### Extra tool calls cause resolution failure

The evaluation allows up to 1.5x the expected number of tool calls. If the agent makes significantly more calls (e.g., retrying with different parameters), it may exceed this limit. Instruct the agent to be deliberate and avoid exploratory calls.

### Category filter returns no results

Category names must match the dataset's category field. Inspect available categories:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('MCPToolBench/MCPToolBenchPP', split='train')
cats = sorted(set(item['category'] for item in ds))
for cat in cats:
    print(cat)
"
```

## Best Practices

- **Use structured output prompts**: MCPToolBench++ evaluation depends on extracting tool calls from the agent's response. JSON-formatted output is most reliable.
- **Start with single-step tasks**: Use `--filter-difficulty easy` to evaluate basic tool selection and invocation before progressing to multi-step tasks.
- **Category-specific evaluation**: Different categories test different tool types. Evaluate categories individually to identify specific strengths and weaknesses.
- **Monitor all three metrics**: Track tool selection accuracy, parameter accuracy, and sequence match separately. Each reveals different aspects of tool-use capability.
- **Provide tool schemas**: Ensure your agent has access to the full MCP tool definitions. The `mcp_tools_dict` field in each task contains the complete schemas.
- **Test with your actual MCP server**: MCPToolBench++ is particularly valuable when used with your actual MCP server configuration, as it tests the real tool discovery and invocation pipeline.
- **Increase timeout for multi-step tasks**: Multi-step tasks require sequential tool calls and result interpretation. Use at least 300 seconds for multi-step evaluations.

## Related Links

- [Benchmarks Overview](index.md)
- [ToolBench](toolbench.md) - General API tool use benchmark
- [GAIA](gaia.md) - General AI assistant benchmark with tool use
- [AgentBench](agentbench.md) - Multi-environment agent benchmark
- [MCPToolBench++ Dataset](https://huggingface.co/datasets/MCPToolBench/MCPToolBenchPP)
- [MCP Integration Guide](../mcp-integration.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
