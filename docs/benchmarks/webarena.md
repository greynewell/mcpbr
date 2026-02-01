---
description: "WebArena evaluates autonomous web agents in realistic web environments featuring functional e-commerce sites, forums, content management systems, and maps with multi-step interaction tasks."
benchmark_howto:
  name: "WebArena"
  description: "Realistic web environment benchmark with functional websites for evaluating autonomous agents on multi-step web navigation, form filling, and information retrieval tasks."
  benchmark_id: "webarena"
faq:
  - q: "What types of websites does WebArena include?"
    a: "WebArena includes functional replicas of real websites covering e-commerce (shopping), social forums (Reddit-like), content management systems (GitLab-like), and mapping services. Agents must navigate these sites to complete multi-step tasks."
  - q: "How does WebArena evaluate task completion?"
    a: "Evaluation verifies the agent's output against reference answers using substring matching for string answers or structured comparison for complex answers that include must_include fields. The agent must produce output that contains the expected information."
  - q: "Can I filter WebArena tasks by website type?"
    a: "Yes. Use the filter_category option to select tasks involving specific website domains. For example, '--filter-category shopping' will select only e-commerce tasks. The filter matches against the sites field in each task."
---

# WebArena

| Property | Value |
|----------|-------|
| **Benchmark ID** | `webarena` |
| **Dataset** | [WebArena/WebArena](https://huggingface.co/datasets/WebArena/WebArena) |
| **Tasks** | Web navigation and interaction tasks across multiple website types |
| **Evaluation** | Verifies agent output against reference answers (substring matching or structured comparison) |
| **Output Type** | Web interaction results (text extracted from browser state) |
| **Timeout** | 300-600s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark webarena
    ```

## Overview

[WebArena](https://webarena.dev/) is a benchmark for evaluating autonomous web agents in realistic, self-hosted web environments. Unlike synthetic or simplified web tasks, WebArena provides fully functional websites that mirror real-world platforms -- including e-commerce storefronts, discussion forums, content management systems, and map services. Agents must navigate these websites, interact with forms and buttons, extract information, and complete multi-step tasks that require understanding web page structure and content.

Each task specifies an intent (what the user wants to accomplish), a set of target websites, and a reference answer describing the expected outcome. The agent interacts with the web environment and produces a response, which is then compared against the reference answer to determine success.

WebArena is particularly useful for evaluating MCP servers that provide web browsing, DOM manipulation, or browser automation capabilities. It tests whether agents can effectively use these tools to accomplish real web-based tasks.

## Task Structure

Each WebArena task contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Unique numeric identifier for the task |
| **intent** | Natural language description of what the user wants to accomplish |
| **sites** | List of websites involved in the task (e.g., shopping, forum, gitlab, map) |
| **eval** | Evaluation configuration including `eval_types` and `reference_answers` |
| **reference_answers** | Expected outcome, either a string or a structured object with `must_include` fields |

**Example task:**

The agent receives a problem statement like:

```text
Complete the following web task:

Find the price of the most expensive laptop on the shopping site.

Available websites: shopping
```

The agent must navigate the shopping site, search or browse for laptops, identify the most expensive one, and return its price. The evaluation checks whether the reference answer (e.g., "$2,499.99") appears in the agent's response.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run WebArena with default settings
    mcpbr run -c config.yaml --benchmark webarena

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark webarena -n 20

    # Filter by website type
    mcpbr run -c config.yaml --benchmark webarena --filter-category shopping

    # Filter by multiple website types
    mcpbr run -c config.yaml --benchmark webarena \
      --filter-category shopping --filter-category forum

    # Run with verbose output
    mcpbr run -c config.yaml --benchmark webarena -n 10 -v

    # Save results to JSON
    mcpbr run -c config.yaml --benchmark webarena -n 20 -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "webarena"
    sample_size: 10
    timeout_seconds: 300

    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]

    model: "sonnet"

    # Optional: Filter by website type
    filter_category:
      - "shopping"
      - "forum"
    ```

    Configuration with extended timeout for complex multi-site tasks:

    ```yaml
    benchmark: "webarena"
    sample_size: 5
    timeout_seconds: 600
    max_iterations: 30

    model: "opus"
    ```

## Evaluation Methodology

WebArena evaluation compares the agent's output against reference answers through the following process:

1. **Task Execution**: The agent receives the intent and available website information as a problem statement. It interacts with the web environment using available tools (browser automation, HTTP requests, DOM inspection, etc.) to complete the task.

2. **Output Collection**: The agent's final response is captured as its solution output, which should contain the information or confirmation requested by the task intent.

3. **Reference Comparison**: The solution is compared against the task's reference answers:
   - **String references**: A case-insensitive substring match checks whether the reference answer appears anywhere in the agent's output.
   - **Structured references**: For `must_include` style references, every required item must appear (case-insensitive) in the agent's output.

4. **Resolution**: A task is marked as **resolved** if the agent's output satisfies the reference answer criteria. Tasks without reference answers are marked as unresolved.

The evaluation is performed offline against the agent's textual output. No live browser state inspection is performed during the evaluation step.

## Example Output

**Successful resolution:**

```json
{
  "resolved": true,
  "agent_output": "The most expensive laptop on the shopping site is the ProBook Ultra at $2,499.99...",
  "reference": "$2,499.99"
}
```

**Failed resolution (incorrect answer):**

```json
{
  "resolved": false,
  "agent_output": "I found several laptops. The cheapest one costs $599.99...",
  "reference": "$2,499.99"
}
```

**Failed resolution (no reference answer):**

```json
{
  "resolved": false,
  "error": "No reference answer available"
}
```

## Troubleshooting

**Agent output does not contain the reference answer**

The agent may find the correct information but present it in a different format. For example, the reference may be "$1,234" but the agent outputs "1234 dollars." Ensure the agent prompt instructs precise, verbatim answers. Substring matching is case-insensitive, but formatting differences like commas or currency symbols can cause mismatches.

**Tasks involving multiple websites**

Some WebArena tasks require navigating across multiple sites (e.g., comparing prices between shopping and a forum recommendation). These tasks are more complex and may need longer timeouts (600s) and more agent iterations. Use `max_iterations: 30` for multi-site tasks.

**Agent cannot navigate web interface**

WebArena tasks require browser interaction capabilities. Ensure your MCP server provides tools for web browsing, page navigation, form submission, and content extraction. Without these tools, the agent will be unable to complete most tasks.

**Timeout on complex tasks**

Multi-step web interactions can take significant time, especially when the agent needs to search, paginate, or fill out forms across several pages. Start with a generous timeout of 300-600 seconds and adjust based on your observed completion times.

## Best Practices

- **Provide browser automation tools** through your MCP server, as WebArena tasks fundamentally require web interaction capabilities.
- **Start with a small sample** (`-n 5` or `-n 10`) to verify your agent can successfully navigate the web environments before scaling up.
- **Use `filter_category`** to focus evaluation on specific website types that match your MCP server's strengths (e.g., e-commerce for shopping-related tools).
- **Set generous timeouts** (300-600s) since multi-step web interactions involve page loads, form submissions, and content extraction.
- **Instruct precise answers** in your agent prompt to ensure output matches reference answers. Verbatim answers are more likely to pass substring matching.
- **Increase `max_iterations`** for tasks that require multiple page navigations or complex workflows.
- **Monitor agent interactions** with `-vv` to observe which web pages the agent visits and identify navigation bottlenecks.

## Related Links

- [WebArena Project](https://webarena.dev/)
- [WebArena Dataset on HuggingFace](https://huggingface.co/datasets/WebArena/WebArena)
- [WebArena Paper (arXiv)](https://arxiv.org/abs/2307.13854)
- [Benchmarks Overview](index.md)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
