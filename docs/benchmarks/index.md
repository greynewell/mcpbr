---
description: "Overview of all benchmarks supported by mcpbr for evaluating MCP servers and AI agent capabilities across software engineering, code generation, math reasoning, knowledge, tool use, and security."
faq:
  - q: "What benchmarks does mcpbr support?"
    a: "mcpbr supports 25+ benchmarks across software engineering (SWE-bench, APPS, CodeContests), code generation (HumanEval, MBPP), math reasoning (GSM8K, MATH), knowledge & QA (TruthfulQA, ARC), tool use (MCPToolBench++, ToolBench), ML research (MLAgentBench), code understanding (RepoQA), and security (CyberGym)."
  - q: "How do I list available benchmarks in mcpbr?"
    a: "Run 'mcpbr benchmarks' to see all available benchmarks with their descriptions and output types."
  - q: "How do I select a benchmark to run?"
    a: "Use the --benchmark flag with the benchmark ID: 'mcpbr run -c config.yaml --benchmark humaneval'. Or set 'benchmark: humaneval' in your YAML config file."
---

# Benchmarks

mcpbr supports a comprehensive suite of benchmarks for evaluating MCP servers and AI agent capabilities. Each benchmark targets different skills - from bug fixing and code generation to math reasoning, tool use, and security exploit generation.

## Quick Start

```bash
# List all available benchmarks
mcpbr benchmarks

# Run a specific benchmark
mcpbr run -c config.yaml --benchmark humaneval -n 20

# Run default benchmark (SWE-bench Verified)
mcpbr run -c config.yaml
```

## All Benchmarks

| Benchmark | ID | Tasks | Category | Evaluation | Docs |
|-----------|-----|-------|----------|------------|------|
| SWE-bench Verified | `swe-bench-verified` | 500 | Software Engineering | Test suite pass/fail | [Details](swe-bench.md) |
| SWE-bench Lite | `swe-bench-lite` | 300 | Software Engineering | Test suite pass/fail | [Details](swe-bench.md) |
| SWE-bench Full | `swe-bench-full` | 2,294 | Software Engineering | Test suite pass/fail | [Details](swe-bench.md) |
| APPS | `apps` | 10,000 | Software Engineering | stdin/stdout tests | [Details](apps.md) |
| CodeContests | `codecontests` | Varies | Software Engineering | Test case comparison | [Details](codecontests.md) |
| BigCodeBench | `bigcodebench` | 1,140 | Software Engineering | Test pass/fail | [Details](bigcodebench.md) |
| LeetCode | `leetcode` | Varies | Software Engineering | Code execution | [Details](leetcode.md) |
| CoderEval | `codereval` | Varies | Software Engineering | Language-specific tests | [Details](codereval.md) |
| Aider Polyglot | `aider-polyglot` | Varies | Software Engineering | Language-specific tests | [Details](aider-polyglot.md) |
| HumanEval | `humaneval` | 164 | Code Generation | Unit tests | [Details](humaneval.md) |
| MBPP | `mbpp` | ~1,000 | Code Generation | Test pass/fail | [Details](mbpp.md) |
| GSM8K | `gsm8k` | 1,319 | Math & Reasoning | Numeric answer matching | [Details](gsm8k.md) |
| MATH | `math` | 12,500 | Math & Reasoning | LaTeX answer extraction | [Details](math.md) |
| BigBench-Hard | `bigbench-hard` | 27 subtasks | Math & Reasoning | Exact match | [Details](bigbench-hard.md) |
| TruthfulQA | `truthfulqa` | ~800 | Knowledge & QA | Substring matching | [Details](truthfulqa.md) |
| HellaSwag | `hellaswag` | Varies | Knowledge & QA | Option selection | [Details](hellaswag.md) |
| ARC | `arc` | 7,787 | Knowledge & QA | Multiple choice | [Details](arc.md) |
| GAIA | `gaia` | Varies | Knowledge & QA | Exact match | [Details](gaia.md) |
| MCPToolBench++ | `mcptoolbench` | Varies | Tool Use & Agents | Tool accuracy metrics | [Details](mcptoolbench.md) |
| ToolBench | `toolbench` | Varies | Tool Use & Agents | Tool call comparison | [Details](toolbench.md) |
| AgentBench | `agentbench` | Varies | Tool Use & Agents | String matching | [Details](agentbench.md) |
| WebArena | `webarena` | Varies | Tool Use & Agents | Reference matching | [Details](webarena.md) |
| TerminalBench | `terminalbench` | Varies | Tool Use & Agents | Validation command | [Details](terminalbench.md) |
| InterCode | `intercode` | Varies | Tool Use & Agents | Output comparison | [Details](intercode.md) |
| MLAgentBench | `mlagentbench` | Varies | ML Research | Score comparison | [Details](mlagentbench.md) |
| RepoQA | `repoqa` | Varies | Code Understanding | Function name match | [Details](repoqa.md) |
| CyberGym | `cybergym` | Varies | Security | Crash detection | [Details](cybergym.md) |

## Benchmarks by Category

### Software Engineering

Benchmarks that test an agent's ability to work with real codebases, fix bugs, and solve programming challenges.

| Benchmark | Focus | Difficulty | Best For |
|-----------|-------|-----------|----------|
| [SWE-bench](swe-bench.md) | Real GitHub bug fixes | High | MCP server evaluation, production benchmarking |
| [APPS](apps.md) | Coding problems (intro → competition) | Low → High | Broad code generation assessment |
| [CodeContests](codecontests.md) | Competitive programming | High | Algorithmic reasoning evaluation |
| [BigCodeBench](bigcodebench.md) | Multi-library function composition | Medium | Real-world API usage testing |
| [LeetCode](leetcode.md) | Algorithmic problems | Low → High | Data structure and algorithm evaluation |
| [CoderEval](codereval.md) | Code generation in project context | Medium | Contextual code generation |
| [Aider Polyglot](aider-polyglot.md) | Multi-language code editing | Medium | Cross-language editing capability |

### Code Generation

Focused benchmarks for evaluating pure code generation from specifications.

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [HumanEval](humaneval.md) | Python function completion | Quick smoke tests, baseline metrics |
| [MBPP](mbpp.md) | Entry-level Python problems | Entry-level code generation |

### Math & Reasoning

Benchmarks testing mathematical reasoning and multi-step problem solving.

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [GSM8K](gsm8k.md) | Grade-school math word problems | Chain-of-thought evaluation |
| [MATH](math.md) | Competition mathematics (AMC/AIME) | Advanced math reasoning |
| [BigBench-Hard](bigbench-hard.md) | 27 hard reasoning tasks | Broad reasoning assessment |

### Knowledge & QA

Benchmarks evaluating knowledge, truthfulness, and question answering.

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [TruthfulQA](truthfulqa.md) | Truthfulness and avoiding misconceptions | Truthfulness evaluation |
| [HellaSwag](hellaswag.md) | Commonsense reasoning | Commonsense evaluation |
| [ARC](arc.md) | Grade-school science questions | Science reasoning |
| [GAIA](gaia.md) | General AI assistant tasks | Multi-modal, tool-use evaluation |

### Tool Use & Agents

Benchmarks specifically testing tool use, API interaction, and agentic capabilities.

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [MCPToolBench++](mcptoolbench.md) | MCP tool discovery and invocation | MCP server evaluation |
| [ToolBench](toolbench.md) | Real-world API tool use | API tool selection testing |
| [AgentBench](agentbench.md) | Multi-environment agent tasks | Broad agent evaluation |
| [WebArena](webarena.md) | Web browsing and interaction | Web automation testing |
| [TerminalBench](terminalbench.md) | Terminal/shell task completion | CLI and shell evaluation |
| [InterCode](intercode.md) | Interactive code environments | Multi-turn code interaction |

### ML Research

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [MLAgentBench](mlagentbench.md) | ML research tasks (Kaggle) | ML pipeline evaluation |

### Code Understanding

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [RepoQA](repoqa.md) | Long-context code understanding | Repository comprehension |

### Security

| Benchmark | Focus | Best For |
|-----------|-------|----------|
| [CyberGym](cybergym.md) | Vulnerability exploitation (PoC) | Security analysis evaluation |

## Comparing Benchmarks

| Aspect | SWE-bench | HumanEval | GSM8K | CyberGym | MCPToolBench++ |
|--------|-----------|-----------|-------|----------|----------------|
| **Goal** | Fix bugs | Generate code | Solve math | Exploit vulnerabilities | Use MCP tools |
| **Output** | Patch (diff) | Function code | Numeric answer | PoC code | Tool calls |
| **Languages** | Python | Python | N/A | C/C++ | N/A |
| **Evaluation** | Test suite | Unit tests | Answer matching | Crash detection | Tool accuracy |
| **Pre-built Images** | Yes | No | No | No | No |
| **Typical Timeout** | 300-600s | 60-180s | 60-180s | 600-900s | 180-300s |
| **Task Count** | 300-2,294 | 164 | 1,319 | Varies | Varies |
| **Difficulty Levels** | N/A | N/A | N/A | 0-3 | easy/hard |
| **Best For** | MCP evaluation | Quick tests | Reasoning | Security research | Tool use testing |

## Benchmark Abstraction

mcpbr uses a Protocol-based abstraction that makes it easy to add new benchmarks:

```python
from mcpbr.benchmarks import Benchmark

class MyBenchmark:
    """Custom benchmark implementation."""

    name = "my-benchmark"

    def load_tasks(self, sample_size, task_ids, level):
        """Load tasks from dataset."""
        ...

    def normalize_task(self, task):
        """Convert to normalized BenchmarkTask format."""
        ...

    async def create_environment(self, task, docker_manager):
        """Create isolated Docker environment."""
        ...

    async def evaluate(self, env, task, solution):
        """Evaluate the solution."""
        ...

    def get_prebuilt_image(self, task):
        """Return pre-built image name or None."""
        ...

    def get_prompt_template(self):
        """Return agent prompt template."""
        ...
```

Each benchmark implements:

- **`load_tasks()`**: Load tasks from HuggingFace or other sources
- **`normalize_task()`**: Convert to common format
- **`create_environment()`**: Set up Docker container with dependencies
- **`evaluate()`**: Run benchmark-specific evaluation
- **`get_prebuilt_image()`**: Return pre-built image name if available
- **`get_prompt_template()`**: Provide task-appropriate instructions

See [src/mcpbr/benchmarks/](https://github.com/greynewell/mcpbr/tree/main/src/mcpbr/benchmarks) for reference implementations.

## Listing Benchmarks

Use the CLI to see available benchmarks:

```bash
$ mcpbr benchmarks

Available Benchmarks

┌────────────────┬──────────────────────────────────────────────────────┬─────────────────────────┐
│ Benchmark      │ Description                                          │ Output Type             │
├────────────────┼──────────────────────────────────────────────────────┼─────────────────────────┤
│ swe-bench      │ Software bug fixes in GitHub repositories             │ Patch (unified diff)    │
│ cybergym       │ Security vulnerability exploitation (PoC generation)  │ Exploit code            │
│ humaneval      │ Python function completion (code generation)          │ Function code           │
│ mcptoolbench   │ MCP tool use evaluation                               │ Tool call accuracy      │
│ gsm8k          │ Grade-school math reasoning                           │ Numeric answer          │
│ ...            │ ...                                                   │ ...                     │
└────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘

Use --benchmark flag with 'run' command to select a benchmark
Example: mcpbr run -c config.yaml --benchmark humaneval
```

## Filtering Tasks

All benchmarks support task filtering to select specific subsets:

```yaml
# Filter by difficulty level
filter_difficulty:
  - "easy"
  - "medium"

# Filter by category
filter_category:
  - "django"
  - "scikit-learn"

# Filter by tags
filter_tags:
  - "security"
```

```bash
# CLI filtering
mcpbr run -c config.yaml --filter-difficulty easy --filter-category browser
```

See [Configuration](../configuration.md#filtering-configuration) for full filtering documentation.

## Best Practices

### Choosing a Benchmark

- **Testing MCP servers**: Start with [SWE-bench](swe-bench.md) or [MCPToolBench++](mcptoolbench.md)
- **Quick smoke tests**: Use [HumanEval](humaneval.md) (164 tasks, fast)
- **Math reasoning**: Use [GSM8K](gsm8k.md) for basic or [MATH](math.md) for competition-level
- **Security research**: Use [CyberGym](cybergym.md) with appropriate difficulty level
- **Multi-language**: Use [Aider Polyglot](aider-polyglot.md) or [CoderEval](codereval.md)

### General Tips

1. **Start small**: Run with `-n 5` before scaling up
2. **Use pre-built images**: Enabled by default for SWE-bench, much faster
3. **Set appropriate timeouts**: See individual benchmark pages for recommendations
4. **Save results**: Always use `-o results.json` to preserve data
5. **Compare benchmarks**: Run multiple benchmarks to get a comprehensive picture

## Related Links

- [Configuration Guide](../configuration.md) - Full configuration reference
- [CLI Reference](../cli.md) - All command options
- [Best Practices](../best-practices.md) - Tips for effective evaluation
- [Architecture](../architecture.md) - How mcpbr works internally
