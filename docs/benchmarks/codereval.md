---
description: "CoderEval evaluates AI agents on pragmatic code generation from real-world open-source projects, testing the ability to implement functions with real project dependencies and context."
benchmark_howto:
  name: "CoderEval"
  description: "Pragmatic code generation from real-world open-source projects supporting Python, Java, JavaScript, and TypeScript with project-specific dependency contexts."
  benchmark_id: "codereval"
faq:
  - q: "What programming languages does CoderEval support?"
    a: "CoderEval supports Python, Java, JavaScript, and TypeScript. Each task specifies its language, and the evaluation uses the appropriate language-specific test runner (python3, javac/java, node, npx ts-node)."
  - q: "How does CoderEval differ from HumanEval or APPS?"
    a: "CoderEval tests code generation in the context of actual open-source projects with real dependencies on project-specific types, functions, and libraries. Unlike HumanEval's isolated functions or APPS' standalone programs, CoderEval requires understanding and working within an existing codebase."
  - q: "Can I filter CoderEval tasks by programming language?"
    a: "Yes. Use the filter_category option with language names: '--filter-category python' or '--filter-category java'. This selects only tasks written in the specified programming language."
---

# CoderEval

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `codereval` |
| **Dataset** | [CoderEval/CoderEval](https://huggingface.co/datasets/CoderEval/CoderEval) |
| **Tasks** | Varies (real-world code generation tasks) |
| **Evaluation** | Language-specific test execution (Python, Java, JavaScript, TypeScript) |
| **Output Type** | Test pass/fail |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark codereval
    ```

## Overview

CoderEval is a pragmatic code generation benchmark that draws tasks from real-world open-source projects. Unlike benchmarks that test isolated function generation (such as HumanEval) or standalone programming puzzles (such as APPS), CoderEval evaluates whether an AI agent can implement functions within the context of an actual codebase, complete with project-specific types, helper functions, and library dependencies.

Each task presents a function from a real open-source project, stripped of its implementation but retaining its signature, docstring, and surrounding context. The agent must understand the project context, including imported modules, utility functions, and data types, to produce a correct implementation that integrates seamlessly with the existing code.

CoderEval supports multiple programming languages:

- **Python**: Tasks from Python open-source projects, tested with Python 3.
- **Java**: Tasks from Java projects, compiled and tested with javac/java.
- **JavaScript**: Tasks from JavaScript/Node.js projects, tested with Node.js.
- **TypeScript**: Tasks from TypeScript projects, tested with ts-node.

This multi-language support makes CoderEval valuable for evaluating agents across different programming ecosystems and assessing whether they can handle language-specific idioms, type systems, and testing patterns.

## Task Structure

Each CoderEval task includes the following components:

- **Task ID**: A unique identifier for the task from the original dataset.
- **Function Name**: The name of the function to implement.
- **Language**: The programming language (Python, Java, JavaScript, or TypeScript).
- **Docstring**: Documentation describing the function's purpose, parameters, and return value.
- **Context**: Surrounding code from the project, including imports, class definitions, and helper functions that the implementation may depend on.
- **Test Code**: Project-specific test cases that validate the implementation against expected behavior.
- **Repository**: The source repository the task was extracted from.
- **Commit**: The specific commit hash for reproducibility.
- **Instance ID**: An auto-generated identifier in the format `codereval_{task_id}`.

The agent receives the function name, language, documentation, and surrounding context, and must produce an implementation that passes the provided test cases.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run CoderEval with default settings
    mcpbr run -c config.yaml --benchmark codereval

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark codereval -n 20

    # Run a specific task
    mcpbr run -c config.yaml --benchmark codereval -t TASK_ID

    # Filter by programming language
    mcpbr run -c config.yaml --benchmark codereval --filter-category python

    # Run only Java tasks
    mcpbr run -c config.yaml --benchmark codereval --filter-category java

    # Filter by difficulty level
    mcpbr run -c config.yaml --benchmark codereval --filter-difficulty easy
    ```

=== "YAML"

    ```yaml
    benchmark: "codereval"
    sample_size: 10
    timeout_seconds: 300
    ```

    Configuration filtered by language:

    ```yaml
    benchmark: "codereval"
    sample_size: 15
    timeout_seconds: 300

    # Only Python tasks
    filter_category:
      - "python"
    ```

    Multi-language configuration:

    ```yaml
    benchmark: "codereval"
    sample_size: 20
    timeout_seconds: 300

    filter_category:
      - "python"
      - "javascript"
    ```

## Evaluation Methodology

CoderEval evaluation is language-specific, using the appropriate runtime for each task:

1. **Test Assembly**: The agent's solution code is concatenated with the task's test code to create a single test file with the appropriate extension (`.py`, `.java`, `.js`, or `.ts`).
2. **Language-Specific Execution**: The test file is executed using the language-appropriate command:
    - **Python**: `python3 test_solution.py`
    - **Java**: `javac test_solution.java && java test_solution`
    - **JavaScript**: `node test_solution.js`
    - **TypeScript**: `npx ts-node test_solution.ts`
3. **Result Determination**: The task is marked as **resolved** if execution completes with exit code 0. Any compilation error, runtime error, or test assertion failure results in a non-zero exit code and a failed evaluation.
4. **Output Capture**: Both stdout and stderr are captured (truncated to 1,000 characters) for diagnostic purposes.

The evaluation has a 60-second timeout per task for the test execution phase.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "codereval_python_utils_parse_config",
  "resolved": true,
  "exit_code": 0,
  "stdout": "All tests passed",
  "stderr": ""
}
```

### Failed Resolution

```json
{
  "instance_id": "codereval_java_StringHelper_format",
  "resolved": false,
  "exit_code": 1,
  "stdout": "",
  "stderr": "AssertionError: expected 'hello world' but got 'Hello World'"
}
```

### Unsupported Language

```json
{
  "instance_id": "codereval_cpp_Vector_sort",
  "resolved": false,
  "error": "Unsupported language: cpp"
}
```

## Troubleshooting

**Language runtime not available in Docker container**
CoderEval tasks require the appropriate language runtime. Python tasks need Python 3, Java tasks need JDK, JavaScript tasks need Node.js, and TypeScript tasks need Node.js with ts-node. Ensure your Docker environment includes the required runtimes for the languages you are evaluating.

**Context not properly understood by the agent**
CoderEval tasks include surrounding code context that the implementation depends on. If the agent ignores the context and writes standalone code, it will likely fail tests that exercise integration with helper functions or project-specific types. Ensure the prompt encourages the agent to read and understand the provided context.

**Test file compilation fails (Java)**
Java tasks require both compilation and execution. If the agent's implementation has type errors or missing imports that are available in the context, the compilation step will fail. The stderr output will contain the javac error messages pointing to the specific issue.

**Solution file saved with wrong extension**
The evaluation determines the file extension based on the task's language. If the agent saves the solution with an incorrect extension (e.g., `.py` for a Java task), the evaluation will still use the correct extension. However, the agent should be instructed to save to the appropriate filename for clarity.

## Best Practices

- **Filter by language** to focus evaluation on languages relevant to your MCP server and use case.
- **Start with Python tasks** (`--filter-category python`) as they typically have the most straightforward execution environment.
- **Provide context in prompts**: CoderEval's value lies in testing context-aware code generation. Ensure the agent receives and processes the surrounding code context.
- **Use moderate timeouts**: 300 seconds provides adequate time for the agent to analyze the context and generate an implementation, even for complex tasks.
- **Check language runtime availability**: Before running non-Python tasks, verify that the Docker environment has the required compilers and runtimes installed.
- **Compare across languages**: Running evaluations for each supported language separately reveals whether your agent handles language-specific patterns consistently.
- **Review stderr for diagnostics**: Compilation errors (Java) and runtime errors provide specific information about what went wrong, enabling targeted improvements.

## Related Links

- [CoderEval Paper (arXiv)](https://arxiv.org/abs/2302.00288)
- [CoderEval Dataset on HuggingFace](https://huggingface.co/datasets/CoderEval/CoderEval)
- [Benchmarks Overview](index.md)
- [BigCodeBench](bigcodebench.md) | [APPS](apps.md) | [Aider Polyglot](aider-polyglot.md)
