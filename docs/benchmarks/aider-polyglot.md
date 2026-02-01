---
description: "Aider Polyglot evaluates AI agents on code editing tasks across Python, JavaScript, Go, Rust, and Java, using Exercism exercises with language-specific test suites."
benchmark_howto:
  name: "Aider Polyglot"
  description: "Code editing benchmark across Python, JavaScript, Go, Rust, and Java based on Exercism exercises, evaluated with language-specific test runners."
  benchmark_id: "aider-polyglot"
faq:
  - q: "What programming languages does Aider Polyglot support?"
    a: "Aider Polyglot supports Python, JavaScript, Go, Rust, and Java. Each task uses the language-specific test runner: pytest for Python, npm test for JavaScript, go test for Go, cargo test for Rust, and mvn test for Java."
  - q: "How does Aider Polyglot differ from other coding benchmarks?"
    a: "Aider Polyglot focuses specifically on code editing rather than code generation from scratch. The agent receives existing source files that need to be modified to pass the provided test suite, testing the ability to understand and edit existing code across multiple languages."
  - q: "What are Exercism exercises?"
    a: "Exercism is an open-source platform with programming exercises across 60+ languages. The exercises are well-structured with clear instructions and comprehensive test suites, making them ideal for evaluating code editing capabilities in a multi-language context."
---

# Aider Polyglot

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `aider-polyglot` |
| **Dataset** | [aider-ai/polyglot-benchmark](https://huggingface.co/datasets/aider-ai/polyglot-benchmark) |
| **Tasks** | Code editing exercises across 5 languages |
| **Evaluation** | Language-specific test suites (pytest, npm test, go test, cargo test, mvn test) |
| **Output Type** | Test pass/fail |
| **Timeout** | 180-300s recommended |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark aider-polyglot
    ```

## Overview

The Aider Polyglot benchmark evaluates code editing capabilities across multiple programming languages. Based on exercises from [Exercism](https://exercism.org/), a well-established open-source platform for programming practice, this benchmark tests whether an AI agent can correctly modify existing source code to make it pass a provided test suite.

Unlike code generation benchmarks where the agent writes code from scratch, Aider Polyglot focuses on code editing -- the agent receives an existing source file (often a skeleton or partial implementation) and must edit it to implement the required functionality. This mirrors a common real-world development workflow where developers modify existing code rather than writing everything from scratch.

The benchmark covers five programming languages:

- **Python**: Exercises tested with `pytest` (`python3 -m pytest -xvs`).
- **JavaScript**: Exercises tested with `npm test`.
- **Go**: Exercises tested with `go test ./...`.
- **Rust**: Exercises tested with `cargo test`.
- **Java**: Exercises tested with `mvn test`.

Each exercise provides clear instructions, a source file to edit, and a test file that validates the implementation. The exercises cover fundamental programming concepts across all languages, including string manipulation, data structures, mathematical operations, and algorithmic problems.

## Task Structure

Each Aider Polyglot task includes the following components:

- **Exercise**: The name of the Exercism exercise (e.g., "hello-world", "two-fer", "reverse-string").
- **Language**: The programming language for the task (Python, JavaScript, Go, Rust, or Java).
- **Instructions**: A natural language description of what the exercise requires, including examples and expected behavior.
- **Source File**: The path to the source file that the agent should edit (e.g., `hello_world.py`, `hello-world.js`).
- **Test File**: The path to the test file that validates the solution (not to be modified by the agent).
- **Test Command**: The language-specific command to run the test suite (auto-detected based on language if not provided).
- **Instance ID**: An auto-generated identifier in the format `aider_{task_id}`.

The agent receives the exercise instructions and the source file to edit, and must produce a modified version that passes all tests in the test file.

## Running the Benchmark

=== "CLI"

    ```bash
    # Run Aider Polyglot with default settings
    mcpbr run -c config.yaml --benchmark aider-polyglot

    # Run a sample of 20 tasks
    mcpbr run -c config.yaml --benchmark aider-polyglot -n 20

    # Run a specific task
    mcpbr run -c config.yaml --benchmark aider-polyglot -t TASK_ID

    # Filter by programming language
    mcpbr run -c config.yaml --benchmark aider-polyglot --filter-category python

    # Run only Go exercises
    mcpbr run -c config.yaml --benchmark aider-polyglot --filter-category go

    # Run only Rust exercises
    mcpbr run -c config.yaml --benchmark aider-polyglot --filter-category rust
    ```

=== "YAML"

    ```yaml
    benchmark: "aider-polyglot"
    sample_size: 10
    timeout_seconds: 300
    ```

    Configuration filtered by language:

    ```yaml
    benchmark: "aider-polyglot"
    sample_size: 15
    timeout_seconds: 300

    # Only Python exercises
    filter_category:
      - "python"
    ```

    Multi-language configuration:

    ```yaml
    benchmark: "aider-polyglot"
    sample_size: 20
    timeout_seconds: 300

    filter_category:
      - "python"
      - "javascript"
      - "go"
    ```

## Evaluation Methodology

Aider Polyglot evaluation runs language-specific test suites:

1. **Environment Setup**: A Docker container is created with the necessary language runtimes and build tools for the task's language.
2. **Code Editing**: The agent edits the source file based on the exercise instructions, producing a modified version of the code.
3. **Test Execution**: The language-appropriate test command is executed:
    - **Python**: `python3 -m pytest -xvs` (runs pytest with verbose output, stops on first failure).
    - **JavaScript**: `npm test` (runs the npm test script defined in package.json).
    - **Go**: `go test ./...` (runs all Go tests in the module).
    - **Rust**: `cargo test` (builds and runs all Rust tests).
    - **Java**: `mvn test` (compiles and runs all Maven-managed tests).
4. **Result Determination**: The task is marked as **resolved** if the test command exits with code 0, indicating all tests passed. Any test failure, compilation error, or runtime error results in a non-zero exit code.
5. **Output Capture**: Both stdout and stderr are captured (truncated to 1,000 characters) for diagnostic purposes.

The test execution has a 60-second timeout to accommodate compilation-heavy languages like Rust and Java.

## Example Output

### Successful Resolution

```json
{
  "instance_id": "aider_python_hello_world",
  "resolved": true,
  "exit_code": 0,
  "stdout": "test_hello_world.py::test_say_hi PASSED\n\n1 passed in 0.02s",
  "stderr": ""
}
```

### Failed Resolution

```json
{
  "instance_id": "aider_go_reverse_string",
  "resolved": false,
  "exit_code": 1,
  "stdout": "--- FAIL: TestReverse (0.00s)\n    reverse_test.go:15: Reverse(\"hello\") = \"ollhe\", want \"olleh\"",
  "stderr": ""
}
```

### Build Failure

```json
{
  "instance_id": "aider_rust_two_fer",
  "resolved": false,
  "exit_code": 101,
  "stdout": "",
  "stderr": "error[E0308]: mismatched types\n  --> src/lib.rs:3:5\n   |3|     42\n   |       ^^ expected `String`, found integer"
}
```

## Troubleshooting

**Language runtime not available in Docker container**
Each language requires its own runtime and build tools. Python needs pytest, JavaScript needs Node.js with npm, Go needs the Go compiler, Rust needs the Cargo toolchain, and Java needs the JDK with Maven. Ensure the Docker environment has the required tools for the languages you are evaluating.

**Tests fail due to modified test files**
The agent should only edit the source file, not the test file. If the agent modifies the test file to make tests pass, this defeats the purpose of the benchmark. Include clear instructions in the prompt: "Do not modify the test files."

**Rust/Java compilation takes too long**
Compiled languages require additional time for the build step. Rust in particular can have lengthy compilation times for the first build. The 60-second test execution timeout should be sufficient for most exercises, but if you see frequent compilation timeouts, consider using a Docker image with pre-cached build artifacts.

**Agent does not understand the existing code structure**
Aider Polyglot tasks require editing existing files rather than writing new ones. Some agents may attempt to create new files instead of modifying the provided source. Ensure the prompt clearly indicates which file to edit and that the agent should work with the existing code structure.

## Best Practices

- **Filter by language** to focus evaluation on the programming languages most relevant to your use case.
- **Start with Python exercises** (`--filter-category python`) as they have the simplest execution environment and fastest test cycles.
- **Ensure Docker has all required runtimes**: For multi-language evaluation, the Docker image must include Python + pytest, Node.js + npm, Go, Rust + Cargo, and JDK + Maven.
- **Use clear editing instructions**: Since Aider Polyglot tests code editing rather than generation, prompt the agent to modify the existing source file rather than creating a new one.
- **Set appropriate timeouts**: 180 seconds is adequate for Python and JavaScript tasks. Go, Rust, and Java may benefit from 300 seconds due to compilation overhead.
- **Compare across languages**: Running separate evaluations per language provides insight into how consistently your agent handles different programming paradigms and type systems.
- **Monitor test output**: The verbose output from test runners (especially pytest and Go) provides detailed information about which specific test cases passed or failed.

## Related Links

- [Aider Polyglot Benchmark on HuggingFace](https://huggingface.co/datasets/aider-ai/polyglot-benchmark)
- [Aider Project](https://aider.chat/)
- [Exercism Platform](https://exercism.org/)
- [Benchmarks Overview](index.md)
- [CoderEval](codereval.md) | [SWE-bench](swe-bench.md) | [BigCodeBench](bigcodebench.md)
