---
title: "Making Claude an Expert: Building the mcpbr Claude Code Plugin"
description: "A comprehensive guide to building domain-specific Claude Code plugins, with real-world examples from creating specialized skills for MCP benchmarking"
date: 2026-01-22
author: Grey Newell
tags: [Claude Code, AI, MCP, Benchmarking, Developer Tools, Plugins, Anthropic]
keywords: Claude Code plugin, AI developer tools, Claude Code skills, MCP benchmarking, Anthropic Claude, AI code assistant, developer productivity, custom AI tools
canonical_url: https://greynewell.github.io/mcpbr/blog/claude-code-plugin-guide
---

# Making Claude an Expert: Building the mcpbr Claude Code Plugin

When you're working with specialized tools, there's a gap between what a general-purpose AI assistant knows and what your domain expert would know. Claude is brilliant at general coding tasks, but what if you could teach it to become an expert in your specific domain? That's exactly what Claude Code plugins enable.

This is the story of building the mcpbr plugin - a set of specialized skills that transformed Claude from a helpful coding assistant into an expert at benchmarking Model Context Protocol (MCP) servers. Along the way, I learned valuable lessons about plugin architecture, prompt engineering, and what makes a great AI-powered developer tool.

## The Problem: Expertise Bottleneck

[mcpbr](https://github.com/greynewell/mcpbr) (Model Context Protocol Benchmark Runner) is a specialized tool for evaluating MCP servers against real-world software engineering benchmarks like SWE-bench. It has a specific CLI syntax, critical requirements (like Docker), mandatory configuration patterns, and subtle pitfalls that can derail evaluations.

Before the plugin, using mcpbr with Claude Code looked like this:

```bash
User: "Run a benchmark for me"
Claude: "Sure! Let me run: mcpbr run -m claude-sonnet-4"
# FAILS - No config file specified
# FAILS - Model flag syntax wrong
# FAILS - Docker not checked
```

The problem wasn't Claude's fault. Without domain-specific knowledge, even the most advanced AI will make reasonable-sounding mistakes. It's like asking a brilliant generalist to perform brain surgery - they might understand the theory, but they lack the specialized expertise.

## What Are Claude Code Plugins?

Claude Code plugins are domain-specific knowledge packages that transform Claude into an expert for your tools. Think of them as "skill packs" that Claude loads when working in your codebase.

A plugin consists of:

1. **Plugin manifest** (`plugin.json`) - Metadata about your plugin
2. **Skills directory** - Individual specialized capabilities
3. **Skill definitions** (SKILL.md files) - Detailed instructions for each skill

Here's the minimal structure:

```
.claude-plugin/
├── plugin.json           # Plugin metadata
└── skills/
    └── my-skill/
        └── SKILL.md      # Skill instructions
```

The plugin manifest is straightforward:

```json
{
  "name": "mcpbr",
  "version": "0.3.17",
  "description": "Expert benchmark runner for MCP servers using mcpbr. Handles Docker checks, config generation, and result parsing.",
  "schema_version": "1.0"
}
```

When Claude Code detects a `.claude-plugin` directory in your project, it automatically loads the plugin and makes the skills available. Users can then invoke skills with slash commands like `/run-benchmark` or `/generate-config`.

## Designing the mcpbr Plugin: Three Specialized Skills

I identified three core workflows where users needed expert guidance:

### 1. **run-benchmark**: The Expert Evaluator

This skill makes Claude an expert at running valid, reproducible MCP evaluations. The key insight was encoding all the prerequisites and validation steps that humans naturally do but AI assistants often skip:

```yaml
---
name: run-benchmark
description: Run an MCP evaluation using mcpbr on SWE-bench or other datasets.
---

# Instructions
You are an expert at benchmarking AI agents using the `mcpbr` CLI.

## Critical Constraints (DO NOT IGNORE)

1. **Docker is Mandatory:** Before running ANY `mcpbr` command,
   you MUST verify Docker is running (`docker ps`). If not,
   tell the user to start it.

2. **Config is Required:** `mcpbr run` FAILS without a config file.
   - IF no config exists: Run `mcpbr init` first
   - IF config exists: Read it to verify the `mcp_server`
     command is valid
```

Notice the explicit constraints and the ALL CAPS emphasis. This isn't just documentation - it's a carefully crafted prompt that prevents common failure modes. The skill includes:

- **Pre-flight checks**: Docker, API keys, config validation
- **Common pitfalls**: Hallucinated flags, missing workdir placeholders
- **Benchmark support**: SWE-bench, CyberGym, MCPToolBench++
- **Troubleshooting guides**: Specific error → solution mappings

### 2. **generate-config**: The Configuration Expert

MCP server configuration is where most users hit problems. The `{workdir}` placeholder is critical but non-obvious. This skill encodes best practices:

```yaml
---
name: generate-config
description: Generate and validate mcpbr configuration files.
---

## Critical Requirements

1. **Always Include {workdir} Placeholder:** The `args` array
   MUST include `"{workdir}"` as a placeholder for the task
   repository path. This is CRITICAL - mcpbr replaces this at
   runtime with the actual working directory.
```

The skill provides:

- **Templates for common MCP servers**: Filesystem, custom Python, etc.
- **Validation checklists**: Command existence, YAML syntax, env vars
- **Benchmark-specific configs**: Different benchmarks need different settings
- **Anti-patterns**: Explicit warnings about common mistakes

### 3. **swe-bench-lite**: The Quick-Start Expert

This skill addresses a specific user journey: "I just want to try it quickly." It provides sensible defaults and clear expectations:

```yaml
---
name: swe-bench-lite
description: Quick-start command to run SWE-bench Lite evaluation.
---

## Default Command

mcpbr run -c mcpbr.yaml --dataset SWE-bench/SWE-bench_Lite \
  -n 5 -v -o results.json -r report.md

## Expected Runtime & Cost

For 5 tasks with default settings:
- **Runtime:** 15-30 minutes
- **Cost:** $2-5 (depends on task complexity)
```

This skill sets clear expectations upfront, preventing the "I ran it and it's taking forever!" confusion.

## Technical Implementation: From Idea to Working Plugin

### Step 1: Set Up the Plugin Structure

Creating the plugin structure is straightforward:

```bash
# In your project root
mkdir -p .claude-plugin/skills
cd .claude-plugin

# Create plugin manifest
cat > plugin.json <<EOF
{
  "name": "mcpbr",
  "version": "0.3.17",
  "description": "Expert benchmark runner for MCP servers",
  "schema_version": "1.0"
}
EOF
```

The version field should match your tool's version. I automated this with a sync script that reads from `pyproject.toml`:

```python
#!/usr/bin/env python3
"""Sync version from pyproject.toml to plugin.json."""
import json
import re
from pathlib import Path

def sync_version():
    root = Path(__file__).parent.parent

    # Read version from pyproject.toml
    pyproject = (root / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    version = match.group(1)

    # Update plugin.json
    plugin_json = root / ".claude-plugin" / "plugin.json"
    data = json.loads(plugin_json.read_text())
    data["version"] = version
    plugin_json.write_text(json.dumps(data, indent=2) + "\n")

if __name__ == "__main__":
    sync_version()
```

Hook this into your build process to ensure versions stay in sync.

### Step 2: Write Skill Instructions

Each skill is a markdown file with YAML frontmatter. The frontmatter defines metadata, and the markdown contains instructions for Claude:

```markdown
---
name: run-benchmark
description: Run an MCP evaluation using mcpbr
---

# Instructions
You are an expert at [specific domain].

## Critical Constraints
[List hard requirements that MUST be followed]

## Execution Steps
[Step-by-step workflow]

## Troubleshooting
[Common errors and solutions]
```

The key is being **explicit and prescriptive**. Don't write documentation - write instructions for an expert assistant. Use:

- **Imperative mood**: "Verify Docker is running" not "Docker should be running"
- **Specific commands**: Show exact commands, not generic descriptions
- **Clear constraints**: Use bold, ALL CAPS, or other emphasis for critical requirements
- **Error handling**: Map specific errors to specific solutions

### Step 3: Test Thoroughly

Testing a plugin involves both automated tests and real-world usage:

**Automated tests** verify the plugin structure:

```python
def test_plugin_json_has_required_fields():
    """Test that plugin.json has all required fields."""
    with open(".claude-plugin/plugin.json") as f:
        data = json.load(f)

    required_fields = ["name", "version", "description", "schema_version"]
    for field in required_fields:
        assert field in data

def test_skill_has_frontmatter():
    """Test that SKILL.md has valid frontmatter."""
    skill_path = Path("skills/mcpbr-eval/SKILL.md")
    content = skill_path.read_text()

    assert content.startswith("---\n")
    assert "name:" in content
    assert "description:" in content
```

**Real-world testing** is crucial. I tested by:

1. Removing the plugin and asking Claude to run a benchmark (observed failures)
2. Adding the plugin and repeating (verified Claude followed the skill's steps)
3. Intentionally breaking prerequisites (e.g., stopping Docker) to verify Claude caught them
4. Testing with different user phrasings to ensure the skill activated correctly

### Step 4: Distribution

There are three distribution strategies:

**1. Bundled with your tool (recommended for tool authors)**

Include `.claude-plugin` in your repository. Users get the plugin automatically when they clone your repo:

```
my-tool/
├── .claude-plugin/
│   ├── plugin.json
│   └── skills/
├── src/
└── README.md
```

**2. Standalone npm/PyPI package**

Publish the plugin separately for users to install:

```bash
npm install -g @yourtool/claude-plugin
# or
pip install yourtool-claude-plugin
```

**3. Claude Code Plugin Registry**

Submit to the official registry (when available) for discoverability.

For mcpbr, I chose option 1 - bundling the plugin with the tool. This ensures every mcpbr user gets the enhanced Claude Code experience automatically.

## Results & Impact: Before and After

The plugin fundamentally changed how users interact with mcpbr through Claude Code.

**Before the plugin:**
```
User: "Run a benchmark"
Claude: [Guesses at command syntax, forgets Docker check, misses config validation]
Result: Multiple rounds of errors, frustrated user
```

**After the plugin:**
```
User: "Run a benchmark"
Claude: [Activates run-benchmark skill]
Claude: "I'll help you run a benchmark. Let me first verify prerequisites..."
Claude: "Docker is running ✓"
Claude: "API key is set ✓"
Claude: "Reading config file..."
Claude: "Running: mcpbr run -c mcpbr.yaml -n 5 -v"
Result: Successful evaluation on first try
```

The plugin reduced support questions and enabled users to be productive immediately. More importantly, it unlocked workflows that weren't practical before:

- **Iterative evaluation**: Users could easily run benchmarks, adjust configs, and re-run
- **Exploration**: New users could explore different benchmarks without reading docs
- **Troubleshooting**: When things failed, Claude could diagnose based on the skill's troubleshooting guides

## Lessons Learned: Tips for Plugin Authors

### 1. **Encode Prerequisites Explicitly**

Don't assume Claude will "figure out" that Docker needs to be running or that a config file is required. State prerequisites explicitly and make them the first thing Claude checks.

**Bad:**
```markdown
This tool requires Docker.
```

**Good:**
```markdown
## Critical Constraint: Docker is Mandatory

Before running ANY command, you MUST verify Docker is running:
```bash
docker ps
```
If this fails, tell the user to start Docker Desktop.
```

### 2. **Prevent Common Mistakes Proactively**

Study how users actually fail with your tool, then encode those failure modes as explicit warnings:

```markdown
## Common Pitfalls to Avoid

- **DO NOT** hallucinate flags. Only use documented CLI flags.
- **DO NOT** forget the `{workdir}` placeholder in config args
- **DO NOT** use `-m` flag unless the user explicitly asks
```

### 3. **Provide Examples, Not Just Descriptions**

Show Claude exactly what commands look like:

```markdown
## Example Commands

# Full evaluation with 5 tasks
mcpbr run -c config.yaml -n 5 -v

# MCP-only evaluation
mcpbr run -c config.yaml -M -n 10
```

### 4. **Structure Skills Around User Workflows**

Don't organize skills by tool features - organize by what users are trying to accomplish:

- **User workflow**: "I want to run a benchmark" → `/run-benchmark` skill
- **User workflow**: "I need to create a config" → `/generate-config` skill
- **User workflow**: "I just want to try this quickly" → `/swe-bench-lite` skill

### 5. **Test With Real Users, Not Just Automated Tests**

Automated tests verify structure, but real users reveal gaps in your instructions. I discovered that users didn't understand the `{workdir}` placeholder until I emphasized it in multiple places.

### 6. **Version Your Plugin With Your Tool**

Keep plugin versions in sync with your tool versions. This prevents confusion when features or flags change. Use automation to enforce this:

```makefile
.PHONY: sync-version
sync-version:
    ./scripts/sync_version.py

.PHONY: build
build: sync-version
    python -m build
```

### 7. **Write Instructions for an Expert Assistant, Not a Human**

Humans can read between the lines. Claude needs explicit instructions:

**For humans:**
> "Make sure Docker is running before you start."

**For Claude:**
> "Before running ANY `mcpbr` command, you MUST verify Docker is running by executing `docker ps`. If this command fails, instruct the user to start Docker Desktop or the Docker daemon. Do not proceed until Docker is confirmed running."

## The Broader Opportunity: Plugins for Every Tool

The Claude Code plugin system opens up a fascinating opportunity: every specialized tool can have its own expert mode in Claude Code.

Imagine:

- **Infrastructure tools** (Terraform, Kubernetes) with plugins that encode best practices
- **Testing frameworks** (pytest, Jest) with plugins that know how to write proper test patterns
- **Build systems** (Make, Bazel) with plugins that understand optimization strategies
- **Deployment tools** (Vercel, AWS CDK) with plugins that prevent common misconfigurations

The pattern is universal: identify the domain expertise that users need, encode it as skills, and distribute it with your tool.

## Building Your Own Plugin: Getting Started

Ready to build a plugin for your tool? Here's a quick-start checklist:

1. **Identify pain points**: What do users struggle with? What questions do they ask repeatedly?

2. **Define 2-3 core workflows**: Don't try to cover everything - focus on the most common tasks

3. **Create the plugin structure**:
   ```bash
   mkdir -p .claude-plugin/skills/my-skill
   # Create plugin.json and SKILL.md
   ```

4. **Write explicit instructions**: Be prescriptive, not descriptive. Show commands, list prerequisites, warn about pitfalls.

5. **Test with real usage**: Try to break it. Have Claude help you with your tool and see where it fails.

6. **Iterate based on feedback**: Add troubleshooting guides as users hit problems.

7. **Automate version syncing**: Keep plugin version in sync with your tool version.

The mcpbr plugin took about a week to build and test, but it's saved countless hours of support and enabled workflows that weren't practical before. The ROI for tool authors is significant.

## Call to Action: Try It Yourself

If you're working with MCP servers, try the mcpbr plugin:

```bash
# Install mcpbr (includes the plugin)
pip install mcpbr

# Use Claude Code with the plugin skills
cd your-mcp-server-repo
/run-benchmark     # Expert benchmark runner
/generate-config   # Configuration assistant
/swe-bench-lite    # Quick evaluation
```

If you're building your own specialized tool, consider creating a Claude Code plugin. The plugin ecosystem is just getting started, and there's huge opportunity for tool authors to provide better developer experiences.

The source code for the mcpbr plugin is available at [github.com/greynewell/mcpbr](https://github.com/greynewell/mcpbr) under the MIT license. Use it as a reference or starting point for your own plugins.

## Conclusion: From Assistant to Expert

Building the mcpbr Claude Code plugin taught me that the gap between "helpful assistant" and "domain expert" isn't about raw intelligence - it's about encoding the right knowledge in the right format.

Claude is already brilliant at general coding tasks. Plugins make it brilliant at *your* specific domain. They transform error-prone guesswork into confident expertise, turning Claude Code into a true pair programmer who knows your tools as well as you do.

The future of AI-powered development tools isn't just about more powerful models - it's about better domain-specific knowledge integration. Claude Code plugins are a powerful step in that direction.

---

**About the Author**: Grey Newell is the creator of mcpbr, an open-source benchmark runner for evaluating Model Context Protocol servers. Follow the project on [GitHub](https://github.com/greynewell/mcpbr) or read the [documentation](https://greynewell.github.io/mcpbr/).

**Have you built a Claude Code plugin?** Share your experience or questions in the comments below, or open an issue in the mcpbr repository.
