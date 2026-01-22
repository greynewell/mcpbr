# @greynewell/mcpbr-claude-plugin

> Claude Code plugin for mcpbr - Makes Claude an expert at running MCP benchmarks

[![npm version](https://badge.fury.io/js/%40greynewell%2Fmcpbr-claude-plugin.svg)](https://www.npmjs.com/package/@greynewell/mcpbr-claude-plugin)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This Claude Code plugin provides specialized skills that make Claude an expert at using [mcpbr](https://github.com/greynewell/mcpbr) for MCP server benchmarking.

## What is mcpbr?

**Model Context Protocol Benchmark Runner** - Benchmark your MCP server against real GitHub issues. Get hard numbers comparing tool-assisted vs. baseline agent performance.

## Installation

### Option 1: Clone Repository (Automatic Detection)

```bash
git clone https://github.com/greynewell/mcpbr.git
cd mcpbr
# Claude Code automatically detects the plugin
```

### Option 2: Install via npm

```bash
npm install -g @greynewell/mcpbr-claude-plugin
```

### Option 3: Claude Code Plugin Manager

```bash
# In Claude Code, run:
/plugin install github:greynewell/mcpbr
```

## What You Get

When using Claude Code in a project with this plugin, Claude automatically gains expertise in:

### 1. run-benchmark (`mcpbr-eval`)
Expert at running evaluations with proper validation:
- Verifies Docker is running
- Checks for API keys
- Validates configuration files
- Uses correct CLI flags
- Provides actionable error messages

### 2. generate-config (`mcpbr-config`)
Generates valid mcpbr configuration files:
- Ensures `{workdir}` placeholder is included
- Validates MCP server commands
- Provides benchmark-specific templates
- Applies best practices

### 3. swe-bench-lite (`benchmark-swe-lite`)
Quick-start command for demonstrations:
- Pre-configured for 5-task evaluation
- Sensible defaults for output files
- Perfect for testing and demos

## Example Interactions

Simply ask Claude in natural language:

- "Run the SWE-bench Lite benchmark"
- "Generate a config for my MCP server"
- "Run a quick test with 1 task"

Claude will automatically:
- Verify prerequisites before starting
- Generate valid configurations
- Use correct CLI commands
- Handle errors gracefully

## How It Works

The plugin includes:
- **plugin.json** - Claude Code plugin manifest
- **skills/** - Three specialized SKILL.md files

When Claude Code loads this plugin, it injects the skill instructions into Claude's context, making it an expert at mcpbr without you having to explain anything.

## Development

This package is part of the mcpbr project:

```bash
# Clone the repository
git clone https://github.com/greynewell/mcpbr.git
cd mcpbr

# Test the plugin
pytest tests/test_claude_plugin.py -v
```

## Related Packages

- [`mcpbr`](https://pypi.org/project/mcpbr/) - Python package (core implementation)
- [`@greynewell/mcpbr`](https://www.npmjs.com/package/@greynewell/mcpbr) - npm CLI wrapper

## Documentation

For full documentation, visit: <https://greynewell.github.io/mcpbr/>

- [Claude Code Plugin Guide](https://greynewell.github.io/mcpbr/claude-code-plugin/)
- [Installation Guide](https://greynewell.github.io/mcpbr/installation/)
- [Configuration Reference](https://greynewell.github.io/mcpbr/configuration/)
- [CLI Reference](https://greynewell.github.io/mcpbr/cli/)

## License

MIT - see [LICENSE](../LICENSE) for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Support

- **Documentation**: <https://greynewell.github.io/mcpbr/>
- **Issues**: <https://github.com/greynewell/mcpbr/issues>
- **Discussions**: <https://github.com/greynewell/mcpbr/discussions>

---

Built by [Grey Newell](https://greynewell.com)
