# Contributing to mcpbr

Thank you for your interest in contributing to mcpbr! This document provides guidelines and information for contributors.

## Community Guidelines

All contributors are expected to follow our [Community Guidelines](CODE_OF_CONDUCT.md). We maintain a focused, productive technical environment where contributions are judged on technical merit.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the existing issues to avoid duplicates. When creating a bug report, include:

- A clear, descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, Docker version)
- Any relevant logs or error messages

### Suggesting Features

Feature requests are welcome! Please:

- Check existing issues and discussions first
- Clearly describe the feature and its use case
- Explain why this feature would be useful to most users

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11+
- Docker
- An API key for at least one supported provider (OpenRouter, OpenAI, Anthropic, or Google)

### Installation

```bash
# Clone your fork
git clone https://github.com/greynewell/mcpbr.git
cd mcpbr

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run unit tests only (no Docker or API keys required)
pytest -m "not integration"

# Run all tests (requires Docker and API keys)
pytest

# Run with coverage
pytest --cov=mcpbr
```

### Code Style

This project uses:

- [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Type hints throughout the codebase

Before submitting a PR:

```bash
# Run linter
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Development Tools

#### Makefile

The project includes a `Makefile` for common development tasks:

```bash
# Show all available commands
make help

# Install development dependencies
make install

# Run unit tests
make test

# Run all tests (including integration)
make test-all

# Run linting checks
make lint

# Format code
make format

# Sync version across project files
make sync-version

# Clean build artifacts
make clean

# Build distribution packages
make build
```

#### Version Management

The project uses a centralized version in `pyproject.toml` that is automatically synced to other files (like `.claude-plugin/plugin.json` and `package.json`).

**Automated syncing:**
- The version sync runs automatically during `make build`
- CI checks verify versions are in sync on every push/PR
- Pre-commit hooks can sync versions before commits (see below)

**Manual syncing:**
```bash
make sync-version
# or
python3 scripts/sync_version.py
```

#### npm Package Commands

The Claude Code plugin is published to npm as `claude-code-plugin-mcpbr`. Use these commands for npm package management:

```bash
# Prepare npm package (syncs version from pyproject.toml)
make npm-build

# Test npm package contents locally
make npm-test

# Create a tarball for local testing
make npm-pack

# Publish to npm (requires NPM_TOKEN environment variable)
make npm-publish
```

#### Pre-commit Hooks (Optional)

The project includes a `.pre-commit-config.yaml` for automated checks before commits:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

The pre-commit hooks will automatically:
- Sync version files when `pyproject.toml` changes
- Run ruff linting and formatting
- Fix trailing whitespace and file endings
- Check YAML syntax
- Prevent large files and merge conflicts

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Reference issues when applicable (e.g., "Fix #123")

## Project Structure

```
mcpbr/
├── src/mcpbr/          # Main package
│   ├── cli.py          # CLI commands
│   ├── config.py       # Configuration models
│   ├── harness.py      # Main orchestrator
│   ├── harnesses.py    # Agent implementations
│   ├── providers.py    # LLM provider abstractions
│   └── ...
├── tests/              # Test suite
├── config/             # Example configurations
└── docs/               # Documentation
```

## Adding New Features

### Adding a New Provider

1. Create a new class in `src/mcpbr/providers.py` implementing the `ModelProvider` protocol
2. Add it to `PROVIDER_REGISTRY`
3. Update `VALID_PROVIDERS` in `config.py`
4. Add tests
5. Update documentation

### Adding a New Agent Harness

1. Create a new class in `src/mcpbr/harnesses.py` implementing the `AgentHarness` protocol
2. Add it to `HARNESS_REGISTRY`
3. Update `VALID_HARNESSES` in `config.py`
4. Add tests
5. Update documentation

## Release Process

### Publishing a New Release

When a new GitHub release is published, the following happens automatically:

1. **PyPI Publishing** (`.github/workflows/publish.yml`)
   - Builds Python package
   - Publishes to PyPI using trusted publishing

2. **npm Publishing** (`.github/workflows/publish-npm.yml`)
   - Syncs version from release tag to `package.json`
   - Verifies package contents
   - Publishes to npm as `claude-code-plugin-mcpbr`

### Prerequisites for Automated Publishing

**PyPI:**
- Uses GitHub's trusted publishing (no token needed)
- Configured in PyPI project settings

**npm:**
- Requires `NPM_TOKEN` secret in GitHub repository settings
- Token must have publish permissions for `claude-code-plugin-mcpbr`
- **Note:** Repository maintainers must add this secret manually

### Creating a Release

1. Update version in `pyproject.toml`
2. Run `make sync-version` to sync to plugin.json and package.json
3. Commit and push changes
4. Create a new GitHub release with tag `vX.Y.Z` (e.g., `v0.3.18`)
5. Both PyPI and npm packages will be published automatically

### Testing Before Release

**Python package:**
```bash
make build
pip install dist/mcpbr-*.whl  # Test locally
```

**npm package:**
```bash
make npm-pack  # Creates a tarball
npm install ./claude-code-plugin-mcpbr-*.tgz  # Test locally
```

### Manual Publishing (if needed)

**PyPI:**
```bash
make build
pip install twine
twine upload dist/*
```

**npm:**
```bash
export NPM_TOKEN=your-npm-token
make npm-publish
# or
npm publish
```

## Questions?

Feel free to open an issue for any questions about contributing.
