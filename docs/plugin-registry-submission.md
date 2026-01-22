# Plugin Registry Submission Guide

This guide provides step-by-step instructions for submitting the mcpbr Claude Code plugin to the [claude-plugins.dev](https://claude-plugins.dev/) community registry.

## Overview

The claude-plugins.dev registry is a community-maintained directory of Claude Code plugins that makes it easy for users to discover and install plugins using a simple CLI command. Instead of manually cloning repositories and configuring paths, users can install mcpbr with:

```bash
npx @claude-plugins/cli install mcpbr
```

## Prerequisites Checklist

Before submitting to the registry, ensure the plugin meets all requirements:

- [x] **Valid plugin.json manifest** in `.claude-plugin/` directory
- [x] **Semantic versioning** in place (currently v0.3.17)
- [x] **Comprehensive README.md** with installation and usage instructions
- [x] **Skills directory** with properly formatted SKILL.md files
- [x] **Open source license** (MIT)
- [x] **Public GitHub repository** (https://github.com/greynewell/mcpbr)
- [x] **Documentation site** (https://greynewell.github.io/mcpbr/)
- [x] **Test coverage** for plugin functionality
- [x] **Working examples** in repository

## Plugin Information Summary

Use this information when filling out submission forms:

### Basic Information

| Field | Value |
|-------|-------|
| **Plugin Name** | mcpbr |
| **Display Name** | MCPBR - Model Context Protocol Benchmark Runner |
| **Version** | 0.3.17 |
| **Author** | Grey Newell |
| **License** | MIT |
| **Repository** | https://github.com/greynewell/mcpbr |
| **Homepage** | https://greynewell.github.io/mcpbr/ |

### Description

**Short Description (for listings):**
```
Expert benchmark runner for MCP servers. Benchmark your MCP server against real GitHub issues with hard numbers comparing tool-assisted vs. baseline agent performance.
```

**Long Description (for detail pages):**
```
mcpbr is a comprehensive benchmarking harness for Model Context Protocol (MCP) servers. It provides quantitative evaluation of MCP server effectiveness by running controlled experiments comparing tool-assisted vs. baseline agent performance on real software engineering tasks from SWE-bench, CyberGym, and MCPToolBench++.

The Claude Code plugin includes three specialized skills that make Claude an expert at:
- Running evaluations with proper validation (mcpbr-eval)
- Generating valid configuration files (mcpbr-config)
- Quick-start testing with SWE-bench Lite (benchmark-swe-lite)

When working in the mcpbr repository, Claude automatically gains domain expertise about Docker validation, config generation with {workdir} placeholders, correct CLI flags, and troubleshooting common issues.
```

### Categories/Tags

Suggested tags for categorization:
- `mcp`
- `benchmarking`
- `testing`
- `evaluation`
- `swe-bench`
- `development-tools`
- `quality-assurance`
- `devops`

### Features List

Key features to highlight:
- Automated benchmark evaluation for MCP servers
- Support for SWE-bench, CyberGym, and MCPToolBench++ benchmarks
- Controlled experiments with MCP vs. baseline comparison
- Docker-based isolated environments
- Regression detection with CI/CD integration
- Comprehensive reporting (JSON, Markdown, JUnit XML)
- Real-time progress monitoring
- Pre-built configuration templates
- Built-in Claude Code skills for expert assistance

### Installation Command

```bash
# From PyPI (Python package)
pip install mcpbr

# From source
git clone https://github.com/greynewell/mcpbr.git
cd mcpbr
pip install -e .

# With uv
uv pip install mcpbr
```

### Quick Start

```bash
# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Run first evaluation
mcpbr run -c examples/quick-start/getting-started.yaml -v
```

### Skills Included

1. **mcpbr-eval** - Expert at running evaluations with proper validation
2. **mcpbr-config** - Generates valid mcpbr configuration files
3. **benchmark-swe-lite** - Quick-start for SWE-bench Lite evaluation

## Submission Process

### Step 1: Prepare Repository

Ensure the repository is in good shape before submission:

```bash
# Verify plugin manifest
cat .claude-plugin/plugin.json

# Check skills are properly formatted
ls -la skills/*/SKILL.md

# Run tests
pytest tests/test_claude_plugin.py -v

# Verify documentation builds
mkdocs build --strict

# Check all links work
pytest tests/test_integration.py -k test_readme_links
```

### Step 2: Create Submission Assets

Gather all assets needed for the submission:

#### Required Assets

1. **Plugin Logo** (already available)
   - Location: `assets/mcpbr-logo.jpg`
   - Dimensions: 400px width recommended
   - Format: JPG/PNG
   - URL: https://raw.githubusercontent.com/greynewell/mcpbr/main/assets/mcpbr-logo.jpg

2. **Demo GIF/Video** (already available)
   - Location: `assets/mcpbr-demo.gif`
   - Shows plugin in action
   - URL: https://raw.githubusercontent.com/greynewell/mcpbr/main/assets/mcpbr-demo.gif

3. **Screenshot** (already available)
   - Location: `assets/mcpbr-eval-results.png`
   - Shows evaluation results
   - URL: https://raw.githubusercontent.com/greynewell/mcpbr/main/assets/mcpbr-eval-results.png

#### Optional Assets

4. **Tutorial/Walkthrough Video**
   - Consider creating a 2-3 minute screencast
   - Show complete workflow from installation to results
   - Upload to YouTube or similar platform

5. **Additional Screenshots**
   - Configuration generation
   - Real-time progress output
   - Regression detection report
   - Different benchmark results

### Step 3: Submit to Registry

Based on the research, the claude-plugins.dev registry has an informal submission process:

#### Option A: GitHub Pull Request (Recommended)

1. **Fork the registry repository:**
   ```bash
   git clone https://github.com/Kamalnrf/claude-plugins.git
   cd claude-plugins
   ```

2. **Add plugin entry** to the registry:
   - Look for `marketplace.json` or similar registry file
   - Add mcpbr entry with required metadata:
     ```json
     {
       "name": "mcpbr",
       "displayName": "MCPBR",
       "description": "Expert benchmark runner for MCP servers using mcpbr. Handles Docker checks, config generation, and result parsing.",
       "version": "0.3.17",
       "author": "Grey Newell",
       "repository": "https://github.com/greynewell/mcpbr",
       "homepage": "https://greynewell.github.io/mcpbr/",
       "license": "MIT",
       "tags": ["mcp", "benchmarking", "testing", "evaluation", "swe-bench"],
       "icon": "https://raw.githubusercontent.com/greynewell/mcpbr/main/assets/mcpbr-logo.jpg"
     }
     ```

3. **Create pull request** with title:
   ```
   Add mcpbr - MCP Server Benchmark Runner
   ```

   PR description template:
   ```markdown
   ## Plugin Submission: mcpbr

   ### Overview
   Expert benchmark runner for MCP servers that provides quantitative evaluation through controlled experiments on real software engineering tasks.

   ### Details
   - **Repository**: https://github.com/greynewell/mcpbr
   - **Version**: 0.3.17
   - **License**: MIT
   - **Documentation**: https://greynewell.github.io/mcpbr/

   ### Skills Included
   - mcpbr-eval: Run evaluations with validation
   - mcpbr-config: Generate valid configs
   - benchmark-swe-lite: Quick-start testing

   ### Testing
   - [x] Plugin manifest valid
   - [x] Skills load correctly
   - [x] Documentation complete
   - [x] Tests passing
   - [x] Examples working

   ### Additional Information
   mcpbr supports multiple benchmarks (SWE-bench, CyberGym, MCPToolBench++), includes regression detection, and provides comprehensive reporting for CI/CD integration.
   ```

#### Option B: GitHub Issue

If the PR approach doesn't work, open an issue in the registry repository:

1. **Navigate to**: https://github.com/Kamalnrf/claude-plugins/issues
2. **Create new issue** with title: `Plugin Submission: mcpbr`
3. **Fill in details** using the information from the Basic Information section
4. **Include links** to repository, documentation, and assets
5. **Tag** with appropriate labels (if available): `plugin-submission`, `enhancement`

#### Option C: Discord Community

The registry mentions a Discord community:

1. **Join Discord**: https://discord.gg/Pt9uN4FXR4
2. **Navigate** to plugin submission channel
3. **Share plugin information** with community
4. **Get feedback** and guidance on submission process
5. **Follow up** on any requirements

### Step 4: Monitor Submission

After submitting:

1. **Watch for feedback** on PR/Issue
2. **Respond promptly** to any questions or change requests
3. **Make updates** as needed to plugin or documentation
4. **Test installation** once accepted via registry CLI

## Post-Submission Tasks

### Immediate Actions

- [ ] **Update README** to include registry installation instructions
  ```markdown
  ## Installation

  ### From Community Registry (Recommended)
  ```bash
  npx @claude-plugins/cli install mcpbr
  ```

  ### From PyPI
  ```bash
  pip install mcpbr
  ```
  ```

- [ ] **Add badge** to README:
  ```markdown
  [![Available on claude-plugins.dev](https://img.shields.io/badge/claude--plugins.dev-available-blue)](https://claude-plugins.dev/)
  ```

- [ ] **Announce** on social media/blog:
  - Twitter/X
  - LinkedIn
  - Dev.to
  - Reddit (r/ClaudeAI, r/MachineLearning)

- [ ] **Update documentation** with registry-specific install instructions

### Ongoing Maintenance

- [ ] **Keep plugin.json version synced** with package version
- [ ] **Test registry installation** with each release
- [ ] **Monitor registry issues** for user feedback
- [ ] **Update registry metadata** when adding major features
- [ ] **Maintain compatibility** with Claude Code updates

## Troubleshooting Submission

### Common Issues

**Issue: PR automatically closed**
- **Cause**: Registry may redirect to submission form
- **Solution**: Check for automated comment with submission form link

**Issue: Plugin not appearing in search**
- **Cause**: Registry may need to rebuild index
- **Solution**: Wait 24-48 hours, then contact maintainers

**Issue: Installation fails**
- **Cause**: Plugin structure doesn't match registry expectations
- **Solution**: Verify `.claude-plugin/plugin.json` is valid and accessible

**Issue: Skills not loading**
- **Cause**: Skills directory structure incorrect
- **Solution**: Ensure `skills/*/SKILL.md` structure is correct

### Getting Help

1. **Registry GitHub Issues**: https://github.com/Kamalnrf/claude-plugins/issues
2. **Discord Community**: https://discord.gg/Pt9uN4FXR4
3. **Claude Code Docs**: https://code.claude.com/docs/en/plugins
4. **mcpbr Issues**: https://github.com/greynewell/mcpbr/issues

## Verification Checklist

Before marking submission as complete:

- [ ] Submitted to registry (PR/Issue/Discord)
- [ ] Assets uploaded and accessible
- [ ] Documentation updated with registry install
- [ ] Badge added to README
- [ ] Announcement prepared
- [ ] Monitoring submission status
- [ ] Post-submission tasks documented in project tracker

## Timeline Expectations

Based on community-maintained nature of registry:

- **Submission processing**: 2-7 days (depending on maintainer availability)
- **Feedback cycle**: 1-3 days per round
- **Approval to live**: 1-2 days after final approval
- **Total estimated time**: 1-2 weeks

## Success Metrics

Track these metrics after submission to measure success:

- **Installation count** (if registry provides analytics)
- **GitHub stars/forks** increase
- **Issue submissions** with `installed-from-registry` tag
- **Documentation page views**
- **Community engagement** (Discord mentions, social media)
- **Downstream usage** (other projects using mcpbr)

## Alternative Registries

Consider submitting to additional registries:

1. **Anthropic Official Marketplace** (when available)
   - Requires submission form and review
   - Higher visibility but stricter requirements

2. **NPM Package** (for Node.js users)
   - Create wrapper package
   - Publish to npm registry

3. **Awesome Lists**
   - [Awesome Claude](https://github.com/topics/claude)
   - [Awesome MCP](https://github.com/topics/mcp)

## Reference Links

- **Plugin Registry**: https://claude-plugins.dev/
- **Registry GitHub**: https://github.com/Kamalnrf/claude-plugins
- **Discord Community**: https://discord.gg/Pt9uN4FXR4
- **Claude Code Docs**: https://code.claude.com/docs/en/plugins
- **mcpbr Repository**: https://github.com/greynewell/mcpbr
- **mcpbr Documentation**: https://greynewell.github.io/mcpbr/

## Updates and Revisions

This guide will be updated as:
- Registry submission process evolves
- New requirements are discovered
- Community feedback is received
- Best practices emerge

Last updated: 2026-01-22

---

**Related Issues**: #267
**Maintainer**: @greynewell
**Next Steps**: Follow Step 3 submission process and update this document with actual experience
