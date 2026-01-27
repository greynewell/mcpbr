# Quick Start: MCP Advantage Study

## TL;DR

Test if your 8 "MCP-only wins" are consistent or just variance:

```bash
# Install dependencies
pip install click rich scipy

# Run 5 trials on each MCP win
python scripts/consistency_study.py \
    --results /Users/grey/Projects/mcp/swe-bench-eval-results/metrics.json \
    --config /path/to/your/config.yaml \
    --runs 5 \
    --mcp-only

# Results saved to: mcp_advantage_dataset.json
```

**Cost**: ~$3.50 (40 task runs)
**Time**: ~2 hours with concurrency=4
**Output**: Dataset of reliable MCP wins

## The 8 MCP-Only Wins

From your original run, these tasks succeeded with MCP but failed with baseline:

1. `pylint-dev__pylint-7228`
2. `pytest-dev__pytest-6116`
3. `pytest-dev__pytest-7168`
4. `pytest-dev__pytest-7490`
5. `pytest-dev__pytest-9359`
6. `scikit-learn__scikit-learn-13496`
7. `scikit-learn__scikit-learn-14092`
8. `sphinx-doc__sphinx-8435`

**Pattern**: 5/8 are pytest tasks! This suggests MCP may excel at testing framework bugs.

## What You'll Learn

### ✅ If consistency ≥70%:
- **You have genuine MCP advantages**
- Build a showcase dataset
- Market these as "MCP sweet spots"
- Analyze what makes them MCP-friendly

### ⚠️ If consistency 40-70%:
- **Mixed results**
- Some advantage, but with variance
- Run more trials on promising tasks
- Consider temperature=0 for determinism

### ❌ If consistency <40%:
- **Likely just variance/randomness**
- Original wins were probably flukes
- Focus on aggregate metrics instead
- Consider different evaluation approaches

## Expected Results

Based on the data:

**Hypothesis**: The pytest tasks (5/8) will show higher consistency because:
- Testing frameworks benefit from codebase exploration
- `explore_codebase` tool helps understand test structure
- Iterative debugging works well with tool feedback

**Prediction**: 3-5 tasks will show ≥70% consistency

## Step-by-Step

### 1. Setup (5 minutes)

```bash
cd /Users/grey/Projects/mcpbr-worktrees/mcp-advantage-study

# Install dependencies
pip install click rich scipy

# Verify original results
cat /Users/grey/Projects/mcp/swe-bench-eval-results/metrics.json | jq '.mcp_only_wins'
```

### 2. Run Consistency Study (2 hours)

```bash
# Create a test config (or use your existing one)
cp /Users/grey/Projects/mcp/vm-eval-config.yaml test-config.yaml

# Run the study
python scripts/consistency_study.py \
    --results /Users/grey/Projects/mcp/swe-bench-eval-results/metrics.json \
    --config test-config.yaml \
    --runs 5 \
    --mcp-only \
    --output mcp_advantage_dataset.json
```

**Note**: Use `--mcp-only` to skip baseline runs (faster, cheaper). We already know baseline failed these tasks.

### 3. Review Results (10 minutes)

```bash
# View the dataset
cat mcp_advantage_dataset.json | jq '.tasks'

# Check how many are reliable (≥70% consistency)
cat mcp_advantage_dataset.json | jq '.total_reliable_tasks'

# See the most consistent tasks
cat mcp_advantage_dataset.json | jq '.tasks | sort_by(.consistency_rate) | reverse'
```

### 4. Analyze Patterns (30 minutes)

Look for commonalities in reliable tasks:

```bash
# Extract task logs for analysis
for task_id in $(jq -r '.tasks[].instance_id' mcp_advantage_dataset.json); do
    echo "=== $task_id ===" >> reliable_task_analysis.txt
    # Add any specific analysis here
done
```

**Questions to ask:**
- Which projects? (pytest, scikit-learn, sphinx, pylint)
- Which MCP tools were used? (check logs)
- What types of bugs? (test failures, API changes, etc.)
- Any common code patterns?

## Output Files

After running, you'll have:

1. **mcp_advantage_dataset.json**: Curated list of reliable MCP wins
2. **consistency_run_*.json**: Individual run results (temporary, cleaned up automatically)
3. **Console output**: Rich-formatted tables with statistics

## Example Output

```
MCP Advantage Consistency Study

Loading results from metrics.json...
Found 8 MCP-only wins

MCP Advantage Tasks:
  • pylint-dev__pylint-7228
  • pytest-dev__pytest-6116
  • pytest-dev__pytest-7168
  • pytest-dev__pytest-7490
  • pytest-dev__pytest-9359
  • scikit-learn__scikit-learn-13496
  • scikit-learn__scikit-learn-14092
  • sphinx-doc__sphinx-8435

Run 1/5
  Testing pylint-dev__pylint-7228... ✓
  Testing pytest-dev__pytest-6116... ✓
  ...

Consistency Study Results

Total Tasks Tested: 8
Reliable MCP Advantage: 5 (62.5%)
Average Consistency Rate: 68%

┌─────────────────────────────────┬───────────┬────────────┬─────────────┬───────────┐
│ Task ID                         │ Successes │ Total Runs │ Consistency │ Reliable? │
├─────────────────────────────────┼───────────┼────────────┼─────────────┼───────────┤
│ pytest-dev__pytest-6116         │ 5         │ 5          │ 100.0%      │ ✓         │
│ pytest-dev__pytest-7490         │ 4         │ 5          │ 80.0%       │ ✓         │
│ pytest-dev__pytest-7168         │ 4         │ 5          │ 80.0%       │ ✓         │
│ scikit-learn__scikit-learn-...  │ 3         │ 5          │ 60.0%       │ ✗         │
│ ...                             │ ...       │ ...        │ ...         │ ...       │
└─────────────────────────────────┴───────────┴────────────┴─────────────┴───────────┘

Saved MCP advantage dataset to mcp_advantage_dataset.json
Total reliable tasks: 5

Recommendation:
Focus on the 5 reliable tasks for showcasing MCP value.
Consider analyzing what makes these tasks MCP-friendly.
```

## Next Steps Based on Results

### If you find 5+ reliable tasks:

1. **Build showcase**: Create case studies for each reliable task
2. **Deep analysis**: Understand *why* MCP excels (tool usage patterns)
3. **Find similar tasks**: Use task embeddings to find more MCP-friendly problems
4. **Marketing materials**: "MCP excels at pytest debugging" narrative

### If you find 2-4 reliable tasks:

1. **Run more trials**: 10 runs instead of 5 for better confidence
2. **Investigate variance**: Check temperature, prompt variations
3. **Partial success**: Some tasks show MCP value, build on those
4. **Expand search**: Test MCP on more tasks from same projects

### If you find 0-1 reliable tasks:

1. **Check for issues**: Verify MCP server is working correctly
2. **Increase runs**: Try 10-20 runs to reduce noise
3. **Lower threshold**: Consider 60% consistency as "moderate advantage"
4. **Different angle**: Focus on cost/efficiency rather than task wins
5. **Aggregate analysis**: Look at overall patterns across many tasks

## Pro Tips

1. **Start small**: 5 runs gives quick signal, expand if promising
2. **Use --mcp-only**: Skip baseline to save time/money
3. **Check logs**: High variance might indicate MCP server issues
4. **Temperature=0**: Consider using for more deterministic results
5. **Track tool usage**: Monitor which MCP tools correlate with success

## Cost Optimization

```bash
# Quick check (1 run per task): $0.70, 20 minutes
python scripts/consistency_study.py -r metrics.json -c config.yaml --runs 1 --mcp-only

# Standard (5 runs per task): $3.50, 2 hours
python scripts/consistency_study.py -r metrics.json -c config.yaml --runs 5 --mcp-only

# High confidence (10 runs per task): $7.00, 4 hours
python scripts/consistency_study.py -r metrics.json -c config.yaml --runs 10 --mcp-only

# Deep study (20 runs per task): $14.00, 8 hours
python scripts/consistency_study.py -r metrics.json -c config.yaml --runs 20 --mcp-only
```

## Troubleshooting

**Script fails to import click/rich:**
```bash
pip install click rich scipy
```

**Config file not found:**
- Use absolute path to config
- Make sure config has proper MCP server settings

**Tasks timeout:**
- Increase timeout in script (default: 600s)
- Check Docker is running
- Verify API key is set

**No output file created:**
- Check mcpbr is installed: `pip install mcpbr`
- Verify config is valid: `mcpbr validate -c config.yaml`
- Run with `-v` flag for verbose output

## Questions?

See the full methodology in `MCP_ADVANTAGE_STUDY.md` or ask questions in the repo.

Good luck! 🚀
