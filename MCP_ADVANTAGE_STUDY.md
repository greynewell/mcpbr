# MCP Advantage Consistency Study

## Research Question

**Are the tasks where MCP outperformed baseline consistent wins, or just variance?**

## Background

From your Supermodel MCP evaluation:
- **300 total tasks** from SWE-bench Lite
- **MCP agent**: 50 resolved (16.7%)
- **Baseline agent**: 50 resolved (16.7%)
- **Both resolved**: 42 tasks
- **MCP-only wins**: 8 tasks ✨
- **Baseline-only wins**: 8 tasks

The critical insight: **MCP and baseline have identical overall success rates, but different task-level results**. This could mean:
1. **Variance/Non-determinism**: Due to randomness in LLM responses
2. **Task-specific advantage**: MCP genuinely helps on certain task types
3. **Complementary strengths**: Each approach has different strengths

## MCP-Only Wins (Candidates for Advantage Dataset)

```
1. pylint-dev__pylint-7228
2. pytest-dev__pytest-6116
3. pytest-dev__pytest-7168
4. pytest-dev__pytest-7490
5. pytest-dev__pytest-9359
6. scikit-learn__scikit-learn-13496
7. scikit-learn__scikit-learn-14092
8. sphinx-doc__sphinx-8435
```

**Pattern observation**: 5/8 are pytest tasks, suggesting MCP may excel at testing framework issues.

## Methodology

### Phase 1: Consistency Testing (5-10 runs per task)

Run each MCP-only win task multiple times to measure consistency:

```bash
python scripts/consistency_study.py \
    --results /path/to/metrics.json \
    --config your-config.yaml \
    --runs 5 \
    --output mcp_advantage_dataset.json
```

**Metrics to track:**
- **Consistency rate**: % of runs where MCP succeeds
- **Reliability threshold**: Tasks with ≥70% consistency
- **Variance analysis**: Standard deviation of success

**Expected outcomes:**
- **High consistency (≥70%)**: Genuine MCP advantage, add to dataset
- **Medium consistency (40-70%)**: Possible advantage, needs more runs
- **Low consistency (<40%)**: Likely variance, exclude from dataset

### Phase 2: Dataset Curation

Build a gold-standard dataset of reliable MCP wins:

**Inclusion criteria:**
1. MCP succeeds in ≥70% of runs
2. Baseline fails in ≥70% of runs (optional: test baseline consistency too)
3. Minimum 5 runs per task

**Dataset structure:**
```json
{
  "description": "Tasks where MCP reliably outperforms baseline",
  "tasks": [
    {
      "instance_id": "pytest-dev__pytest-6116",
      "consistency_rate": 0.80,
      "success_count": 4,
      "total_runs": 5,
      "category": "testing_framework",
      "mcp_advantage_reason": "Code exploration via explore_codebase"
    }
  ]
}
```

### Phase 3: Analysis

For each reliable MCP win, analyze **why** MCP excels:

**Possible factors:**
1. **Codebase exploration**: Tasks requiring deep code understanding
2. **Tool usage**: Specific MCP tools (e.g., `explore_codebase`) are helpful
3. **Iterative refinement**: MCP benefits from tool feedback loops
4. **Domain-specific**: Certain project types (pytest, scikit-learn, sphinx)

**Analysis script:**
```bash
# Extract logs for reliable tasks
python scripts/analyze_mcp_advantage.py \
    --dataset mcp_advantage_dataset.json \
    --logs-dir /path/to/logs \
    --output advantage_analysis.md
```

**Questions to answer:**
- Which MCP tools were most used on these tasks?
- Did explore_codebase calls correlate with success?
- What patterns exist in task descriptions?
- Are there common code patterns in winning tasks?

### Phase 4: Validation

Test the hypothesis on new tasks:

1. **Similarity search**: Find SWE-bench tasks similar to reliable wins
2. **Prediction**: Predict MCP advantage based on task characteristics
3. **Validation run**: Test predictions with new evaluation
4. **Refinement**: Update dataset and criteria based on results

## Expected Results

### Scenario A: High Consistency (Best Case)

**Result**: 6-8 tasks show ≥70% consistency

**Interpretation**: MCP has genuine, reproducible advantages on specific task types

**Action**:
- Build curated dataset for showcasing MCP
- Analyze commonalities to understand MCP strengths
- Use insights to guide MCP server development
- Market these as "MCP sweet spot" tasks

### Scenario B: Medium Consistency (Mixed)

**Result**: 3-5 tasks show ≥70% consistency, rest show 40-70%

**Interpretation**: Some genuine advantage, but with significant variance

**Action**:
- Focus on high-consistency tasks for marketing
- Investigate sources of variance (temperature, non-determinism)
- Consider deterministic settings for fairer comparison
- Run more trials on medium-consistency tasks

### Scenario C: Low Consistency (Variance)

**Result**: <3 tasks show ≥70% consistency

**Interpretation**: Most "wins" are likely due to variance/randomness

**Action**:
- Increase sample size (run 10+ times per task)
- Consider testing with temperature=0 for determinism
- Look for aggregate patterns across many tasks
- Focus on cost/efficiency rather than task-level wins

## Statistical Considerations

### Sample Size

For reliable conclusions:
- **5 runs**: Quick check, ~90% confidence for 80%+ consistency
- **10 runs**: Better confidence, ~95% confidence for 70%+ consistency
- **20 runs**: High confidence, ~99% confidence for 65%+ consistency

### Binomial Test

Test if MCP advantage is statistically significant:

```python
from scipy.stats import binomtest

# Example: 4 successes in 5 runs
result = binomtest(4, 5, 0.5, alternative='greater')
p_value = result.pvalue  # p < 0.05 suggests genuine advantage
```

### Multiple Testing Correction

Testing 8 tasks requires Bonferroni correction:
- **Significance threshold**: 0.05 / 8 = 0.00625
- Apply to each task's binomial test

## Cost Analysis

**Estimated costs** (assuming $0.086 per task from your results):

- **5 runs × 8 tasks = 40 tasks**: ~$3.50
- **10 runs × 8 tasks = 80 tasks**: ~$7.00
- **20 runs × 8 tasks = 160 tasks**: ~$14.00

**Recommendation**: Start with 5 runs, increase for promising tasks.

## Timeline

**Phase 1** (Consistency Testing): 2-4 hours
- Setup: 30 min
- 5 runs × 8 tasks @ 10 min each: ~6.5 hours (with concurrency: 2 hours)
- Analysis: 30 min

**Phase 2** (Dataset Curation): 1 hour
- Manual review of consistent tasks
- Category assignment
- Documentation

**Phase 3** (Analysis): 2-3 hours
- Log analysis
- Pattern identification
- Report writing

**Phase 4** (Validation): 4-8 hours
- Similarity search
- Validation runs
- Results comparison

**Total**: 9-16 hours spread over 1-2 weeks

## Deliverables

1. **Consistency Report**: Per-task consistency rates with statistical tests
2. **MCP Advantage Dataset**: Curated JSON file of reliable wins
3. **Analysis Document**: Patterns and insights about MCP strengths
4. **Validation Results**: Predictions tested on new tasks
5. **Recommendations**: Guidance for MCP server development

## Usage

### Quick Start

```bash
# 1. Run consistency study
python scripts/consistency_study.py \
    --results /Users/grey/Projects/mcp/swe-bench-eval-results/metrics.json \
    --config vm-eval-config.yaml \
    --runs 5

# 2. Review results
cat mcp_advantage_dataset.json

# 3. If reliable tasks found, analyze them
python scripts/analyze_mcp_advantage.py \
    --dataset mcp_advantage_dataset.json
```

### Full Study

```bash
# Phase 1: Consistency (5 runs)
python scripts/consistency_study.py -r metrics.json -c config.yaml --runs 5

# Phase 2: Extended testing for promising tasks (10 more runs)
python scripts/consistency_study.py -r metrics.json -c config.yaml --runs 10 \
    --tasks $(jq -r '.tasks[] | select(.consistency_rate >= 0.6) | .instance_id' mcp_advantage_dataset.json)

# Phase 3: Analyze patterns
python scripts/analyze_mcp_advantage.py --dataset mcp_advantage_dataset.json

# Phase 4: Validation
python scripts/validate_predictions.py --dataset mcp_advantage_dataset.json
```

## Next Steps

1. **Run the consistency study**: Start with 5 runs to quickly identify reliable tasks
2. **Share initial results**: Post findings with the team
3. **Deep dive on winners**: Analyze why MCP excels on consistent tasks
4. **Expand dataset**: Find similar tasks and test predictions
5. **Build narratives**: Create case studies for marketing

## Questions to Answer

- **Is MCP advantage real or variance?** → Consistency study
- **What makes tasks MCP-friendly?** → Pattern analysis
- **Can we predict MCP wins?** → Validation phase
- **How to optimize MCP for these tasks?** → Tool usage analysis
- **Should we focus development?** → Strategic recommendations

## References

- Original results: `/Users/grey/Projects/mcp/swe-bench-eval-results/metrics.json`
- SWE-bench: https://www.swebench.com/
- Supermodel MCP: https://github.com/supermodeltools/mcp
