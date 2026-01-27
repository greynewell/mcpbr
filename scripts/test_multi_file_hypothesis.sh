#!/bin/bash
# Quick test of multi-file hypothesis on 2 confirmed tasks

set -e

echo "======================================================================"
echo "Testing Multi-File Hypothesis on Confirmed MCP Wins"
echo "======================================================================"
echo ""
echo "Tasks to test:"
echo "  1. sphinx-doc__sphinx-8435 (autodoc type aliases - multi-module)"
echo "  2. pytest-dev__pytest-7168 (repr/getattribute - call chain)"
echo ""
echo "Hypothesis: These tasks should show HIGH consistency (≥80%)"
echo "because they require understanding cross-module relationships."
echo ""
echo "Cost: ~$1.50 for 10 runs each"
echo "Time: ~40 minutes with concurrency=4"
echo ""

# Check if config provided
if [ -z "$1" ]; then
    echo "Error: Please provide config file path"
    echo "Usage: ./test_multi_file_hypothesis.sh <config-file>"
    exit 1
fi

CONFIG=$1

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Check if results file exists
RESULTS="/Users/grey/Projects/mcp/swe-bench-eval-results/metrics.json"
if [ ! -f "$RESULTS" ]; then
    echo "Error: Results file not found: $RESULTS"
    exit 1
fi

echo "Using config: $CONFIG"
echo ""
read -p "Press Enter to start testing..."
echo ""

# Run consistency study on just these 2 tasks
python3 scripts/consistency_study.py \
    --results "$RESULTS" \
    --config "$CONFIG" \
    --runs 10 \
    --mcp-only \
    --output multi_file_hypothesis_results.json

echo ""
echo "======================================================================"
echo "Results Analysis"
echo "======================================================================"
echo ""

# Parse results
python3 <<'EOF'
import json

with open('multi_file_hypothesis_results.json') as f:
    results = json.load(f)

print("Task Consistency Results:")
print()

for task in results['tasks']:
    task_id = task['instance_id']
    consistency = task['consistency_rate']
    successes = task['success_count']
    total = task['total_runs']

    # Determine if hypothesis is supported
    if consistency >= 0.80:
        verdict = "✓ HYPOTHESIS SUPPORTED"
        color = "green"
    elif consistency >= 0.60:
        verdict = "⚠ PARTIAL SUPPORT"
        color = "yellow"
    else:
        verdict = "✗ HYPOTHESIS REJECTED"
        color = "red"

    print(f"{task_id}")
    print(f"  Consistency: {consistency:.1%} ({successes}/{total} successes)")
    print(f"  Verdict: {verdict}")
    print()

# Overall assessment
reliable_count = sum(1 for t in results['tasks'] if t['consistency_rate'] >= 0.70)
total_tasks = len(results['tasks'])

print("=" * 60)
print("Overall Assessment:")
print()

if reliable_count == total_tasks:
    print("✓ STRONG EVIDENCE for multi-file hypothesis")
    print("  Both tasks show high consistency (≥70%)")
    print()
    print("Recommendation:")
    print("  1. Expand consistency testing to other pytest tasks")
    print("  2. Build showcase dataset around these wins")
    print("  3. Consider running SWE-bench Full for more multi-file tasks")
    print("  4. Create 'MCP for Testing Frameworks' marketing angle")
elif reliable_count > 0:
    print("⚠ MIXED EVIDENCE for multi-file hypothesis")
    print(f"  {reliable_count}/{total_tasks} tasks show reliability")
    print()
    print("Recommendation:")
    print("  1. Run more trials on inconsistent task")
    print("  2. Analyze logs to understand variance")
    print("  3. Test with temperature=0 for determinism")
    print("  4. Be cautious about generalizing")
else:
    print("✗ HYPOTHESIS NOT SUPPORTED by current data")
    print("  Neither task shows reliable advantage")
    print()
    print("Possible explanations:")
    print("  1. Original wins were variance/luck")
    print("  2. explore_codebase not helping as expected")
    print("  3. Multi-file tasks may not be the key factor")
    print()
    print("Recommendation:")
    print("  1. Analyze logs: was explore_codebase actually useful?")
    print("  2. Look for other patterns in MCP wins")
    print("  3. Consider different evaluation approach")

EOF

echo ""
echo "Results saved to: multi_file_hypothesis_results.json"
echo ""
