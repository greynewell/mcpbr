#!/bin/bash
# Compare MCP vs Baseline on specific multi-file tasks

set -e

echo "======================================================================"
echo "Multi-File Tasks: MCP vs Baseline Comparison"
echo "======================================================================"
echo ""

if [ -z "$1" ]; then
    echo "Usage: ./compare_multifile_tasks.sh <config-file> [num-runs]"
    echo ""
    echo "This will run N trials (default: 5) of both MCP and baseline"
    echo "on the confirmed multi-file tasks to test consistency."
    echo ""
    echo "Tasks to test:"
    echo "  1. sphinx-doc__sphinx-8435 (autodoc multi-module)"
    echo "  2. pytest-dev__pytest-7168 (call chain analysis)"
    echo ""
    echo "Cost: ~$3.50 for 5 runs (both MCP + baseline)"
    echo "Time: ~2 hours with concurrency=4"
    exit 1
fi

CONFIG=$1
RUNS=${2:-5}

# The two confirmed multi-file MCP wins
TASKS=(
    "sphinx-doc__sphinx-8435"
    "pytest-dev__pytest-7168"
)

echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Runs per task: $RUNS"
echo "  Tasks: ${#TASKS[@]}"
echo ""
echo "Tasks to test:"
for task in "${TASKS[@]}"; do
    echo "  - $task"
done
echo ""
echo "Total runs: $((${#TASKS[@]} * $RUNS * 2)) (MCP + baseline)"
echo ""
read -p "Press Enter to start..."

# Create results directory
RESULTS_DIR="multifile_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run a single trial
run_trial() {
    local task=$1
    local run_num=$2
    local mode=$3  # "mcp" or "baseline"

    local output_file="$RESULTS_DIR/${task}_${mode}_run${run_num}.json"

    echo "  Run $run_num ($mode): $task"

    local cmd="mcpbr run -c $CONFIG -t $task -o $output_file"

    if [ "$mode" = "mcp" ]; then
        cmd="$cmd --mcp-only"
    else
        cmd="$cmd --baseline-only"
    fi

    # Run with timeout
    timeout 600 $cmd > /dev/null 2>&1 || true
}

# Run all trials
for task in "${TASKS[@]}"; do
    echo ""
    echo "Testing: $task"
    echo "----------------------------------------"

    for ((run=1; run<=$RUNS; run++)); do
        run_trial "$task" "$run" "mcp"
        run_trial "$task" "$run" "baseline"
    done
done

echo ""
echo "======================================================================"
echo "Analysis"
echo "======================================================================"

# Analyze results
python3 <<EOF
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path("$RESULTS_DIR")
tasks = ["sphinx-doc__sphinx-8435", "pytest-dev__pytest-7168"]

data = defaultdict(lambda: {"mcp": [], "baseline": []})

# Load all results
for task in tasks:
    for mode in ["mcp", "baseline"]:
        for run in range(1, $RUNS + 1):
            file = results_dir / f"{task}_{mode}_run{run}.json"
            if file.exists():
                try:
                    with open(file) as f:
                        result = json.load(f)

                    # Extract resolution status
                    resolved = False
                    for t in result.get("tasks", []):
                        if t["instance_id"] == task:
                            if mode == "mcp":
                                resolved = t.get("mcp", {}).get("resolved", False)
                            else:
                                resolved = t.get("baseline", {}).get("resolved", False)
                            break

                    data[task][mode].append(resolved)
                except:
                    pass

# Print results
print()
print("=" * 70)
print("RESULTS: Multi-File Tasks Consistency")
print("=" * 70)
print()

overall_mcp_wins = 0
overall_consistent = 0

for task in tasks:
    mcp_results = data[task]["mcp"]
    baseline_results = data[task]["baseline"]

    mcp_success = sum(mcp_results)
    baseline_success = sum(baseline_results)

    mcp_rate = mcp_success / len(mcp_results) if mcp_results else 0
    baseline_rate = baseline_success / len(baseline_results) if baseline_results else 0

    advantage = mcp_rate - baseline_rate

    print(f"Task: {task}")
    print(f"  MCP:      {mcp_success}/{len(mcp_results)} = {mcp_rate:.1%}")
    print(f"  Baseline: {baseline_success}/{len(baseline_results)} = {baseline_rate:.1%}")
    print(f"  Advantage: {advantage:+.1%}")

    # Determine if this is a consistent MCP win
    if mcp_rate >= 0.70 and baseline_rate <= 0.30:
        print(f"  Status: ✓ CONSISTENT MCP WIN")
        overall_consistent += 1
        overall_mcp_wins += 1
    elif mcp_rate > baseline_rate and mcp_rate >= 0.60:
        print(f"  Status: ⚠ POSSIBLE MCP WIN (needs more data)")
        overall_mcp_wins += 1
    elif abs(advantage) <= 0.20:
        print(f"  Status: ~ NO CLEAR ADVANTAGE")
    elif baseline_rate > mcp_rate:
        print(f"  Status: ✗ BASELINE BETTER")
    else:
        print(f"  Status: ? UNCLEAR")
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"Consistent MCP wins: {overall_consistent}/{len(tasks)}")
print(f"Tasks with MCP advantage: {overall_mcp_wins}/{len(tasks)}")
print()

if overall_consistent == len(tasks):
    print("✓ STRONG EVIDENCE: Both multi-file tasks show consistent MCP advantage")
    print()
    print("Next steps:")
    print("  1. These are reliable MCP showcase tasks")
    print("  2. Analyze logs to understand WHY code graphs helped")
    print("  3. Find similar tasks in SWE-bench Full")
    print("  4. Build 'MCP for Testing Frameworks' narrative")
elif overall_consistent > 0:
    print("⚠ PARTIAL EVIDENCE: Some multi-file tasks show advantage")
    print()
    print("Next steps:")
    print("  1. Run more trials on inconsistent tasks")
    print("  2. Check logs for failure modes")
    print("  3. Test with temperature=0 for determinism")
elif overall_mcp_wins > 0:
    print("~ WEAK EVIDENCE: Some advantage but low consistency")
    print()
    print("Possible explanations:")
    print("  1. High variance in results")
    print("  2. Code graphs help but not reliably")
    print("  3. Need larger sample size")
else:
    print("✗ NO EVIDENCE: Original wins appear to be variance")
    print()
    print("Possible explanations:")
    print("  1. explore_codebase not actually helpful")
    print("  2. Multi-file hypothesis incorrect")
    print("  3. Original evaluation had lucky runs")

# Save summary
summary = {
    "tasks_tested": tasks,
    "runs_per_task": $RUNS,
    "results": {}
}

for task in tasks:
    summary["results"][task] = {
        "mcp_successes": sum(data[task]["mcp"]),
        "mcp_total": len(data[task]["mcp"]),
        "mcp_rate": sum(data[task]["mcp"]) / len(data[task]["mcp"]) if data[task]["mcp"] else 0,
        "baseline_successes": sum(data[task]["baseline"]),
        "baseline_total": len(data[task]["baseline"]),
        "baseline_rate": sum(data[task]["baseline"]) / len(data[task]["baseline"]) if data[task]["baseline"] else 0,
    }

with open("$RESULTS_DIR/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print(f"Detailed summary saved to: $RESULTS_DIR/summary.json")
EOF

echo ""
echo "======================================================================"
echo "Complete!"
echo "======================================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
