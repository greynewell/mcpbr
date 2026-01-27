#!/usr/bin/env python3
"""Consistency study for MCP advantage tasks.

This script helps identify tasks where MCP reliably outperforms baseline
by running consistency tests and building a curated dataset.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()


def load_results(results_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def extract_mcp_advantage_tasks(results: dict[str, Any]) -> list[str]:
    """Extract task IDs where MCP succeeded but baseline failed."""
    return results.get("mcp_only_wins", [])


def extract_baseline_advantage_tasks(results: dict[str, Any]) -> list[str]:
    """Extract task IDs where baseline succeeded but MCP failed."""
    return results.get("baseline_only_wins", [])


def run_consistency_test(
    config_path: Path,
    task_ids: list[str],
    num_runs: int,
    mcp_only: bool = False,
) -> dict[str, Any]:
    """Run tasks multiple times to test consistency.

    Args:
        config_path: Path to mcpbr config file
        task_ids: List of task IDs to test
        num_runs: Number of times to run each task
        mcp_only: If True, only run MCP agent (faster)

    Returns:
        Dictionary with consistency metrics
    """
    results = {task_id: {"runs": [], "mcp_success_count": 0} for task_id in task_ids}

    for run_num in range(1, num_runs + 1):
        console.print(f"\n[bold cyan]Run {run_num}/{num_runs}[/bold cyan]")

        for task_id in task_ids:
            console.print(f"  Testing {task_id}...")

            # Build mcpbr command
            cmd = [
                "mcpbr",
                "run",
                "-c",
                str(config_path),
                "-t",
                task_id,
                "-o",
                f"consistency_run_{run_num}_{task_id}.json",
            ]

            if mcp_only:
                cmd.append("--mcp-only")

            try:
                # Run the task
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 min timeout per task
                    check=False,  # Don't raise on non-zero exit
                )

                # Parse the result
                output_file = Path(f"consistency_run_{run_num}_{task_id}.json")
                if output_file.exists():
                    with open(output_file) as f:
                        data = json.load(f)

                    # Check if task was resolved
                    mcp_resolved = False
                    for task in data.get("tasks", []):
                        if task["instance_id"] == task_id:
                            mcp_resolved = task.get("mcp", {}).get("resolved", False)
                            break

                    results[task_id]["runs"].append({"run": run_num, "resolved": mcp_resolved})

                    if mcp_resolved:
                        results[task_id]["mcp_success_count"] += 1

                    # Clean up
                    output_file.unlink()

            except subprocess.TimeoutExpired:
                console.print(f"    [yellow]Timeout on {task_id}[/yellow]")
                results[task_id]["runs"].append({"run": run_num, "resolved": False})
            except Exception as e:
                console.print(f"    [red]Error on {task_id}: {e}[/red]")
                results[task_id]["runs"].append(
                    {"run": run_num, "resolved": False, "error": str(e)}
                )

    # Calculate consistency metrics
    for task_id, data in results.items():
        total_runs = len(data["runs"])
        success_count = data["mcp_success_count"]
        data["consistency_rate"] = success_count / total_runs if total_runs > 0 else 0
        data["reliable"] = data["consistency_rate"] >= 0.7  # 70% threshold

    return results


def print_consistency_report(
    consistency_results: dict[str, Any], original_results: dict[str, Any]
) -> None:
    """Print a formatted consistency report."""
    console.print("\n[bold]Consistency Study Results[/bold]\n")

    # Summary stats
    total_tasks = len(consistency_results)
    reliable_tasks = sum(1 for r in consistency_results.values() if r["reliable"])
    avg_consistency = (
        sum(r["consistency_rate"] for r in consistency_results.values()) / total_tasks
        if total_tasks > 0
        else 0
    )

    console.print(f"Total Tasks Tested: {total_tasks}")
    console.print(
        f"Reliable MCP Advantage: {reliable_tasks} ({reliable_tasks / total_tasks * 100:.1f}%)"
    )
    console.print(f"Average Consistency Rate: {avg_consistency:.2%}\n")

    # Detailed table
    table = Table(title="Per-Task Consistency")
    table.add_column("Task ID", style="cyan")
    table.add_column("Successes", justify="right")
    table.add_column("Total Runs", justify="right")
    table.add_column("Consistency", justify="right")
    table.add_column("Reliable?", justify="center")

    for task_id, data in sorted(
        consistency_results.items(), key=lambda x: x[1]["consistency_rate"], reverse=True
    ):
        success_count = data["mcp_success_count"]
        total_runs = len(data["runs"])
        consistency = data["consistency_rate"]
        reliable = "✓" if data["reliable"] else "✗"
        style = "green" if data["reliable"] else "yellow"

        table.add_row(
            task_id,
            str(success_count),
            str(total_runs),
            f"{consistency:.1%}",
            reliable,
            style=style,
        )

    console.print(table)


def build_advantage_dataset(
    consistency_results: dict[str, Any], output_path: Path, min_consistency: float = 0.7
) -> None:
    """Build a curated dataset of reliable MCP advantage tasks.

    Args:
        consistency_results: Results from consistency study
        output_path: Where to save the dataset
        min_consistency: Minimum consistency rate to include (default: 70%)
    """
    dataset = {
        "description": "Curated dataset of tasks where MCP reliably outperforms baseline",
        "methodology": {
            "selection_criteria": f"Tasks where MCP succeeds with >={min_consistency:.0%} consistency",
            "consistency_threshold": min_consistency,
        },
        "tasks": [],
    }

    for task_id, data in consistency_results.items():
        if data["consistency_rate"] >= min_consistency:
            dataset["tasks"].append(
                {
                    "instance_id": task_id,
                    "consistency_rate": data["consistency_rate"],
                    "success_count": data["mcp_success_count"],
                    "total_runs": len(data["runs"]),
                }
            )

    # Sort by consistency rate
    dataset["tasks"].sort(key=lambda x: x["consistency_rate"], reverse=True)
    dataset["total_reliable_tasks"] = len(dataset["tasks"])

    # Save dataset
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    console.print(f"\n[green]Saved MCP advantage dataset to {output_path}[/green]")
    console.print(f"Total reliable tasks: {dataset['total_reliable_tasks']}")


@click.command()
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to original evaluation results JSON",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to mcpbr config file for re-runs",
)
@click.option(
    "--runs",
    "-n",
    type=int,
    default=5,
    help="Number of times to run each task (default: 5)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("mcp_advantage_dataset.json"),
    help="Output path for curated dataset",
)
@click.option(
    "--min-consistency",
    type=float,
    default=0.7,
    help="Minimum consistency rate to include in dataset (default: 0.7)",
)
@click.option(
    "--mcp-only",
    is_flag=True,
    help="Only test MCP agent (faster, skip baseline)",
)
def main(
    results: Path,
    config: Path,
    runs: int,
    output: Path,
    min_consistency: float,
    mcp_only: bool,
) -> None:
    """Run consistency study to identify reliable MCP advantage tasks.

    This tool helps answer the question: "Are MCP wins consistent or just variance?"

    \b
    Example:
        python consistency_study.py \\
            --results metrics.json \\
            --config config.yaml \\
            --runs 5 \\
            --output mcp_advantage_dataset.json
    """
    console.print("[bold]MCP Advantage Consistency Study[/bold]\n")

    # Load original results
    console.print(f"Loading results from {results}...")
    original_results = load_results(results)

    # Extract MCP advantage tasks
    mcp_advantage_tasks = extract_mcp_advantage_tasks(original_results)
    console.print(f"Found {len(mcp_advantage_tasks)} MCP-only wins\n")

    if not mcp_advantage_tasks:
        console.print("[yellow]No MCP advantage tasks found. Exiting.[/yellow]")
        sys.exit(0)

    # Display tasks
    console.print("[bold]MCP Advantage Tasks:[/bold]")
    for task_id in mcp_advantage_tasks:
        console.print(f"  • {task_id}")

    console.print(f"\nWill run each task {runs} times to test consistency...")
    input("\nPress Enter to continue...")

    # Run consistency test
    consistency_results = run_consistency_test(config, mcp_advantage_tasks, runs, mcp_only)

    # Print report
    print_consistency_report(consistency_results, original_results)

    # Build dataset
    build_advantage_dataset(consistency_results, output, min_consistency)

    # Recommendations
    reliable_count = sum(1 for r in consistency_results.values() if r["reliable"])
    if reliable_count > 0:
        console.print("\n[bold green]Recommendation:[/bold green]")
        console.print(f"Focus on the {reliable_count} reliable tasks for showcasing MCP value.")
        console.print("Consider analyzing what makes these tasks MCP-friendly.")
    else:
        console.print("\n[bold yellow]Recommendation:[/bold yellow]")
        console.print("No reliable MCP advantage found. This suggests:")
        console.print("  1. Results may be due to variance/non-determinism")
        console.print("  2. MCP may not provide consistent advantage on these tasks")
        console.print("  3. Consider testing with more runs or different tasks")


if __name__ == "__main__":
    main()
