"""Reporting utilities for evaluation results."""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from .harness import EvaluationResults


def print_summary(results: "EvaluationResults", console: Console) -> None:
    """Print a summary of evaluation results to the console.

    Args:
        results: Evaluation results.
        console: Rich console for output.
    """
    console.print()
    console.print("[bold]Evaluation Results[/bold]")
    console.print()

    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("MCP Agent", style="green")
    table.add_column("Baseline", style="yellow")

    mcp = results.summary["mcp"]
    baseline = results.summary["baseline"]

    table.add_row(
        "Resolved",
        f"{mcp['resolved']}/{mcp['total']}",
        f"{baseline['resolved']}/{baseline['total']}",
    )
    table.add_row(
        "Resolution Rate",
        f"{mcp['rate']:.1%}",
        f"{baseline['rate']:.1%}",
    )

    console.print(table)
    console.print()
    console.print(f"[bold]Improvement:[/bold] {results.summary['improvement']}")

    console.print()
    console.print("[bold]Per-Task Results[/bold]")

    task_table = Table()
    task_table.add_column("Instance ID", style="dim")
    task_table.add_column("MCP", justify="center")
    task_table.add_column("Baseline", justify="center")
    task_table.add_column("Error", style="red", max_width=50)

    for task in results.tasks:
        mcp_status = (
            "[green]PASS[/green]" if task.mcp and task.mcp.get("resolved") else "[red]FAIL[/red]"
        )
        if task.mcp is None:
            mcp_status = "[dim]-[/dim]"

        baseline_status = (
            "[green]PASS[/green]"
            if task.baseline and task.baseline.get("resolved")
            else "[red]FAIL[/red]"
        )
        if task.baseline is None:
            baseline_status = "[dim]-[/dim]"

        error_msg = ""
        if task.mcp and task.mcp.get("error"):
            error_msg = task.mcp.get("error", "")
        elif task.baseline and task.baseline.get("error"):
            error_msg = task.baseline.get("error", "")

        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."

        task_table.add_row(task.instance_id, mcp_status, baseline_status, error_msg)

    console.print(task_table)


def save_json_results(results: "EvaluationResults", output_path: Path) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Evaluation results.
        output_path: Path to save the JSON file.
    """
    data = {
        "metadata": results.metadata,
        "summary": results.summary,
        "tasks": [],
    }

    for task in results.tasks:
        task_data = {
            "instance_id": task.instance_id,
        }
        if task.mcp:
            task_data["mcp"] = task.mcp
        if task.baseline:
            task_data["baseline"] = task.baseline
        data["tasks"].append(task_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_markdown_report(results: "EvaluationResults", output_path: Path) -> None:
    """Save evaluation results as a Markdown report.

    Args:
        results: Evaluation results.
        output_path: Path to save the Markdown file.
    """
    lines = []

    lines.append("# SWE-bench MCP Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {results.metadata['timestamp']}")
    lines.append(f"**Model:** {results.metadata['config']['model']}")
    lines.append(f"**Dataset:** {results.metadata['config']['dataset']}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")

    mcp = results.summary["mcp"]
    baseline = results.summary["baseline"]

    lines.append("| Metric | MCP Agent | Baseline |")
    lines.append("|--------|-----------|----------|")
    lines.append(
        f"| Resolved | {mcp['resolved']}/{mcp['total']} | {baseline['resolved']}/{baseline['total']} |"
    )
    lines.append(f"| Resolution Rate | {mcp['rate']:.1%} | {baseline['rate']:.1%} |")
    lines.append("")
    lines.append(f"**Improvement:** {results.summary['improvement']}")
    lines.append("")

    lines.append("## MCP Server Configuration")
    lines.append("")
    lines.append("```")
    lines.append(f"command: {results.metadata['mcp_server']['command']}")
    lines.append(f"args: {results.metadata['mcp_server']['args']}")
    lines.append("```")
    lines.append("")

    lines.append("## Per-Task Results")
    lines.append("")
    lines.append("| Instance ID | MCP | Baseline |")
    lines.append("|-------------|-----|----------|")

    for task in results.tasks:
        mcp_status = "PASS" if task.mcp and task.mcp.get("resolved") else "FAIL"
        if task.mcp is None:
            mcp_status = "-"

        baseline_status = "PASS" if task.baseline and task.baseline.get("resolved") else "FAIL"
        if task.baseline is None:
            baseline_status = "-"

        lines.append(f"| {task.instance_id} | {mcp_status} | {baseline_status} |")

    lines.append("")

    mcp_only = []
    baseline_only = []
    both = []
    neither = []

    for task in results.tasks:
        mcp_resolved = task.mcp and task.mcp.get("resolved")
        baseline_resolved = task.baseline and task.baseline.get("resolved")

        if mcp_resolved and baseline_resolved:
            both.append(task.instance_id)
        elif mcp_resolved:
            mcp_only.append(task.instance_id)
        elif baseline_resolved:
            baseline_only.append(task.instance_id)
        else:
            neither.append(task.instance_id)

    lines.append("## Analysis")
    lines.append("")
    lines.append(f"- **Resolved by both:** {len(both)}")
    lines.append(f"- **Resolved by MCP only:** {len(mcp_only)}")
    lines.append(f"- **Resolved by Baseline only:** {len(baseline_only)}")
    lines.append(f"- **Resolved by neither:** {len(neither)}")
    lines.append("")

    if mcp_only:
        lines.append("### Tasks Resolved by MCP Only")
        lines.append("")
        for task_id in mcp_only:
            lines.append(f"- {task_id}")
        lines.append("")

    if baseline_only:
        lines.append("### Tasks Resolved by Baseline Only")
        lines.append("")
        for task_id in baseline_only:
            lines.append(f"- {task_id}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def save_yaml_results(results: "EvaluationResults", output_path: Path) -> None:
    """Save evaluation results to a YAML file.

    Args:
        results: Evaluation results.
        output_path: Path to save the YAML file.
    """
    data = {
        "metadata": results.metadata,
        "summary": results.summary,
        "tasks": [],
    }

    for task in results.tasks:
        task_data = {
            "instance_id": task.instance_id,
        }
        if task.mcp:
            task_data["mcp"] = task.mcp
        if task.baseline:
            task_data["baseline"] = task.baseline
        data["tasks"].append(task_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def save_csv_results(
    results: "EvaluationResults", output_path: Path, format: str = "summary"
) -> None:
    """Save evaluation results to a CSV file.

    Args:
        results: Evaluation results.
        output_path: Path to save the CSV file.
        format: CSV format - 'summary' or 'detailed'.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "summary":
        _save_summary_csv(results, output_path)
    elif format == "detailed":
        _save_detailed_csv(results, output_path)
    else:
        raise ValueError(f"Invalid CSV format: {format}. Must be 'summary' or 'detailed'.")


def _save_summary_csv(results: "EvaluationResults", output_path: Path) -> None:
    """Save summary CSV with key metrics per task.

    Columns:
        instance_id, mcp_resolved, baseline_resolved,
        mcp_tokens_in, mcp_tokens_out, baseline_tokens_in, baseline_tokens_out,
        mcp_iterations, baseline_iterations, mcp_tool_calls, baseline_tool_calls,
        mcp_patch_generated, baseline_patch_generated, mcp_error, baseline_error
    """
    fieldnames = [
        "instance_id",
        "mcp_resolved",
        "baseline_resolved",
        "mcp_tokens_in",
        "mcp_tokens_out",
        "baseline_tokens_in",
        "baseline_tokens_out",
        "mcp_iterations",
        "baseline_iterations",
        "mcp_tool_calls",
        "baseline_tool_calls",
        "mcp_patch_generated",
        "baseline_patch_generated",
        "mcp_error",
        "baseline_error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in results.tasks:
            row = {
                "instance_id": task.instance_id,
            }

            # MCP columns
            if task.mcp:
                row["mcp_resolved"] = task.mcp.get("resolved", False)
                row["mcp_tokens_in"] = task.mcp.get("tokens", {}).get("input", 0)
                row["mcp_tokens_out"] = task.mcp.get("tokens", {}).get("output", 0)
                row["mcp_iterations"] = task.mcp.get("iterations", 0)
                row["mcp_tool_calls"] = task.mcp.get("tool_calls", 0)
                row["mcp_patch_generated"] = task.mcp.get("patch_generated", False)
                row["mcp_error"] = task.mcp.get("error", "")
            else:
                row["mcp_resolved"] = ""
                row["mcp_tokens_in"] = ""
                row["mcp_tokens_out"] = ""
                row["mcp_iterations"] = ""
                row["mcp_tool_calls"] = ""
                row["mcp_patch_generated"] = ""
                row["mcp_error"] = ""

            # Baseline columns
            if task.baseline:
                row["baseline_resolved"] = task.baseline.get("resolved", False)
                row["baseline_tokens_in"] = task.baseline.get("tokens", {}).get("input", 0)
                row["baseline_tokens_out"] = task.baseline.get("tokens", {}).get("output", 0)
                row["baseline_iterations"] = task.baseline.get("iterations", 0)
                row["baseline_tool_calls"] = task.baseline.get("tool_calls", 0)
                row["baseline_patch_generated"] = task.baseline.get("patch_generated", False)
                row["baseline_error"] = task.baseline.get("error", "")
            else:
                row["baseline_resolved"] = ""
                row["baseline_tokens_in"] = ""
                row["baseline_tokens_out"] = ""
                row["baseline_iterations"] = ""
                row["baseline_tool_calls"] = ""
                row["baseline_patch_generated"] = ""
                row["baseline_error"] = ""

            writer.writerow(row)


def _save_detailed_csv(results: "EvaluationResults", output_path: Path) -> None:
    """Save detailed CSV with all available metrics per task.

    Includes all summary columns plus:
        mcp_patch_applied, baseline_patch_applied,
        mcp_fail_to_pass_passed, mcp_fail_to_pass_total,
        baseline_fail_to_pass_passed, baseline_fail_to_pass_total,
        mcp_pass_to_pass_passed, mcp_pass_to_pass_total,
        baseline_pass_to_pass_passed, baseline_pass_to_pass_total,
        mcp_eval_error, baseline_eval_error,
        mcp_tool_usage, baseline_tool_usage
    """
    fieldnames = [
        "instance_id",
        "mcp_resolved",
        "baseline_resolved",
        "mcp_tokens_in",
        "mcp_tokens_out",
        "baseline_tokens_in",
        "baseline_tokens_out",
        "mcp_iterations",
        "baseline_iterations",
        "mcp_tool_calls",
        "baseline_tool_calls",
        "mcp_patch_generated",
        "baseline_patch_generated",
        "mcp_patch_applied",
        "baseline_patch_applied",
        "mcp_fail_to_pass_passed",
        "mcp_fail_to_pass_total",
        "baseline_fail_to_pass_passed",
        "baseline_fail_to_pass_total",
        "mcp_pass_to_pass_passed",
        "mcp_pass_to_pass_total",
        "baseline_pass_to_pass_passed",
        "baseline_pass_to_pass_total",
        "mcp_eval_error",
        "baseline_eval_error",
        "mcp_error",
        "baseline_error",
        "mcp_tool_usage",
        "baseline_tool_usage",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in results.tasks:
            row = {
                "instance_id": task.instance_id,
            }

            # MCP columns
            if task.mcp:
                row["mcp_resolved"] = task.mcp.get("resolved", False)
                row["mcp_tokens_in"] = task.mcp.get("tokens", {}).get("input", 0)
                row["mcp_tokens_out"] = task.mcp.get("tokens", {}).get("output", 0)
                row["mcp_iterations"] = task.mcp.get("iterations", 0)
                row["mcp_tool_calls"] = task.mcp.get("tool_calls", 0)
                row["mcp_patch_generated"] = task.mcp.get("patch_generated", False)
                row["mcp_patch_applied"] = task.mcp.get("patch_applied", False)
                row["mcp_error"] = task.mcp.get("error", "")
                row["mcp_eval_error"] = task.mcp.get("eval_error", "")

                # Test results
                fail_to_pass = task.mcp.get("fail_to_pass", {})
                row["mcp_fail_to_pass_passed"] = fail_to_pass.get("passed", "")
                row["mcp_fail_to_pass_total"] = fail_to_pass.get("total", "")

                pass_to_pass = task.mcp.get("pass_to_pass", {})
                row["mcp_pass_to_pass_passed"] = pass_to_pass.get("passed", "")
                row["mcp_pass_to_pass_total"] = pass_to_pass.get("total", "")

                # Tool usage (convert dict to JSON string for CSV)
                tool_usage = task.mcp.get("tool_usage", {})
                row["mcp_tool_usage"] = json.dumps(tool_usage) if tool_usage else ""
            else:
                for key in fieldnames:
                    if key.startswith("mcp_") and key not in row:
                        row[key] = ""

            # Baseline columns
            if task.baseline:
                row["baseline_resolved"] = task.baseline.get("resolved", False)
                row["baseline_tokens_in"] = task.baseline.get("tokens", {}).get("input", 0)
                row["baseline_tokens_out"] = task.baseline.get("tokens", {}).get("output", 0)
                row["baseline_iterations"] = task.baseline.get("iterations", 0)
                row["baseline_tool_calls"] = task.baseline.get("tool_calls", 0)
                row["baseline_patch_generated"] = task.baseline.get("patch_generated", False)
                row["baseline_patch_applied"] = task.baseline.get("patch_applied", False)
                row["baseline_error"] = task.baseline.get("error", "")
                row["baseline_eval_error"] = task.baseline.get("eval_error", "")

                # Test results
                fail_to_pass = task.baseline.get("fail_to_pass", {})
                row["baseline_fail_to_pass_passed"] = fail_to_pass.get("passed", "")
                row["baseline_fail_to_pass_total"] = fail_to_pass.get("total", "")

                pass_to_pass = task.baseline.get("pass_to_pass", {})
                row["baseline_pass_to_pass_passed"] = pass_to_pass.get("passed", "")
                row["baseline_pass_to_pass_total"] = pass_to_pass.get("total", "")

                # Tool usage (convert dict to JSON string for CSV)
                tool_usage = task.baseline.get("tool_usage", {})
                row["baseline_tool_usage"] = json.dumps(tool_usage) if tool_usage else ""
            else:
                for key in fieldnames:
                    if key.startswith("baseline_") and key not in row:
                        row[key] = ""

            writer.writerow(row)
