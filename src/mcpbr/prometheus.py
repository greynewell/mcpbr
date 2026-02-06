"""Prometheus metrics formatting and file export.

Produces metrics in the Prometheus exposition format for scraping or file-based
consumption.
"""

from pathlib import Path
from typing import Any


def format_metric(
    name: str,
    value: int | float,
    labels: dict[str, str] | None = None,
    help_text: str | None = None,
    metric_type: str | None = None,
) -> str:
    """Format a single metric in Prometheus exposition format.

    Args:
        name: Metric name (e.g., 'mcpbr_resolution_rate').
        value: Metric value.
        labels: Optional label key-value pairs.
        help_text: Optional HELP line text.
        metric_type: Optional TYPE line (gauge, counter, histogram, summary).

    Returns:
        Formatted Prometheus metric string.
    """
    lines = []

    if help_text:
        lines.append(f"# HELP {name} {help_text}")

    if metric_type:
        lines.append(f"# TYPE {name} {metric_type}")

    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        lines.append(f"{name}{{{label_str}}} {value}")
    else:
        lines.append(f"{name} {value}")

    return "\n".join(lines)


def export_metrics(
    results: dict[str, Any],
    output_path: Path | None = None,
) -> str:
    """Export evaluation results as Prometheus metrics.

    Args:
        results: Evaluation results dictionary with metadata and summary.
        output_path: Optional file path to write metrics to.

    Returns:
        Complete Prometheus exposition format string.
    """
    config = results.get("metadata", {}).get("config", {})
    labels = {
        "benchmark": config.get("benchmark", "unknown"),
        "model": config.get("model", "unknown"),
    }

    mcp = results.get("summary", {}).get("mcp", {})
    baseline = results.get("summary", {}).get("baseline", {})

    lines = []

    # MCP metrics
    lines.append(
        format_metric(
            "mcpbr_resolution_rate",
            mcp.get("rate", 0),
            labels={**labels, "agent": "mcp"},
            help_text="Task resolution rate",
            metric_type="gauge",
        )
    )

    lines.append(
        format_metric(
            "mcpbr_tasks_total",
            mcp.get("total", 0),
            labels={**labels, "agent": "mcp"},
            help_text="Total tasks evaluated",
            metric_type="gauge",
        )
    )

    lines.append(
        format_metric(
            "mcpbr_tasks_resolved",
            mcp.get("resolved", 0),
            labels={**labels, "agent": "mcp"},
            help_text="Tasks resolved successfully",
            metric_type="gauge",
        )
    )

    lines.append(
        format_metric(
            "mcpbr_total_cost",
            mcp.get("total_cost", 0),
            labels={**labels, "agent": "mcp"},
            help_text="Total cost in USD",
            metric_type="gauge",
        )
    )

    # Baseline metrics (if present)
    if baseline:
        lines.append(
            format_metric(
                "mcpbr_resolution_rate",
                baseline.get("rate", 0),
                labels={**labels, "agent": "baseline"},
            )
        )

        lines.append(
            format_metric(
                "mcpbr_tasks_total",
                baseline.get("total", 0),
                labels={**labels, "agent": "baseline"},
            )
        )

        lines.append(
            format_metric(
                "mcpbr_tasks_resolved",
                baseline.get("resolved", 0),
                labels={**labels, "agent": "baseline"},
            )
        )

        lines.append(
            format_metric(
                "mcpbr_total_cost",
                baseline.get("total_cost", 0),
                labels={**labels, "agent": "baseline"},
            )
        )

    output = "\n\n".join(lines) + "\n"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)

    return output
