"""Tests for CLI exit codes."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from mcpbr.cli import main
from mcpbr.harness import EvaluationResults, TaskResult


class TestExitCodes:
    """Tests for exit codes returned by the run command."""

    def create_mock_results(
        self,
        mcp_resolved: int = 0,
        mcp_total: int = 10,
        baseline_resolved: int = 0,
        baseline_total: int = 10,
        evaluated_tasks: int = 10,
        cached_tasks: int = 0,
        enabled_incremental: bool = True,
    ) -> EvaluationResults:
        """Create mock evaluation results.

        Args:
            mcp_resolved: Number of tasks resolved by MCP agent.
            mcp_total: Total number of tasks for MCP agent.
            baseline_resolved: Number of tasks resolved by baseline.
            baseline_total: Total number of tasks for baseline.
            evaluated_tasks: Number of tasks actually evaluated (not cached).
            cached_tasks: Number of tasks that were cached/skipped.
            enabled_incremental: Whether incremental evaluation is enabled.

        Returns:
            Mock EvaluationResults object.
        """
        return EvaluationResults(
            metadata={
                "incremental": {
                    "enabled": enabled_incremental,
                    "total_tasks": evaluated_tasks + cached_tasks,
                    "evaluated_tasks": evaluated_tasks,
                    "cached_tasks": cached_tasks,
                }
            },
            summary={
                "mcp": {
                    "resolved": mcp_resolved,
                    "total": mcp_total,
                    "rate": mcp_resolved / mcp_total if mcp_total > 0 else 0,
                },
                "baseline": {
                    "resolved": baseline_resolved,
                    "total": baseline_total,
                    "rate": baseline_resolved / baseline_total if baseline_total > 0 else 0,
                },
                "improvement": f"+{mcp_resolved - baseline_resolved}",
            },
            tasks=[
                TaskResult(instance_id=f"task-{i}", mcp={}, baseline={})
                for i in range(max(mcp_total, baseline_total))
            ],
        )

    def test_exit_code_0_success_with_mcp_resolutions(self) -> None:
        """Test exit code 0 when MCP agent resolves tasks."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            mock_results = self.create_mock_results(
                mcp_resolved=5,
                mcp_total=10,
                baseline_resolved=3,
                baseline_total=10,
                evaluated_tasks=10,
            )

            # Create an async mock that returns our results
            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(main, ["run", "-c", str(config_path), "--skip-health-check"])

                assert result.exit_code == 0

    def test_exit_code_0_success_with_baseline_resolutions(self) -> None:
        """Test exit code 0 when baseline resolves tasks."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            mock_results = self.create_mock_results(
                mcp_resolved=0,
                mcp_total=10,
                baseline_resolved=3,
                baseline_total=10,
                evaluated_tasks=10,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(main, ["run", "-c", str(config_path), "--skip-health-check"])

                assert result.exit_code == 0

    def test_exit_code_2_no_resolutions_both_agents(self) -> None:
        """Test exit code 2 when no tasks are resolved by either agent."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            mock_results = self.create_mock_results(
                mcp_resolved=0,
                mcp_total=10,
                baseline_resolved=0,
                baseline_total=10,
                evaluated_tasks=10,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(main, ["run", "-c", str(config_path), "--skip-health-check"])

                assert result.exit_code == 2
                assert "No tasks resolved by either agent" in result.output

    def test_exit_code_2_no_resolutions_mcp_only(self) -> None:
        """Test exit code 2 when MCP-only run resolves nothing."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            mock_results = self.create_mock_results(
                mcp_resolved=0,
                mcp_total=10,
                baseline_resolved=0,
                baseline_total=0,
                evaluated_tasks=10,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(
                    main, ["run", "-c", str(config_path), "-M", "--skip-health-check"]
                )

                assert result.exit_code == 2
                assert "No tasks resolved (0% success)" in result.output

    def test_exit_code_2_no_resolutions_baseline_only(self) -> None:
        """Test exit code 2 when baseline-only run resolves nothing."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            mock_results = self.create_mock_results(
                mcp_resolved=0,
                mcp_total=0,
                baseline_resolved=0,
                baseline_total=10,
                evaluated_tasks=10,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(
                    main, ["run", "-c", str(config_path), "-B", "--skip-health-check"]
                )

                assert result.exit_code == 2
                assert "No tasks resolved (0% success)" in result.output

    def test_exit_code_3_all_cached(self) -> None:
        """Test exit code 3 when all tasks are cached (nothing evaluated)."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            mock_results = self.create_mock_results(
                mcp_resolved=5,
                mcp_total=10,
                baseline_resolved=3,
                baseline_total=10,
                evaluated_tasks=0,  # Nothing evaluated
                cached_tasks=10,  # All cached
                enabled_incremental=True,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(main, ["run", "-c", str(config_path), "--skip-health-check"])

                assert result.exit_code == 3
                assert "No tasks evaluated (all cached)" in result.output
                assert "Use --reset-state or --no-incremental to re-run" in result.output

    def test_exit_code_3_takes_precedence_over_exit_code_2(self) -> None:
        """Test that exit code 3 takes precedence over exit code 2."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            # All cached AND no resolutions
            mock_results = self.create_mock_results(
                mcp_resolved=0,
                mcp_total=10,
                baseline_resolved=0,
                baseline_total=10,
                evaluated_tasks=0,  # Nothing evaluated
                cached_tasks=10,  # All cached
                enabled_incremental=True,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(main, ["run", "-c", str(config_path), "--skip-health-check"])

                # Should return 3 (nothing evaluated) not 2 (no resolutions)
                assert result.exit_code == 3
                assert "No tasks evaluated (all cached)" in result.output

    def test_exit_code_1_invalid_config(self) -> None:
        """Test exit code 1 for invalid configuration."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("invalid: yaml: content: [")

            result = runner.invoke(main, ["run", "-c", str(config_path)])

            assert result.exit_code == 1
            assert "Error loading config" in result.output

    def test_exit_code_1_conflicting_flags(self) -> None:
        """Test exit code 1 when both --mcp-only and --baseline-only are specified."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            result = runner.invoke(main, ["run", "-c", str(config_path), "-M", "-B"])

            assert result.exit_code == 1
            assert "Cannot specify both --mcp-only and --baseline-only" in result.output

    def test_exit_code_130_keyboard_interrupt(self) -> None:
        """Test exit code 130 when evaluation is interrupted by user."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            async def mock_run_evaluation(*args, **kwargs):
                raise KeyboardInterrupt()

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(main, ["run", "-c", str(config_path), "--skip-health-check"])

                assert result.exit_code == 130
                assert "interrupted by user" in result.output

    def test_exit_code_0_incremental_disabled(self) -> None:
        """Test that exit code 3 is not triggered when incremental is disabled."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
provider: anthropic
model: claude-sonnet-4-5-20250514
mcp_server:
  command: npx
  args:
    - "@modelcontextprotocol/server-filesystem"
    - /tmp
"""
            )

            # Incremental disabled, so evaluated_tasks = 0 should not trigger exit code 3
            mock_results = self.create_mock_results(
                mcp_resolved=5,
                mcp_total=10,
                baseline_resolved=3,
                baseline_total=10,
                evaluated_tasks=10,
                cached_tasks=0,
                enabled_incremental=False,
            )

            async def mock_run_evaluation(*args, **kwargs):
                return mock_results

            with patch("mcpbr.cli.run_evaluation", side_effect=mock_run_evaluation):
                result = runner.invoke(
                    main, ["run", "-c", str(config_path), "--no-incremental", "--skip-health-check"]
                )

                # Should be 0 (success) since incremental is disabled
                assert result.exit_code == 0
