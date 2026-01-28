"""Tests for Docker validation in CLI run command."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from mcpbr.cli import run


class TestDockerValidation:
    """Test Docker validation before evaluation starts."""

    @pytest.fixture
    def temp_config(self, tmp_path: Path) -> Path:
        """Create a temporary config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
mcp_server:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "{workdir}"]
  env: {}

provider: anthropic
agent_harness: claude-code
model: claude-sonnet-4-5-20250514
benchmark: swe-bench-verified
sample_size: 1
timeout_seconds: 300
max_concurrent: 1
max_iterations: 10
"""
        )
        return config_path

    @patch("mcpbr.cli.asyncio.run")
    @patch("docker.from_env")
    def test_docker_validation_success(self, mock_docker, mock_asyncio_run, temp_config: Path):
        """Test successful Docker validation before evaluation."""
        # Mock Docker client
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.info.return_value = {"ServerVersion": "24.0.0"}
        mock_docker.return_value = mock_client

        # Mock run_evaluation to prevent actual evaluation
        mock_asyncio_run.return_value = Mock(tasks=[])

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--config",
                str(temp_config),
                "--skip-health-check",
                "--no-incremental",
                "--sample",
                "1",
            ],
        )

        # Should call Docker validation
        mock_docker.assert_called_once()
        mock_client.ping.assert_called_once()
        mock_client.info.assert_called_once()

        # Should succeed (at least Docker validation passed)
        assert "Docker 24.0.0 running" in result.output
        assert "Docker validation passed" in result.output

    @patch("docker.from_env")
    def test_docker_validation_failure_docker_exception(self, mock_docker, temp_config: Path):
        """Test Docker validation failure when Docker daemon is not running."""
        from docker.errors import DockerException

        # Mock Docker to raise exception (daemon not running)
        mock_docker.side_effect = DockerException("Cannot connect to Docker daemon")

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--config",
                str(temp_config),
                "--skip-health-check",
                "--no-incremental",
            ],
        )

        # Should fail with exit code 1
        assert result.exit_code == 1

        # Should show error message
        assert "Docker validation failed" in result.output
        assert "Docker not available" in result.output
        assert "Make sure Docker Desktop is running" in result.output

    @patch("docker.from_env")
    def test_docker_validation_failure_general_exception(self, mock_docker, temp_config: Path):
        """Test Docker validation failure with general exception."""
        # Mock Docker to raise general exception
        mock_docker.side_effect = Exception("Unexpected error")

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--config",
                str(temp_config),
                "--skip-health-check",
                "--no-incremental",
            ],
        )

        # Should fail with exit code 1
        assert result.exit_code == 1

        # Should show error message
        assert "Docker validation failed" in result.output
        assert "Make sure Docker Desktop is running" in result.output

    @patch("mcpbr.cli.asyncio.run")
    @patch("docker.from_env")
    def test_docker_validation_runs_before_mcp_check(
        self, mock_docker, mock_asyncio_run, temp_config: Path
    ):
        """Test that Docker validation runs before MCP health check."""
        # Mock Docker client
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.info.return_value = {"ServerVersion": "24.0.0"}
        mock_docker.return_value = mock_client

        # Mock run_evaluation to prevent actual evaluation
        mock_asyncio_run.return_value = Mock(tasks=[])

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--config",
                str(temp_config),
                "--no-incremental",
                "--sample",
                "1",
            ],
        )

        # Both Docker and MCP checks should appear
        output_lines = result.output.split("\n")
        docker_line_idx = None
        mcp_line_idx = None

        for idx, line in enumerate(output_lines):
            if "Docker validation passed" in line:
                docker_line_idx = idx
            if "MCP Pre-flight Check" in line:
                mcp_line_idx = idx

        # Docker check should run before MCP check
        if docker_line_idx is not None and mcp_line_idx is not None:
            assert docker_line_idx < mcp_line_idx

    @patch("docker.from_env")
    def test_docker_validation_fails_fast(self, mock_docker, temp_config: Path):
        """Test that Docker validation fails fast without running evaluation."""
        from docker.errors import DockerException

        # Mock Docker to fail
        mock_docker.side_effect = DockerException("Docker not running")

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--config",
                str(temp_config),
                "--skip-health-check",
            ],
        )

        # Should exit immediately with code 1
        assert result.exit_code == 1

        # Should not reach evaluation stage
        assert "mcpbr Evaluation" not in result.output

        # Should show clear error message
        assert "Docker validation failed" in result.output
