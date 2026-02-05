"""Tests for GPU support module."""

import subprocess
from unittest.mock import MagicMock, patch

from mcpbr.gpu_support import (
    check_gpu_runtime,
    detect_gpus,
    format_gpu_info,
    get_docker_gpu_config,
)


class TestDetectGpus:
    """Tests for detect_gpus() function."""

    @patch("mcpbr.gpu_support.check_gpu_runtime")
    @patch("mcpbr.gpu_support.subprocess.run")
    def test_detect_gpus_with_nvidia(self, mock_run, mock_runtime):
        """Test GPU detection with mock nvidia-smi output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA A100-SXM4-80GB, 535.129.03\nNVIDIA A100-SXM4-80GB, 535.129.03\n",
        )
        mock_runtime.return_value = True

        info = detect_gpus()

        assert info["nvidia_available"] is True
        assert info["gpu_count"] == 2
        assert info["gpu_names"] == ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"]
        assert info["driver_version"] == "535.129.03"
        assert info["docker_runtime_available"] is True

        mock_run.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

    @patch("mcpbr.gpu_support.check_gpu_runtime")
    @patch("mcpbr.gpu_support.subprocess.run")
    def test_detect_gpus_single_gpu(self, mock_run, mock_runtime):
        """Test GPU detection with a single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 545.23.08\n",
        )
        mock_runtime.return_value = True

        info = detect_gpus()

        assert info["nvidia_available"] is True
        assert info["gpu_count"] == 1
        assert info["gpu_names"] == ["NVIDIA GeForce RTX 4090"]
        assert info["driver_version"] == "545.23.08"

    @patch("mcpbr.gpu_support.check_gpu_runtime")
    @patch("mcpbr.gpu_support.subprocess.run")
    def test_detect_gpus_no_nvidia_smi(self, mock_run, mock_runtime):
        """Test GPU detection when nvidia-smi is not found."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        mock_runtime.return_value = False

        info = detect_gpus()

        assert info["nvidia_available"] is False
        assert info["gpu_count"] == 0
        assert info["gpu_names"] == []
        assert info["driver_version"] == ""
        assert info["docker_runtime_available"] is False

    @patch("mcpbr.gpu_support.check_gpu_runtime")
    @patch("mcpbr.gpu_support.subprocess.run")
    def test_detect_gpus_nvidia_smi_fails(self, mock_run, mock_runtime):
        """Test GPU detection when nvidia-smi returns non-zero exit code."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
        )
        mock_runtime.return_value = False

        info = detect_gpus()

        assert info["nvidia_available"] is False
        assert info["gpu_count"] == 0
        assert info["gpu_names"] == []

    @patch("mcpbr.gpu_support.check_gpu_runtime")
    @patch("mcpbr.gpu_support.subprocess.run")
    def test_detect_gpus_nvidia_smi_timeout(self, mock_run, mock_runtime):
        """Test GPU detection when nvidia-smi times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)
        mock_runtime.return_value = False

        info = detect_gpus()

        assert info["nvidia_available"] is False
        assert info["gpu_count"] == 0

    @patch("mcpbr.gpu_support.check_gpu_runtime")
    @patch("mcpbr.gpu_support.subprocess.run")
    def test_detect_gpus_empty_output(self, mock_run, mock_runtime):
        """Test GPU detection when nvidia-smi returns empty output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
        )
        mock_runtime.return_value = False

        info = detect_gpus()

        assert info["nvidia_available"] is False
        assert info["gpu_count"] == 0


class TestGetDockerGpuConfig:
    """Tests for get_docker_gpu_config() function."""

    def test_gpu_enabled(self):
        """Test Docker GPU config when GPU is enabled."""
        config = get_docker_gpu_config(gpu_enabled=True)

        assert "device_requests" in config
        assert len(config["device_requests"]) == 1

        device_request = config["device_requests"][0]
        assert device_request.count == -1
        assert device_request.capabilities == [["gpu"]]

    def test_gpu_disabled(self):
        """Test Docker GPU config when GPU is disabled."""
        config = get_docker_gpu_config(gpu_enabled=False)

        assert config == {}

    def test_gpu_config_is_valid_docker_type(self):
        """Test that the device request is a proper Docker DeviceRequest type."""
        import docker.types

        config = get_docker_gpu_config(gpu_enabled=True)
        device_request = config["device_requests"][0]
        assert isinstance(device_request, docker.types.DeviceRequest)


class TestCheckGpuRuntime:
    """Tests for check_gpu_runtime() function."""

    @patch("mcpbr.gpu_support.docker.from_env")
    def test_nvidia_runtime_available(self, mock_from_env):
        """Test when NVIDIA runtime is available in Docker."""
        mock_client = MagicMock()
        mock_client.info.return_value = {
            "Runtimes": {
                "nvidia": {"path": "/usr/bin/nvidia-container-runtime"},
                "runc": {"path": "runc"},
            }
        }
        mock_from_env.return_value = mock_client

        assert check_gpu_runtime() is True

    @patch("mcpbr.gpu_support.docker.from_env")
    def test_nvidia_runtime_not_available(self, mock_from_env):
        """Test when NVIDIA runtime is not available in Docker."""
        mock_client = MagicMock()
        mock_client.info.return_value = {
            "Runtimes": {
                "runc": {"path": "runc"},
            }
        }
        mock_from_env.return_value = mock_client

        assert check_gpu_runtime() is False

    @patch("mcpbr.gpu_support.docker.from_env")
    def test_no_runtimes_key(self, mock_from_env):
        """Test when Docker info has no Runtimes key."""
        mock_client = MagicMock()
        mock_client.info.return_value = {}
        mock_from_env.return_value = mock_client

        assert check_gpu_runtime() is False

    @patch("mcpbr.gpu_support.docker.from_env")
    def test_docker_not_available(self, mock_from_env):
        """Test when Docker daemon is not reachable."""
        import docker.errors

        mock_from_env.side_effect = docker.errors.DockerException("Cannot connect")

        assert check_gpu_runtime() is False

    @patch("mcpbr.gpu_support.docker.from_env")
    def test_unexpected_error(self, mock_from_env):
        """Test when an unexpected error occurs."""
        mock_from_env.side_effect = RuntimeError("Unexpected")

        assert check_gpu_runtime() is False


class TestFormatGpuInfo:
    """Tests for format_gpu_info() function."""

    def test_format_with_gpus(self):
        """Test formatting with GPU info present."""
        info = {
            "nvidia_available": True,
            "gpu_count": 2,
            "gpu_names": ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"],
            "driver_version": "535.129.03",
            "docker_runtime_available": True,
        }

        result = format_gpu_info(info)

        assert "NVIDIA GPUs detected: 2" in result
        assert "GPU 0: NVIDIA A100-SXM4-80GB" in result
        assert "GPU 1: NVIDIA A100-SXM4-80GB" in result
        assert "Driver version: 535.129.03" in result
        assert "Docker NVIDIA runtime: available" in result

    def test_format_no_gpus(self):
        """Test formatting when no GPUs are detected."""
        info = {
            "nvidia_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "driver_version": "",
            "docker_runtime_available": False,
        }

        result = format_gpu_info(info)

        assert "No NVIDIA GPUs detected." in result
        assert "Docker NVIDIA runtime: not available" in result

    def test_format_gpu_without_runtime(self):
        """Test formatting with GPUs but no Docker runtime."""
        info = {
            "nvidia_available": True,
            "gpu_count": 1,
            "gpu_names": ["NVIDIA GeForce RTX 4090"],
            "driver_version": "545.23.08",
            "docker_runtime_available": False,
        }

        result = format_gpu_info(info)

        assert "NVIDIA GPUs detected: 1" in result
        assert "GPU 0: NVIDIA GeForce RTX 4090" in result
        assert "Docker NVIDIA runtime: not available" in result

    def test_format_empty_dict(self):
        """Test formatting with empty dict (defensive)."""
        info: dict = {}

        result = format_gpu_info(info)

        assert "No NVIDIA GPUs detected." in result
        assert "Docker NVIDIA runtime: not available" in result

    def test_format_no_driver_version(self):
        """Test formatting when driver version is missing."""
        info = {
            "nvidia_available": True,
            "gpu_count": 1,
            "gpu_names": ["NVIDIA T4"],
            "driver_version": "",
            "docker_runtime_available": True,
        }

        result = format_gpu_info(info)

        assert "Driver version:" not in result
        assert "NVIDIA GPUs detected: 1" in result
