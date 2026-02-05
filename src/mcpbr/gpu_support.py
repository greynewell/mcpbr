"""GPU support for Docker containers used in ML benchmark evaluations.

Provides detection of available GPUs (NVIDIA), Docker GPU runtime checks,
and Docker container configuration for GPU access.
"""

import logging
import subprocess

import docker
import docker.types

logger = logging.getLogger(__name__)


def detect_gpus() -> dict:
    """Detect available GPUs on the host system.

    Checks for NVIDIA GPUs via nvidia-smi and verifies the Docker GPU runtime
    is available.

    Returns:
        Dictionary with GPU detection results:
            - nvidia_available (bool): Whether NVIDIA GPUs were detected.
            - gpu_count (int): Number of GPUs found.
            - gpu_names (list[str]): Names of detected GPUs.
            - driver_version (str): NVIDIA driver version, or empty string.
            - docker_runtime_available (bool): Whether Docker NVIDIA runtime is available.
    """
    info: dict = {
        "nvidia_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "driver_version": "",
        "docker_runtime_available": False,
    }

    # Detect NVIDIA GPUs via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            gpu_names = []
            driver_version = ""
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpu_names.append(parts[0])
                    driver_version = parts[1]
                elif len(parts) == 1:
                    gpu_names.append(parts[0])

            info["nvidia_available"] = True
            info["gpu_count"] = len(gpu_names)
            info["gpu_names"] = gpu_names
            info["driver_version"] = driver_version
    except FileNotFoundError:
        logger.debug("nvidia-smi not found; no NVIDIA GPUs detected.")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out while detecting GPUs.")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    # Check Docker GPU runtime availability
    info["docker_runtime_available"] = check_gpu_runtime()

    return info


def get_docker_gpu_config(gpu_enabled: bool) -> dict:
    """Return Docker container creation kwargs for GPU access.

    When gpu_enabled is True, returns a dictionary containing a DeviceRequest
    that grants access to all available NVIDIA GPUs. This dict can be merged
    into the kwargs passed to ``docker.containers.run()`` or
    ``docker.containers.create()``.

    Args:
        gpu_enabled: Whether to enable GPU access in the container.

    Returns:
        Dictionary of Docker container kwargs. Empty dict if gpu_enabled is False.
        When True, contains ``device_requests`` with a DeviceRequest for all GPUs.
    """
    if not gpu_enabled:
        return {}

    return {
        "device_requests": [
            docker.types.DeviceRequest(
                count=-1,
                capabilities=[["gpu"]],
            )
        ],
    }


def check_gpu_runtime() -> bool:
    """Check if Docker has the NVIDIA runtime available.

    Queries the Docker daemon info for registered runtimes and checks
    whether the ``nvidia`` runtime is among them.

    Returns:
        True if the NVIDIA Docker runtime is available, False otherwise.
    """
    try:
        client = docker.from_env()
        docker_info = client.info()
        runtimes = docker_info.get("Runtimes", {})
        return "nvidia" in runtimes
    except docker.errors.DockerException as e:
        logger.debug(f"Could not query Docker for GPU runtime: {e}")
        return False
    except Exception as e:
        logger.debug(f"Unexpected error checking Docker GPU runtime: {e}")
        return False


def format_gpu_info(info: dict) -> str:
    """Format GPU detection info as a human-readable string.

    Args:
        info: Dictionary returned by ``detect_gpus()``.

    Returns:
        Human-readable multi-line string describing the GPU environment.
    """
    lines: list[str] = []

    if not info.get("nvidia_available"):
        lines.append("No NVIDIA GPUs detected.")
    else:
        gpu_count = info.get("gpu_count", 0)
        lines.append(f"NVIDIA GPUs detected: {gpu_count}")

        gpu_names = info.get("gpu_names", [])
        for i, name in enumerate(gpu_names):
            lines.append(f"  GPU {i}: {name}")

        driver_version = info.get("driver_version", "")
        if driver_version:
            lines.append(f"Driver version: {driver_version}")

    runtime_available = info.get("docker_runtime_available", False)
    lines.append(f"Docker NVIDIA runtime: {'available' if runtime_available else 'not available'}")

    return "\n".join(lines)
