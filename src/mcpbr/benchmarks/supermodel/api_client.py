"""Async Supermodel API client with polling and idempotency support."""

import asyncio
import hashlib
import json
import logging
import time

logger = logging.getLogger("mcpbr.supermodel")


async def call_supermodel_api(
    endpoint_path: str,
    zip_path: str,
    api_base: str,
    api_key: str | None = None,
    idempotency_key: str | None = None,
    max_poll_time: int = 300,
) -> dict:
    """Call a Supermodel API endpoint with a zipped repo.

    Uses curl subprocess for the HTTP request and polls for async results.

    Args:
        endpoint_path: API endpoint path (e.g. '/v1/analysis/dead-code').
        zip_path: Path to the zipped repository archive.
        api_base: Base URL for the Supermodel API.
        api_key: Optional API key.
        idempotency_key: Optional idempotency key (auto-generated from zip hash if not provided).
        max_poll_time: Maximum time to poll for results in seconds.

    Returns:
        Parsed API response dict.

    Raises:
        RuntimeError: If the API request fails or times out.
    """
    url = f"{api_base}{endpoint_path}"
    logger.info(f"Calling Supermodel API: {url}")

    if not idempotency_key:
        with open(zip_path, "rb") as f:
            zip_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        ep_name = endpoint_path.strip("/").replace("/", "-")
        idempotency_key = f"bench:{ep_name}:{zip_hash}"

    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        url,
        "-F",
        f"file=@{zip_path}",
        "-H",
        "Accept: application/json",
        "-H",
        f"Idempotency-Key: {idempotency_key}",
    ]
    if api_key:
        cmd.extend(["-H", f"X-Api-Key: {api_key}"])

    start_time = time.time()

    # Initial request
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

    if proc.returncode != 0:
        raise RuntimeError(f"Supermodel API request failed: {stderr.decode()}")

    response = json.loads(stdout.decode())

    # Poll if async
    while response.get("status") in ("pending", "processing"):
        elapsed = time.time() - start_time
        if elapsed > max_poll_time:
            raise RuntimeError(f"Supermodel API timed out after {max_poll_time}s")

        retry_after = response.get("retryAfter", 10)
        logger.info(f"Job pending, polling in {retry_after}s... ({elapsed:.0f}s elapsed)")
        await asyncio.sleep(retry_after)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            raise RuntimeError(f"Supermodel API poll failed: {stderr.decode()}")
        response = json.loads(stdout.decode())

    elapsed = time.time() - start_time

    if response.get("status") == "error" or response.get("error"):
        raise RuntimeError(f"Supermodel API error: {response.get('error')}")

    api_result = response.get("result", response)
    logger.info(f"Supermodel API completed in {elapsed:.1f}s")
    return api_result
