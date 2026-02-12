"""Async Supermodel API client with polling and idempotency support."""

import asyncio
import hashlib
import json
import logging
import sys
import time

logger = logging.getLogger("mcpbr.supermodel")


async def call_supermodel_api(
    endpoint_path: str,
    zip_path: str,
    api_base: str,
    api_key: str | None = None,
    idempotency_key: str | None = None,
    max_poll_time: int = 600,
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

    if not idempotency_key:
        with open(zip_path, "rb") as f:
            zip_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        ep_name = endpoint_path.strip("/").replace("/", "-")
        idempotency_key = f"bench:{ep_name}:{zip_hash}"

    headers = [
        "-H",
        "Accept: application/json",
        "-H",
        f"Idempotency-Key: {idempotency_key}",
    ]
    if api_key:
        headers.extend(["-H", f"X-Api-Key: {api_key}"])

    # Initial request with file upload
    upload_cmd = ["curl", "-s", "-X", "POST", url, "-F", f"file=@{zip_path}"] + headers

    start_time = time.time()
    print(
        f"  Supermodel API: uploading {zip_path} to {endpoint_path}...", file=sys.stderr, flush=True
    )

    proc = await asyncio.create_subprocess_exec(
        *upload_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

    if proc.returncode != 0:
        raise RuntimeError(f"Supermodel API request failed: {stderr.decode()}")

    response = json.loads(stdout.decode())

    # Poll if async — use lightweight requests (1-byte dummy file instead of
    # re-uploading the full zip). The API recognizes the idempotency key and
    # returns the cached job status without reprocessing.
    import tempfile

    poll_dummy = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    poll_dummy.write(b"\n")
    poll_dummy.close()
    poll_cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        url,
        "-F",
        f"file=@{poll_dummy.name}",
    ] + headers
    poll_count = 0

    while response.get("status") in ("pending", "processing"):
        elapsed = time.time() - start_time
        if elapsed > max_poll_time:
            raise RuntimeError(f"Supermodel API timed out after {max_poll_time}s")

        retry_after = response.get("retryAfter", 10)
        poll_count += 1
        print(
            f"  Supermodel API: {response.get('status')} "
            f"(poll #{poll_count}, {elapsed:.0f}s elapsed, retry in {retry_after}s)",
            file=sys.stderr,
            flush=True,
        )
        await asyncio.sleep(retry_after)

        proc = await asyncio.create_subprocess_exec(
            *poll_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            raise RuntimeError(f"Supermodel API poll failed: {stderr.decode()}")
        response = json.loads(stdout.decode())

    elapsed = time.time() - start_time

    # Clean up poll dummy file
    import os as _os

    _os.unlink(poll_dummy.name)

    # Check for error responses (status can be string "error" or HTTP status int)
    status = response.get("status")
    if status == "error" or response.get("error"):
        raise RuntimeError(
            f"Supermodel API error: {response.get('error', response.get('message'))}"
        )
    if isinstance(status, int) and status >= 400:
        raise RuntimeError(f"Supermodel API HTTP {status}: {response.get('message', response)}")

    api_result = response.get("result", response)
    print(f"  Supermodel API: completed in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return api_result
