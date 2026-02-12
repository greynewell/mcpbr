"""Git utilities for cloning repos and creating zip archives."""

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger("mcpbr.supermodel")


async def clone_repo_at_commit(repo: str, commit: str, dest: str) -> None:
    """Clone a repo and checkout a specific commit.

    Args:
        repo: GitHub repo in 'owner/name' format.
        commit: Git commit SHA to checkout.
        dest: Destination directory path.
    """
    logger.info(f"Cloning {repo} at {commit[:8]} -> {dest}")

    proc = await asyncio.create_subprocess_exec(
        "git",
        "clone",
        "--quiet",
        "--depth",
        "1",
        f"https://github.com/{repo}.git",
        dest,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Clone failed: {stderr.decode()}")

    proc = await asyncio.create_subprocess_exec(
        "git",
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        commit,
        cwd=dest,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Fetch failed: {stderr.decode()}")

    proc = await asyncio.create_subprocess_exec(
        "git",
        "checkout",
        "--quiet",
        commit,
        cwd=dest,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"Checkout failed: {stderr.decode()}")


def get_pre_merge_commit(repo: str, merge_commit: str) -> str:
    """Get the first parent of a merge commit (pre-merge state).

    Args:
        repo: GitHub repo in 'owner/name' format.
        merge_commit: Merge commit SHA.

    Returns:
        SHA of the first parent commit.
    """
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/commits/{merge_commit}", "--jq", ".parents[0].sha"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get parent of {merge_commit}: {result.stderr}")
    return result.stdout.strip()


async def zip_repo(repo_dir: str, output_zip: str, scope_prefix: str | None = None) -> str:
    """Create a zip of the repo for Supermodel API using git archive.

    Args:
        repo_dir: Path to the repository directory.
        output_zip: Path for the output zip file.
        scope_prefix: Optional subdirectory to scope the archive to.

    Returns:
        Path to the created zip file.
    """
    git_dir = os.path.join(repo_dir, ".git")
    if os.path.isdir(git_dir):
        cmd = ["git", "archive", "-o", output_zip, "HEAD"]
        if scope_prefix:
            cmd.append(scope_prefix)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"git archive failed: {stderr.decode()}")
    else:
        target = os.path.join(repo_dir, scope_prefix) if scope_prefix else repo_dir
        proc = await asyncio.create_subprocess_exec(
            "zip",
            "-r",
            "-q",
            output_zip,
            ".",
            "-x",
            "node_modules/*",
            "-x",
            ".git/*",
            "-x",
            "dist/*",
            "-x",
            "build/*",
            "-x",
            "*.pyc",
            "-x",
            "__pycache__/*",
            cwd=target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"zip failed: {stderr.decode()}")
    return output_zip
