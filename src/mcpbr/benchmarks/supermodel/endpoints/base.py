"""Abstract base class for Supermodel endpoint plugins."""

import re
import subprocess
from abc import ABC, abstractmethod


class EndpointPlugin(ABC):
    """Base class for all endpoint benchmark plugins.

    Each endpoint defines:
    - How to call the Supermodel API
    - What prompts to give the baseline vs enhanced agent
    - How to extract ground truth from a PR
    - What tuple format to use for evaluation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'dead_code'."""

    @property
    @abstractmethod
    def api_path(self) -> str:
        """API endpoint path, e.g. '/v1/analysis/dead-code'."""

    @property
    @abstractmethod
    def baseline_prompt(self) -> str:
        """Prompt for the baseline agent (no graph data)."""

    @property
    @abstractmethod
    def enhanced_prompt(self) -> str:
        """Prompt for the graph-enhanced agent."""

    @property
    def analysis_filename(self) -> str:
        """Filename for the analysis JSON placed in the workdir."""
        return f"supermodel_{self.name}_analysis.json"

    @property
    def key_fields(self) -> tuple[str, str]:
        """Tuple field names for evaluation set comparison.

        Default: ("file", "name") -- works for dead code, impact analysis, test coverage.
        Override for endpoints with different tuple shapes (e.g. circular deps).
        """
        return ("file", "name")

    @abstractmethod
    def extract_ground_truth(
        self,
        repo: str,
        pr_number: int,
        language: str = "typescript",
        scope_prefix: str | None = None,
    ) -> list[dict]:
        """Extract ground truth from a GitHub PR diff.

        Returns a list of dicts with keys matching self.key_fields.
        """

    def parse_api_response(self, response: dict) -> dict:
        """Transform raw API response into the JSON file placed in workdir for Claude.

        Default: pass through as-is. Override if the API response needs reshaping.
        """
        return response

    def scope_prompt(self, prompt: str, scope_prefix: str | None) -> str:
        """Append scope context to a prompt if a scope_prefix is set."""
        if scope_prefix:
            return prompt + f"\n\nNote: Focus your analysis on the {scope_prefix} directory."
        return prompt

    @staticmethod
    def get_pr_diff(repo: str, pr_number: int) -> str:
        """Fetch a PR diff from GitHub using the gh CLI."""
        result = subprocess.run(
            ["gh", "pr", "diff", str(pr_number), "--repo", repo],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get diff for {repo}#{pr_number}: {result.stderr}")
        return result.stdout

    @staticmethod
    def should_skip_file(filepath: str, skip_patterns: list[str]) -> bool:
        """Check if a file should be skipped based on patterns."""
        return any(re.search(p, filepath) for p in skip_patterns)
