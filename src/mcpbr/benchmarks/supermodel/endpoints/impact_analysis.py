"""Impact analysis endpoint plugin.

Ground truth: PRs where a change in one area required updates across multiple files.
The PR diff shows exactly which files/symbols were touched -- that IS the impact.
Tuple format: (file, symbol_name)
"""

import re

from .base import EndpointPlugin

TS_MODIFIED_PATTERNS = [
    (r"^[+-]\s*export\s+(?:async\s+)?function\s+(\w+)", "function"),
    (r"^[+-]\s*export\s+class\s+(\w+)", "class"),
    (r"^[+-]\s*export\s+const\s+(\w+)\s*[=:]", "const"),
    (r"^[+-]\s*(?:async\s+)?function\s+(\w+)", "function"),
    (r"^[+-]\s*class\s+(\w+)", "class"),
]

PY_MODIFIED_PATTERNS = [
    (r"^[+-]\s*def\s+(\w+)\s*\(", "function"),
    (r"^[+-]\s*async\s+def\s+(\w+)\s*\(", "function"),
    (r"^[+-]\s*class\s+(\w+)[\s(:]", "class"),
]

SKIP_FILE_PATTERNS = [
    r"\.test\.",
    r"\.spec\.",
    r"__tests__/",
    r"\.stories\.",
    r"\.d\.ts$",
    r"__mocks__/",
    r"package\.json",
    r"package-lock\.json",
    r"\.md$",
    r"\.lock$",
    r"yarn\.lock",
]


class ImpactAnalysisPlugin(EndpointPlugin):
    @property
    def name(self) -> str:
        return "impact_analysis"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/impact"

    @property
    def baseline_prompt(self) -> str:
        return """You are an expert software architect performing impact analysis.

Given this codebase, identify ALL files and symbols that would be affected if key
modules were modified. Think about:
- Direct importers of each module
- Transitive dependencies (files that import the importers)
- Shared types/interfaces that cross module boundaries

CRITICAL: Update the existing REPORT.json file with your findings.
Format: a JSON object with "dead_code" array containing objects with file, name, type, and reason.
Set "analysis_complete" to true when done.
"""

    @property
    def enhanced_prompt(self) -> str:
        return """Read the file supermodel_impact_analysis_analysis.json in the current directory.

It contains a pre-computed impact analysis with:
- impacts: per-file blast radius with affectedFunctions and affectedFiles
- globalMetrics: most critical files ranked by dependent count

Extract all affected functions from the impacts array. Each impact has a target
file and an affectedFunctions list with {file, name, type, distance, relationship}.

CRITICAL: Update the existing REPORT.json file with the affected functions.
Format: a JSON object with "dead_code" array containing objects with file, name, type, and reason.
Set "analysis_complete" to true when done.

Do NOT search the codebase. Just read the file and update REPORT.json.
"""

    def parse_api_response(self, response: dict) -> dict:
        """Flatten impacts into a simpler structure for Claude."""
        impacts = response.get("impacts", [])
        all_affected = []
        for impact in impacts:
            for af in impact.get("affectedFunctions", []):
                all_affected.append(af)
        response = dict(response)
        response["allAffectedFunctions"] = all_affected
        return response

    def extract_ground_truth(
        self,
        repo: str,
        pr_number: int,
        language: str = "typescript",
        scope_prefix: str | None = None,
    ) -> list[dict]:
        diff = self.get_pr_diff(repo, pr_number)
        return _parse_impacted_symbols(diff, language, scope_prefix)


def _parse_impacted_symbols(
    diff_text: str,
    language: str = "typescript",
    scope_prefix: str | None = None,
) -> list[dict]:
    """Parse diff to extract all modified symbols across all touched files."""
    patterns = TS_MODIFIED_PATTERNS if language == "typescript" else PY_MODIFIED_PATTERNS
    results = []
    current_file = None
    seen: set[tuple[str, str]] = set()

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) >= 2:
                current_file = parts[-1]
            continue

        if current_file is None:
            continue
        if EndpointPlugin.should_skip_file(current_file, SKIP_FILE_PATTERNS):
            continue
        if scope_prefix and not current_file.startswith(scope_prefix):
            continue

        if not (line.startswith("+") or line.startswith("-")):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue

        for pattern, decl_type in patterns:
            match = re.match(pattern, line)
            if match:
                name = match.group(1)
                key = (current_file, name)
                if key not in seen:
                    seen.add(key)
                    results.append({"file": current_file, "name": name, "type": decl_type})
                break

    return results
