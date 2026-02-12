"""Test coverage endpoint plugin.

Ground truth: PRs that add tests for previously uncovered code.
The PR diff shows which functions gained test coverage -- those were the untested ones.
Tuple format: (file, function_name)
"""

import re

from .base import EndpointPlugin

TS_FUNC_PATTERNS = [
    (r"^[+-]\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
    (r"^[+-]\s*(?:export\s+)?class\s+(\w+)", "class"),
    (r"^[+-]\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),
]

PY_FUNC_PATTERNS = [
    (r"^[+-]\s*def\s+(\w+)\s*\(", "function"),
    (r"^[+-]\s*async\s+def\s+(\w+)\s*\(", "function"),
    (r"^[+-]\s*class\s+(\w+)[\s(:]", "class"),
]

TEST_FILE_PATTERNS = [
    r"\.test\.",
    r"\.spec\.",
    r"__tests__/",
    r"test_\w+\.py$",
    r"tests/",
    r"test/",
]

SKIP_NAMES = {
    "describe",
    "it",
    "test",
    "expect",
    "beforeEach",
    "afterEach",
    "setUp",
    "tearDown",
}


class TestCoveragePlugin(EndpointPlugin):
    @property
    def name(self) -> str:
        return "test_coverage"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/test-coverage-map"

    @property
    def baseline_prompt(self) -> str:
        return """You are an expert software architect analyzing test coverage.

Analyze this codebase and identify ALL exported functions, classes, and methods
that lack test coverage -- i.e., they are never called or referenced in any test file.

Focus on:
- Functions/classes in source files that have no corresponding test
- Public API surface that is untested
- Complex logic paths with no test assertions

Do NOT include:
- Type definitions or interfaces
- Config files or constants
- Functions that are tested indirectly through integration tests

CRITICAL: Update the existing REPORT.json file with your findings.
Format: a JSON object with "dead_code" array containing objects with file, name, type, and reason.
Set "analysis_complete" to true when done.
"""

    @property
    def enhanced_prompt(self) -> str:
        return """Read the file supermodel_test_coverage_analysis.json in the current directory.

It contains a pre-computed test coverage analysis with:
- untestedFunctions: functions with no test coverage ({file, name, type, reason})
- testedFunctions: functions with confirmed test coverage
- coverageByFile: per-file coverage percentages

Extract the untestedFunctions list. Filter out obvious false positives:
- Type definitions or interfaces
- Config/setup files
- Framework lifecycle methods

CRITICAL: Update the existing REPORT.json file with the untested functions.
Format: a JSON object with "dead_code" array containing objects with file, name, type, and reason.
Set "analysis_complete" to true when done.

Do NOT search the codebase. Just read the file, filter, and update REPORT.json.
"""

    def extract_ground_truth(
        self,
        repo: str,
        pr_number: int,
        language: str = "typescript",
        scope_prefix: str | None = None,
    ) -> list[dict]:
        diff = self.get_pr_diff(repo, pr_number)
        return _parse_newly_tested_symbols(diff, language, scope_prefix)


def _is_test_file(filepath: str) -> bool:
    return any(re.search(p, filepath) for p in TEST_FILE_PATTERNS)


def _parse_newly_tested_symbols(
    diff_text: str,
    language: str = "typescript",
    scope_prefix: str | None = None,
) -> list[dict]:
    """Parse diff to extract source-file symbols that gained test coverage."""
    patterns = TS_FUNC_PATTERNS if language == "typescript" else PY_FUNC_PATTERNS
    source_symbols = []
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
        if _is_test_file(current_file):
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
                if name in SKIP_NAMES:
                    continue
                key = (current_file, name)
                if key not in seen:
                    seen.add(key)
                    source_symbols.append({"file": current_file, "name": name, "type": decl_type})
                break

    return source_symbols
