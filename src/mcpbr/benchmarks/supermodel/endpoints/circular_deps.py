"""Circular dependencies endpoint plugin.

Ground truth: PRs that refactor to break circular imports.
The removed import edges ARE the cycles that existed.
Tuple format: (module_a, module_b) -- directed edges in a cycle.
"""

import re
from pathlib import PurePosixPath

from .base import EndpointPlugin


class CircularDepsPlugin(EndpointPlugin):
    @property
    def name(self) -> str:
        return "circular_deps"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/circular-dependencies"

    @property
    def key_fields(self) -> tuple[str, str]:
        return ("module_a", "module_b")

    @property
    def baseline_prompt(self) -> str:
        return """You are an expert software architect analyzing module dependencies.

Analyze this codebase and identify ALL circular import/dependency chains.
A circular dependency exists when module A imports module B, and module B
(directly or transitively) imports module A.

For each cycle, report the directed edges that form the cycle.

Focus on:
- Direct circular imports (A -> B -> A)
- Transitive cycles (A -> B -> C -> A)
- Re-export chains that create hidden cycles

Do NOT include:
- Type-only imports (import type { ... })
- Test files importing source files
- Dynamic imports that don't create load-time cycles

CRITICAL: Update the existing REPORT.json file with your findings.
Format: a JSON object with "dead_code" array containing objects with module_a, module_b, and cycle.
Set "analysis_complete" to true when done.
"""

    @property
    def enhanced_prompt(self) -> str:
        return """Read the file supermodel_circular_deps_analysis.json in the current directory.

It contains a pre-computed circular dependency analysis with:
- cycles: detected circular import chains with edges and severity
- summary: counts of cycles by severity level
- metadata: analysis method and file/import counts

Extract the cycles and report each directed edge. Filter out:
- Type-only import cycles (import type { ... })
- Test file cycles

CRITICAL: Update the existing REPORT.json file with the cycle edges.
Format: a JSON object with "dead_code" array containing objects with module_a, module_b, and cycle.
Set "analysis_complete" to true when done.

Do NOT search the codebase. Just read the file and update REPORT.json.
"""

    def extract_ground_truth(
        self,
        repo: str,
        pr_number: int,
        language: str = "typescript",
        scope_prefix: str | None = None,
    ) -> list[dict]:
        diff = self.get_pr_diff(repo, pr_number)
        return _parse_broken_cycles(diff, language, scope_prefix)


def _parse_broken_cycles(
    diff_text: str,
    language: str = "typescript",
    scope_prefix: str | None = None,
) -> list[dict]:
    """Parse diff to extract import edges that were removed to break cycles."""
    results = []
    current_file = None
    seen: set[tuple[str, str]] = set()

    if language == "typescript":
        import_patterns = [
            r"""^-\s*import\s+\{[^}]*\}\s+from\s+['"]([^'"]+)['"]""",
            r"""^-\s*import\s+\w+\s+from\s+['"]([^'"]+)['"]""",
            r"""^-\s*import\s+\*\s+as\s+\w+\s+from\s+['"]([^'"]+)['"]""",
        ]
        type_import = r"^-\s*import\s+type\s+"
    else:
        import_patterns = [
            r"^-\s*from\s+(\S+)\s+import\s+",
            r"^-\s*import\s+(\S+)",
        ]
        type_import = None

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) >= 2:
                current_file = parts[-1]
            continue

        if current_file is None:
            continue
        if scope_prefix and not current_file.startswith(scope_prefix):
            continue

        if not line.startswith("-") or line.startswith("---"):
            continue

        # Skip type-only imports
        if type_import and re.match(type_import, line):
            continue

        for pattern in import_patterns:
            match = re.match(pattern, line)
            if match:
                imported = match.group(1)
                if imported.startswith("."):
                    imported = _resolve_relative(current_file, imported)
                key = (current_file, imported)
                if key not in seen:
                    seen.add(key)
                    results.append({"module_a": current_file, "module_b": imported})
                break

    return results


def _resolve_relative(from_file: str, import_path: str) -> str:
    """Rough resolution of relative import paths."""
    from_dir = str(PurePosixPath(from_file).parent)
    parts = import_path.split("/")
    resolved_parts = from_dir.split("/") if from_dir != "." else []

    for part in parts:
        if part == ".":
            continue
        elif part == "..":
            if resolved_parts:
                resolved_parts.pop()
        else:
            resolved_parts.append(part)

    resolved = "/".join(resolved_parts)
    if not any(resolved.endswith(ext) for ext in [".ts", ".tsx", ".js", ".jsx", ".py"]):
        resolved += ".ts"
    return resolved
