"""Endpoint plugin registry for Supermodel benchmarks."""

from .base import EndpointPlugin
from .circular_deps import CircularDepsPlugin
from .dead_code import DeadCodePlugin
from .impact_analysis import ImpactAnalysisPlugin
from .test_coverage import TestCoveragePlugin

ENDPOINT_REGISTRY: dict[str, type[EndpointPlugin]] = {
    "dead-code": DeadCodePlugin,
    "impact": ImpactAnalysisPlugin,
    "test-coverage": TestCoveragePlugin,
    "circular-deps": CircularDepsPlugin,
}


def get_endpoint(name: str) -> EndpointPlugin:
    """Get an endpoint plugin instance by name."""
    if name not in ENDPOINT_REGISTRY:
        available = ", ".join(ENDPOINT_REGISTRY.keys())
        raise ValueError(f"Unknown endpoint: {name}. Available: {available}")
    return ENDPOINT_REGISTRY[name]()


__all__ = [
    "EndpointPlugin",
    "DeadCodePlugin",
    "ImpactAnalysisPlugin",
    "TestCoveragePlugin",
    "CircularDepsPlugin",
    "ENDPOINT_REGISTRY",
    "get_endpoint",
]
