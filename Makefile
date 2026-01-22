.PHONY: help install test test-all lint format sync-version clean build npm-build npm-test npm-publish npm-pack

help:
	@echo "Available commands:"
	@echo "  make install       - Install the package in development mode"
	@echo "  make test          - Run unit tests"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with ruff"
	@echo "  make sync-version  - Sync version across project files"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make build         - Build distribution packages"
	@echo ""
	@echo "npm commands:"
	@echo "  make npm-build     - Prepare npm package (sync version)"
	@echo "  make npm-test      - Test npm package locally"
	@echo "  make npm-pack      - Create npm package tarball"
	@echo "  make npm-publish   - Publish to npm (requires NPM_TOKEN)"

install:
	pip install -e ".[dev]"

test:
	pytest -m "not integration" -v

test-all:
	pytest -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

sync-version:
	python3 scripts/sync_version.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: sync-version
	python -m build

npm-build: sync-version
	@echo "npm package prepared (version synced)"

npm-test: npm-build
	@echo "Testing npm package contents..."
	@test -f package.json || (echo "Error: package.json not found" && exit 1)
	@test -f .claude-plugin/plugin.json || (echo "Error: plugin.json not found" && exit 1)
	@test -d skills || (echo "Error: skills directory not found" && exit 1)
	@echo "npm package contents verified"

npm-pack: npm-build
	npm pack

npm-publish: npm-build
	@if [ -z "$$NPM_TOKEN" ]; then \
		echo "Error: NPM_TOKEN environment variable not set"; \
		echo "Set it with: export NPM_TOKEN=your-token"; \
		exit 1; \
	fi
	npm publish
