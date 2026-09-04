.PHONY: sync check test lint format typecheck smoke figures clean help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  sync        Install dependencies via uv"
	@echo "  check       Run formatting checks, linting, typechecking, and tests"
	@echo "  test        Run pytest suite"
	@echo "  lint        Run ruff linter"
	@echo "  format      Run ruff formatter"
	@echo "  typecheck   Run pyright type checker"
	@echo "  smoke       Run fast smoke test across all experiments"
	@echo "  figures     Run experiments and generate publication-quality figures"
	@echo "  clean       Remove cache and build artifacts"

sync:
	uv sync --extra dev

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest -v tests/

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run pytest -v tests/

smoke:
	uv run python experiments/run_all.py --smoke

figures:
	uv run python experiments/run_all.py

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .venv dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
