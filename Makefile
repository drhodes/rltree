export PYTHONPATH := .

.PHONY: help install run test coverage lint typecheck format clean samples

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies and sync the virtual environment using uv"
	@echo "  make run        - Run the end-to-end RL training and inference pipeline"
	@echo "  make samples    - Pretty-print a few random dataset SVGs to inspect vocabulary"
	@echo "  make test       - Run pytest test suite"
	@echo "  make coverage   - Run pytest and generate a code coverage report"
	@echo "  make lint       - Run Ruff linter checks"
	@echo "  make typecheck  - Run Mypy static type checker"
	@echo "  make format     - Automatically format codebase using Ruff"
	@echo "  make clean      - Clean up temporary files, coverage database, and caches"

install:
	uv sync

run:
	uv run python main.py

samples:
	uv run python scripts/show_samples.py

test:
	uv run pytest -v

coverage:
	uv run coverage run -m pytest -v
	uv run coverage report

lint:
	uv run ruff check

typecheck:
	uv run mypy .

format:
	uv run ruff check --fix
	uv run ruff format

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
