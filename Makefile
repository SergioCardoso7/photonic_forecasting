.PHONY: install fmt lint type test test-all check clean

install:
	uv sync --group notebooks
	uv run pre-commit install

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

type:
	uv run mypy src

test:
	uv run pytest -m "not integration"

test-all:
	uv run pytest

check: lint type test

update:
	uv lock --upgrade
	uv sync

silver:
	uv run python -m pv.transform.silver

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
