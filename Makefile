.PHONY: setup install dev test lint fmt

setup:
	bash scripts/setup.sh

install:
	pip install -e .

dev:
	pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check src || true

fmt:
	black src tests || true
