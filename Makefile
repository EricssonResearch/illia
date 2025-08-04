# Declare all phony targets
.PHONY: install clean lint code_check tests doc pipeline all

# Default command
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= illia
SRC_PROJECT_TESTS_TF ?= tests/tf
SRC_PROJECT_TESTS_TORCH ?= tests/torch
SRC_PROJECT_TESTS_JAX ?= tests/jax
SRC_ALL ?= $(SRC_PROJECT_NAME)/ $(SRC_PROJECT_TESTS_TF)/ \
    $(SRC_PROJECT_TESTS_TORCH)/ $(SRC_PROJECT_TESTS_JAX)/

# Allows the installation of project dependencies
install:
	@echo "Installing dependencies..."
	@uv sync --all-extras
	@echo "✅ Dependencies installed."

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@echo "✅ Clean complete."

# Check code formatting and linting
lint:
	@echo "Running lint checks..."
	@uv run black $(SRC_ALL)/
	@uv run isort $(SRC_ALL)/
	@uv run flake8 $(SRC_ALL)/
	@uv run pylint --fail-under=8 $(SRC_PROJECT_NAME)/
	@echo "✅ Linting complete."

# Static analysis checks
code_check:
	@echo "Running static code checks..."
	@uv run complexipy -d low $(SRC_PROJECT_NAME)/
	@uv run mypy $(SRC_PROJECT_NAME)/
	@uv run bandit -r $(SRC_PROJECT_NAME)/ --exclude tests/
	@echo "✅ Code checks complete."

# Test the code, only if the tests directory exists
tests:
	@echo "Runing test per each backend..."
	@uv run pytest $(SRC_PROJECT_TESTS_TF)/ && \
	uv run pytest $(SRC_PROJECT_TESTS_TORCH)/ && \
	uv run pytest $(SRC_PROJECT_TESTS_JAX)/
	@echo "✅ Tests complete."

# Serve documentation locally
doc:
	@echo "Serving documentation..."
	@uv run mkdocs serve

# Run code checks and tests
pipeline: clean lint code_check tests
	@echo "✅ Pipeline complete."

# Run full workflow including install and docs
all: install pipeline doc
	@echo "✅ All tasks complete."