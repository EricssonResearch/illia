# Declare all phony targets
.PHONY: install clean lint code_check tests doc pipeline all

# Default command
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= illia
SRC_NOTEBOOKS_DL ?= docs/deep_learning_frameworks/examples
TEST_FILE ?= tests/torch tests/tf

# Allows the installation of project dependencies
install:
	@echo "Upgrading pip..."
	pip install --upgrade pip
	@echo "Installing uv..."
	pip install uv
	@echo "Installing dependecies with uv..."
	uv pip install -r pyproject.toml
	@echo ""

# Allows cache clearing
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	@echo ""

# Check format and quality of the code
lint:
	@echo "Checking code format with Black..."
	black --check $(SRC_PROJECT_NAME)/ $(TEST_FILE)/
	@echo "Checking code style with Flake8..."
	flake8 $(SRC_PROJECT_NAME)/
	@echo "Checking code quality with Pylint..."
	pylint --fail-under=8 $(SRC_PROJECT_NAME)/
	@echo ""

# Check
code_check:
	@echo "Checking code complexity with complexipy..."
	complexipy -d low $(SRC_PROJECT_NAME)/
	@echo "Checking type annotations with Mypy..."
	mypy $(SRC_PROJECT_NAME)/ $(TEST_FILE)/
	@echo ""

# Test the code
tests:
	@echo "Running tests..."
	pytest $(TEST_FILE)
	@echo ""

# Allows run the MkDocs
doc:
	@echo "Running MkDocs..."
	mkdocs serve

# Run all tasks of the pipeline without install and doc
pipeline: clean lint code_check tests
	@echo ""

# Run all tasks in sequence
all: install clean lint code_check tests doc
	@echo ""