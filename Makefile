# Allows you to indicate that the targets you define are executed as commands
# commands and do not refer to project files
.PHONY: install format type-check clean tests requirements all

# Default command
.DEFAULT_GOAL := init

# Variable for the test file
TEST_FILE ?= ./tests

# Allows the installation of Poetry dependencies for the project
install: pyproject.toml
	@echo "Installing Poetry dependencies..."
	poetry install

# Check format of the code using Black
format:
	@echo "Checking format of the code using Black..."
	poetry run black --check .

# Allows to use Mypy to detect errors 
type-check:
	@echo "Checking errors of the code using Mypy..."
	poetry run mypy --cache-dir=/dev/null .

# Allows cache clearing
clean:
	@echo "Cleaning cache..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

# Allows testing the code
tests: clean delete_logs format type-check
	@echo "Testing code..."
	poetry run pytest -v $(TEST_FILE)

# Generates a requirements.txt file with the project dependencies
requirements: pyproject.toml
	@echo "Generating requirements.txt file..."
	poetry export -f requirements.txt --without-hashes > requirements.txt

# Run all tasks in sequence
all: install tests requirements
