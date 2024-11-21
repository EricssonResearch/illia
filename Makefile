# Allows you to indicate that the targets you define are executed as commands
# commands and do not refer to project files
.PHONY: install format type-check clean tests wiki-up requirements all 

# Default command
.DEFAULT_GOAL := all

# Variable for the test file
TEST_FILE ?= ./tests

# Allows the installation of project dependencies using Poetry
install: pyproject.toml
	@echo "Installing Poetry dependencies..."
	poetry install

# Allows the installation of all dependencies for the project using Poetry
# This is for development
install-all: pyproject.toml
	@echo "Installing Poetry dependencies..."
	poetry install --with dev,docs 

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
	@echo "Cleaning pycache..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

# Allows testing the code
tests: clean format type-check
	@echo "Testing code..."
	poetry run pytest -v $(TEST_FILE)

# Allows run the MkDocs
wiki-up:
	@echo "Runing MkDocs..."
	poetry run mkdocs serve

# Generates a requirements.txt file with the project dependencies using Poetry
requirements: pyproject.toml
	@echo "Generating requirements.txt file..."
	poetry export -f requirements.txt --without-hashes > requirements.txt

# Run all tasks in sequence
all: install tests requirements
