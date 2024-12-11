# Allows you to indicate that the targets you define are executed as commands
# commands and do not refer to project files
.PHONY: install install-all format type-check clean tests wiki-up all 

# Default command
.DEFAULT_GOAL := all

# Variable for the test file
TEST_FILE ?= ./tests

# Allows the installation of project dependencies
install: requirements.txt
	@echo "Installing illia dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

# Allows the installation of all dependencies for the project
# This is for development
install-all: requirements.txt requirements-dev.txt requirements-wiki.txt
	@echo "Installing illia dependencies for development..."
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt -r requirements-wiki.txt

# Check format of the code using Black
format:
	@echo "Checking format of the code using Black..."
	black --check .

# Allows to use Mypy to detect errors 
type-check:
	@echo "Checking errors of the code using Mypy..."
	mypy --cache-dir=/dev/null .

# Allows cache clearing
clean:
	@echo "Cleaning pycache..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

# Allows testing the code
tests: clean format type-check
	@echo "Testing code..."
	pytest -v $(TEST_FILE)

# Allows run the MkDocs
wiki-up:
	@echo "Runing MkDocs..."
	mkdocs serve

# Run all tasks in sequence
all: install-all tests
