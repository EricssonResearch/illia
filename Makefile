# Declare all phony targets
.PHONY: install lint type-check clean tests wiki-up all

# Default command
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= illia
TEST_FILE ?= tests

# Allows the installation of project dependencies
install:
	@echo "Updating PIP..."
	pip install --upgrade pip
	@if [ "$(ALL)" = "1" ]; then \
	    echo "Installing all dependencies..."; \
	    pip install -r requirements.txt -r requirements-dev.txt -r requirements-wiki.txt; \
	else \
	    echo "Installing main dependencies..."; \
	    pip install -r requirements.txt; \
	fi

# Check format, quality and more, of the code
lint:
	@echo "Checking Code Format with Black..."
	black --check $(SRC_PROJECT_NAME)/ $(TEST_FILE)/
	@echo "Checking Code Style and Quality with Flake8..."
	flake8 $(SRC_PROJECT_NAME)/
	@echo "Checking Code Complexity with complexipy..."
	complexipy -d low $(SRC_PROJECT_NAME)/
	@echo "Checking Code Annotations with Mypy..."
	mypy $(SRC_PROJECT_NAME)/ $(TEST_FILE)/
	@echo "Checking Code Style and Quality with Pylint..."
	pylint --fail-under=8 $(SRC_PROJECT_NAME)/ $(TEST_FILE)/

# Allows cache clearing
clean:
	@echo "Cleaning pycache..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

# Allows testing the code
tests: clean lint
	@echo "Testing code..."
	pytest -v $(TEST_FILE)

# Allows run the MkDocs
wiki-up:
	@echo "Running MkDocs..."
	mkdocs serve

# Run all tasks in sequence
all: install tests