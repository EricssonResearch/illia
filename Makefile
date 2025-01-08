# Declare all phony targets
.PHONY: install lint type-check clean tests wiki-up all

# Default command
.DEFAULT_GOAL := all

# Variable for the test file
TEST_FILE ?= ./tests

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

# Check format of the code using Black and Mypy to detect errors 
lint:
	@echo "Checking format of the code using Black..."
	black --check .
	@echo "Checking errors of the code using Mypy..."
	mypy --cache-dir=/dev/null .

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