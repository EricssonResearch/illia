# Declare all phony targets
.PHONY: install lint clean tests wiki-up all

# Default command
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= illia
SRC_NOTEBOOKS_DL ?= docs/deep_learning_frameworks/examples
TEST_FILE ?= tests/torch

# Allows the installation of project dependencies
install:
	@echo "Upgrading pip..."
	pip install --upgrade pip
	@echo "Installing requirements..."
	pip install -r requirements.txt

# Check format, quality and more, of the code
lint:
	@echo "Apply code format with Black..."
	black $(SRC_PROJECT_NAME)/ $(TEST_FILE)/ $(SRC_NOTEBOOKS_DL)/
	@echo "Checking code style and quality with Flake8..."
	flake8 $(SRC_PROJECT_NAME)/
	@echo "Checking code complexity with complexipy..."
	complexipy -d low $(SRC_PROJECT_NAME)/
	@echo "Checking code annotations with Mypy..."
	mypy $(SRC_PROJECT_NAME)/ $(TEST_FILE)/
	@echo "Checking code style and quality with Pylint..."
	pylint --fail-under=8 $(SRC_PROJECT_NAME)/

# Allows cache clearing
clean:
	@echo "Cleaning cache..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name nohup.out -exec rm -rf {} +

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