# Declare all phony targets
.PHONY: install clean lint tests doc all

# Default command
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= illia
SRC_NOTEBOOKS_DL ?= docs/deep_learning_frameworks/examples
TEST_FILE ?= tests/torch tests/tf/distributions

# Allows the installation of project dependencies
install: requirements.txt
	@echo "Upgrading pip..."
	pip install --upgrade pip
	@echo "Installing requirements..."
	pip install -r requirements.txt

# Allows cache clearing
clean:
	@echo "Cleaning cache..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name nohup.out -exec rm -rf {} +

# Check format, quality and more, of the code
lint:
	@echo "Apply code format with Black..."
	black $(SRC_PROJECT_NAME)/ $(TEST_FILE)/ $(SRC_NOTEBOOKS_DL)/
	@echo "Checking code style and quality with Flake8..."
	flake8 $(SRC_PROJECT_NAME)/
	@echo "Checking code complexity with complexipy..."
	complexipy -d low $(SRC_PROJECT_NAME)/
	@echo "Checking code annotations with Mypy..."
	mypy $(SRC_PROJECT_NAME)/ $(TEST_FILE)
	@echo "Checking code style and quality with Pylint..."
	pylint --fail-under=8 $(SRC_PROJECT_NAME)/

# Test the code
tests: clean
	@echo "Testing code..."
	pytest $(TEST_FILE)

# Allows run the MkDocs
doc:
	@echo "Running MkDocs..."
	mkdocs serve

# Run all tasks in sequence
all: install lint tests
