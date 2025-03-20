# Step 1: Check for code style errors with Black
black --check illia/

# Step 2: Check for syntax errors with Flake8
flake8 illia/

# Step 3: Measure code complexity with Complexipy
complexipy -d low illia/

# Step 4: Check for type errors with MyPy
mypy illia/

# Step 5: Check for style errors with Pylint
pylint --fail-under=8 illia/