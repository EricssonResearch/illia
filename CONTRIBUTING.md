# Contributing to illia

Hello and welcome! We're excited that you're interested in contributing to the
illia project.

We appreciate all types of contributions whether it's code, documentation, bug reports,
or feature requests.

## Want to Join the Team?

If you’re interested in contributing or collaborating more closely, feel free to reach
out!.

## How to Contribute Code

There are two main ways to contribute:

- Request a **new feature** or report a **bug**
- Submit a **code contribution** (bug fix, feature, improvement, etc.)

> Please note: we're actively supporting several projects, so while we’ll do our best to
> respond promptly, we can't always guarantee an immediate reply.

### Step 1: Open an Issue

Start by opening an issue to describe your proposal or the problem you've found. This
helps us evaluate and guide the work before coding begins.

> If your change is small (e.g., a typo fix or minor bug), feel free to skip this and
> open a pull request directly.

### Step 2: Make Your Changes

1. Fork this repository.
2. Clone it locally.
3. Create a new branch for your change.
4. Set up your development environment (see [Setup Environment](#setup-environment)).
5. Make your changes and ensure all tests pass.

### Step 3: Submit a Pull Request

Once your changes are ready, open a pull request (PR) from your forked branch to the
`main` branch.

### Step 4: Code Review

A maintainer will review your PR and may leave feedback. You may be asked to revise the
code, improve formatting, or adjust test coverage.

> Fix any failing tests and address review comments before re-requesting review.

### Step 5: Merge

Once approved, a maintainer will merge the pull request.

## Setup Environment

You can set up your dev environment in two ways:

### Option 1: Dev Container

We support Visual Studio Code, GitHub Codespaces, and JetBrains IDEs dev containers for
an easy, preconfigured development setup.

### Option 2: Local Setup

To work locally:

1. Install `git`, `python`, and preferably [`uv`](https://github.com/astral-sh/uv).
2. Clone the repository.
3. Run:

   ```bash
   make install
   ```

> The required Python version is defined in `pyproject.toml`. If using `uv`, it will
> create a compatible virtual environment automatically.

## Running Tests

We use [pytest](https://docs.pytest.org/) for testing. Run all tests with:

```bash
make tests
```

## Code Formatting & Quality Checks

We use `make` to automate formatting, linting, and checks. Just run:

```bash
make
```

This will:

- Install dependencies
- Run linters (`black`, `isort`, `ruff`, ...)
- Run static analysis (`complexipy`)
- Run tests
- Serve the documentation locally

## Docstring Style

A good function docstring typically includes:

- A short one-line description
- Optional longer explanation
- `Args:` section with parameter descriptions
- `Returns:` section
- Optional `Raises:` section
- Optional `Examples:` section

Please write clear, concise, and consistent docstrings.

## Other Ways to Contribute

Not a fan of writing code? You can still help by:

- Improving documentation or examples
- Fixing typos or grammar
- Suggesting features or usability improvements
- Helping triage and respond to issues

Thank you for being part of illia's journey. We’re thrilled to have
you here. If you have questions, feel free to reach out or open an issue.
