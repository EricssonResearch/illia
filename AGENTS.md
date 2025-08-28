# AGENTS.md — Guide for Illia

This document provides clear instructions for developers and code agents working with the
**Illia** codebase.

## 1. Project Structure

- **`/illia`** — Core source code for the library:

  - `__init__.py`: Initializes the multi-backend manager.
  - `support.py`: Contains backend class support definitions.
  - `version.py`: Specifies the project version (taken directly from `pyproject.toml`).

- **`/tests`** — Core test classes for the library. These are usually inherited rather
  than directly run.

- **`/docs`** — Documentation for the library:
  - Built with **MkDocs**.
  - Versioning managed with **mike**.

## 2. Development Environment

- **Dev Container**:

  - `.devcontainer/` contains development environment configuration.
  - Includes a `setup.sh` script (Bash) that installs `uv` and all project dependencies.

- **Automation with Makefile**:
  - The `Makefile` provides shortcuts for development tasks.

### Available Commands:

| Command           | Description                                                      |
| ----------------- | ---------------------------------------------------------------- |
| `make install`    | Install dependencies.                                            |
| `make clean`      | Remove cache and temporary files.                                |
| `make lint`       | Run code formatting and linting checks.                          |
| `make code_check` | Perform static code analysis.                                    |
| `make tests`      | Run tests for all supported backends (TensorFlow, PyTorch, JAX). |
| `make doc`        | Serve documentation locally (MkDocs).                            |
| `make pipeline`   | Run clean → lint → code_check → tests                            |
| `make all`        | Full workflow (install + pipeline + docs)                        |

## 3. Contributing

- Please read **`CONTRIBUTING.md`** before making contributions.
- Keep **Pull Requests (PRs)** minimal:
  - Bugfix PRs should be as small as possible (even 1–2 lines if enough).
  - Avoid unnecessary comments, docstrings, or new functions unless needed.
- When writing tests:
  - Add them to existing files whenever possible.
  - Create a new test directory **only** when adding a new model, layer, or feature.
- Code style is checked automatically in **CI**.

## 4. Testing

- Continuous Integration (CI) workflows are in `.github/workflows/`.
- You can run tests locally with:

  ```bash
  make tests
  ```

This will execute all defined backend test suites.
