"""
This file contains the pytests calls with different versions of python.
The versions included are more or less up-to-date with official release support.
    https://devguide.python.org/versions/, as such end-of-life versions are not tested
    nor supported.
"""

# Standard libraries
import sys

# 3pps
import nox

nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.10", "3.12"]
TORCH_COMPAT = {
    "2.1.2": {"3.8", "3.9", "3.10", "3.11"},
    "2.2.2": {"3.8", "3.9", "3.10", "3.11", "3.12"},
    "2.5.1": {"3.8", "3.9", "3.10", "3.11", "3.12"},
}
TF_COMPAT = {
    "2.11.0": {"3.8", "3.9", "3.10", "3.11"},
    "2.16.1": {"3.10", "3.11", "3.12"},
    "2.19.0": {"3.10", "3.11", "3.12"},
}


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("torch", ["2.1.2", "2.2.2", "2.5.1"])
def pytorch(session, torch):
    # Ensure correct compatibility test
    py_version = session.python
    if py_version not in TORCH_COMPAT[torch]:
        session.skip(f"{torch=} is not compatible with {py_version=}")

    # Install dependencies and specific torch version
    # session.install(*nox.project.dependency_groups(PYPROJECT, "pipeline"))
    torch_version = f"torch=={torch}"
    session.install("pytest", "pytest-order", torch_version)
    session.run("pytest", "tests/torch/")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("tf", ["2.16.1", "2.19.0"])
def tensorflow(session, tf):
    # Ensure only correct matrix compatibility
    py_version = session.python
    if py_version not in TF_COMPAT[tf]:
        session.skip(f"{tf=} not compatible with {py_version=}")

    # Install dependencies & tf specific version
    tf_version = f"tensorflow=={tf}"
    session.install("pytest", "pytest-order", tf_version)
    session.run("pytest", "tests/tf/")
