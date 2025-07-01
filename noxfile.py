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


# Framework-specific sessions for GitHub Actions matrix
@nox.session(python=PYTHON_VERSIONS, name="test-torch")
def test_torch(session):
    """Run all PyTorch tests across compatible versions."""
    # Install dependencies
    # session.install(*nox.project.dependency_groups(PYPROJECT, "pipeline"))
    session.install("pytest", "pytest-order")
    
    # Test with latest compatible torch version for the Python version
    py_version = session.python
    latest_torch = None
    
    # Find the latest torch version compatible with current Python version
    for torch_ver in sorted(TORCH_COMPAT.keys(), reverse=True):
        if py_version in TORCH_COMPAT[torch_ver]:
            latest_torch = torch_ver
            break
    
    if latest_torch:
        session.install(f"torch=={latest_torch}")
        session.run("pytest", "tests/torch/")
    else:
        session.skip(f"No compatible PyTorch version found for Python {py_version}")


@nox.session(python=PYTHON_VERSIONS, name="test-tf")
def test_tf(session):
    """Run all TensorFlow tests across compatible versions."""
    # Install dependencies
    # session.install(*nox.project.dependency_groups(PYPROJECT, "pipeline"))
    session.install("pytest", "pytest-order")
    
    # Test with latest compatible tf version for the Python version
    py_version = session.python
    latest_tf = None
    
    # Find the latest tf version compatible with current Python version
    for tf_ver in sorted(TF_COMPAT.keys(), reverse=True):
        if py_version in TF_COMPAT[tf_ver]:
            latest_tf = tf_ver
            break
    
    if latest_tf:
        session.install(f"tensorflow=={latest_tf}")
        session.run("pytest", "tests/tf/")
    else:
        session.skip(f"No compatible TensorFlow version found for Python {py_version}")