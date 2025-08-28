"""
This file contains the pytests calls with different versions of python.
"""

# 3pps
import nox

# Own modules
from illia.support import PYTHON_VERSIONS, TF_COMPAT, TORCH_COMPAT


nox.options.default_venv_backend = "uv"


# Framework-specific sessions for GitHub Actions matrix
@nox.session(python=PYTHON_VERSIONS, name="test-torch")
def test_torch(session: nox.Session) -> None:
    """
    Run all PyTorch tests across compatible versions.

    Args:
        session: The Nox session object.

    Returns:
        None.
    """

    # Install dependencies
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
        session.run("pytest", "torch/")
    else:
        session.skip(f"No compatible PyTorch version found for Python {py_version}")


@nox.session(python=PYTHON_VERSIONS, name="test-tf")
def test_tf(session: nox.Session) -> None:
    """
    Run all TensorFlow tests across compatible versions.

    Args:
        session: The Nox session object.

    Returns:
        None.
    """

    # Install dependencies
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
        session.run("pytest", "tf/")
    else:
        session.skip(f"No compatible TensorFlow version found for Python {py_version}")


@nox.session(python=PYTHON_VERSIONS, name="test-jax", tags=["test-jax-backend"])
def test_jax(session: nox.Session) -> None:
    """
    Test compatibility and execute tests for specified jax version.

    Args:
        session: The Nox session object.

    Returns:
        None.
    """

    # Install dependencies with tensorflow extras
    session.run("uv", "sync", "--extra", "jax", "--active")

    # Run pytest
    session.run("pytest", "jax/", external=True)
