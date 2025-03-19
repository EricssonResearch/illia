from typing import Optional

import pytest
import torch

from illia.torch.nn import Linear
from illia.torch.distributions import Distribution, GaussianDistribution


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution",
    [
        (30, 20, None, None),
        (20, 30, GaussianDistribution((20, 30)), GaussianDistribution((20, 30))),
    ],
)
def test_linear_init(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
) -> None:
    """
    This function is the test for the Linear constructor.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.

    Returns:
        None.
    """

    # define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # check parameters length
    len_parameters: int = len(list(model.parameters()))
    assert (
        len_parameters == 4
    ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution, batch_size",
    [
        (30, 20, None, None, 64),
        (20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,)), 32),
    ],
)
def test_linear_forward(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
) -> None:
    """
    This function is the test for the Linear forward pass.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.
        batch_size: batch size to use in the tests.

    Returns:
        None.
    """

    # define linear layer
    original_model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs = torch.rand((batch_size, input_size))

    # iter over models
    for model in (original_model, model_scripted):
        # check parameters length
        outputs: torch.Tensor = model(inputs)

        # check type of outputs
        assert isinstance(
            outputs, torch.Tensor
        ), f"Incorrect outputs class, expected {torch.Tensor} and got {type(outputs)}"

        # check outputs shape
        assert outputs.shape == (batch_size, output_size), (
            f"Incorrect outputs shape, expected {(batch_size, output_size)} and got "
            f"{outputs.shape}"
        )

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution, batch_size",
    [
        (30, 20, None, None, 64),
        (20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,)), 32),
    ],
)
def test_linear_backward(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
) -> None:
    """
    This function is the test for the Linear backward pass.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.
        batch_size: batch size to use in the tests.

    Returns:
        None.
    """

    # define linear layer
    original_model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs = torch.rand((batch_size, input_size))

    # iter over models
    for model in (original_model, model_scripted):
        # check parameters length
        outputs: torch.Tensor = model(inputs)
        outputs.sum().backward()

        # check type of outputs
        for name, parameter in model.named_parameters():
            # check if parameter is none
            assert parameter.grad is not None, (
                f"Incorrect backward computation, gradient of {name} shouldn't be "
                f"None"
            )

    return None


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution, batch_size",
    [
        (30, 20, None, None, 64),
        (20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,)), 32),
    ],
)
def test_linear_freeze(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
) -> None:
    """
    This function is the test for the freeze and unfreeze layers from
    Linear layer.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.
        batch_size: batch size to use in the tests.

    Returns:
        None.
    """

    # define linear layer
    original_model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs = torch.rand((batch_size, input_size))

    # iter over models
    for model in (original_model, model_scripted):
        # compute outputs
        outputs_first: torch.Tensor = model(inputs)
        outputs_second: torch.Tensor = model(inputs)

        # check if both outputs are equal
        assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect outputs, different forwards are equal when at the "
            "initialization the layer should be unfrozen"
        )

        # freeze layer
        model.freeze()

        # compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # check if both outputs are equal
        assert torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect freezing, when layer is frozen outputs are not the same in "
            "different forward passes"
        )

        # unfreeze layer
        model.unfreeze()

        # compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
            "different forward passes"
        )

    return None


@pytest.mark.order(5)
@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution, batch_size",
    [
        (30, 20, None, None, 64),
        (20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,)), 32),
    ],
)
def test_linear_kl_cost(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
) -> None:
    """
    This function is the test for the kl_cost method of Linear layer.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.
        batch_size: batch size to use in the tests.

    Returns:
        None.
    """

    # define linear layer
    original_model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # iter over models
    for model in (original_model, model_scripted):
        # compute outputs
        outputs: tuple[torch.Tensor, int] = model.kl_cost()

        # check type of output
        assert isinstance(
            outputs, tuple
        ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

        # check type of kl cost
        assert isinstance(outputs[0], torch.Tensor), (
            f"Incorrect output type in the first element, expected {torch.Tensor} and "
            f"got {type(outputs[0])}"
        )

        # check type of num params
        assert isinstance(outputs[1], int), (
            f"Incorrect output type in the second element, expected {int} and got "
            f"{type(outputs[1])}"
        )

        # check shape of kl cost
        assert outputs[0].shape == (), (
            f"Incorrect shape of outputs first element, expected () and got "
            f"{outputs[0].shape}"
        )

    return None
