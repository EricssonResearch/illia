# Standard libraries
from typing import Optional, Union

# 3pp
import tensorflow as tf
import pytest

# Own modules
from illia.tf.nn import Linear
from illia.tf.distributions import GaussianDistribution


linear_test_keys = [
    "input_size",
    "output_size",
    "weights_distribution",
    "bias_distribution",
    "batch_size",
]
combos_ids = ["normal_default", "normal_distr"]
COMBOS_WOMBOS = (
    {
        "input_size": 30,
        "output_size": 20,
        "weights_distribution": None,
        "bias_distribution": None,
        "batch_size": 1,
    },
    {
        "input_size": 30,
        "output_size": 20,
        "weights_distribution": GaussianDistribution((30, 20)),
        "bias_distribution": GaussianDistribution((20,)),
        "batch_size": 1,
    },
)


@pytest.mark.order(1)
@pytest.mark.parametrize(
    linear_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
)
def test_linear_init(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[GaussianDistribution],
    bias_distribution: Optional[GaussianDistribution],
    batch_size: Optional[int],
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

    # Define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Check parameters length
    len_parameters: int = len(model.trainable_variables)
    assert (
        len_parameters == 4
    ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    linear_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
)
def test_linear_forward(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[GaussianDistribution],
    bias_distribution: Optional[GaussianDistribution],
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

    # Define linear layer
    original_model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Get scripted version
    model_scripted = tf.function(
        original_model,
    )
    # Get inputs
    inputs = tf.random.uniform((batch_size, input_size))

    # iter over models
    for model in (original_model, original_model):  # , model_scripted):
        # Check parameters length
        outputs: tf.Tensor = model(inputs)

        # Check type of outputs
        assert isinstance(
            outputs, tf.Tensor
        ), f"Incorrect outputs class, expected {tf.Tensor} and got {type(outputs)}"

        # Check outputs shape
        assert outputs.shape == (batch_size, output_size), (
            f"Incorrect outputs shape, expected {(batch_size, output_size)} and got "
            f"{outputs.shape}"
        )

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    linear_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
)
def test_linear_backward(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[GaussianDistribution],
    bias_distribution: Optional[GaussianDistribution],
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

    # Define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Get scripted version
    model_scripted = tf.function(
        model,
    )

    # Get inputs
    inputs = tf.random.uniform((batch_size, input_size))
    inputs = tf.Variable(inputs)
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        outputs = model(inputs)
        loss, nparams = model.kl_cost()

    # Compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Check gradients
    for name, grad in zip([v.name for v in model.trainable_variables], grads):
        assert (
            grad is not None
        ), f"Incorrect backward computation, gradient of {name} shouldn't be None"
        print(f"Gradient for {name}: {grad}")

    return None


@pytest.mark.order(4)
@pytest.mark.parametrize(
    linear_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
)
def test_linear_freeze(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[GaussianDistribution],
    bias_distribution: Optional[GaussianDistribution],
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

    # Define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Get scripted version
    model_scripted = tf.function(
        model,
    )

    # Get inputs
    inputs = tf.random.uniform((batch_size, input_size))

    # Compute outputs
    outputs_first: tf.Tensor = model(inputs)
    outputs_second: tf.Tensor = model(inputs)

    # Check if both outputs are equal
    assert outputs_first.numpy() != pytest.approx(
        outputs_second.numpy(), rel=1e-8, nan_ok=False
    ), "Incorrect outputs, different forwards are equal when at the initialization the layer should be unfrozen"

    # Freeze layer
    model.freeze()

    # Compute outputs
    outputs_first = model(inputs)
    outputs_second = model(inputs)

    # Check if both outputs are equal
    assert outputs_first.numpy() == pytest.approx(
        outputs_second.numpy(), rel=1e-8, nan_ok=False
    ), "Incorrect freezing, when layer is frozen outputs are not the same in different forward passes"

    # Unfreeze layer
    model.unfreeze()

    # Compute outputs
    outputs_first = model(inputs)
    outputs_second = model(inputs)

    assert outputs_first.numpy() != pytest.approx(
        outputs_second.numpy(), rel=1e-8, nan_ok=False
    ), "Incorrect unfreezing, when layer is unfrozen outputs are the same in different forward passes"

    return None


@pytest.mark.order(5)
@pytest.mark.parametrize(
    linear_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
)
def test_linear_kl_cost(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[GaussianDistribution],
    bias_distribution: Optional[GaussianDistribution],
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

    # Define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Get scripted version
    # Compile_jit = True #TODO
    model_scripted = tf.function(model)

    # Compute outputs
    outputs: tuple[tf.Tensor, int] = model.kl_cost()

    # Check type of output
    assert isinstance(
        outputs, tuple
    ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

    # Check type of kl cost
    assert isinstance(outputs[0], tf.Tensor), (
        f"Incorrect output type in the first element, expected {tf.Tensor} and "
        f"got {type(outputs[0])}"
    )

    # Work around 'tensorflow.python.framework.ops.EagerTensor'
    nparams = outputs[1]
    assert isinstance(nparams, int), (
        f"Incorrect output type in the second element, expected {int} and "
        f"got {type(outputs[0])}"
    )

    # Check shape of kl cost
    assert outputs[0].shape == (), (
        f"Incorrect shape of outputs first element, expected () and got "
        f"{outputs[0].shape}"
    )

    return None
