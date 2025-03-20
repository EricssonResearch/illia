# standard libraries
from typing import Optional, Union

# 3pp
import tensorflow as tf
import pytest

# own modules
from illia.tf.nn import Conv2d
from illia.tf.distributions import Distribution, GaussianDistribution

# parametrized testing functions
conv_test_keys = [
    "input_channels",
    "output_channels",
    "kernel_size",
    "stride",
    "padding",
    "data_format",
    "dilation_rate",
    "groups",
    "expected_input_shape",
    "expected_output_shape",
    "batch_size",
    "height",
    "width",
    "weights_distribution",
    "bias_distribution",
]
combos_ids = ["normal", "normal2", "normal3"]
COMBOS_WOMBOS = (
    {
        "input_channels": 4,
        "output_channels": 5,  # filters
        "kernel_size": 2,
        "stride": 1,
        "padding": "valid",
        "data_format": "NHWC",
        "dilation_rate": 1,
        "groups": 1,
        "expected_input_shape": (3, 5, 5, 4),
        "expected_output_shape": (3, 4, 4, 5),
        "batch_size": 3,
        "height": 5,
        "width": 5,
        "weights_distribution": None,
        "bias_distribution": None,
    },
    {
        "input_channels": 4,
        "output_channels": 6,
        "kernel_size": 2,
        "stride": 1,
        "padding": "same",
        "data_format": "NHWC",
        "dilation_rate": (2, 2),
        "groups": 2,
        "expected_input_shape": (3, 4, 4, 4),
        "expected_output_shape": (3, 4, 4, 6),
        "batch_size": 3,
        "height": 4,
        "width": 4,
        "weights_distribution": None,
        "bias_distribution": None,
    },
    {
        "input_channels": 4,
        "output_channels": 6,
        "kernel_size": (2, 2),
        "stride": (2, 1),
        "padding": "valid",
        "data_format": "NHWC",
        "dilation_rate": (1, 1),
        "groups": 2,
        "expected_input_shape": (3, 5, 5, 4),
        "expected_output_shape": (3, 2, 4, 6),
        "batch_size": 3,
        "height": 5,
        "width": 5,
        "weights_distribution": None,
        "bias_distribution": None,
    },
)


@pytest.mark.order(1)
@pytest.mark.parametrize(conv_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS])
def test_conv_init(
    output_channels: int,
    input_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    data_format: str,
    dilation_rate: Union[int, tuple[int, int]],
    groups: int,
    expected_input_shape: tuple,
    expected_output_shape: tuple,
    batch_size: int,
    height: int,
    width: int,
    weights_distribution,
    bias_distribution,
) -> None:
    """
    Pytest for correct initialization of the layer.
    Args: initial convolutions args, defined in fixture.
    """
    # Dynamic args passing.
    original_model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation_rate,
        groups,
        weights_distribution,
        bias_distribution,
        data_format,
    )

    # check parameters length (of distribution)
    len_parameters: int = len(original_model.trainable_variables)
    assert (
        len_parameters == 4
    ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    # https://www.tensorflow.org/api_docs/python/tf/function
    model_scripted = tf.function(original_model, jit_compile=True)


@pytest.mark.order(2)
@pytest.mark.parametrize(conv_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS])
def test_conv_freeze(
    output_channels: int,
    input_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    data_format: str,
    dilation_rate: Union[int, tuple[int, int]],
    groups: int,
    expected_input_shape: tuple,
    expected_output_shape: tuple,
    batch_size: int,
    height: int,
    width: int,
    weights_distribution,
    bias_distribution,
) -> None:
    model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation_rate,
        groups,
        weights_distribution,
        bias_distribution,
        data_format,
    )

    input = tf.random.stateless_uniform(expected_input_shape, seed=(0, 0))

    output_1 = model(input)
    output_2 = model(input)

    # model_scripted = tf.function(model, jit_compile=True)

    assert output_2.numpy() != pytest.approx(
        output_1.numpy(), rel=1e-6, nan_ok=False
    ), """While weights not frozen. \
        The output difference wrt second fordwards is smaller than relative tolerance of 1e-6"""

    model.freeze()

    output_1 = model(input)
    output_2 = model(input)

    # model_scripted = tf.function(model, jit_compile=True)

    assert output_2.numpy() == pytest.approx(
        output_1.numpy(), rel=1e-6, nan_ok=False
    ), """
        While weights frozen. \
        The output difference wrt second fordwards is larger than relative tolerance of 1e-6
        """


@pytest.mark.order(3)
@pytest.mark.parametrize(conv_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS])
def test_conv_forward(
    output_channels: int,
    input_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    data_format: str,
    dilation_rate: Union[int, tuple[int, int]],
    groups: int,
    expected_input_shape: tuple,
    expected_output_shape: tuple,
    batch_size: int,
    height: int,
    width: int,
    weights_distribution,
    bias_distribution,
) -> None:
    model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation_rate,
        groups,
        weights_distribution,
        bias_distribution,
        data_format,
    )

    input = tf.random.stateless_uniform(expected_input_shape, seed=(0, 0))

    model.freeze()
    output = model(input)
    assert output.shape == expected_output_shape, "Incorrect output shape"

    model_scripted = tf.function(model, jit_compile=True)
    compiled_output = model_scripted(input)
    assert output.numpy() == pytest.approx(
        compiled_output.numpy(), rel=1e-8, nan_ok=False
    ), """The output difference wrt compiled version is larger than relative tolerance of 1e-6"""


@pytest.mark.order(4)
@pytest.mark.parametrize(conv_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS])
def test_conv_kl_loss(
    output_channels: int,
    input_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    data_format: str,
    dilation_rate: Union[int, tuple[int, int]],
    groups: int,
    expected_input_shape: tuple,
    expected_output_shape: tuple,
    batch_size: int,
    height: int,
    width: int,
    weights_distribution,
    bias_distribution,
) -> None:
    model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation_rate,
        groups,
        weights_distribution,
        bias_distribution,
        data_format,
    )

    input = tf.random.stateless_uniform(expected_input_shape, seed=(0, 0))

    # compute outputs
    outputs: tuple[tf.Tensor, int] = model.kl_cost()

    # check type of output
    assert isinstance(
        outputs, tuple
    ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

    # check type of kl cost
    assert isinstance(outputs[0], tf.Tensor), (
        f"Incorrect output type in the first element, expected {tf.Tensor} and "
        f"got {type(outputs[0])}"
    )

    # check type of num params
    # work around 'tensorflow.python.framework.ops.EagerTensor'
    nparams = outputs[1]
    assert isinstance(
        nparams, int
    ), f"Incorrect output type in the second element, expected {int}"

    # check shape of kl cost
    assert outputs[0].shape == (), (
        f"Incorrect shape of outputs first element, expected () and got "
        f"{outputs[0].shape}"
    )


@pytest.mark.order(5)
@pytest.mark.parametrize(conv_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS])
def test_conv_backward(
    output_channels: int,
    input_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    data_format: str,
    dilation_rate: Union[int, tuple[int, int]],
    groups: int,
    expected_input_shape: tuple,
    expected_output_shape: tuple,
    batch_size: int,
    height: int,
    width: int,
    weights_distribution,
    bias_distribution,
) -> None:
    model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation_rate,
        groups,
        weights_distribution,
        bias_distribution,
        data_format,
    )

    inputs = tf.random.stateless_uniform(expected_input_shape, seed=(0, 0))
    inputs = tf.Variable(inputs)

    # No need for a scripted version in TensorFlow
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
