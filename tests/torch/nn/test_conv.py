# standard libraries
from typing import Optional, Union

# 3pp
import pytest
import torch

# own modules
from illia.torch.nn import Conv2d
from illia.torch.distributions import Distribution, GaussianDistribution


@pytest.mark.order(1)
@pytest.mark.parametrize(
    (
        "input_channels, output_channels, kernel_size, stride, padding,"
        "dilation, groups, weights_distribution, bias_distribution"
    ),
    [
        (3, 16, 3, 1, 0, 1, 1, None, None),
        (
            4,
            32,
            7,
            2,
            1,
            2,
            2,
            GaussianDistribution(((32, 4 // 2, 7, 7))),
            GaussianDistribution((32,)),
        ),
    ],
)
def test_conv_init(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
) -> None:
    """
    This function is the test for the Conv2d constructor.

    Args:
        input_channels: Number of channels in the input image.
        output_channels: Number of channels produced by the
            convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.
        weights_distribution: The distribution for the weights.
        bias_distribution: The distribution for the bias.

    Returns:
        None.
    """

    # define conv layer
    original_model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # iter over models
    for model in (original_model, model_scripted):
        # check parameters length
        len_parameters: int = len(list(model.parameters()))
        assert (
            len_parameters == 4
        ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    (
        "input_channels, output_channels, kernel_size, stride, padding,"
        "dilation, groups, weights_distribution, bias_distribution, "
        "batch_size, height, width"
    ),
    [
        (3, 16, 3, 1, 0, 1, 1, None, None, 64, 32, 32),
        (
            4,
            32,
            7,
            2,
            1,
            2,
            2,
            GaussianDistribution(((32, 4 // 2, 7, 7))),
            GaussianDistribution((32,)),
            32,
            16,
            16,
        ),
    ],
)
def test_conv_forward(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """
    This function is the test for the Conv2d forward.

    Args:
        input_channels: Number of channels in the input image.
        output_channels: Number of channels produced by the
            convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.
        weights_distribution: The distribution for the weights.
        bias_distribution: The distribution for the bias.
        batch_size: batch size to use in the tests.
        height: height of the inputs.
        width: width of the inputs.

    Returns:
        None.
    """

    # define conv layer
    original_model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs = torch.rand((batch_size, input_channels, height, width))

    # get correct outputs shape
    outputs_shape: tuple[int, int, int, int] = get_outputs_shape(
        batch_size,
        output_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    )

    # iter over models
    for model in (original_model, model_scripted):
        # check parameters length
        outputs: torch.Tensor = model(inputs)

        # check type of outputs
        assert isinstance(
            outputs, torch.Tensor
        ), f"Incorrect outputs class, expected {torch.Tensor} and got {type(outputs)}"

        # check outputs shape
        assert outputs.shape == outputs_shape, (
            f"Incorrect outputs shape, expected {outputs_shape} and got "
            f"{outputs.shape}"
        )

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    (
        "input_channels, output_channels, kernel_size, stride, padding,"
        "dilation, groups, weights_distribution, bias_distribution, "
        "batch_size, height, width"
    ),
    [
        (3, 16, 3, 1, 0, 1, 1, None, None, 64, 32, 32),
        (
            4,
            32,
            7,
            2,
            1,
            2,
            2,
            GaussianDistribution(((32, 4 // 2, 7, 7))),
            GaussianDistribution((32,)),
            32,
            16,
            16,
        ),
    ],
)
def test_conv_backward(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """
    This function is the test for the Conv2d backward.

    Args:
        input_channels: Number of channels in the input image.
        output_channels: Number of channels produced by the
            convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.
        weights_distribution: The distribution for the weights.
        bias_distribution: The distribution for the bias.
        batch_size: batch size to use in the tests.
        height: height of the inputs.
        width: width of the inputs.

    Returns:
        None.
    """

    # define conv layer
    original_model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs = torch.rand((batch_size, input_channels, height, width))

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
    (
        "input_channels, output_channels, kernel_size, stride, padding,"
        "dilation, groups, weights_distribution, bias_distribution, "
        "batch_size, height, width"
    ),
    [
        (3, 16, 3, 1, 0, 1, 1, None, None, 64, 32, 32),
        (
            4,
            32,
            7,
            2,
            1,
            2,
            2,
            GaussianDistribution(((32, 4 // 2, 7, 7))),
            GaussianDistribution((32,)),
            32,
            16,
            16,
        ),
    ],
)
def test_conv_freeze(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """
    This function is the test for the Conv2d freeze and unfreeze.

    Args:
        input_channels: Number of channels in the input image.
        output_channels: Number of channels produced by the
            convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.
        weights_distribution: The distribution for the weights.
        bias_distribution: The distribution for the bias.
        batch_size: batch size to use in the tests.
        height: height of the inputs.
        width: width of the inputs.

    Returns:
        None.
    """

    # define conv layer
    original_model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs = torch.rand((batch_size, input_channels, height, width))

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
    (
        "input_channels, output_channels, kernel_size, stride, padding,"
        "dilation, groups, weights_distribution, bias_distribution, "
        "batch_size, height, width"
    ),
    [
        (3, 16, 3, 1, 0, 1, 1, None, None, 64, 32, 32),
        (
            4,
            32,
            7,
            2,
            1,
            2,
            2,
            GaussianDistribution(((32, 4 // 2, 7, 7))),
            GaussianDistribution((32,)),
            32,
            16,
            16,
        ),
    ],
)
def test_conv_kl_cost(
    input_channels: int,
    output_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """
    This function is the test for the Conv2d KL cost.

    Args:
        input_channels: Number of channels in the input image.
        output_channels: Number of channels produced by the
            convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.
        weights_distribution: The distribution for the weights.
        bias_distribution: The distribution for the bias.
        batch_size: batch size to use in the tests.
        height: height of the inputs.
        width: width of the inputs.

    Returns:
        None.
    """

    # define conv layer
    original_model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
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


def get_outputs_shape(
    batch_size: int,
    output_channels: int,
    height: int,
    width: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
) -> tuple[int, int, int, int]:
    """
    This function returns the output shape

    Args:
        batch_size: batch size to use in the tests.
        output_channels: Number of channels produced by the
            convolution.
        height: height of the inputs.
        width: width of the inputs.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.

    Returns:
        batch size.
        output channels.
        outputs height.
        outputs width.
    """

    # expand ints to tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # get outputs height and width
    outputs_height: int = (
        height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    outputs_width: int = (
        width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    # define outputs shape
    outputs_shape: tuple[int, int, int, int] = (
        batch_size,
        output_channels,
        outputs_height,
        outputs_width,
    )

    return outputs_shape
