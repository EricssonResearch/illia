# standard libraries
from typing import Optional

# 3pp
import pytest
import torch

# own modules
from illia.torch.nn import Embedding
from illia.torch.distributions import Distribution, GaussianDistribution


@pytest.mark.order(1)
@pytest.mark.parametrize(
    (
        "num_embeddings, embeddings_dim, weights_distribution, padding_idx, max_norm, "
        "norm_type, scale_grad_by_freq, sparse"
    ),
    [
        (8, 64, None, None, None, 2.0, False, False),
        (10, 32, GaussianDistribution((10, 32)), 0, 2.0, 2.0, True, False),
    ],
)
def test_embedding_init(
    num_embeddings: int,
    embeddings_dim: int,
    weights_distribution: Optional[Distribution],
    padding_idx: Optional[int],
    max_norm: Optional[float],
    norm_type: float,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> None:
    """
    This function is the test for the constructor of the Embedding
    layer.

    Args:
        num_embeddings: size of the dictionary of embeddings.
        embeddings_dim: the size of each embedding vector.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient. Defaults to None.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Defaults to None.
        norm_type: The p of the p-norm to compute for the max_norm
            option. Defaults to 2.0.
        scale_grad_by_freq: If given, this will scale gradients by
            the inverse of frequency of the words in the
            mini-batch. Defaults to False.
        sparse: If True, gradient w.r.t. weight matrix will be a
            sparse tensor. Defaults to False.
        batch_size: batch size for the inputs.

    Returns:
        None.
    """

    # define embedding layer
    model: Embedding = Embedding(
        num_embeddings,
        embeddings_dim,
        weights_distribution,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )

    # check parameters length
    len_parameters: int = len(list(model.parameters()))
    assert (
        len_parameters == 2
    ), f"Incorrect parameters length, expected 2 and got {len_parameters}"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    (
        "num_embeddings, embeddings_dim, weights_distribution, padding_idx, max_norm, "
        "norm_type, scale_grad_by_freq, sparse, batch_size"
    ),
    [
        (8, 64, None, None, None, 2.0, False, False, 64),
        (10, 32, GaussianDistribution((10, 32)), 0, 2.0, 2.0, True, False, 128),
    ],
)
def test_embedding_forward(
    num_embeddings: int,
    embeddings_dim: int,
    weights_distribution: Optional[Distribution],
    padding_idx: Optional[int],
    max_norm: Optional[float],
    norm_type: float,
    scale_grad_by_freq: bool,
    sparse: bool,
    batch_size: int,
) -> None:
    """
    This function is the test for the forward pass of the Embedding
    layer.

    Args:
        num_embeddings: size of the dictionary of embeddings.
        embeddings_dim: the size of each embedding vector.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient. Defaults to None.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Defaults to None.
        norm_type: The p of the p-norm to compute for the max_norm
            option. Defaults to 2.0.
        scale_grad_by_freq: If given, this will scale gradients by
            the inverse of frequency of the words in the
            mini-batch. Defaults to False.
        sparse: If True, gradient w.r.t. weight matrix will be a
            sparse tensor. Defaults to False.
        batch_size: batch size for the inputs.

    Returns:
        None.
    """

    # define embedding layer
    original_model: Embedding = Embedding(
        num_embeddings,
        embeddings_dim,
        weights_distribution,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )

    # get scripted model
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs: torch.Tensor = torch.randint(0, num_embeddings, (batch_size,))

    # iter over models
    for model in (original_model, model_scripted):
        # compute outputs
        outputs: torch.Tensor = model(inputs)

        # check type
        assert isinstance(
            outputs, torch.Tensor
        ), f"Incorrect type of outputs, expected {torch.Tensor} and got {type(outputs)}"

        # check shape
        assert outputs.shape == (batch_size, embeddings_dim), (
            f"Incorrect outputs shape, expected {(batch_size, embeddings_dim)} and got "
            f"{outputs.shape}"
        )

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    (
        "num_embeddings, embeddings_dim, weights_distribution, padding_idx, max_norm, "
        "norm_type, scale_grad_by_freq, sparse, batch_size"
    ),
    [
        (8, 64, None, None, None, 2.0, False, False, 64),
        (10, 32, GaussianDistribution((10, 32)), 0, 2.0, 2.0, True, False, 128),
    ],
)
def test_embedding_backward(
    num_embeddings: int,
    embeddings_dim: int,
    weights_distribution: Optional[Distribution],
    padding_idx: Optional[int],
    max_norm: Optional[float],
    norm_type: float,
    scale_grad_by_freq: bool,
    sparse: bool,
    batch_size: int,
) -> None:
    """
    This function is the test for the backward pass of the Embedding
    layer.

    Args:
        num_embeddings: size of the dictionary of embeddings.
        embeddings_dim: the size of each embedding vector.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient. Defaults to None.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Defaults to None.
        norm_type: The p of the p-norm to compute for the max_norm
            option. Defaults to 2.0.
        scale_grad_by_freq: If given, this will scale gradients by
            the inverse of frequency of the words in the
            mini-batch. Defaults to False.
        sparse: If True, gradient w.r.t. weight matrix will be a
            sparse tensor. Defaults to False.
        batch_size: batch size for the inputs.

    Returns:
        None.
    """

    # define embedding layer
    original_model: Embedding = Embedding(
        num_embeddings,
        embeddings_dim,
        weights_distribution,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )

    # get scripted model
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs: torch.Tensor = torch.randint(0, num_embeddings, (batch_size,))

    # iter over models
    for model in (original_model, model_scripted):
        # compute outputs
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
        "num_embeddings, embeddings_dim, weights_distribution, padding_idx, max_norm, "
        "norm_type, scale_grad_by_freq, sparse, batch_size"
    ),
    [
        (8, 64, None, None, None, 2.0, False, False, 64),
        (10, 32, GaussianDistribution((10, 32)), 0, 2.0, 2.0, True, False, 128),
    ],
)
def test_embedding_freeze(
    num_embeddings: int,
    embeddings_dim: int,
    weights_distribution: Optional[Distribution],
    padding_idx: Optional[int],
    max_norm: Optional[float],
    norm_type: float,
    scale_grad_by_freq: bool,
    sparse: bool,
    batch_size: int,
) -> None:
    """
    This function is the test for the freeze and unfreeze layers from
    Embedding layer.

    Args:
        num_embeddings: size of the dictionary of embeddings.
        embeddings_dim: the size of each embedding vector.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient. Defaults to None.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Defaults to None.
        norm_type: The p of the p-norm to compute for the max_norm
            option. Defaults to 2.0.
        scale_grad_by_freq: If given, this will scale gradients by
            the inverse of frequency of the words in the
            mini-batch. Defaults to False.
        sparse: If True, gradient w.r.t. weight matrix will be a
            sparse tensor. Defaults to False.
        batch_size: batch size for the inputs.

    Returns:
        None.
    """

    # define embedding layer
    original_model: Embedding = Embedding(
        num_embeddings,
        embeddings_dim,
        weights_distribution,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )

    # get scripted version
    model_scripted = torch.jit.script(original_model)

    # get inputs
    inputs: torch.Tensor = torch.randint(0, num_embeddings, (batch_size,))

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
        "num_embeddings, embeddings_dim, weights_distribution, padding_idx, max_norm, "
        "norm_type, scale_grad_by_freq, sparse, batch_size"
    ),
    [
        (8, 64, None, None, None, 2.0, False, False, 64),
        (10, 32, GaussianDistribution((10, 32)), 0, 2.0, 2.0, True, False, 128),
    ],
)
def test_linear_kl_cost(
    num_embeddings: int,
    embeddings_dim: int,
    weights_distribution: Optional[Distribution],
    padding_idx: Optional[int],
    max_norm: Optional[float],
    norm_type: float,
    scale_grad_by_freq: bool,
    sparse: bool,
    batch_size: int,
) -> None:
    """
    This function is the test for the kl_cost method of Embedding layer.

    Args:
        num_embeddings: size of the dictionary of embeddings.
        embeddings_dim: the size of each embedding vector.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient. Defaults to None.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Defaults to None.
        norm_type: The p of the p-norm to compute for the max_norm
            option. Defaults to 2.0.
        scale_grad_by_freq: If given, this will scale gradients by
            the inverse of frequency of the words in the
            mini-batch. Defaults to False.
        sparse: If True, gradient w.r.t. weight matrix will be a
            sparse tensor. Defaults to False.
        batch_size: batch size for the inputs.

    Returns:
        None.
    """

    # define embedding layer
    original_model: Embedding = Embedding(
        num_embeddings,
        embeddings_dim,
        weights_distribution,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
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
