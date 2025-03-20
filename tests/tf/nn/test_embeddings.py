# standard libraries
from typing import Optional, Union

# 3pp
import tensorflow as tf
import pytest

# own modules
from illia.tf.nn import Embedding
from illia.tf.distributions import Distribution, GaussianDistribution

embedding_test_keys = [
    "num_embeddings",
    "embeddings_dim",
    "weights_distribution",
    "padding_idx",
    "max_norm",
    "norm_type",
    "scale_grad_by_freq",
    "sparse",
    "batch_size",
]
combos_ids = ["normal_default", "normal_distr"]
COMBOS_WOMBOS = (
    {
        "num_embeddings": 8,
        "embeddings_dim": 64,
        "weights_distribution": None,
        "padding_idx": None,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "sparse": False,
        "batch_size": 64,
    },
    {
        "num_embeddings": 10,
        "embeddings_dim": 32,
        "weights_distribution": GaussianDistribution((10, 32)),
        "padding_idx": 0,
        "max_norm": 2.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "sparse": False,
        "batch_size": 64,
    },
)


@pytest.mark.order(1)
@pytest.mark.parametrize(
    embedding_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
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
    batch_size: int,
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
    len_parameters: int = len(model.trainable_variables)
    assert (
        len_parameters == 2
    ), f"Incorrect parameters length, expected 2 and got {len_parameters}"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    embedding_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
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

    # get inputs
    inputs = tf.random.stateless_uniform(
        (batch_size,), minval=0, maxval=num_embeddings, dtype=tf.int32, seed=(0, 0)
    )

    # check parameters length
    outputs: tf.Tensor = model(inputs)

    # check type of outputs
    assert isinstance(
        outputs, tf.Tensor
    ), f"Incorrect outputs class, expected {tf.Tensor} and got {type(outputs)}"

    # check shape
    assert outputs.shape == (batch_size, embeddings_dim), (
        f"Incorrect outputs shape, expected {(batch_size, embeddings_dim)} and got "
        f"{outputs.shape}"
    )

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    embedding_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
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

    # get inputs
    inputs = tf.random.stateless_uniform(
        (batch_size,), minval=0, maxval=num_embeddings, dtype=tf.int32, seed=(0, 0)
    )

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


@pytest.mark.order(4)
@pytest.mark.parametrize(
    embedding_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
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

    # get inputs
    inputs = tf.random.stateless_uniform(
        (batch_size,), minval=0, maxval=num_embeddings, dtype=tf.int32, seed=(0, 0)
    )

    # compute outputs
    outputs_first: tf.Tensor = model(inputs)
    outputs_second: tf.Tensor = model(inputs)

    # check if both outputs are equal
    assert outputs_first.numpy() != pytest.approx(
        outputs_second.numpy(), rel=1e-8, nan_ok=False
    ), "Incorrect outputs, different forwards are equal when at the initialization the layer should be unfrozen"

    # freeze layer
    model.freeze()

    # compute outputs
    outputs_first = model(inputs)
    outputs_second = model(inputs)

    # check if both outputs are equal
    assert outputs_first.numpy() == pytest.approx(
        outputs_second.numpy(), rel=1e-8, nan_ok=False
    ), "Incorrect freezing, when layer is frozen outputs are not the same in different forward passes"

    # unfreeze layer
    model.unfreeze()

    # compute outputs
    outputs_first = model(inputs)
    outputs_second = model(inputs)

    assert outputs_first.numpy() != pytest.approx(
        outputs_second.numpy(), rel=1e-8, nan_ok=False
    ), "Incorrect unfreezing, when layer is unfrozen outputs are the same in different forward passes"

    return None


@pytest.mark.order(5)
@pytest.mark.parametrize(
    embedding_test_keys, [tuple(d.values()) for d in COMBOS_WOMBOS], ids=combos_ids
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

    # get inputs
    inputs = tf.random.stateless_uniform(
        (batch_size,), minval=0, maxval=num_embeddings, dtype=tf.int32, seed=(0, 0)
    )

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

    return None
