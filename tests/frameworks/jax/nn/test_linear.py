# 3pp
import jax

# own modules
from illia.nn.jax import Linear


def test_linear_call() -> None:
    # get sample
    model: Linear = Linear(16, 32)
    input_key = jax.random.key(0)
    inputs: jax.Array = jax.random.normal(input_key, (64, 16))
    outputs: jax.Array = model(inputs)

    assert isinstance(outputs, jax.Array), "output is not a jax array"

    return None
