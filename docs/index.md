# Welcome to illia

!!! warning

    This library is still in early development and there could be breaking in the future.

### What is illia?

illia is a library focused on Bayesian Neural Networks with support for multiple backends. 

### Availables backends

On the deep learning platforms side, illia has the goal to support:

+ [PyTorch](https://pytorch.org/).
+ [Tensorflow](https://www.tensorflow.org/).
+ [Jax](https://jax.readthedocs.io/en/latest/index.html).

For the use case of graph neural networks, illia has the goal to support:

+ [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#).
+ [Deep Graph Library](https://www.dgl.ai/).
+ [Spektral](https://graphneural.network/).

### Example

```python
# Include the library
from illia.nn import Linear

# Define the layer with torch backend
linear_layer_torch = Linear(
    input_size=3,
    output_size=3,
    backend="torch",
)

# Define the layer with tensorflow backend
linear_layer_tf = Linear(
    input_size=3,
    output_size=3,
    backend="tf",
)
```

A more extensive usage can be found in the [Package Reference](./distributions/distributions.md).

### Contributing

To simplify the process, we have created a Makefile that allows for a quick and easy installation. Follow these steps:

1. Clone the current repository, update your system packages, and install `make`. On Linux, though it may vary by operating system or distribution, the commands are typically:

    ```bash
    sudo apt-get update
    sudo apt-get install build-essential
    ```

2. Once the repository has been downloaded locally, navigate to its location. Ensure you have created and activated a Python environment. Then, install all the required dependencies using the following command:

    ```bash
    make install-all
    ```

To create a Python wheel from the repository, execute the following command:

```python
python setup.py bdist_wheel
```

### License

This project is licensed under the terms of the [MIT license](https://github.com/EricssonResearch/illia/blob/main/LICENSE).