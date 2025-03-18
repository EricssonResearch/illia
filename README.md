**Warning:**  

Illia is currently undergoing active development to achieve stability across all
frameworks and backends. The library is subject to ongoing changes.

## Introduction

Illia is an innovative library designed for Bayesian Neural Networks, offering support
for multiple backends. Our goal is to integrate seamlessly with popular deep learning
platforms:

- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Jax](https://jax.readthedocs.io/en/latest/index.html)

For graph neural networks, Illia aims to support:

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#)
- [Deep Graph Library](https://www.dgl.ai/)
- [Spektral](https://graphneural.network/)

## Quick start example

```python
# Import the Illia library
from illia.torch.nn.linear import Linear as TorchLinear
from illia.tf.nn.linear import Linear as TFLinear

# Create a linear layer using the PyTorch backend
linear_layer_torch = TorchLinear(
    input_size=3,
    output_size=3,
)

# Create a linear layer using the TensorFlow backend
linear_layer_tf = TFLinear(
    input_size=3,
    output_size=3,
)
```

Explore further usage examples in the backend-specific documentation.

## Contributing

We welcome contributions! To streamline the setup process, follow these steps using our
convenient Makefile:

1. **Clone the repository:**  
   Update your system and install `make` (commands may vary based on your OS or
   distribution). For Linux:

   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   ```

2. **Set up your environment:**  
   Navigate to the cloned repository, create a Python environment, activate it, and
   install dependencies:

   ```bash
   make install-all
   ```

3. **Build a Python wheel:**  
   To package the repository, execute:

   ```bash
   python setup.py bdist_wheel
   ```

## License

This project is distributed under the
[MIT License](https://github.com/EricssonResearch/illia/blob/main/LICENSE). Enjoy
exploring and using Illia!
