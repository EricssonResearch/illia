# Home

<p align="center">
  <img src="./assets/images/white_logo_illia.png" class="logo-white" height="200" width="200"/>
  <img src="./assets/images/black_logo_illia.png" class="logo-black" height="200" width="200"/>
  <br />
</p>

!!! warning

    Illia is currently undergoing active development to achieve stability across all
    frameworks and backends. The library is subject to ongoing changes.

## Introduction

**Illia** is an innovative library for **Bayesian Neural Networks**, designed to support
multiple backends and integrate seamlessly with popular deep learning ecosystems such as
**PyTorch**, **TensorFlow**, and **JAX**, as well as graph neural network libraries
including **PyTorch Geometric**, **Deep Graph Library (DGL)**, and **Spektral**.

## Quick start example

```python
# Import the Illia library
from illia.nn import Linear

# Create a linear layer using the PyTorch backend
linear_layer_torch = Linear(
    input_size=3,
    output_size=3,
)
```

Explore further usage examples in the backend-specific documentation.

## Contributing

We welcome contributions from the community! To get started quickly, follow these steps
using our streamlined `Makefile`:

1. **Clone the repository** Ensure your system is updated and `make` is installed. On
   most Linux systems:

   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   ```

2. **Set up your environment** Navigate to the cloned directory, create a Python
   environment, activate it, and run:

   ```bash
   make
   ```

   This command updates `pip`, installs `uv`, and sets up all project dependencies
   automatically.

## License

Illia is released under the
[MIT License](https://github.com/EricssonResearch/illia/blob/main/LICENSE). We hope you
find it useful and inspiring.
