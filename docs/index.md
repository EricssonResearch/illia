<p align="center">
  <img src="./assets/images/white_logo_illia.png" class="logo-white" height="200" width="200"/>
  <img src="./assets/images/black_logo_illia.png" class="logo-black" height="200" width="200"/>
  <br />
</p>

<p align="center">
  <a href="https://github.com/EricssonResearch/illia/actions/workflows/workflow.yml"><img src="https://github.com/EricssonResearch/illia/actions/workflows/workflow.yml/badge.svg"></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue">
  <a href="https://github.com/EricssonResearch/illia/blob/main/LICENSE" target="_blank">
      <img src="https://img.shields.io/github/license/EricssonResearch/illia" alt="License">
  </a>
</p>

# Home

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
# Import standard 'os' library to change the environment variable
import os

# By default PyTorch is the backend
os.environ["ILLIA_BACKEND"] = "torch"

# Import the Illia library
import illia

print(f"Illia version: {illia.__version__}")
print(f"Illia backend: {illia.__get_backend__}")
print(f"Illia available backends: {illia.__get_available_backends__}")

# Create a convolutional layer using the PyTorch backend
conv_layer = Conv2D(
    input_channels=3,
    output_channels=64,
    kernel_size=3,
)

# Create a random tensor (B, C, H, W)
tensor = torch.rand((1, 3, 12, 12))

# Obtain the output
output = conv_layer(tensor)
output.shape
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
