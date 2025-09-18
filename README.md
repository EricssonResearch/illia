<p align="center">
  <img src="./docs/assets/images/white_logo_illia.png" height="250"/>
  <br/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue" alt="Python versions">
  <a href="https://github.com/EricssonResearch/illia/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/EricssonResearch/illia" alt="License">
  </a>
  <a href="https://github.com/EricssonResearch/illia/releases/latest">
    <img src="https://img.shields.io/github/release-date/EricssonResearch/illia?display_date=published_at" alt="Last Release">
  </a>
  <a href="https://github.com/EricssonResearch/illia/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/EricssonResearch/illia.svg"></a>
</p>

> [!WARNING]
>
> **illia is under active development.** The library is evolving rapidly to ensure
> stable support across all frameworks. Expect ongoing changes as we improve
> functionality and performance.

## Introduction

**illia** is a library for **Bayesian Neural Networks** that brings uncertainty
quantification to deep learning, a capability that is critical in sectors such as
telecommunications, medicine, and beyond. Designed with flexibility in mind, it
integrates seamlessly with multiple backends and popular frameworks, enabling a single
codebase to support multiple backends with minimal modifications.

For full documentation, please visit the site:
[https://ericssonresearch.github.io/illia/](https://ericssonresearch.github.io/illia/)

## Why Choose illia?

- **Multi-Backend Support**: Works with PyTorch, TensorFlow, and JAX.
- **Graph Neural Networks**: Currently integrated with PyTorch Geometric, with planned
  support for DGL and/or Spektral in future releases.
- **Developer Friendly**: Intuitive API design and comprehensive documentation.

## Quick Start

To show how easy it is to use **illia**, here’s a quick example to get started. In this
case, we explicitly choose the backend PyTorch, the underlying framework, and define a
convolutional layer:

```python
import os
import torch

# Configure backend (PyTorch is default)
os.environ["ILLIA_BACKEND"] = "torch"

import illia
from illia.nn import Conv2d

# Create a Bayesian convolutional layer
conv_layer = Conv2d(
    input_channels=1,
    output_channels=1,
    kernel_size=3,
)

# Define input tensor
input_tensor = torch.rand(1, 1, 4, 4)

# Define the number of iterations to apply the forward pass
num_passes = 10
outputs = [conv_layer(input_tensor) for _ in range(num_passes)]

# Stack outputs into a single tensor
outputs = torch.stack(outputs)

print(f"Output shape: {outputs.shape}")
print(f"Output std: {outputs.std()}")
print(f"Output var: {outputs.var()}")
```

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding
features, or improving documentation:

1. **Read our
   [contributing guide](https://github.com/EricssonResearch/illia/blob/main/CONTRIBUTING.md)**
   for development setup.
2. **Check [open issues](https://github.com/EricssonResearch/illia/issues)** for ways to
   help.
3. **Submit bug reports** using our issue templates.

## License

illia is released under the
[MIT License](https://github.com/EricssonResearch/illia/blob/main/LICENSE). We hope you
find it useful and inspiring for your projects!
