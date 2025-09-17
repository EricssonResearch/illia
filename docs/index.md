<p align="center">
  <img src="./assets/images/white_logo_illia.png" class="logo-white" height="200" width="200"/>
  <img src="./assets/images/black_logo_illia.png" class="logo-black" height="200" width="200"/>
  <br />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue" alt="Python versions">
  <a href="https://github.com/EricssonResearch/illia/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/EricssonResearch/illia" alt="License">
  </a>
  <a href="https://github.com/EricssonResearch/illia/releases/latest">
    <img src="https://img.shields.io/github/release-date/EricssonResearch/illia?display_date=published_at" alt="Last Release">
  </a>
  <a href="https://github.com/EricssonResearch/illia/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/EricssonResearch/illia.svg">
  </a>
</p>

!!! warning

    The library is evolving rapidly to ensure stable support across all
    frameworks and backends. Expect ongoing changes as we improve
    functionality and performance.

## Introduction

**illia** is a cutting-edge library for **Bayesian Neural Networks** that brings
uncertainty quantification to deep learning. Designed with flexibility in mind, it
seamlessly integrates with multiple backends and popular frameworks.

For full documentation, please visit the site:
[https://ericssonresearch.github.io/illia/](https://ericssonresearch.github.io/illia/)

## Why Choose illia?

- **Multi-Backend Support**: Works with PyTorch, TensorFlow, and JAX.
- **Graph Neural Networks**: Currently integrated with PyTorch Geometric, with planned
  support for DGL and/or Spektral in future releases.
- **Developer Friendly**: Intuitive API design and comprehensive documentation.

## Quick Start

Get started with illia in just a few lines of code:

```python
import os
import torch

# Configure backend (PyTorch is default)
os.environ["ILLIA_BACKEND"] = "torch"

import illia
from illia.nn import Conv2d

# Create a Bayesian convolutional layer
conv_layer = Conv2d(
    input_channels=3,
    output_channels=64,
    kernel_size=3,
    bias=True
)

# Forward pass with uncertainty
input_tensor = torch.rand(1, 3, 32, 32)
output_mean, output_std = conv_layer(input_tensor)

print(f"Output shape: {output_mean.shape}")
print(f"Uncertainty quantified: {output_std.mean():.4f}")
```

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features,
or improving documentation:

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
