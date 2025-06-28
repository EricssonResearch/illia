<p align="center">
  <img src="./docs/assets/images/white_logo_illia.png" height="250"/>
  <br/>
</p>

<p align="center">
  <a href="https://github.com/EricssonResearch/illia/actions/workflows/workflow.yml"><img src="https://github.com/EricssonResearch/illia/actions/workflows/workflow.yml/badge.svg"></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue">
  <a href="https://github.com/EricssonResearch/illia/blob/main/LICENSE" target="_blank">
      <img src="https://img.shields.io/github/license/EricssonResearch/illia" alt="License">
  </a>
</p>

### ⚠️ **Warning**

**Illia is under active development.** The library is evolving rapidly to ensure stable
support across all frameworks and backends. Expect ongoing changes as we improve
functionality and performance.

## Introduction

**Illia** is an innovative library for **Bayesian Neural Networks**, designed to support
multiple backends and integrate seamlessly with popular deep learning ecosystems such as
**PyTorch**, **TensorFlow**, and **JAX**, as well as graph neural network libraries
including **PyTorch Geometric**, **Deep Graph Library (DGL)**, and **Spektral**.

For full documentation, please visit the site:
[https://ericssonresearch.github.io/illia/](https://ericssonresearch.github.io/illia/)

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
