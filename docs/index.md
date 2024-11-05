# illia

### What is illia?

illia is a library focused on Bayesian Neural Networks with support for multiple backends. 
The main goal is to enable the use of backends such as PyTorch, Tensorflow, and Jax.

### Generating a Python Wheel

To create a Python wheel from the repository, execute the following command:

```python
python setup.py bdist_wheel
```

### Some useful commands

Currently the documentation will be done with MkDocs, to start the service locally run the
Next command:

```bash
poetry run mkdocs serve
```

For more information on how to set up MkDocs [visit the following page](https://mkdocstrings.github.io/usage/).

### Contribute

To start contributing to the repository, follow these steps:

1. **Clone the repository using SSH**: Ensure you have the permissions to clone the 
    repository via SSH.

2. **Update system packages and install `make`**: In some cases, you may need to update 
    your operating system packages and install `make`. Run the following commands:

    ```bash
    sudo apt-get update
    sudo apt-get install build-essential
    ```

3. **Navigate to the project folder and run `make`**: Once in the project directory, 
    execute the following command:
    ```sh
    make
    ```
    
4. **Install dependencies using Poetry**: Create a virtual environment with Python >=3.10 and <3.12
    and then run:
    ```sh
    make install
    ```