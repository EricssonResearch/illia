# illia

## What is illia?

illia is a library focused on Bayesian Neural Networks with support for multiple backends. 
The main goal is to enable the use of backends such as PyTorch, Tensorflow, and Jax.

### Generating a Python Wheel

To create a Python wheel from the repository, execute the following command:

```python
python setup.py bdist_wheel
```

## Contribute

To start contributing to the repository, follow these steps:

1. **Clone the repository using SSH**: Ensure you have the permissions to clone the 
    repository via SSH.

2. **Update system packages and install `make`**: In some cases, you may need to update 
    your operating system packages and install `make`. Run the following commands:
    ```sh
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
    Poetry is configured to be used as a way to install the dependencies, and manage package versions, 
    needed for this repository without creating a new virtual environment.