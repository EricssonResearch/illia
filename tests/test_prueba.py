import pytest
import logging
import os
from logging.handlers import RotatingFileHandler

# Configura el logger
os.makedirs(".", exist_ok=True)
log_file = os.path.join(".", f"illia_tests.log")

# Set up rotating file handler (max 1MB per file, keep 5 backups)
handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def initialize_model(backend):
    if backend == 'tf':
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    elif backend == 'torch':
        import torch
        import torch.nn as nn
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleModel()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return model

@pytest.fixture(params=['tf', 'torch'])
def backend(request):
    current_backend = request.param
    logger.info(f"Running test with backend: {current_backend}")
    return current_backend

def test_model_initialization(backend):
    model = initialize_model(backend)
    assert model is not None
    logger.info(f"Model initialized successfully with backend: {backend}")

@pytest.mark.tf
def test_tf_specific_feature():
    import tensorflow as tf
    logger.info("Running TensorFlow-specific test")
    assert tf.__version__ is not None

@pytest.mark.torch
def test_torch_specific_feature():
    import torch
    logger.info("Running PyTorch-specific test")
    assert torch.__version__ is not None

# Para ejecutar las pruebas, puedes hacer:
# pytest -m tf  # para ejecutar solo las pruebas de TensorFlow
# pytest -m torch  # para ejecutar solo las pruebas de PyTorch
# pytest  # para ejecutar todas las pruebas
