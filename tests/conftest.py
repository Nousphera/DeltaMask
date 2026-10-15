"""Bounded CPU fixtures retain a complete paper architecture."""

import pytest
import torch
from deltamask.config import defaults

torch.set_num_threads(1)


@pytest.fixture
def image_check_config():
    return defaults() | {
        "dataset": "synthetic-images",
        "model": "dinov2-small",
        "pretrained": False,
        "num-clients": 2,
        "num-rounds": 2,
        "samples-per-client": 1,
        "batch-size": 1,
    }
