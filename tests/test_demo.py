import pytest
import torch
from jaxtyping import Float, jaxtyped
from loguru import logger
from torch import Tensor


@pytest.fixture(scope="session")
@jaxtyped(typechecker=None)
def setup() -> Float[Tensor, "m n"]:
    # Some demo fixture for reusable data or whatever
    result = torch.randn(10, 12)
    logger.info("Setting up")
    return result


def test_basic_true() -> None:
    logger.info("Hello from pytest basic true")
    assert True
