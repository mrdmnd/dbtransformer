import time

import numpy as np
import torch
import tqdm
from beartype import beartype
from jaxtyping import Float, jaxtyped
from loguru import logger
from torch import Tensor


# Enable runtime type checking for this function as a demo
@jaxtyped(typechecker=beartype)
def checked_matmul(a: Float[Tensor, "m n"], b: Float[Tensor, "n p"]) -> Float[Tensor, "m p"]:
    return torch.matmul(a, b)


def main() -> None:
    logger.info("Hello from dbtransformer!")

    n1 = np.random.default_rng().normal(size=(10, 12))
    logger.info(f"n1: {n1.shape}")

    t1 = torch.randn(10, 12)
    t2 = torch.randn(12, 14)
    result = checked_matmul(t1, t2)
    logger.success(f"result: {result}")

    t3 = torch.randn(5, 14)
    try:
        checked_matmul(t1, t3)
    except Exception as e:
        logger.error(f"Caught expected error due to shape mismatch:\n{e}")

    for _ in tqdm.tqdm(range(100)):
        time.sleep(0.01)


if __name__ == "__main__":
    main()
