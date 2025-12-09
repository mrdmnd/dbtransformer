"""
Download all RelBench datasets and tasks.
"""

# mypy: ignore-errors

import pooch
from loguru import logger
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names
from tqdm import tqdm

if __name__ == "__main__":
    logger.info("Downloading all RelBench datasets and tasks")

    cache_dir = f"{pooch.os_cache('relbench')}"
    logger.info(f"Cache: {cache_dir}")

    for dataset_name in tqdm(get_dataset_names(), colour="green"):
        logger.info(f"Downloading dataset: {dataset_name}")
        get_dataset(dataset_name, download=True)

        for task_name in get_task_names(dataset_name):
            logger.info(f"  Downloading task: {task_name} from dataset: {dataset_name}")
            get_task(dataset_name, task_name, download=True)
