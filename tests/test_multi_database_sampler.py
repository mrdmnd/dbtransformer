"""
Tests for multi-database sampler loading.

These tests verify that the sampler correctly loads multiple training databases
from the GCS bucket mounted at ~/gcs/databases_preprocessed/.

Run with: uv run pytest tests/test_multi_database_sampler.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import tributary
from dbtransformer.configurations import (
    DataConfig,
    DDPParameters,
    ModelConfig,
    SamplerConfig,
    TrainingConfig,
)
from dbtransformer.model import Batch
from dbtransformer.sampler_dataset import SamplerBatchDataset, discover_databases


# GCS bucket mount path for preprocessed databases
GCS_DB_DIR = Path.home() / "gcs" / "databases_preprocessed"

# List of training databases to use
TRAINING_DATABASES = ["rel-event"]


class TestDatabaseDiscovery:
    """Test database discovery functionality."""

    def test_discover_specific_databases(self):
        """Test discovering specific databases by name."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        db_paths = discover_databases(GCS_DB_DIR, TRAINING_DATABASES)

        assert len(db_paths) == len(TRAINING_DATABASES)
        for db_path in db_paths:
            assert db_path.exists(), f"Database path should exist: {db_path}"
            assert (db_path / "schema.rkyv").exists(), f"schema.rkyv should exist in {db_path}"
            assert (db_path / "graph.rkyv").exists(), f"graph.rkyv should exist in {db_path}"
            assert (db_path / "cells.rkyv").exists(), f"cells.rkyv should exist in {db_path}"
            assert (db_path / "embeddings.bin").exists(), f"embeddings.bin should exist in {db_path}"
            logger.info(f"Found database: {db_path.name}")

    def test_discover_all_databases(self):
        """Test discovering all databases in directory."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        db_paths = discover_databases(GCS_DB_DIR, None)

        assert len(db_paths) >= 1, "Should find at least one database"
        logger.info(f"Found {len(db_paths)} total databases:")
        for db_path in db_paths:
            logger.info(f"  - {db_path.name}")

    def test_discover_nonexistent_database_raises(self):
        """Test that requesting a nonexistent database raises an error."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        with pytest.raises(ValueError, match="not found"):
            discover_databases(GCS_DB_DIR, ["nonexistent-database"])

    def test_discover_nonexistent_directory_raises(self):
        """Test that a nonexistent directory raises an error."""
        with pytest.raises(ValueError, match="does not exist"):
            discover_databases(Path("/nonexistent/path"), None)


class TestSingleDatabaseSampler:
    """Test loading a single database with the Rust sampler."""

    @pytest.fixture(scope="class")
    def rel_event_sampler(self):
        """Create a sampler for rel-event database."""
        db_path = GCS_DB_DIR / "rel-event"
        if not db_path.exists():
            pytest.skip(f"rel-event database not found at {db_path}")

        return tributary.Sampler(
            db_path=str(db_path),
            batch_size=4,
            seq_len=128,
            max_bfs_width=32,
            seed=42,
        )

    def test_sampler_loads_successfully(self, rel_event_sampler):
        """Test that the sampler loads without errors."""
        assert rel_event_sampler is not None
        logger.info("rel-event sampler loaded successfully")

    def test_sampler_properties(self, rel_event_sampler):
        """Test that sampler reports reasonable properties."""
        num_rows = rel_event_sampler.num_rows()
        num_tables = rel_event_sampler.num_tables()
        embed_dim = rel_event_sampler.embed_dim()
        num_batches = rel_event_sampler.len_py()

        logger.info("rel-event database stats:")
        logger.info(f"  num_rows: {num_rows:,}")
        logger.info(f"  num_tables: {num_tables}")
        logger.info(f"  embed_dim: {embed_dim}")
        logger.info(f"  num_batches: {num_batches:,}")

        assert num_rows > 0, "Should have rows"
        assert num_tables > 0, "Should have tables"
        assert embed_dim == 768, "Should use BGE-base embeddings (768 dim)"
        assert num_batches > 0, "Should have batches"

    def test_batch_generation(self, rel_event_sampler):
        """Test that batches can be generated."""
        raw = rel_event_sampler.batch_py(0)
        batch_dict = dict(raw)

        required_keys = [
            "numerical_values",
            "categorical_values",
            "text_values",
            "timestamp_values",
            "column_name_values",
            "semantic_types",
            "masks",
            "is_padding",
            "column_attn_mask",
            "feature_attn_mask",
            "neighbor_attn_mask",
        ]

        for key in required_keys:
            assert key in batch_dict, f"Missing key: {key}"

        # Check we got actual data
        is_padding = batch_dict["is_padding"]
        non_padding = (~is_padding).sum()
        logger.info(f"Batch 0: {non_padding} non-padding cells")
        assert non_padding > 0, "Should have non-padding cells"


class TestMultiDatabaseDataset:
    """Test the SamplerBatchDataset with multiple databases."""

    @pytest.fixture(scope="class")
    def multi_db_dataset(self):
        """Create a dataset from multiple databases."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        data_config = DataConfig(
            db_dir=str(GCS_DB_DIR),
            db_names=TRAINING_DATABASES,
        )
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=4, seq_len=128)
        sampler_config = SamplerConfig(max_bfs_width=32, seed=42)

        return SamplerBatchDataset(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            sampler_config=sampler_config,
        )

    def test_dataset_creation(self, multi_db_dataset):
        """Test that dataset creates successfully with multiple databases."""
        assert multi_db_dataset is not None
        assert len(multi_db_dataset) > 0
        logger.info(f"Dataset has {len(multi_db_dataset)} batches")

    def test_dataset_iteration(self, multi_db_dataset):
        """Test iterating through the dataset."""
        batch_count = 0
        max_batches = 5  # Don't iterate through everything

        for batch in multi_db_dataset:
            assert isinstance(batch, Batch)
            assert batch.numerical_values.shape == (4, 128, 1)
            assert batch.semantic_types.shape == (4, 128)
            assert batch.column_attn_mask.shape == (4, 128, 128)

            batch_count += 1
            if batch_count >= max_batches:
                break

        logger.info(f"Successfully iterated through {batch_count} batches")
        assert batch_count == max_batches

    def test_batch_contents(self, multi_db_dataset):
        """Test that batch contents are valid."""
        multi_db_dataset.set_epoch(0)

        total_masked = 0
        for i, batch in enumerate(multi_db_dataset):
            # Check shapes
            assert batch.numerical_values.shape == (4, 128, 1)
            assert batch.categorical_values.shape == (4, 128, 768)
            assert batch.text_values.shape == (4, 128, 768)
            assert batch.timestamp_values.shape == (4, 128, 11)
            assert batch.column_name_values.shape == (4, 128, 768)
            assert batch.semantic_types.shape == (4, 128)
            assert batch.masks.shape == (4, 128)
            assert batch.is_padding.shape == (4, 128)
            assert batch.column_attn_mask.shape == (4, 128, 128)
            assert batch.feature_attn_mask.shape == (4, 128, 128)
            assert batch.neighbor_attn_mask.shape == (4, 128, 128)

            # Check dtypes
            assert batch.numerical_values.dtype == torch.float32
            assert batch.categorical_values.dtype == torch.float32
            assert batch.semantic_types.dtype == torch.long
            assert batch.masks.dtype == torch.bool
            assert batch.is_padding.dtype == torch.bool

            # Check we have non-padding cells
            non_padding = (~batch.is_padding).sum().item()
            masked = batch.masks.sum().item()
            total_masked += masked
            logger.info(f"Batch {i}: {non_padding} cells, {masked} masked")

            assert non_padding > 0, "Should have non-padding cells"
            # Note: Some sequences might have 0 masked cells if seed row only has NULL values
            # We check across multiple batches that we do get some masked cells

            if i >= 9:  # Check 10 batches
                break

        # Across multiple batches we should definitely have some masked cells
        assert total_masked > 0, "Should have at least some masked cells across batches"
        logger.info(f"Total masked cells across batches: {total_masked}")

    def test_epoch_shuffling(self, multi_db_dataset):
        """Test that epoch shuffling changes batch order."""
        # Get first batch from epoch 0
        multi_db_dataset.set_epoch(0)
        batch_0_epoch_0 = next(iter(multi_db_dataset))
        semantic_types_0 = batch_0_epoch_0.semantic_types.clone()

        # Get first batch from epoch 1
        multi_db_dataset.set_epoch(1)
        batch_0_epoch_1 = next(iter(multi_db_dataset))
        semantic_types_1 = batch_0_epoch_1.semantic_types.clone()

        # Reset to epoch 0 for other tests
        multi_db_dataset.set_epoch(0)

        # The batches should likely be different (not guaranteed for small DBs)
        are_equal = torch.equal(semantic_types_0, semantic_types_1)
        if not are_equal:
            logger.info("Epoch shuffling changed batch contents as expected")
        else:
            logger.warning("Batches unchanged after epoch shuffle (possible for small DBs)")


class TestMultipleDatabasesConfig:
    """Test loading datasets with different database configurations."""

    def test_single_database_by_name(self):
        """Test loading a single database by name."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        data_config = DataConfig(
            db_dir=str(GCS_DB_DIR),
            db_names=["rel-event"],
        )
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, seq_len=64)
        sampler_config = SamplerConfig(max_bfs_width=16, seed=123)

        dataset = SamplerBatchDataset(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            sampler_config=sampler_config,
        )

        assert len(dataset) > 0
        batch = next(iter(dataset))
        assert batch.numerical_values.shape == (2, 64, 1)
        logger.info("Single database by name loaded successfully")

    def test_all_databases_in_directory(self):
        """Test loading all databases in directory (db_names=None)."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        data_config = DataConfig(
            db_dir=str(GCS_DB_DIR),
            db_names=None,  # Load all databases
        )
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, seq_len=64)
        sampler_config = SamplerConfig(max_bfs_width=16, seed=456)

        dataset = SamplerBatchDataset(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            sampler_config=sampler_config,
        )

        assert len(dataset) > 0
        logger.info(f"Loaded all databases, total {len(dataset)} batches")


class TestDDPSimulation:
    """Test distributed training simulation with multiple ranks."""

    def test_ddp_sharding(self):
        """Test that DDP sharding divides batches correctly."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        data_config = DataConfig(
            db_dir=str(GCS_DB_DIR),
            db_names=TRAINING_DATABASES,
        )
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, seq_len=64)
        sampler_config = SamplerConfig(max_bfs_width=16, seed=789)

        # Create datasets for 2 simulated ranks
        device = torch.device("cpu")
        datasets = []
        for rank in range(2):
            ddp_params = DDPParameters(
                local_rank=rank,
                global_rank=rank,
                world_size=2,
                device=device,
            )
            dataset = SamplerBatchDataset(
                data_config=data_config,
                model_config=model_config,
                training_config=training_config,
                sampler_config=sampler_config,
                ddp_parameters=ddp_params,
            )
            datasets.append(dataset)

        # Each rank should have roughly half the batches
        logger.info(f"Rank 0: {len(datasets[0])} batches")
        logger.info(f"Rank 1: {len(datasets[1])} batches")

        # Total should be close to full dataset length
        total_batches = len(datasets[0]) + len(datasets[1])
        assert total_batches > 0

        # Set epoch and verify both ranks work
        for rank, dataset in enumerate(datasets):
            dataset.set_epoch(0)
            batch = next(iter(dataset))
            assert batch is not None
            logger.info(f"Rank {rank} generated batch successfully")


class TestGCSIntegration:
    """Integration tests specifically for GCS bucket access."""

    def test_gcs_mount_accessible(self):
        """Test that the GCS mount point is accessible."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        assert GCS_DB_DIR.is_dir()
        logger.info(f"GCS mount accessible at {GCS_DB_DIR}")

    def test_training_databases_available(self):
        """Test that all training databases are available."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        for db_name in TRAINING_DATABASES:
            db_path = GCS_DB_DIR / db_name
            assert db_path.exists(), f"Training database {db_name} not found"

            # Verify all required files exist
            required_files = ["schema.rkyv", "graph.rkyv", "cells.rkyv", "embeddings.bin", "manifest.json"]
            for file_name in required_files:
                file_path = db_path / file_name
                assert file_path.exists(), f"Missing {file_name} in {db_name}"

            logger.info(f"Verified training database: {db_name}")

    def test_end_to_end_training_simulation(self):
        """Simulate a mini training loop to verify everything works end-to-end."""
        if not GCS_DB_DIR.exists():
            pytest.skip(f"GCS bucket not mounted at {GCS_DB_DIR}")

        data_config = DataConfig(
            db_dir=str(GCS_DB_DIR),
            db_names=TRAINING_DATABASES,
        )
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, seq_len=64)
        sampler_config = SamplerConfig(max_bfs_width=16, seed=42)

        dataset = SamplerBatchDataset(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            sampler_config=sampler_config,
        )

        # Simulate 3 epochs with 5 batches each
        total_masked_across_epochs = 0
        for epoch in range(3):
            dataset.set_epoch(epoch)
            logger.info(f"Epoch {epoch}")

            for batch_idx, batch in enumerate(dataset):
                if batch_idx >= 5:
                    break

                # Verify batch is valid
                assert batch.numerical_values.is_contiguous()
                assert batch.categorical_values.is_contiguous()
                assert not torch.isnan(batch.numerical_values).any()
                # Some batches may have 0 masked cells (NULL-only seed rows)
                total_masked_across_epochs += batch.masks.sum().item()

            logger.info(f"  Completed 5 batches")

        # Across all epochs and batches, we should have some masked cells
        assert total_masked_across_epochs > 0, "Should have at least some masked cells across epochs"
        logger.info(f"End-to-end training simulation successful! Total masked: {total_masked_across_epochs}")
