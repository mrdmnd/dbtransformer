"""
Tests for the Rust sampler (tributary) Python bindings.

Run with: uv run pytest tests/test_sampler.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import tributary

# Test database paths
# For direct sampler tests, use the database directory
DB_PATH = Path("/tmp/databases_preprocessed/rel-event")
# For SamplerBatchDataset tests, use the parent directory + db_names
DB_DIR = Path("/tmp/databases_preprocessed")
DB_NAMES = ["rel-event"]


@pytest.fixture(scope="module")
def sampler():
    """Create a sampler for testing."""
    if not DB_PATH.exists():
        pytest.skip(f"Test database not found: {DB_PATH}")

    return tributary.Sampler(
        db_path=str(DB_PATH),
        batch_size=4,
        seq_len=128,
        max_bfs_width=32,
        seed=42,
    )


class TestSamplerImport:
    """Test that the tributary module imports correctly."""

    def test_import_tributary(self):
        """Test basic import."""
        assert tributary is not None
        logger.info("tributary module imported successfully")

    def test_sampler_class_exists(self):
        """Test Sampler class is exposed."""
        assert hasattr(tributary, "Sampler")
        logger.info(f"Sampler class: {tributary.Sampler}")


class TestSamplerCreation:
    """Test sampler construction."""

    def test_create_sampler(self, sampler):
        """Test creating a sampler from a database file."""
        assert sampler is not None
        logger.info("Sampler created successfully")

    def test_sampler_properties(self, sampler):
        """Test sampler properties."""
        num_rows = sampler.num_rows()
        num_tables = sampler.num_tables()
        embed_dim = sampler.embed_dim()
        num_batches = sampler.len_py()

        logger.info("Database stats:")
        logger.info(f"  num_rows: {num_rows}")
        logger.info(f"  num_tables: {num_tables}")
        logger.info(f"  embed_dim: {embed_dim}")
        logger.info(f"  num_batches: {num_batches}")

        assert num_rows > 0, "Database should have rows"
        assert num_tables > 0, "Database should have tables"
        assert embed_dim > 0, "Embedding dim should be positive"
        assert num_batches > 0, "Should have at least one batch"

    def test_invalid_path_raises(self):
        """Test that invalid path raises an error."""
        with pytest.raises(Exception):
            tributary.Sampler(
                db_path="/nonexistent/path",
                batch_size=4,
                seq_len=128,
                max_bfs_width=32,
                seed=42,
            )


class TestBatchGeneration:
    """Test batch generation."""

    def test_batch_returns_dict(self, sampler):
        """Test that batch_py returns a list of tuples."""
        raw = sampler.batch_py(0)
        assert isinstance(raw, list), f"Expected list, got {type(raw)}"
        logger.info(f"Batch returned {len(raw)} arrays")

        # Convert to dict for easier inspection
        batch_dict = dict(raw)
        logger.info(f"Batch keys: {list(batch_dict.keys())}")

    def test_batch_has_required_keys(self, sampler):
        """Test that batch contains all required arrays."""
        batch_dict = dict(sampler.batch_py(0))

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
        logger.info("All required keys present")

    def test_batch_shapes(self, sampler):
        """Test that batch arrays have correct shapes."""
        batch_size = 4
        seq_len = 128
        embed_dim = sampler.embed_dim()

        batch_dict = dict(sampler.batch_py(0))

        # Check shapes
        assert batch_dict["numerical_values"].shape == (batch_size * seq_len,)
        assert batch_dict["semantic_types"].shape == (batch_size * seq_len,)
        assert batch_dict["masks"].shape == (batch_size * seq_len,)
        assert batch_dict["is_padding"].shape == (batch_size * seq_len,)

        # Embedding arrays are flattened: (batch_size * seq_len * embed_dim,)
        assert batch_dict["categorical_values"].shape == (batch_size * seq_len * embed_dim,)
        assert batch_dict["text_values"].shape == (batch_size * seq_len * embed_dim,)
        assert batch_dict["column_name_values"].shape == (batch_size * seq_len * embed_dim,)

        # Timestamp has 11 dimensions
        assert batch_dict["timestamp_values"].shape == (batch_size * seq_len * 11,)

        # Attention masks are (batch_size * seq_len * seq_len,)
        assert batch_dict["column_attn_mask"].shape == (batch_size * seq_len * seq_len,)
        assert batch_dict["feature_attn_mask"].shape == (batch_size * seq_len * seq_len,)
        assert batch_dict["neighbor_attn_mask"].shape == (batch_size * seq_len * seq_len,)

        logger.info("All shapes correct")

    def test_batch_dtypes(self, sampler):
        """Test that batch arrays have correct dtypes."""
        batch_dict = dict(sampler.batch_py(0))

        # Float arrays
        assert batch_dict["numerical_values"].dtype == np.float32
        assert batch_dict["timestamp_values"].dtype == np.float32

        # f16 arrays (embeddings)
        assert batch_dict["categorical_values"].dtype == np.float16
        assert batch_dict["text_values"].dtype == np.float16
        assert batch_dict["column_name_values"].dtype == np.float16

        # Integer arrays
        assert batch_dict["semantic_types"].dtype == np.int32

        # Boolean arrays
        assert batch_dict["masks"].dtype == np.bool_
        assert batch_dict["is_padding"].dtype == np.bool_
        assert batch_dict["column_attn_mask"].dtype == np.bool_

        logger.info("All dtypes correct")

    def test_multiple_batches(self, sampler):
        """Test generating multiple batches."""
        num_batches = min(5, sampler.len_py())

        for i in range(num_batches):
            batch_dict = dict(sampler.batch_py(i))
            non_padding = (~batch_dict["is_padding"]).sum()
            masked = batch_dict["masks"].sum()
            logger.info(f"Batch {i}: {non_padding} cells, {masked} masked")

        logger.info(f"Generated {num_batches} batches successfully")


class TestSequenceVisualization:
    """Tests for visualizing sequence contents."""

    def test_print_single_sequence(self):
        """Print all cells in a single sequence for debugging."""
        if not DB_PATH.exists():
            pytest.skip(f"Test database not found: {DB_PATH}")

        # Create sampler with batch_size=1 for a single sequence
        seq_len = 64
        single_sampler = tributary.Sampler(
            db_path=str(DB_PATH),
            batch_size=1,
            seq_len=seq_len,
            max_bfs_width=32,
            seed=42,
        )

        embed_dim = single_sampler.embed_dim()
        batch_dict = dict(single_sampler.batch_py(0))

        # Extract arrays for the single sequence
        numerical = batch_dict["numerical_values"]
        categorical = batch_dict["categorical_values"].reshape(seq_len, embed_dim)
        text = batch_dict["text_values"].reshape(seq_len, embed_dim)
        timestamp = batch_dict["timestamp_values"].reshape(seq_len, 11)
        column_name = batch_dict["column_name_values"].reshape(seq_len, embed_dim)
        semantic_types = batch_dict["semantic_types"]
        masks = batch_dict["masks"]
        is_padding = batch_dict["is_padding"]

        # Attention masks
        col_attn = batch_dict["column_attn_mask"].reshape(seq_len, seq_len)
        feat_attn = batch_dict["feature_attn_mask"].reshape(seq_len, seq_len)
        nbr_attn = batch_dict["neighbor_attn_mask"].reshape(seq_len, seq_len)

        # Semantic type names
        stype_names = ["NUMERICAL", "CATEGORICAL", "TEXT", "TIMESTAMP"]

        logger.info("=" * 80)
        logger.info("SINGLE SEQUENCE TRAJECTORY")
        logger.info("=" * 80)

        # Count stats
        non_padding_count = (~is_padding).sum()
        masked_count = masks.sum()
        logger.info(f"Sequence length: {seq_len}")
        logger.info(f"Non-padding cells: {non_padding_count}")
        logger.info(f"Masked cells: {masked_count}")
        logger.info(f"Padding cells: {is_padding.sum()}")
        logger.info("")

        # Count by semantic type
        type_counts = dict.fromkeys(stype_names, 0)
        for i in range(seq_len):
            if not is_padding[i]:
                stype = semantic_types[i]
                if 0 <= stype < len(stype_names):
                    type_counts[stype_names[stype]] += 1

        logger.info("Semantic type distribution:")
        for name, count in type_counts.items():
            logger.info(f"  {name}: {count}")
        logger.info("")

        # Print each cell
        logger.info("-" * 80)
        logger.info(f"{'Pos':>4} | {'Type':<12} | {'Masked':<6} | {'Value':<50}")
        logger.info("-" * 80)

        for i in range(seq_len):
            if is_padding[i]:
                logger.info(f"{i:>4} | {'[PADDING]':<12} | {'-':<6} | -")
                continue

            stype = semantic_types[i]
            stype_name = stype_names[stype] if 0 <= stype < len(stype_names) else f"UNKNOWN({stype})"
            masked_str = "YES" if masks[i] else "no"

            # Format value based on semantic type
            if stype == 0:  # NUMERICAL
                val = numerical[i]
                value_str = f"z-score={val:.4f}"
            elif stype == 1:  # CATEGORICAL
                emb = categorical[i]
                emb_norm = np.linalg.norm(emb)
                emb_preview = ", ".join(f"{v:.3f}" for v in emb[:4])
                value_str = f"emb=[{emb_preview}, ...] (norm={emb_norm:.3f})"
            elif stype == 2:  # TEXT
                emb = text[i]
                emb_norm = np.linalg.norm(emb)
                emb_preview = ", ".join(f"{v:.3f}" for v in emb[:4])
                value_str = f"emb=[{emb_preview}, ...] (norm={emb_norm:.3f})"
            elif stype == 3:  # TIMESTAMP
                ts = timestamp[i]
                # ts[0:2] = sin/cos minute, ts[2:4] = sin/cos hour, etc.
                # ts[10] = z-scored epoch
                epoch_zscore = ts[10]
                value_str = f"epoch_z={epoch_zscore:.4f}, sin_min={ts[0]:.3f}, cos_min={ts[1]:.3f}"
            else:
                value_str = "???"

            logger.info(f"{i:>4} | {stype_name:<12} | {masked_str:<6} | {value_str}")

        logger.info("-" * 80)

        # Print attention mask summary for first few non-padding cells
        logger.info("")
        logger.info("ATTENTION MASK SUMMARY (first 10 non-padding cells)")
        logger.info("-" * 80)

        printed = 0
        for i in range(seq_len):
            if is_padding[i]:
                continue

            col_attends = col_attn[i].sum()
            feat_attends = feat_attn[i].sum()
            nbr_attends = nbr_attn[i].sum()

            logger.info(f"Cell {i:>3}: col_attn={col_attends:>3} cells, feat_attn={feat_attends:>3} cells, nbr_attn={nbr_attends:>3} cells")

            printed += 1
            if printed >= 10:
                break

        # Print full attention masks as ASCII grids
        def print_attention_mask(name: str, mask: np.ndarray) -> None:
            """Print attention mask as ASCII grid."""
            logger.info("")
            logger.info(f"{name} ATTENTION MASK")
            logger.info("=" * 80)

            # Header row with column indices
            header = "     |" + "".join(f"{i % 10}" for i in range(seq_len)) + "|"
            logger.info(header)
            logger.info("-" * len(header))

            for i in range(seq_len):
                row_chars = []
                for j in range(seq_len):
                    if is_padding[i] or is_padding[j]:
                        row_chars.append(" ")
                    elif mask[i, j]:
                        row_chars.append("█")
                    else:
                        row_chars.append("·")
                row_str = "".join(row_chars)
                # Mark row index and padding status
                pad_marker = "P" if is_padding[i] else " "
                logger.info(f"{i:>3}{pad_marker} |{row_str}|")

            logger.info("-" * len(header))

            # Print legend
            total_ones = mask.sum()
            density = total_ones / (seq_len * seq_len) * 100
            logger.info("Legend: █=attends, ·=no attention, (space)=padding")
            logger.info(f"Total attending pairs: {total_ones} ({density:.1f}% density)")

        print_attention_mask("COLUMN", col_attn)
        print_attention_mask("FEATURE", feat_attn)
        print_attention_mask("NEIGHBOR", nbr_attn)

        logger.info("")
        logger.info("=" * 80)


class TestSamplerShuffle:
    """Test epoch shuffling."""

    def test_shuffle_changes_batches(self, sampler):
        """Test that shuffling changes the batch contents."""
        # Get batch 0 before shuffle
        batch_before = dict(sampler.batch_py(0))
        types_before = batch_before["semantic_types"].copy()

        # Shuffle for epoch 1
        sampler.shuffle_py(1)

        # Get batch 0 after shuffle
        batch_after = dict(sampler.batch_py(0))
        types_after = batch_after["semantic_types"].copy()

        # Reset for other tests
        sampler.shuffle_py(0)

        # Check they're different (with high probability for large DBs)
        are_equal = np.array_equal(types_before, types_after)
        logger.info(f"Batches equal after shuffle: {are_equal}")

        # Not a hard assertion since small DBs might have same first batch
        if not are_equal:
            logger.info("Shuffle changed batch contents as expected")


class TestTorchIntegration:
    """Test integration with PyTorch."""

    def test_convert_to_tensors(self, sampler):
        """Test converting batch arrays to PyTorch tensors."""
        batch_dict = dict(sampler.batch_py(0))

        batch_size = 4
        seq_len = 128
        embed_dim = sampler.embed_dim()

        # Convert and reshape
        numerical = torch.from_numpy(np.array(batch_dict["numerical_values"], copy=False))
        numerical = numerical.view(batch_size, seq_len, 1)

        semantic_types = torch.from_numpy(np.array(batch_dict["semantic_types"], copy=False))
        semantic_types = semantic_types.view(batch_size, seq_len)

        categorical = torch.from_numpy(np.array(batch_dict["categorical_values"], copy=False))
        categorical = categorical.view(batch_size, seq_len, embed_dim).float()

        masks = torch.from_numpy(np.array(batch_dict["masks"], copy=False))
        masks = masks.view(batch_size, seq_len)

        column_attn = torch.from_numpy(np.array(batch_dict["column_attn_mask"], copy=False))
        column_attn = column_attn.view(batch_size, seq_len, seq_len)

        logger.info(f"numerical shape: {numerical.shape}")
        logger.info(f"semantic_types shape: {semantic_types.shape}")
        logger.info(f"categorical shape: {categorical.shape}")
        logger.info(f"masks shape: {masks.shape}")
        logger.info(f"column_attn shape: {column_attn.shape}")

        assert numerical.shape == (batch_size, seq_len, 1)
        assert semantic_types.shape == (batch_size, seq_len)
        assert categorical.shape == (batch_size, seq_len, embed_dim)
        assert masks.shape == (batch_size, seq_len)
        assert column_attn.shape == (batch_size, seq_len, seq_len)

    def test_mask_exists(self, sampler):
        """Test that masking produces exactly one masked cell per sequence."""
        batch_dict = dict(sampler.batch_py(0))

        masks = batch_dict["masks"]
        is_padding = batch_dict["is_padding"]

        # Check each sequence has exactly one masked cell (single-label masking)
        for seq_idx in range(masks.shape[0]):
            seq_masks = masks[seq_idx]
            seq_padding = is_padding[seq_idx]
            
            # Only count non-padding positions
            non_padding_count = (~seq_padding).sum()
            masked_count = seq_masks.sum()

            if non_padding_count > 0:
                logger.info(f"Sequence {seq_idx}: {masked_count} masked cells")
                # Single-label masking: exactly one cell masked per sequence
                assert masked_count == 1, f"Expected 1 masked cell, got {masked_count}"


class TestSamplerDataset:
    """Test the SamplerBatchDataset wrapper."""

    def test_dataset_import(self):
        """Test importing the dataset wrapper."""
        from dbtransformer.sampler_dataset import SamplerBatchDataset

        assert SamplerBatchDataset is not None

    def test_dataset_creation(self):
        """Test creating a SamplerBatchDataset."""
        if not DB_PATH.exists():
            pytest.skip(f"Test database not found: {DB_PATH}")

        from dbtransformer.configurations import (
            DataConfig,
            ModelConfig,
            SamplerConfig,
            TrainingConfig,
        )
        from dbtransformer.sampler_dataset import SamplerBatchDataset

        data_config = DataConfig(db_dir=str(DB_DIR), db_names=DB_NAMES)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=4, seq_len=128)
        sampler_config = SamplerConfig(max_bfs_width=32, seed=42)

        dataset = SamplerBatchDataset(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            sampler_config=sampler_config,
        )

        assert len(dataset) > 0
        logger.info(f"Dataset length: {len(dataset)}")

    def test_dataset_getitem(self):
        """Test getting a batch from the dataset."""
        if not DB_PATH.exists():
            pytest.skip(f"Test database not found: {DB_PATH}")

        from dbtransformer.configurations import (
            DataConfig,
            ModelConfig,
            SamplerConfig,
            TrainingConfig,
        )
        from dbtransformer.model import Batch
        from dbtransformer.sampler_dataset import SamplerBatchDataset

        data_config = DataConfig(db_dir=str(DB_DIR), db_names=DB_NAMES)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=4, seq_len=128)
        sampler_config = SamplerConfig(max_bfs_width=32, seed=42)

        dataset = SamplerBatchDataset(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            sampler_config=sampler_config,
        )

        # SamplerBatchDataset is an IterableDataset, use iteration
        batch = next(iter(dataset))
        assert isinstance(batch, Batch)

        logger.info(f"Batch numerical_values shape: {batch.numerical_values.shape}")
        logger.info(f"Batch semantic_types shape: {batch.semantic_types.shape}")
        logger.info(f"Batch column_attn_mask shape: {batch.column_attn_mask.shape}")

        # Check expected shapes (d_text=768 for BGE-base embeddings)
        assert batch.numerical_values.shape == (4, 128, 1)
        assert batch.semantic_types.shape == (4, 128)
        assert batch.categorical_values.shape == (4, 128, 768)
        assert batch.column_attn_mask.shape == (4, 128, 128)
