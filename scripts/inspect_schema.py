"""Inspect the preprocessed schema to verify is_prediction_target flags."""

import tributary

# Load the database
sampler = tributary.Sampler(
    db_path="/home/mrdmnd/data/databases_preprocessed/synthetic-xor",
    batch_size=1,
    seq_len=16,
    max_bfs_width=256,
    seed=42,
    num_threads=1,
    split=None,
    train_frac=0.8,
    val_frac=0.1,
    split_seed=12345,
)

# Try to get schema info if available
print("Sampler info:")
print(f"  Embedding dim: {sampler.embed_dim()}")
print(f"  Number of seeds: {sampler.num_seeds()}")
print(f"  Number of rows: {sampler.num_rows()}")
print(f"  Number of batches: {sampler.len_py()}")

# Check if we can access database properties
# This might not be exposed via PyO3, but let's see what's available
print("\nSampler methods:")
for attr in dir(sampler):
    if not attr.startswith("_"):
        print(f"  {attr}")
