"""Test the sampler with the synthetic teams dataset.

Verify:
1. Batch structure is correct
2. Each sample has exactly 1 masked cell (the prediction target)
3. Masked cells are numerical (since score and win_rate are numerical)
"""

from pathlib import Path

import numpy as np

import tributary

DB_PATH = Path.home() / "data" / "databases_preprocessed" / "synthetic-teams"
BATCH_SIZE = 8  # Small for inspection
SEQ_LEN = 16  # Allow room for FK traversal
D_TEXT = 768

sampler = tributary.Sampler(
    db_path=str(DB_PATH),
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    max_bfs_width=16,
    seed=42,
    num_threads=1,
    split="train",
    train_frac=0.8,
    val_frac=0.1,
    split_seed=42,
)

print("=" * 70)
print("SAMPLER INFO")
print("=" * 70)
print(f"Embedding dim: {sampler.embed_dim()}")
print(f"Number of tables: {sampler.num_tables()}")
print(f"Number of seeds: {sampler.num_seeds()}")
print(f"Number of rows: {sampler.num_rows()}")
print(f"Number of batches: {sampler.len_py()}")

# Sample a batch
batch = dict(sampler.batch_py(0))

print("\n" + "=" * 70)
print("BATCH STRUCTURE")
print("=" * 70)
print(f"Keys: {sorted(batch.keys())}")

for key in sorted(batch.keys()):
    arr = np.array(batch[key])
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# Reshape to (batch_size, seq_len, ...)
NUMERICAL = 0
CATEGORICAL = 1
TEXT = 2
TIMESTAMP = 3
STYPE_NAMES = {0: "NUM", 1: "CAT", 2: "TEXT", 3: "TIME"}

masks = np.array(batch["masks"]).reshape(BATCH_SIZE, SEQ_LEN)
is_padding = np.array(batch["is_padding"]).reshape(BATCH_SIZE, SEQ_LEN)
stypes = np.array(batch["semantic_types"]).reshape(BATCH_SIZE, SEQ_LEN)
numerical_values = np.array(batch["numerical_values"]).reshape(BATCH_SIZE, SEQ_LEN)

print("\n" + "=" * 70)
print("SAMPLE ANALYSIS")
print("=" * 70)

for b in range(min(4, BATCH_SIZE)):
    active = ~is_padding[b]
    masked = masks[b] & active

    num_active = active.sum()
    num_numerical = ((stypes[b] == NUMERICAL) & active).sum()
    num_masked = masked.sum()

    masked_types = stypes[b][masked]

    print(f"\nSample {b}:")
    print(f"  Active cells: {num_active}")
    print(f"  Numerical cells: {num_numerical}")
    print(f"  Masked cells: {num_masked}")
    print(f"  Masked types: {[STYPE_NAMES[t] for t in masked_types]}")

    # Show the sequence
    print("  Sequence:")
    for s in range(SEQ_LEN):
        if is_padding[b, s]:
            continue
        stype = stypes[b, s]
        val = numerical_values[b, s]
        mask_str = " <<< MASKED" if masks[b, s] else ""
        print(f"    pos[{s:2d}] {STYPE_NAMES.get(stype, '?'):4s} val={val:8.4f}{mask_str}")

print("\n" + "=" * 70)
print("MASK VERIFICATION")
print("=" * 70)

# Check that:
# 1. Each sample has exactly 1 masked cell
# 2. Masked cells are numerical (since score and win_rate are numerical)

errors = 0
correct = 0
total_masked = 0
masked_type_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for batch_idx in range(sampler.len_py()):
    batch = dict(sampler.batch_py(batch_idx))
    masks_b = np.array(batch["masks"]).reshape(BATCH_SIZE, SEQ_LEN)
    is_padding_b = np.array(batch["is_padding"]).reshape(BATCH_SIZE, SEQ_LEN)
    stypes_b = np.array(batch["semantic_types"]).reshape(BATCH_SIZE, SEQ_LEN)

    for i in range(BATCH_SIZE):
        active = ~is_padding_b[i]
        masked = masks_b[i] & active
        num_masked = masked.sum()

        if num_masked == 1:
            correct += 1
            masked_type = stypes_b[i][masked][0]
            masked_type_counts[masked_type] = masked_type_counts.get(masked_type, 0) + 1
        elif num_masked == 0:
            # Some samples may not have a masked cell if they don't have a prediction target
            pass
        else:
            errors += 1
            if errors <= 5:
                print(f"  Batch {batch_idx}, Sample {i}: {num_masked} masked cells (expected 1)")

        total_masked += num_masked

total_samples = sampler.len_py() * BATCH_SIZE
print(f"\nResults: {correct} samples with exactly 1 masked cell")
print(f"         {total_samples - correct - errors} samples with 0 masked cells")
print(f"         {errors} samples with >1 masked cells (errors)")
print(f"Total masked cells: {total_masked}")

print("\nMasked cells by semantic type:")
for stype, count in sorted(masked_type_counts.items()):
    print(f"  {STYPE_NAMES.get(stype, f'type_{stype}')}: {count}")

# Prediction targets are score (numerical) and win_rate (numerical)
# So all masked cells should be numerical
if masked_type_counts.get(NUMERICAL, 0) == total_masked:
    print("\n✅ All masked cells are numerical (as expected for score/win_rate targets)!")
else:
    print("\n⚠️  Some masked cells are not numerical")
