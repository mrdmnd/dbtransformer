"""Test the sampler on the synthetic XOR dataset."""

import numpy as np

import tributary

BATCH_SIZE = 4
SEQ_LEN = 16
D_TEXT = 768

# Create sampler for the XOR dataset
sampler = tributary.Sampler(
    db_path="/home/mrdmnd/data/databases_preprocessed/synthetic-xor",
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    max_bfs_width=256,
    seed=42,
    num_threads=1,
    split=None,  # Use all data
    train_frac=0.8,
    val_frac=0.1,
    split_seed=12345,
)

print(f"Number of batches: {sampler.len_py()}")
print(f"Embedding dim: {sampler.embed_dim()}")
print(f"Number of seeds (rows): {sampler.num_seeds()}")
print(f"Number of rows: {sampler.num_rows()}")

# Get a batch
batch = dict(sampler.batch_py(0))

print("\n" + "=" * 80)
print("BATCH CONTENTS (raw shapes)")
print("=" * 80)

for key in sorted(batch.keys()):
    arr = np.array(batch[key])
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# Reshape arrays properly
masks = np.array(batch["masks"]).reshape(BATCH_SIZE, SEQ_LEN)
is_padding = np.array(batch["is_padding"]).reshape(BATCH_SIZE, SEQ_LEN)
semantic_types = np.array(batch["semantic_types"]).reshape(BATCH_SIZE, SEQ_LEN)
numerical_values = np.array(batch["numerical_values"]).reshape(BATCH_SIZE, SEQ_LEN)
categorical_values = np.array(batch["categorical_values"]).reshape(BATCH_SIZE, SEQ_LEN, D_TEXT)

print("\n" + "=" * 80)
print("RESHAPED DATA")
print("=" * 80)
print(f"\nmasks (shape={masks.shape}):")
print(masks)
print(f"\nis_padding (shape={is_padding.shape}):")
print(is_padding)
print(f"\nsemantic_types (shape={semantic_types.shape}):")
print(semantic_types)

# Semantic types: 0=numerical, 1=categorical, 2=text, 3=timestamp
NUMERICAL = 0
CATEGORICAL = 1
TEXT = 2
TIMESTAMP = 3

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nSemantic type counts per sample:")
for i in range(BATCH_SIZE):
    active = ~is_padding[i]
    num_active = active.sum()
    num_numerical = ((semantic_types[i] == NUMERICAL) & active).sum()
    num_categorical = ((semantic_types[i] == CATEGORICAL) & active).sum()
    num_masked = (masks[i] & active).sum()

    # What semantic types are masked?
    masked_types = semantic_types[i][masks[i] & active]

    print(
        f"  Sample {i}: {num_active} active cells, "
        f"{num_numerical} numerical, {num_categorical} categorical, "
        f"{num_masked} masked (types: {masked_types})"
    )

print("\nChecking that only CATEGORICAL cells are masked (prediction_targets=['color']):")
all_correct = True
for i in range(BATCH_SIZE):
    active = ~is_padding[i]
    masked_non_cat = masks[i] & active & (semantic_types[i] != CATEGORICAL)
    if masked_non_cat.any():
        all_correct = False
        print(f"  ❌ Sample {i}: Non-categorical cells are masked!")
        print(f"     Masked positions: {np.where(masks[i] & active)[0]}")
        print(f"     Types at masked positions: {semantic_types[i][masks[i] & active]}")
    else:
        masked_cat = masks[i] & active & (semantic_types[i] == CATEGORICAL)
        if masked_cat.any():
            print(f"  ✓ Sample {i}: Only categorical cells masked ({masked_cat.sum()} cells)")
        else:
            print(f"  - Sample {i}: No cells masked")

if all_correct:
    print("\n✅ All samples have correct masking!")
else:
    print("\n❌ Some samples have incorrect masking!")

# Check numerical values (x and y should be z-scored, so mean~0, std~1)
print("\nNumerical value stats (x, y are z-scored, so should be mean~0, std~1):")
for i in range(BATCH_SIZE):
    active = ~is_padding[i]
    num_mask = (semantic_types[i] == NUMERICAL) & active
    num_vals = numerical_values[i, num_mask]
    if len(num_vals) > 0:
        print(f"  Sample {i}: values={num_vals}")

# Print a more readable sample
print("\n" + "=" * 80)
print("SAMPLE VISUALIZATION (first batch element)")
print("=" * 80)
print("Position | Type       | Masked | Padding | Value")
print("-" * 60)
TYPE_NAMES = {0: "NUMERICAL", 1: "CATEGORICAL", 2: "TEXT", 3: "TIMESTAMP"}
for pos in range(SEQ_LEN):
    stype = semantic_types[0, pos]
    masked = masks[0, pos]
    padding = is_padding[0, pos]

    if padding:
        val = "(padding)"
    elif stype == NUMERICAL:
        val = f"{numerical_values[0, pos]:.4f}"
    elif stype == CATEGORICAL:
        # Categorical is an embedding, just show if it's nonzero
        emb = categorical_values[0, pos]
        val = f"embedding (norm={np.linalg.norm(emb):.3f})"
    else:
        val = "N/A"

    print(
        f"  {pos:3d}    | {TYPE_NAMES.get(stype, '?'):11s} | {'Y' if masked else 'N':6s} | {'Y' if padding else 'N':7s} | {val}"
    )

# Check multiple batches to see the pattern
print("\n" + "=" * 80)
print("CHECKING MULTIPLE BATCHES")
print("=" * 80)

errors = 0
correct = 0
for batch_idx in range(min(100, sampler.len_py())):
    batch = dict(sampler.batch_py(batch_idx))
    masks_b = np.array(batch["masks"]).reshape(BATCH_SIZE, SEQ_LEN)
    is_padding_b = np.array(batch["is_padding"]).reshape(BATCH_SIZE, SEQ_LEN)
    semantic_types_b = np.array(batch["semantic_types"]).reshape(BATCH_SIZE, SEQ_LEN)

    for i in range(BATCH_SIZE):
        active = ~is_padding_b[i]
        masked = masks_b[i] & active
        masked_types = semantic_types_b[i][masked]

        # All masked cells should be categorical
        if (masked_types != CATEGORICAL).any():
            errors += 1
            if errors <= 5:
                print(f"  Batch {batch_idx}, Sample {i}: Masked types = {masked_types}")
        else:
            correct += 1

print(f"\nResults: {correct} correct, {errors} errors (out of {min(100, sampler.len_py()) * BATCH_SIZE} samples)")
