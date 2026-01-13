"""Create a synthetic two-moons dataset for testing the pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

# Set seed for reproducibility
np.random.seed(42)

# Generate 10000 samples using sklearn's make_moons
n_samples = 10000
noise = 0.25  # Higher noise for a harder classification task

X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

# Create a simple row_id as primary key
row_id = np.arange(n_samples)

# Convert labels to boolean strings for categorical treatment
# Class 0 -> "False", Class 1 -> "True"
label = np.where(y == 0, "False", "True")

# Create DataFrame
df = pd.DataFrame({
    "row_id": row_id,
    "x1": X[:, 0],
    "x2": X[:, 1],
    "label": label,
})

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print("\nFirst 10 rows:")
print(df.head(10))

# Quick stats
print(f"\nFeature ranges:")
print(f"  x1: [{df['x1'].min():.3f}, {df['x1'].max():.3f}]")
print(f"  x2: [{df['x2'].min():.3f}, {df['x2'].max():.3f}]")

# Save to parquet
output_dir = Path.home() / "data" / "databases_raw" / "synthetic-moons" / "db"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "moons_data.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")
