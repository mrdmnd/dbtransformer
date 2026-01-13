"""Create a synthetic dataset for testing numerical prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate 10000 samples
n_samples = 10000

# Input features: uniform in [-1, 1]
x1 = np.random.uniform(-1, 1, n_samples)
x2 = np.random.uniform(-1, 1, n_samples)

# Target: a simpler non-linear function
# y = x1^2 + x2^2 + noise (distance from origin squared)
# This is learnable but still requires non-linearity
noise_std = 0.05
y = x1**2 + x2**2 + np.random.normal(0, noise_std, n_samples)

# Create a simple row_id as primary key
row_id = np.arange(n_samples)

# Create DataFrame
df = pd.DataFrame({
    "row_id": row_id,
    "x1": x1,
    "x2": x2,
    "y": y,
})

print(f"Dataset shape: {df.shape}")
print("\nTarget stats:")
print(f"  y min: {y.min():.4f}")
print(f"  y max: {y.max():.4f}")
print(f"  y mean: {y.mean():.4f}")
print(f"  y std: {y.std():.4f}")
print("\nFirst 10 rows:")
print(df.head(10))

# Verify the function (without noise)
print("\nVerifying function (expected: x1^2 + x2^2):")
for i in range(5):
    xi1, xi2, yi = df.iloc[i][["x1", "x2", "y"]]
    expected = xi1**2 + xi2**2
    print(f"  x1={xi1:.3f}, x2={xi2:.3f} -> y={yi:.3f} (noiseless: {expected:.3f})")

# Save to parquet
output_dir = Path.home() / "data" / "databases_raw" / "synthetic-numerical" / "db"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "numerical_data.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")
