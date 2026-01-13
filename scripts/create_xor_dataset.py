"""Create a synthetic XOR dataset for testing the pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate 10000 rows
n_rows = 10000

# Generate x and y uniformly in [-1, 1]
x = np.random.uniform(-1, 1, n_rows)
y = np.random.uniform(-1, 1, n_rows)

# XOR logic: quadrants 1 and 3 are blue, quadrants 2 and 4 are yellow
# Q1: x > 0, y > 0 -> blue
# Q2: x < 0, y > 0 -> yellow
# Q3: x < 0, y < 0 -> blue
# Q4: x > 0, y < 0 -> yellow
# This is equivalent to: (x > 0) XOR (y > 0) -> yellow, else blue
color = np.where((x > 0) ^ (y > 0), "yellow", "blue")

# Create a simple row_id as primary key
row_id = np.arange(n_rows)

# Create DataFrame
df = pd.DataFrame({
    "row_id": row_id,
    "x": x,
    "y": y,
    "color": color,
})

print(f"Dataset shape: {df.shape}")
print(f"Color distribution:\n{df['color'].value_counts()}")
print("\nFirst 10 rows:")
print(df.head(10))

# Verify XOR logic
print("\nVerifying XOR logic:")
for i in range(5):
    xi, yi, ci = df.iloc[i][["x", "y", "color"]]
    expected = "yellow" if (xi > 0) ^ (yi > 0) else "blue"
    print(f"  x={xi:.3f}, y={yi:.3f} -> {ci} (expected: {expected})")

# Save to parquet
output_dir = Path.home() / "data" / "databases_raw" / "synthetic-xor" / "db"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "xor_data.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")
