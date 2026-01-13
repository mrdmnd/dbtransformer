"""Visualize synthetic datasets and optionally regenerate with more noise."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============================================================================
# Load and plot XOR dataset
# ============================================================================
xor_path = Path.home() / "data" / "databases_raw" / "synthetic-xor" / "db" / "xor_data.parquet"
df_xor = pd.read_parquet(xor_path)

ax = axes[0]
colors_xor = {"blue": "#4B0082", "yellow": "#FFD700"}  # Indigo and Gold
for label, color in colors_xor.items():
    mask = df_xor["color"] == label
    ax.scatter(df_xor.loc[mask, "x"], df_xor.loc[mask, "y"], 
               c=color, label=label, alpha=0.6, s=10, edgecolors='none')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Synthetic XOR Dataset\n(Quadrants 1,3 = blue; Quadrants 2,4 = yellow)")
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# ============================================================================
# Load and plot Moons dataset
# ============================================================================
moons_path = Path.home() / "data" / "databases_raw" / "synthetic-moons" / "db" / "moons_data.parquet"
df_moons = pd.read_parquet(moons_path)

ax = axes[1]
colors_moons = {"False": "#4B0082", "True": "#FFD700"}  # Indigo and Gold
for label, color in colors_moons.items():
    mask = df_moons["label"] == label
    ax.scatter(df_moons.loc[mask, "x1"], df_moons.loc[mask, "x2"], 
               c=color, label=label, alpha=0.6, s=10, edgecolors='none')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Synthetic Two-Moons Dataset\n(Two interleaved crescents)")
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path.home() / "dbtransformer" / "synthetic_datasets.png", dpi=150, bbox_inches='tight')
print(f"Saved visualization to ~/dbtransformer/synthetic_datasets.png")
plt.show()

# Print stats
print("\n" + "=" * 60)
print("XOR Dataset Stats:")
print(f"  Samples: {len(df_xor)}")
print(f"  Class distribution: {df_xor['color'].value_counts().to_dict()}")

print("\nMoons Dataset Stats:")
print(f"  Samples: {len(df_moons)}")
print(f"  Class distribution: {df_moons['label'].value_counts().to_dict()}")
