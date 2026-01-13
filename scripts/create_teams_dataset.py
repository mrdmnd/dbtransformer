"""Create a synthetic two-table relational dataset: teams & players."""

from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# Table 1: Teams
# =============================================================================
n_teams = 50

team_id = np.arange(n_teams)
budget = np.random.uniform(0.5, 2.0, n_teams)  # Budget in millions

# win_rate depends on budget (teams with more money win more)
# win_rate = 0.3 * budget + noise, clipped to [0, 1]
noise_win = np.random.normal(0, 0.1, n_teams)
win_rate = np.clip(0.2 + 0.3 * budget + noise_win, 0, 1)

teams_df = pd.DataFrame({
    "team_id": team_id,
    "budget": budget,
    "win_rate": win_rate,
})

print("=" * 60)
print("TEAMS TABLE")
print("=" * 60)
print(f"Shape: {teams_df.shape}")
print("\nFirst 10 rows:")
print(teams_df.head(10))
print("\nStats:")
print(teams_df.describe())

# =============================================================================
# Table 2: Players
# =============================================================================
# Each team has 3-5 players (random)
players_per_team = np.random.randint(3, 6, n_teams)
n_players = players_per_team.sum()

player_id = np.arange(n_players)
player_team_id = np.repeat(team_id, players_per_team)

# Player skill is random
skill = np.random.uniform(0.3, 1.0, n_players)

# Player score depends on BOTH skill AND team's budget
# This requires FK traversal to learn!
# score = 0.5 * skill + 0.3 * team_budget + noise
team_budgets_for_players = budget[player_team_id]
noise_score = np.random.normal(0, 0.1, n_players)
score = 0.5 * skill + 0.3 * team_budgets_for_players + noise_score

players_df = pd.DataFrame({
    "player_id": player_id,
    "team_id": player_team_id,
    "skill": skill,
    "score": score,
})

print("\n" + "=" * 60)
print("PLAYERS TABLE")
print("=" * 60)
print(f"Shape: {players_df.shape}")
print(
    f"Players per team: min={players_per_team.min()}, max={players_per_team.max()}, mean={players_per_team.mean():.1f}"
)
print("\nFirst 15 rows:")
print(players_df.head(15))
print("\nStats:")
print(players_df.describe())

# =============================================================================
# Verify the cross-table relationship
# =============================================================================
print("\n" + "=" * 60)
print("VERIFYING CROSS-TABLE RELATIONSHIP")
print("=" * 60)
print("\nPlayer score should correlate with team budget:")
print("(This requires FK traversal to learn!)")

# Show a few examples
for i in [0, 50, 100, 150]:
    if i < n_players:
        p = players_df.iloc[i]
        t = teams_df[teams_df["team_id"] == p["team_id"]].iloc[0]
        expected_score = 0.5 * p["skill"] + 0.3 * t["budget"]
        print(
            f"  Player {int(p['player_id'])}: skill={p['skill']:.2f}, team_budget={t['budget']:.2f} -> score={p['score']:.2f} (expected≈{expected_score:.2f})"
        )

# Correlation analysis
merged = players_df.merge(teams_df, on="team_id")
corr_skill = np.corrcoef(merged["skill"], merged["score"])[0, 1]
corr_budget = np.corrcoef(merged["budget"], merged["score"])[0, 1]
print("\nCorrelations with player score:")
print(f"  skill: {corr_skill:.3f}")
print(f"  team_budget: {corr_budget:.3f}")

# =============================================================================
# Save to parquet files
# =============================================================================
output_dir = Path.home() / "data" / "databases_raw" / "synthetic-teams" / "db"
output_dir.mkdir(parents=True, exist_ok=True)

teams_path = output_dir / "teams.parquet"
players_path = output_dir / "players.parquet"

teams_df.to_parquet(teams_path, index=False)
players_df.to_parquet(players_path, index=False)

print("\nSaved to:")
print(f"  {teams_path}")
print(f"  {players_path}")
