import pandas as pd
import os

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Load Glicko ratings
try:
    batting_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_batting_ratings.csv'))
    bowling_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_bowling_ratings.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure 'glicko_batting_ratings.csv' and 'glicko_bowling_ratings.csv' are in the '{DATA_DIR}' directory.")
    exit()

# Get the latest Glicko ratings for each player
latest_batting_glicko = batting_ratings_df.groupby('player').last().reset_index()
latest_bowling_glicko = bowling_ratings_df.groupby('player').last().reset_index()

# Combine batting and bowling ratings for an overall view
# For simplicity, let's average their ratings if they are both batsmen and bowlers
# Or, we can just show them separately.
# For a quick rank, let's combine and prioritize batting rating if both exist.

combined_ratings = pd.merge(
    latest_batting_glicko[['player', 'mu', 'phi']].rename(columns={'mu': 'batting_mu', 'phi': 'batting_phi'}),
    latest_bowling_glicko[['player', 'mu', 'phi']].rename(columns={'mu': 'bowling_mu', 'phi': 'bowling_phi'}),
    on='player', how='outer'
)

# Fill NaN for players who only bat or only bowl
combined_ratings['batting_mu'].fillna(1500, inplace=True) # Default Glicko rating
combined_ratings['batting_phi'].fillna(350, inplace=True)
combined_ratings['bowling_mu'].fillna(1500, inplace=True)
combined_ratings['bowling_phi'].fillna(350, inplace=True)

# Calculate an overall rating (simple average for now, can be weighted)
combined_ratings['overall_mu'] = (combined_ratings['batting_mu'] + combined_ratings['bowling_mu']) / 2
combined_ratings['overall_phi'] = (combined_ratings['batting_phi'] + combined_ratings['bowling_phi']) / 2

# Sort by overall rating
ranked_players = combined_ratings.sort_values(by='overall_mu', ascending=False).reset_index(drop=True)

# Print the ranked table
print("\n--- Overall Player Ranks (Glicko Rating) ---")
print(ranked_players[['player', 'overall_mu', 'overall_phi', 'batting_mu', 'bowling_mu']].round(2).to_string(index=False))

print("\n--- Top 10 Batsmen by Glicko Rating ---")
print(latest_batting_glicko.sort_values(by='mu', ascending=False).head(10)[['player', 'mu', 'phi']].round(2).to_string(index=False))

print("\n--- Top 10 Bowlers by Glicko Rating ---")
print(latest_bowling_glicko.sort_values(by='mu', ascending=False).head(10)[['player', 'mu', 'phi']].round(2).to_string(index=False))