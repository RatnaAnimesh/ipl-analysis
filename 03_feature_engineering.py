
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Load the cleaned dataset
ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_cleaned.csv'))

# --- Feature Engineering ---

# 1. Match Context Features

# Match Phase
def get_phase(over):
    if over <= 6:
        return 'Powerplay'
    elif over <= 15:
        return 'Middle'
    else:
        return 'Death'
ipl_df['phase'] = ipl_df['over'].apply(get_phase)

# Is Chase?
ipl_df['is_chase'] = np.where(ipl_df['inning'] == 2, 1, 0)

# 2. Player Role Identification (Simple Heuristic)

batsman_df = ipl_df.groupby('batter')['ball'].count().reset_index()
bowler_df = ipl_df.groupby('bowler')['ball'].count().reset_index()

# Rename columns for clarity
batsman_df.rename(columns={'ball': 'balls_faced'}, inplace=True)
bowler_df.rename(columns={'ball': 'balls_bowled'}, inplace=True)

player_roles = pd.merge(batsman_df, bowler_df, left_on='batter', right_on='bowler', how='outer')
player_roles.fillna(0, inplace=True)

# Simple role classification
player_roles['is_batsman'] = np.where(player_roles['balls_faced'] > player_roles['balls_bowled'], 1, 0)
player_roles['is_bowler'] = np.where(player_roles['balls_bowled'] >= player_roles['balls_faced'], 1, 0)

# Merge roles back into main df
ipl_df = pd.merge(ipl_df, player_roles[['batter', 'is_batsman', 'is_bowler']], on='batter', how='left')

# 3. Batting Performance Metrics

# Strike Rate
batsman_grp = ipl_df.groupby(['match_id', 'batter'])
cumulative_runs = batsman_grp['batsman_runs'].cumsum()
cumulative_balls = batsman_grp['ball'].cumcount() + 1
ipl_df['strike_rate'] = (cumulative_runs / cumulative_balls) * 100

# Runs and Balls in Phase
ipl_df['runs_in_phase'] = ipl_df.groupby(['match_id', 'batter', 'phase'])['batsman_runs'].cumsum()
ipl_df['balls_in_phase'] = ipl_df.groupby(['match_id', 'batter', 'phase'])['ball'].cumcount() + 1

# 4. Bowling Performance Metrics

# Economy Rate
bowler_grp = ipl_df.groupby(['match_id', 'bowler'])
cumulative_runs_conceded = bowler_grp['total_runs'].cumsum()
cumulative_overs_bowled = (bowler_grp['ball'].cumcount() + 1) / 6
ipl_df['economy_rate'] = cumulative_runs_conceded / cumulative_overs_bowled

# Wickets in Phase
ipl_df['wickets_in_phase'] = ipl_df.groupby(['match_id', 'bowler', 'phase'])['is_wicket'].cumsum()

# --- Save Enriched Data ---
enriched_data_path = os.path.join(DATA_DIR, 'ipl_data_enriched.csv')
ipl_df.to_csv(enriched_data_path, index=False)

print(f"Feature engineering complete. Enriched data saved to: {enriched_data_path}")
print("\nEnriched DataFrame Info:")
ipl_df.info()
print("\nEnriched DataFrame Head:")
print(ipl_df.head())
