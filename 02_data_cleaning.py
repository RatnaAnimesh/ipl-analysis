
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Load the datasets
matches_df = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))
deliveries_df = pd.read_csv(os.path.join(DATA_DIR, 'deliveries.csv'))

# --- Data Cleaning on matches_df ---

# 1. Clean Team Names
team_name_mapping = {
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad'
}
matches_df['team1'] = matches_df['team1'].replace(team_name_mapping)
matches_df['team2'] = matches_df['team2'].replace(team_name_mapping)
matches_df['winner'] = matches_df['winner'].replace(team_name_mapping)
matches_df['toss_winner'] = matches_df['toss_winner'].replace(team_name_mapping)

# 2. Impute Missing City from Venue
venue_city_mapping = {
    'Dubai International Cricket Stadium': 'Dubai',
    'Sharjah Cricket Stadium': 'Sharjah'
}
matches_df['city'].fillna(matches_df['venue'].map(venue_city_mapping), inplace=True)

# 3. Handle No-Result Matches (where winner is NaN)
matches_df = matches_df[matches_df['winner'].notna()]

# 4. Correct Data Types
matches_df['date'] = pd.to_datetime(matches_df['date'])

# --- Data Cleaning on deliveries_df ---

deliveries_df['batting_team'] = deliveries_df['batting_team'].replace(team_name_mapping)
deliveries_df['bowling_team'] = deliveries_df['bowling_team'].replace(team_name_mapping)

# --- Merge DataFrames ---
# Rename 'id' in matches_df to 'match_id' for a consistent merge key
matches_df.rename(columns={'id': 'match_id'}, inplace=True)

ipl_df = pd.merge(deliveries_df, matches_df, on='match_id', how='left')

# --- Save Cleaned Data ---
cleaned_data_path = os.path.join(DATA_DIR, 'ipl_data_cleaned.csv')
ipl_df.to_csv(cleaned_data_path, index=False)

print(f"Data cleaning complete. Cleaned data saved to: {cleaned_data_path}")
print("\nCleaned DataFrame Info:")
ipl_df.info()
print("\nCleaned DataFrame Head:")
print(ipl_df.head())
