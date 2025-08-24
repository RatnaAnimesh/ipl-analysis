
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Load data
ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_enriched.csv'))
matches_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_cleaned.csv')) # Use cleaned matches for match-level info

batting_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_batting_ratings.csv'))
bowling_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_bowling_ratings.csv'))
player_impact_batter_df = pd.read_csv(os.path.join(DATA_DIR, 'player_impact_batter.csv'))
player_impact_bowler_df = pd.read_csv(os.path.join(DATA_DIR, 'player_impact_bowler.csv'))

# Convert date to datetime for proper sorting
ipl_df['date'] = pd.to_datetime(ipl_df['date'])
matches_df['date'] = pd.to_datetime(matches_df['date'])

# --- Sub-Phase 6.1: Player Performance Forecasting ---
print("\n--- Player Performance Forecasting ---")

# Aggregate player performance per match
player_match_perf = ipl_df.groupby(['match_id', 'date', 'batter', 'bowler']).agg(
    total_runs_scored=('batsman_runs', 'sum'),
    balls_faced=('ball', 'count'),
    wickets_taken=('is_wicket', 'sum'),
    runs_conceded=('total_runs', 'sum')
).reset_index()

# Calculate strike rate and economy rate for the match
player_match_perf['match_strike_rate'] = (player_match_perf['total_runs_scored'] / player_match_perf['balls_faced']) * 100
player_match_perf['match_economy_rate'] = (player_match_perf['runs_conceded'] / player_match_perf['balls_faced']) * 6

# Merge Glicko ratings and impact scores
# For simplicity, use the latest Glicko/impact score for each player at the time of the match
# This requires a more complex merge or a time-series join. For now, let's use the overall latest.

# Get latest Glicko ratings (as of the end of the dataset)
latest_batting_ratings = batting_ratings_df.groupby('player').last().reset_index()
latest_bowling_ratings = bowling_ratings_df.groupby('player').last().reset_index()

player_match_perf = pd.merge(player_match_perf, latest_batting_ratings[['player', 'mu']], 
                             left_on='batter', right_on='player', how='left')
player_match_perf.rename(columns={'mu': 'batter_glicko_mu'}, inplace=True)
player_match_perf.drop('player', axis=1, inplace=True)

player_match_perf = pd.merge(player_match_perf, latest_bowling_ratings[['player', 'mu']], 
                             left_on='bowler', right_on='player', how='left')
player_match_perf.rename(columns={'mu': 'bowler_glicko_mu'}, inplace=True)
player_match_perf.drop('player', axis=1, inplace=True)

# Merge impact scores
player_match_perf = pd.merge(player_match_perf, player_impact_batter_df, 
                             left_on='batter', right_on='batter', how='left')
player_match_perf = pd.merge(player_match_perf, player_impact_bowler_df, 
                             left_on='bowler', right_on='bowler', how='left')

# Fill NaNs for new players or those not in Glicko/impact data
player_match_perf.fillna(0, inplace=True) # Simple fill for now

# Create lagged features for player performance
# Sort by player and date to ensure correct lagging
player_match_perf.sort_values(by=['batter', 'date'], inplace=True)
player_match_perf['prev_match_runs'] = player_match_perf.groupby('batter')['total_runs_scored'].shift(1)
player_match_perf['prev_match_sr'] = player_match_perf.groupby('batter')['match_strike_rate'].shift(1)

player_match_perf.sort_values(by=['bowler', 'date'], inplace=True)
player_match_perf['prev_match_wickets'] = player_match_perf.groupby('bowler')['wickets_taken'].shift(1)
player_match_perf['prev_match_economy'] = player_match_perf.groupby('bowler')['match_economy_rate'].shift(1)

player_match_perf.fillna(0, inplace=True) # Fill NaNs created by shifting

# Define features and targets for player forecasting
batter_features = [
    'batter_glicko_mu', 'avg_impact_residual_batter',
    'prev_match_runs', 'prev_match_sr'
]
batter_target = 'total_runs_scored' # Predict runs in current match

bowler_features = [
    'bowler_glicko_mu', 'avg_impact_residual_bowler',
    'prev_match_wickets', 'prev_match_economy'
]
bowler_target = 'wickets_taken' # Predict wickets in current match

# Train Batter Performance Model
X_batter = player_match_perf[batter_features]
y_batter = player_match_perf[batter_target]

X_batter_train, X_batter_test, y_batter_train, y_batter_test = train_test_split(X_batter, y_batter, test_size=0.2, random_state=42)

print("Training Batter Performance Model...")
batter_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
batter_model.fit(X_batter_train, y_batter_train, 
                   eval_set=[(X_batter_test, y_batter_test)],
                   eval_metric='mae',
                   callbacks=[lgb.early_stopping(50, verbose=False)])
joblib.dump(batter_model, os.path.join(DATA_DIR, 'batter_performance_model.pkl'))
print("Batter Performance Model saved.")

# Train Bowler Performance Model
X_bowler = player_match_perf[bowler_features]
y_bowler = player_match_perf[bowler_target]

X_bowler_train, X_bowler_test, y_bowler_train, y_bowler_test = train_test_split(X_bowler, y_bowler, test_size=0.2, random_state=42)

print("Training Bowler Performance Model...")
bowler_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
bowler_model.fit(X_bowler_train, y_bowler_train, 
                   eval_set=[(X_bowler_test, y_bowler_test)],
                   eval_metric='mae',
                   callbacks=[lgb.early_stopping(50, verbose=False)])
joblib.dump(bowler_model, os.path.join(DATA_DIR, 'bowler_performance_model.pkl'))
print("Bowler Performance Model saved.")

# --- Sub-Phase 6.2: Match Outcome Prediction ---
print("\n--- Match Outcome Prediction ---")

# Prepare match-level features
# Aggregate player ratings/impacts to team level for each match

# Get match-level data from cleaned_data (which has winner)
match_data = matches_df.drop_duplicates(subset='match_id').copy()

# Merge latest Glicko ratings for all players
all_latest_glicko = pd.concat([
    latest_batting_ratings.rename(columns={'mu': 'glicko_mu'}),
    latest_bowling_ratings.rename(columns={'mu': 'glicko_mu'})
]).drop_duplicates(subset='player')

# Function to get average team rating for a match
def get_team_avg_glicko(team_name, match_id, player_type='batter'):
    # Get players who played for this team in this match
    players_in_team = ipl_df[(ipl_df['match_id'] == match_id) & 
                             ((ipl_df['batting_team'] == team_name) | (ipl_df['bowling_team'] == team_name))]['batter'].unique()
    
    # Filter for relevant player type (e.g., top 6 batters, top 4 bowlers)
    # For simplicity, let's just take all players who played for the team in that match
    
    team_glicko_mus = []
    for player in players_in_team:
        if player in all_latest_glicko['player'].values:
            team_glicko_mus.append(all_latest_glicko[all_latest_glicko['player'] == player]['glicko_mu'].iloc[0])
    
    return np.mean(team_glicko_mus) if team_glicko_mus else 1500 # Default if no players found

# Apply this function to get team average Glicko for team1 and team2
match_data['team1_avg_glicko'] = match_data.apply(lambda row: get_team_avg_glicko(row['team1'], row['match_id']), axis=1)
match_data['team2_avg_glicko'] = match_data.apply(lambda row: get_team_avg_glicko(row['team2'], row['match_id']), axis=1)

# Team form (simple win streak/loss streak)
# This requires iterating through matches chronologically for each team
# For simplicity, let's use a rolling average of wins for each team

# Sort matches by date
match_data.sort_values(by='date', inplace=True)

team_win_rates = {}
team_form_features = []

for index, row in match_data.iterrows():
    team1 = row['team1']
    team2 = row['team2']
    winner = row['winner']

    # Get current win rates (or initialize if new team)
    team1_win_rate = team_win_rates.get(team1, 0.5) # Default to 0.5 for new teams
    team2_win_rate = team_win_rates.get(team2, 0.5)

    team_form_features.append({
        'match_id': row['match_id'],
        'team1_win_rate': team1_win_rate,
        'team2_win_rate': team2_win_rate
    })

    # Update win rates for next iteration (simple moving average or just last outcome)
    # For simplicity, let's just update based on current match outcome
    if winner == team1: 
        team_win_rates[team1] = (team_win_rates.get(team1, 0.5) * 0.9) + (1 * 0.1) # Simple decay
        team_win_rates[team2] = (team_win_rates.get(team2, 0.5) * 0.9) + (0 * 0.1)
    elif winner == team2:
        team_win_rates[team1] = (team_win_rates.get(team1, 0.5) * 0.9) + (0 * 0.1)
        team_win_rates[team2] = (team_win_rates.get(team2, 0.5) * 0.9) + (1 * 0.1)

team_form_df = pd.DataFrame(team_form_features)
match_data = pd.merge(match_data, team_form_df, on='match_id', how='left')

# Encode winner for classification
le = LabelEncoder()
match_data['winner_encoded'] = le.fit_transform(match_data['winner'])

# Features for match outcome prediction
match_features = [
    'team1_avg_glicko', 'team2_avg_glicko',
    'team1_win_rate', 'team2_win_rate',
    'toss_decision', 'venue' # Add toss decision and venue as categorical
]
match_target = 'winner_encoded'

X_match = match_data[match_features]
y_match = match_data[match_target]

# Convert categorical features to category type for LightGBM
for col in ['toss_decision', 'venue']:
    X_match[col] = X_match[col].astype('category')

# Split data
X_match_train, X_match_test, y_match_train, y_match_test = train_test_split(X_match, y_match, test_size=0.2, random_state=42)

print("Training Match Outcome Prediction Model...")
match_outcome_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
match_outcome_model.fit(X_match_train, y_match_train, 
                          eval_set=[(X_match_test, y_match_test)],
                          eval_metric='multi_logloss',
                          callbacks=[lgb.early_stopping(50, verbose=False)])
joblib.dump(match_outcome_model, os.path.join(DATA_DIR, 'match_outcome_model.pkl'))
print("Match Outcome Prediction Model saved.")

# Print accuracy
y_pred = match_outcome_model.predict(X_match_test)
print(f"Match Outcome Prediction Accuracy: {accuracy_score(y_match_test, y_pred):.4f}")
