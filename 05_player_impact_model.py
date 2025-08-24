import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib # For saving the model
import warnings

warnings.filterwarnings('ignore')

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Load the enriched dataset
ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_enriched.csv'))

# Load Glicko ratings
batting_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_batting_ratings.csv'))
bowling_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_bowling_ratings.csv'))

# Get the latest Glicko ratings for each player
latest_batting_ratings = batting_ratings_df.groupby('player').last().reset_index()
latest_bowling_ratings = bowling_ratings_df.groupby('player').last().reset_index()

# --- Corrected Merge Operations ---
# Rename columns in latest_batting_ratings before merging
latest_batting_ratings.rename(columns={'mu': 'mu_batter_glicko', 'phi': 'phi_batter_glicko'}, inplace=True)

# Merge latest Glicko batting ratings into the main DataFrame
ipl_df = pd.merge(ipl_df, latest_batting_ratings[['player', 'mu_batter_glicko', 'phi_batter_glicko']], 
                  left_on='batter', right_on='player', how='left')
ipl_df.drop('player', axis=1, inplace=True)

# Rename columns in latest_bowling_ratings before merging
latest_bowling_ratings.rename(columns={'mu': 'mu_bowler_glicko', 'phi': 'phi_bowler_glicko'}, inplace=True)

ipl_df = pd.merge(ipl_df, latest_bowling_ratings[['player', 'mu_bowler_glicko', 'phi_bowler_glicko']], 
                  left_on='bowler', right_on='player', how='left')
ipl_df.drop('player', axis=1, inplace=True)

# Fill NaN Glicko ratings with default (for players who haven't played enough to get a rating)
ipl_df['mu_batter_glicko'].fillna(1500, inplace=True)
ipl_df['phi_batter_glicko'].fillna(350, inplace=True)
ipl_df['mu_bowler_glicko'].fillna(1500, inplace=True)
ipl_df['phi_bowler_glicko'].fillna(350, inplace=True)

# --- Feature Selection and Preprocessing ---
features = [
    'inning', 'over', 'ball', 'phase', 'is_chase',
    'batting_team', 'bowling_team', 'batter', 'bowler',
    'mu_batter_glicko', 'phi_batter_glicko',
    'mu_bowler_glicko', 'phi_bowler_glicko',
    'strike_rate', 'economy_rate' # Include some engineered features
]
target = 'batsman_runs' # Predicting runs scored on a ball

X = ipl_df[features]
y = ipl_df[target]

# Label Encode categorical features for LightGBM
# LightGBM can handle categorical features directly if specified, but Label Encoding is safer for general use
categorical_features = [
    'phase', 'batting_team', 'bowling_team', 'batter', 'bowler'
]

for col in categorical_features:
    X[col] = X[col].astype('category')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
print("Training LightGBM model...")
model = lgb.LGBMRegressor(objective='regression_l1', # MAE objective
                          metric='mae',
                          n_estimators=1000,
                          learning_rate=0.05,
                          num_leaves=31,
                          max_depth=-1,
                          min_child_samples=20,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          random_state=42,
                          n_jobs=-1)

model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(100, verbose=False)])

print("Model training complete.")

# --- Calculate Feature Importance ---
print("Feature Importances:")
feature_importances = pd.DataFrame({'feature': model.feature_name_,
                                    'importance': model.feature_importances_})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print(feature_importances)

# --- Define Impact Score (Initial Concept) ---
# For an initial impact score, we can use the model's prediction error.
# A player who consistently performs better than the model predicts (positive residual) 
# is having a higher impact. We can aggregate this.

ipl_df['predicted_runs'] = model.predict(X)
ipl_df['impact_residual'] = ipl_df['batsman_runs'] - ipl_df['predicted_runs']

# Aggregate impact residual by player
player_impact_batter = ipl_df.groupby('batter')['impact_residual'].mean().reset_index()
player_impact_batter.rename(columns={'impact_residual': 'avg_impact_residual_batter'}, inplace=True)

player_impact_bowler = ipl_df.groupby('bowler')['impact_residual'].mean().reset_index()
player_impact_bowler.rename(columns={'impact_residual': 'avg_impact_residual_bowler'}, inplace=True)

# --- Save Model and Impact Scores ---
model_path = os.path.join(DATA_DIR, 'player_impact_model.pkl')
joblib.dump(model, model_path)
print(f"Player impact model saved to: {model_path}")

player_impact_batter.to_csv(os.path.join(DATA_DIR, 'player_impact_batter.csv'), index=False)
player_impact_bowler.to_csv(os.path.join(DATA_DIR, 'player_impact_bowler.csv'), index=False)
print("Player impact scores (residuals) saved.")

print("\nPlayer Impact (Batter) Head:")
print(player_impact_batter.head())
print("\nPlayer Impact (Bowler) Head:")
print(player_impact_bowler.head())