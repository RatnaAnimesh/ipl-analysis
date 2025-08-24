import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# --- Load Data and Models ---
def load_all_data_and_models():
    try:
        ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_enriched.csv'))
        matches_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_cleaned.csv'))
        batting_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_batting_ratings.csv'))
        bowling_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_bowling_ratings.csv'))
        player_impact_batter_df = pd.read_csv(os.path.join(DATA_DIR, 'player_impact_batter.csv'))
        player_impact_bowler_df = pd.read_csv(os.path.join(DATA_DIR, 'player_impact_bowler.csv'))
        
        # Convert date columns to datetime
        ipl_df['date'] = pd.to_datetime(ipl_df['date'])
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        batting_ratings_df['date'] = pd.to_datetime(batting_ratings_df['date'])
        bowling_ratings_df['date'] = pd.to_datetime(bowling_ratings_df['date'])

        # Load models
        player_impact_model = joblib.load(os.path.join(DATA_DIR, 'player_impact_model.pkl'))
        batter_performance_model = joblib.load(os.path.join(DATA_DIR, 'batter_performance_model.pkl'))
        bowler_performance_model = joblib.load(os.path.join(DATA_DIR, 'bowler_performance_model.pkl'))
        match_outcome_model = joblib.load(os.path.join(DATA_DIR, 'match_outcome_model.pkl'))

        return {
            "ipl_df": ipl_df,
            "matches_df": matches_df,
            "batting_ratings_df": batting_ratings_df,
            "bowling_ratings_df": bowling_ratings_df,
            "player_impact_batter_df": player_impact_batter_df,
            "player_impact_bowler_df": player_impact_bowler_df,
            "player_impact_model": player_impact_model,
            "batter_performance_model": batter_performance_model,
            "bowler_performance_model": bowler_performance_model,
            "match_outcome_model": match_outcome_model,
        }
    except FileNotFoundError as e:
        print(f"Error loading data or model: {e}. Please ensure all previous scripts were run successfully.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data/model loading: {e}")
        exit()

data_and_models = load_all_data_and_models()

ipl_df = data_and_models["ipl_df"]
matches_df = data_and_models["matches_df"]
batting_ratings_df = data_and_models["batting_ratings_df"]
bowling_ratings_df = data_and_models["bowling_ratings_df"]
player_impact_batter_df = data_and_models["player_impact_batter_df"]
player_impact_bowler_df = data_and_models["player_impact_bowler_df"]
player_impact_model = data_and_models["player_impact_model"]
batter_performance_model = data_and_models["batter_performance_model"]
bowler_performance_model = data_and_models["bowler_performance_model"]
match_outcome_model = data_and_models["match_outcome_model"]

# --- Replicate Glicko Rating Merge Logic (from 05_player_impact_model.py) ---
# Get the latest Glicko ratings for each player
latest_batting_glicko = batting_ratings_df.groupby('player').last().reset_index()
latest_bowling_glicko = bowling_ratings_df.groupby('player').last().reset_index()

# Rename columns in latest_batting_ratings before merging
latest_batting_glicko.rename(columns={'mu': 'mu_batter_glicko', 'phi': 'phi_batter_glicko'}, inplace=True)

# Merge latest Glicko batting ratings into the main DataFrame
ipl_df = pd.merge(ipl_df, latest_batting_glicko[['player', 'mu_batter_glicko', 'phi_batter_glicko']], 
                  left_on='batter', right_on='player', how='left')
ipl_df.drop('player', axis=1, inplace=True)

# Rename columns in latest_bowling_ratings before merging
latest_bowling_glicko.rename(columns={'mu': 'mu_bowler_glicko', 'phi': 'phi_bowler_glicko'}, inplace=True)

ipl_df = pd.merge(ipl_df, latest_bowling_glicko[['player', 'mu_bowler_glicko', 'phi_bowler_glicko']], 
                  left_on='bowler', right_on='player', how='left')
ipl_df.drop('player', axis=1, inplace=True)

# Fill NaN Glicko ratings with default (for players who haven't played enough to get a rating)
ipl_df['mu_batter_glicko'].fillna(1500, inplace=True)
ipl_df['phi_batter_glicko'].fillna(350, inplace=True)
ipl_df['mu_bowler_glicko'].fillna(1500, inplace=True)
ipl_df['phi_bowler_glicko'].fillna(350, inplace=True)

# --- Reporting Section ---
print("\n--- IPL Player Analytics Report ---")
print(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n## Top Players by Glicko Rating (Latest)")

# Top Batsmen
latest_batting_glicko_report = batting_ratings_df.groupby('player').last().reset_index()
latest_batting_glicko_report = latest_batting_glicko_report.sort_values(by='mu', ascending=False).head(10)
print("### Top 10 Batsmen (Glicko Rating):")
print(latest_batting_glicko_report[['player', 'mu', 'phi']].round(2).to_string(index=False))

# Top Bowlers
latest_bowling_glicko_report = bowling_ratings_df.groupby('player').last().reset_index()
latest_bowling_glicko_report = latest_bowling_glicko_report.sort_values(by='mu', ascending=False).head(10)
print("\n### Top 10 Bowlers (Glicko Rating):")
print(latest_bowling_glicko_report[['player', 'mu', 'phi']].round(2).to_string(index=False))

print("\n## Top Players by ML Impact Score (Average Residual)")

# Top Impact Batsmen
top_impact_batsmen = player_impact_batter_df.sort_values(by='avg_impact_residual_batter', ascending=False).head(10)
print("### Top 10 Impact Batsmen:")
print(top_impact_batsmen.round(2).to_string(index=False))

# Top Impact Bowlers
top_impact_bowlers = player_impact_bowler_df.sort_values(by='avg_impact_residual_bowler', ascending=False).head(10)
print("\n### Top 10 Impact Bowlers:")
print(top_impact_bowlers.round(2).to_string(index=False))

# --- Explainability Section (SHAP) ---
print("\n--- Model Explainability (SHAP Values) ---")

# Example: Player Impact Model Explanation
print("\n## Player Impact Model (Predicting Runs per Ball) SHAP Explanation")

# Prepare data for SHAP
impact_features = [
    'inning', 'over', 'ball', 'phase', 'is_chase',
    'batting_team', 'bowling_team', 'batter', 'bowler',
    'mu_batter_glicko', 'phi_batter_glicko',
    'mu_bowler_glicko', 'phi_bowler_glicko',
    'strike_rate', 'economy_rate'
]

X_impact = ipl_df[impact_features].copy()

# Convert categorical features to category type for SHAP Explainer
categorical_features_impact = [
    'phase', 'batting_team', 'bowling_team', 'batter', 'bowler'
]
for col in categorical_features_impact:
    X_impact[col] = X_impact[col].astype('category')

# Create a SHAP Explainer
explainer_impact = shap.TreeExplainer(player_impact_model)

# Select a subset of data for explanation (e.g., first 1000 rows for speed)
# For a full report, you might explain more or specific instances.
shap_values_impact = explainer_impact.shap_values(X_impact.sample(n=1000, random_state=42))

print("\nSHAP Summary Plot (Feature Importance for Player Impact Model):\n")
# shap.summary_plot(shap_values_impact, X_impact.sample(n=1000, random_state=42), show=False)
# plt.savefig(os.path.join(DATA_DIR, 'shap_summary_impact.png'), bbox_inches='tight')
# print(f"SHAP summary plot saved to {os.path.join(DATA_DIR, 'shap_summary_impact.png')}")

# Due to environment limitations (no display for matplotlib), we'll print text summary
# Get feature importances from SHAP values (mean absolute SHAP value)
shap_feature_importance = pd.DataFrame({
    'feature': X_impact.columns,
    'mean_abs_shap_value': np.abs(shap_values_impact).mean(0)
})
shap_feature_importance = shap_feature_importance.sort_values(by='mean_abs_shap_value', ascending=False)
print(shap_feature_importance.to_string(index=False))

print("\n--- Report Generation Complete ---")
print("Note: SHAP plots require a graphical backend and are commented out. Text summaries are provided.")