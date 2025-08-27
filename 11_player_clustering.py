import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# --- Load Data ---
def load_clustering_data():
    try:
        ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_enriched.csv'))
        batting_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_batting_ratings.csv'))
        bowling_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_bowling_ratings.csv'))
        player_impact_batter_df = pd.read_csv(os.path.join(DATA_DIR, 'player_impact_batter.csv'))
        player_impact_bowler_df = pd.read_csv(os.path.join(DATA_DIR, 'player_impact_bowler.csv'))
        return ipl_df, batting_ratings_df, bowling_ratings_df, player_impact_batter_df, player_impact_bowler_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure all necessary CSV files are in the '{DATA_DIR}' directory.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()

ipl_df, batting_ratings_df, bowling_ratings_df, player_impact_batter_df, player_impact_bowler_df = load_clustering_data()

# --- Prepare Player-Level Features ---

# Aggregate overall player statistics from ipl_df
player_stats = ipl_df.groupby('batter').agg(
    total_runs=('batsman_runs', 'sum'),
    total_balls_faced=('ball', 'count'),
    total_fours=('batsman_runs', lambda x: (x == 4).sum()),
    total_sixes=('batsman_runs', lambda x: (x == 6).sum())
).reset_index()
player_stats.rename(columns={'batter': 'player'}, inplace=True)

bowler_stats = ipl_df.groupby('bowler').agg(
    total_wickets=('is_wicket', 'sum'),
    total_runs_conceded=('total_runs', 'sum'),
    total_balls_bowled=('ball', 'count')
).reset_index()
bowler_stats.rename(columns={'bowler': 'player'}, inplace=True)

# Calculate derived metrics
player_stats['strike_rate'] = (player_stats['total_runs'] / player_stats['total_balls_faced']) * 100
bowler_stats['economy_rate'] = (bowler_stats['total_runs_conceded'] / bowler_stats['total_balls_bowled']) * 6

# Merge Glicko ratings (latest)
latest_batting_glicko = batting_ratings_df.groupby('player').last().reset_index()
latest_bowling_glicko = bowling_ratings_df.groupby('player').last().reset_index()

# Start with batting stats and merge Glicko ratings
player_features = pd.merge(player_stats, latest_batting_glicko[['player', 'mu', 'phi']], 
                           on='player', how='left', suffixes=('', '_batting_glicko'))

# Merge bowling stats into player_features
player_features = pd.merge(player_features, bowler_stats, on='player', how='outer', suffixes=('', '_bowling_stats'))

player_features = pd.merge(player_features, latest_bowling_glicko[['player', 'mu', 'phi']], 
                           on='player', how='left', suffixes=('', '_bowling_glicko'))

# Merge Impact Scores
# Rename 'batter'/'bowler' columns to 'player' before merging
player_impact_batter_df.rename(columns={'batter': 'player'}, inplace=True)
player_impact_bowler_df.rename(columns={'bowler': 'player'}, inplace=True)

player_features = pd.merge(player_features, player_impact_batter_df, on='player', how='left')
player_features = pd.merge(player_features, player_impact_bowler_df, on='player', how='left')

# --- Ensure all required columns exist and fill NaNs ---
required_glicko_cols = [
    'mu_batting_glicko', 'phi_batting_glicko',
    'mu_bowling_glicko', 'phi_bowling_glicko'
]

for col in required_glicko_cols:
    if col not in player_features.columns:
        player_features[col] = np.nan # Add column if missing

# Fill specific columns with 0 where appropriate before general fillna
player_features['total_runs'].fillna(0, inplace=True)
player_features['total_balls_faced'].fillna(0, inplace=True)
player_features['total_fours'].fillna(0, inplace=True)
player_features['total_sixes'].fillna(0, inplace=True)

player_features['total_wickets'].fillna(0, inplace=True)
player_features['total_runs_conceded'].fillna(0, inplace=True)
player_features['total_balls_bowled'].fillna(0, inplace=True)

player_features['strike_rate'].fillna(0, inplace=True)
player_features['economy_rate'].fillna(0, inplace=True)

player_features['mu_batting_glicko'].fillna(1500, inplace=True)
player_features['phi_batting_glicko'].fillna(350, inplace=True)
player_features['mu_bowling_glicko'].fillna(1500, inplace=True)
player_features['phi_bowling_glicko'].fillna(350, inplace=True)

player_features['avg_impact_residual_batter'].fillna(0, inplace=True)
player_features['avg_impact_residual_bowler'].fillna(0, inplace=True)

# Select features for clustering
# Choose features that represent different aspects of a player's game
clustering_features = [
    'total_runs', 'strike_rate', 'total_fours', 'total_sixes',
    'total_wickets', 'economy_rate', 'total_runs_conceded',
    'mu_batting_glicko', 'phi_batting_glicko',
    'mu_bowling_glicko', 'phi_bowling_glicko',
    'avg_impact_residual_batter', 'avg_impact_residual_bowler'
]

# Filter out players with very few balls faced/bowled (e.g., less than 30 balls faced or 30 balls bowled)
# This removes players who haven't played enough to have meaningful stats
min_balls_faced = 30
min_balls_bowled = 30

# Filter players who have either batted or bowled a significant number of balls
filtered_players = player_features[(player_features['total_balls_faced'] >= min_balls_faced) | 
                                   (player_features['total_balls_bowled'] >= min_balls_bowled)].copy()

X = filtered_players[clustering_features]

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Determine Optimal K (Conceptual) ---
# For a real project, you would use methods like the Elbow Method or Silhouette Score
# to determine the optimal number of clusters. For demonstration, we'll pick K=5.

# wcss = [] # Within-cluster sum of squares
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X_scaled)
#     wcss.append(kmeans.inertia_)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('WCSS')
# plt.grid(True)
# plt.savefig(os.path.join(DATA_DIR, 'elbow_method.png'))
# print(f"Elbow method plot saved to {os.path.join(DATA_DIR, 'elbow_method.png')}")

# --- Apply K-Means Clustering ---
K = 5 # Chosen number of clusters
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42, n_init=10) # n_init to suppress warning
filtered_players['cluster'] = kmeans.fit_predict(X_scaled)

# --- Analyze Clusters ---
print("\n--- Player Clustering Analysis ---")
print(f"Clustering players into {K} groups based on their playing style.\n")

# Characterize each cluster by its mean feature values
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=clustering_features)
cluster_centers['cluster'] = range(K)
print("Cluster Centers (Mean Feature Values for Each Cluster):\n")
print(cluster_centers.round(2).to_string(index=False))

print("\n--- Sample Players from Each Cluster ---")
for i in range(K):
    print(f"\nCluster {i} (Sample Players):")
    sample_players = filtered_players[filtered_players['cluster'] == i]['player'].sample(min(5, len(filtered_players[filtered_players['cluster'] == i]))).tolist()
    print(sample_players)

# --- Save Cluster Assignments ---
cluster_assignments_path = os.path.join(DATA_DIR, 'player_cluster_assignments.csv')
filtered_players[['player', 'cluster'] + clustering_features].to_csv(cluster_assignments_path, index=False)
print(f"\nPlayer cluster assignments saved to: {cluster_assignments_path}")

print("\n--- Player Clustering Complete ---")
