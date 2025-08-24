import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Streamlit App Layout (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="IPL Player Analytics")

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# --- Load Data and Models ---
@st.cache_data
def load_data():
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

    return ipl_df, matches_df, batting_ratings_df, bowling_ratings_df, \
           player_impact_batter_df, player_impact_bowler_df

@st.cache_resource
def load_models():
    batter_performance_model = joblib.load(os.path.join(DATA_DIR, 'batter_performance_model.pkl'))
    bowler_performance_model = joblib.load(os.path.join(DATA_DIR, 'bowler_performance_model.pkl'))
    match_outcome_model = joblib.load(os.path.join(DATA_DIR, 'match_outcome_model.pkl'))
    return batter_performance_model, bowler_performance_model, match_outcome_model

ipl_df, matches_df, batting_ratings_df, bowling_ratings_df, \
player_impact_batter_df, player_impact_bowler_df = load_data()
batter_performance_model, bowler_performance_model, match_outcome_model = load_models()

# --- Helper Functions ---

def get_latest_glicko_ratings(ratings_df):
    return ratings_df.groupby('player').last().reset_index()

latest_batting_glicko = get_latest_glicko_ratings(batting_ratings_df)
latest_bowling_glicko = get_latest_glicko_ratings(bowling_ratings_df)

st.title("🏏 IPL Player Ranking & Performance Analytics")

# Sidebar for filters
st.sidebar.header("Filters")
selected_season = st.sidebar.selectbox(
    "Select Season", 
    options=sorted(ipl_df['season'].unique(), reverse=True)
)

# Filter data by selected season
ipl_df_season = ipl_df[ipl_df['season'] == selected_season]
matches_df_season = matches_df[matches_df['season'] == selected_season]

# --- Tabs for Navigation ---
tab1, tab2, tab3 = st.tabs(["Player Leaderboards", "Player Profile", "Match Predictions"])

with tab1:
    st.header("Player Leaderboards")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Batsmen (Glicko Rating)")
        top_batsmen = latest_batting_glicko.sort_values(by='mu', ascending=False).head(10)
        st.dataframe(top_batsmen[['player', 'mu', 'phi']].round(2), hide_index=True)

        st.subheader("Top Batsmen (Impact Score)")
        top_impact_batsmen = player_impact_batter_df.sort_values(by='avg_impact_residual_batter', ascending=False).head(10)
        st.dataframe(top_impact_batsmen.round(2), hide_index=True)

    with col2:
        st.subheader("Top Bowlers (Glicko Rating)")
        top_bowlers = latest_bowling_glicko.sort_values(by='mu', ascending=False).head(10)
        st.dataframe(top_bowlers[['player', 'mu', 'phi']].round(2), hide_index=True)

        st.subheader("Top Bowlers (Impact Score)")
        top_impact_bowlers = player_impact_bowler_df.sort_values(by='avg_impact_residual_bowler', ascending=False).head(10)
        st.dataframe(top_impact_bowlers.round(2), hide_index=True)

with tab2:
    st.header("Player Profile")

    all_players = sorted(list(ipl_df['batter'].unique()) + list(ipl_df['bowler'].unique()))
    selected_player = st.selectbox("Select a Player", options=all_players)

    if selected_player:
        st.subheader(f"Profile for {selected_player}")

        # Glicko Rating Trajectory
        player_batting_history = batting_ratings_df[batting_ratings_df['player'] == selected_player]
        player_bowling_history = bowling_ratings_df[bowling_ratings_df['player'] == selected_player]

        if not player_batting_history.empty:
            fig_batting = px.line(player_batting_history, x='date', y='mu', title=f"{selected_player}'s Batting Glicko Rating Over Time")
            st.plotly_chart(fig_batting, use_container_width=True)
        else:
            st.info(f"No batting history found for {selected_player}.")

        if not player_bowling_history.empty:
            fig_bowling = px.line(player_bowling_history, x='date', y='mu', title=f"{selected_player}'s Bowling Glicko Rating Over Time")
            st.plotly_chart(fig_bowling, use_container_width=True)
        else:
            st.info(f"No bowling history found for {selected_player}.")

        # Situational Performance (Average runs/wickets in phases)
        st.subheader("Situational Performance")
        player_df = ipl_df[ (ipl_df['batter'] == selected_player) | (ipl_df['bowler'] == selected_player) ]
        
        if not player_df.empty:
            # Batting
            player_batting_perf = player_df[player_df['batter'] == selected_player].groupby('phase').agg(
                avg_runs=('batsman_runs', 'mean'),
                balls_faced=('ball', 'count')
            ).reset_index()
            if not player_batting_perf.empty:
                st.write(f"**{selected_player} - Batting Performance by Phase:**")
                st.dataframe(player_batting_perf.round(2), hide_index=True)

            # Bowling
            player_bowling_perf = player_df[player_df['bowler'] == selected_player].groupby('phase').agg(
                avg_runs_conceded=('total_runs', 'mean'),
                wickets_taken=('is_wicket', 'sum'),
                balls_bowled=('ball', 'count')
            ).reset_index()
            if not player_bowling_perf.empty:
                st.write(f"**{selected_player} - Bowling Performance by Phase:**")
                st.dataframe(player_bowling_perf.round(2), hide_index=True)
        else:
            st.info(f"No detailed performance data found for {selected_player}.")

        # Player Impact Score
        st.subheader("Player Impact Score")
        impact_batter = player_impact_batter_df[player_impact_batter_df['batter'] == selected_player]['avg_impact_residual_batter'].values
        impact_bowler = player_impact_bowler_df[player_impact_bowler_df['bowler'] == selected_player]['avg_impact_residual_bowler'].values

        if impact_batter.size > 0: st.write(f"Average Batting Impact Residual: {impact_batter[0]:.2f}")
        if impact_bowler.size > 0: st.write(f"Average Bowling Impact Residual: {impact_bowler[0]:.2f}")
        if impact_batter.size == 0 and impact_bowler.size == 0: st.info(f"No impact score found for {selected_player}.")

        # Predict Next Match Performance (Simplified)
        st.subheader("Next Match Performance Prediction")
        st.info("This is a simplified prediction based on the last available data point.")

        # Get latest performance for the player
        latest_player_perf = ipl_df[(ipl_df['batter'] == selected_player) | (ipl_df['bowler'] == selected_player)].sort_values(by='date', ascending=False).head(1)
        
        if not latest_player_perf.empty:
            # Prepare features for batter prediction
            batter_features_df = pd.DataFrame({
                'batter_glicko_mu': [latest_batting_glicko[latest_batting_glicko['player'] == selected_player]['mu'].iloc[0] if selected_player in latest_batting_glicko['player'].values else 1500],
                'avg_impact_residual_batter': [player_impact_batter_df[player_impact_batter_df['batter'] == selected_player]['avg_impact_residual_batter'].iloc[0] if selected_player in player_impact_batter_df['batter'].values else 0],
                'prev_match_runs': [latest_player_perf['batsman_runs'].sum() if selected_player == latest_player_perf['batter'].iloc[0] else 0],
                'prev_match_sr': [(latest_player_perf['batsman_runs'].sum() / latest_player_perf['ball'].count()) * 100 if selected_player == latest_player_perf['batter'].iloc[0] and latest_player_perf['ball'].count() > 0 else 0]
            })
            # Ensure column names match training features
            batter_features_df.columns = batter_performance_model.feature_name_

            # Predict batter runs
            predicted_runs = batter_performance_model.predict(batter_features_df)[0]
            st.write(f"Predicted Runs in Next Match: {predicted_runs:.2f}")

            # Prepare features for bowler prediction
            bowler_features_df = pd.DataFrame({
                'bowler_glicko_mu': [latest_bowling_glicko[latest_bowling_glicko['player'] == selected_player]['mu'].iloc[0] if selected_player in latest_bowling_glicko['player'].values else 1500],
                'avg_impact_residual_bowler': [player_impact_bowler_df[player_impact_bowler_df['bowler'] == selected_player]['avg_impact_residual_bowler'].iloc[0] if selected_player in player_impact_bowler_df['bowler'].values else 0],
                'prev_match_wickets': [latest_player_perf['is_wicket'].sum() if selected_player == latest_player_perf['bowler'].iloc[0] else 0],
                'prev_match_economy': [(latest_player_perf['total_runs'].sum() / latest_player_perf['ball'].count()) * 6 if selected_player == latest_player_perf['bowler'].iloc[0] and latest_player_perf['ball'].count() > 0 else 0]
            })
            # Ensure column names match training features
            bowler_features_df.columns = bowler_performance_model.feature_name_

            # Predict bowler wickets
            predicted_wickets = bowler_performance_model.predict(bowler_features_df)[0]
            st.write(f"Predicted Wickets in Next Match: {predicted_wickets:.2f}")

        else:
            st.info(f"Not enough data to predict next match performance for {selected_player}.")

with tab3:
    st.header("Match Predictions")

    teams = sorted(list(matches_df['team1'].unique()))
    venues = sorted(list(matches_df['venue'].unique()))

    col1_mp, col2_mp = st.columns(2)
    with col1_mp:
        team1_pred = st.selectbox("Select Team 1", options=teams, key='team1_pred')
    with col2_mp:
        team2_pred = st.selectbox("Select Team 2", options=teams, key='team2_pred')
    
    selected_venue = st.selectbox("Select Venue", options=venues)
    toss_decision = st.selectbox("Toss Decision", options=matches_df['toss_decision'].unique())

    if st.button("Predict Match Winner"):
        # Prepare features for prediction
        # Need to get average Glicko and win rates for selected teams
        # This is a simplified approach for the dashboard. In a real-time system,
        # these would be calculated dynamically based on the latest available data.

        # Get latest Glicko for teams (simplified: average of all players in team)
        # This is a very rough approximation for dashboard display
        team1_players = ipl_df[ipl_df['batting_team'] == team1_pred]['batter'].unique()
        team2_players = ipl_df[ipl_df['batting_team'] == team2_pred]['batter'].unique()

        team1_glicko_mu = latest_batting_glicko[latest_batting_glicko['player'].isin(team1_players)]['mu'].mean()
        team2_glicko_mu = latest_batting_glicko[latest_batting_glicko['player'].isin(team2_players)]['mu'].mean()

        # Get latest win rates (simplified: from the last match they played)
        team1_last_match = matches_df[ (matches_df['team1'] == team1_pred) | (matches_df['team2'] == team1_pred) ].sort_values(by='date', ascending=False).head(1)
        team2_last_match = matches_df[ (matches_df['team1'] == team2_pred) | (matches_df['team2'] == team2_pred) ].sort_values(by='date', ascending=False).head(1)

        team1_win_rate = 0.5 # Placeholder
        team2_win_rate = 0.5 # Placeholder

        # For a proper win rate, we need to re-run the team form calculation from 06_predictive_analytics.py
        # or store the team win rates over time. For dashboard simplicity, we'll use a placeholder.

        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'team1_avg_glicko': [team1_glicko_mu if not pd.isna(team1_glicko_mu) else 1500],
            'team2_avg_glicko': [team2_glicko_mu if not pd.isna(team2_glicko_mu) else 1500],
            'team1_win_rate': [team1_win_rate],
            'team2_win_rate': [team2_win_rate],
            'toss_decision': [toss_decision],
            'venue': [selected_venue]
        })

        # Ensure categorical features are of 'category' dtype
        for col in ['toss_decision', 'venue']:
            pred_df[col] = pred_df[col].astype('category')
        
        # Ensure columns match training features
        pred_df.columns = match_outcome_model.feature_name_

        # Predict winner
        predicted_winner_encoded = match_outcome_model.predict(pred_df)[0]
        
        # Get the LabelEncoder from the training script to decode the winner
        # This is a bit tricky as LabelEncoder is not saved with the model.
        # For now, we'll assume the order of teams in matches_df is consistent.
        # A robust solution would save the LabelEncoder or map manually.
        
        # Re-create LabelEncoder (assuming same data and order as training)
        le = LabelEncoder()
        le.fit(matches_df['winner'])
        predicted_winner = le.inverse_transform([predicted_winner_encoded])[0]

        st.success(f"Predicted Winner: **{predicted_winner}**")

# --- Instructions to Run ---
 st.sidebar.markdown(
    """
---
### How to Run 
1.  Save this file as `07_dashboard.py`. 
2.  Open your terminal in the `ipl-analytics` directory. 
3.  Run: `streamlit run 07_dashboard.py` 
"""
)