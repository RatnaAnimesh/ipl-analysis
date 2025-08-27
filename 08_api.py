from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any

# --- FastAPI App Initialization ---
app = FastAPI(
    title="IPL Player Analytics API",
    description="API for accessing IPL player ratings, impact scores, and match predictions.",
    version="1.0.0",
)

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# --- Global Data and Model Loading ---
# These will be loaded once when the API starts

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
        batter_performance_model = joblib.load(os.path.join(DATA_DIR, 'batter_performance_model.pkl'))
        bowler_performance_model = joblib.load(os.path.join(DATA_DIR, 'bowler_performance_model.pkl'))
        match_outcome_model = joblib.load(os.path.join(DATA_DIR, 'match_outcome_model.pkl'))

        # Get latest Glicko ratings for quick lookup
        latest_batting_glicko = batting_ratings_df.groupby('player').last().reset_index()
        latest_bowling_glicko = bowling_ratings_df.groupby('player').last().reset_index()

        # Re-create LabelEncoder for match winner (as it's not saved with model)
        le_winner = LabelEncoder()
        le_winner.fit(matches_df['winner'])

        return {
            "ipl_df": ipl_df,
            "matches_df": matches_df,
            "batting_ratings_df": batting_ratings_df,
            "bowling_ratings_df": bowling_ratings_df,
            "player_impact_batter_df": player_impact_batter_df,
            "player_impact_bowler_df": player_impact_bowler_df,
            "batter_performance_model": batter_performance_model,
            "bowler_performance_model": bowler_performance_model,
            "match_outcome_model": match_outcome_model,
            "latest_batting_glicko": latest_batting_glicko,
            "latest_bowling_glicko": latest_bowling_glicko,
            "le_winner": le_winner
        }
    except FileNotFoundError as e:
        raise RuntimeError(f"Required data or model file not found: {e}. Please ensure all previous scripts were run successfully.")
    except Exception as e:
        raise RuntimeError(f"An error occurred during data/model loading: {e}")

# Load data and models globally
data_and_models = load_all_data_and_models()

ipl_df = data_and_models["ipl_df"]
matches_df = data_and_models["matches_df"]
batting_ratings_df = data_and_models["batting_ratings_df"]
bowling_ratings_df = data_and_models["bowling_ratings_df"]
player_impact_batter_df = data_and_models["player_impact_batter_df"]
player_impact_bowler_df = data_and_models["player_impact_bowler_df"]
batter_performance_model = data_and_models["batter_performance_model"]
bowler_performance_model = data_and_models["bowler_performance_model"]
match_outcome_model = data_and_models["match_outcome_model"]
latest_batting_glicko = data_and_models["latest_batting_glicko"]
latest_bowling_glicko = data_and_models["latest_bowling_glicko"]
le_winner = data_and_models["le_winner"]

# --- Helper Functions for API Endpoints ---

def get_player_glicko_history(player_name: str, ratings_df: pd.DataFrame) -> List[Dict[str, Any]]:
    history = ratings_df[ratings_df['player'] == player_name].copy()
    if history.empty:
        return []
    history['date'] = history['date'].dt.strftime('%Y-%m-%d')
    return history[['date', 'match_id', 'mu', 'phi']].to_dict(orient='records')

def get_player_impact_score(player_name: str, impact_df: pd.DataFrame, player_col: str) -> float:
    impact = impact_df[impact_df[player_col] == player_name][f'avg_impact_residual_{player_col}'].values
    return float(impact[0]) if impact.size > 0 else 0.0

def predict_player_performance(player_name: str, model, is_batter: bool) -> Dict[str, float]:
    latest_player_perf = ipl_df[(ipl_df['batter'] == player_name) | (ipl_df['bowler'] == player_name)].sort_values(by='date', ascending=False).head(1)
    
    if latest_player_perf.empty:
        return {"error": f"No recent performance data for {player_name}."}

    if is_batter:
        features_df = pd.DataFrame({
            'batter_glicko_mu': [latest_batting_glicko[latest_batting_glicko['player'] == player_name]['mu'].iloc[0] if player_name in latest_batting_glicko['player'].values else 1500],
            'avg_impact_residual_batter': [get_player_impact_score(player_name, player_impact_batter_df, 'batter')],
            'prev_match_runs': [latest_player_perf['batsman_runs'].sum() if player_name == latest_player_perf['batter'].iloc[0] else 0],
            'prev_match_sr': [(latest_player_perf['batsman_runs'].sum() / latest_player_perf['ball'].count()) * 100 if player_name == latest_player_perf['batter'].iloc[0] and latest_player_perf['ball'].count() > 0 else 0]
        })
        features_df.columns = model.feature_name_
        prediction = model.predict(features_df)[0]
        return {"predicted_runs": float(prediction)}
    else:
        features_df = pd.DataFrame({
            'bowler_glicko_mu': [latest_bowling_glicko[latest_bowling_glicko['player'] == player_name]['mu'].iloc[0] if player_name in latest_bowling_glicko['player'].values else 1500],
            'avg_impact_residual_bowler': [get_player_impact_score(player_name, player_impact_bowler_df, 'bowler')],
            'prev_match_wickets': [latest_player_perf['is_wicket'].sum() if player_name == latest_player_perf['bowler'].iloc[0] else 0],
            'prev_match_economy': [(latest_player_perf['total_runs'].sum() / latest_player_perf['ball'].count()) * 6 if player_name == latest_player_perf['bowler'].iloc[0] and latest_player_perf['ball'].count() > 0 else 0]
        })
        features_df.columns = model.feature_name_
        prediction = model.predict(features_df)[0]
        return {"predicted_wickets": float(prediction)}

# --- API Endpoints ---

@app.get("/players/list", response_model=List[str])
async def get_all_players():
    all_players = sorted(list(ipl_df['batter'].unique()) + list(ipl_df['bowler'].unique()))
    return all_players

@app.get("/teams/list", response_model=List[str])
async def get_all_teams():
    # Ensure team columns are string type and handle NaNs before getting unique values
    if not matches_df.empty:
        matches_df['team1'] = matches_df['team1'].astype(str).fillna('')
        matches_df['team2'] = matches_df['team2'].astype(str).fillna('')
        teams = sorted(list(matches_df['team1'].unique()) + list(matches_df['team2'].unique()))
        teams = [t for t in teams if t != ''] # Remove empty strings if any
        teams = sorted(list(set(teams))) # Get unique and sort again
    else:
        teams = []
    return teams

@app.get("/seasons/list", response_model=List[str])
async def get_all_seasons():
    # Convert season to string to ensure consistent sorting
    ipl_df['season'] = ipl_df['season'].astype(str)
    seasons = sorted(ipl_df['season'].unique(), reverse=True)
    return seasons

@app.get("/players/{player_name}/ratings", response_model=Dict[str, Dict[str, float]])
async def get_player_ratings(player_name: str):
    batting_rating = latest_batting_glicko[latest_batting_glicko['player'] == player_name]
    bowling_rating = latest_bowling_glicko[latest_bowling_glicko['player'] == player_name]

    if batting_rating.empty and bowling_rating.empty:
        raise HTTPException(status_code=404, detail="Player not found or no ratings available.")
    
    response = {}
    if not batting_rating.empty:
        response["batting"] = {"mu": float(batting_rating['mu'].iloc[0]), "phi": float(batting_rating['phi'].iloc[0])}
    if not bowling_rating.empty:
        response["bowling"] = {"mu": float(bowling_rating['mu'].iloc[0]), "phi": float(bowling_rating['phi'].iloc[0])}
    
    return response

@app.get("/players/{player_name}/history", response_model=Dict[str, List[Dict[str, Any]]])
async def get_player_history(player_name: str):
    batting_history = get_player_glicko_history(player_name, batting_ratings_df)
    bowling_history = get_player_glicko_history(player_name, bowling_ratings_df)

    if not batting_history and not bowling_history:
        raise HTTPException(status_code=404, detail="Player not found or no history available.")
    
    return {"batting_history": batting_history, "bowling_history": bowling_history}

@app.get("/players/{player_name}/impact", response_model=Dict[str, float])
async def get_player_impact(player_name: str):
    impact_batter = get_player_impact_score(player_name, player_impact_batter_df, 'batter')
    impact_bowler = get_player_impact_score(player_name, player_impact_bowler_df, 'bowler')

    if impact_batter == 0.0 and impact_bowler == 0.0: # Assuming 0.0 means not found for impact
        raise HTTPException(status_code=404, detail="Player not found or no impact score available.")
    
    return {"batting_impact": impact_batter, "bowling_impact": impact_bowler}

@app.get("/players/{player_name}/predict", response_model=Dict[str, float])
async def predict_player(player_name: str, player_type: str = "batter"): # player_type can be "batter" or "bowler"
    if player_type not in ["batter", "bowler"]:
        raise HTTPException(status_code=400, detail="player_type must be 'batter' or 'bowler'.")

    if player_type == "batter":
        prediction_result = predict_player_performance(player_name, batter_performance_model, is_batter=True)
    else:
        prediction_result = predict_player_performance(player_name, bowler_performance_model, is_batter=False)
    
    if "error" in prediction_result:
        raise HTTPException(status_code=404, detail=prediction_result["error"])
    
    return prediction_result

class MatchPredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_decision: str

@app.post("/matches/predict", response_model=Dict[str, str])
async def predict_match(request: MatchPredictionRequest):
    team1 = request.team1
    team2 = request.team2
    venue = request.venue
    toss_decision = request.toss_decision

    # --- Prepare features for prediction (similar to 06_predictive_analytics.py) ---
    # Get latest Glicko for teams (simplified: average of all players in team)
    team1_players = ipl_df[ipl_df['batting_team'] == team1]['batter'].unique()
    team2_players = ipl_df[ipl_df['batting_team'] == team2]['batter'].unique()

    team1_glicko_mu = latest_batting_glicko[latest_batting_glicko['player'].isin(team1_players)]['mu'].mean()
    team2_glicko_mu = latest_batting_glicko[latest_batting_glicko['player'].isin(team2_players)]['mu'].mean()

    # Get latest win rates (simplified: from the last match they played)
    # This is a placeholder. A robust solution would require pre-calculating and storing team form.
    team1_win_rate = 0.5 
    team2_win_rate = 0.5 

    # Create prediction dataframe
    pred_df = pd.DataFrame({
            'team1_avg_glicko': [team1_glicko_mu if not pd.isna(team1_glicko_mu) else 1500],
            'team2_avg_glicko': [team2_glicko_mu if not pd.isna(team2_glicko_mu) else 1500],
            'team1_win_rate': [team1_win_rate],
            'team2_win_rate': [team2_win_rate],
            'toss_decision': [toss_decision],
            'venue': [venue]
        })

    # Ensure categorical features are of 'category' dtype
    for col in ['toss_decision', 'venue']:
        pred_df[col] = pred_df[col].astype('category')
    
    # Ensure columns match training features
    pred_df.columns = match_outcome_model.feature_name_

    # Predict winner
    predicted_winner_encoded = match_outcome_model.predict(pred_df)[0]
    predicted_winner = le_winner.inverse_transform([predicted_winner_encoded])[0]

    return {"predicted_winner": predicted_winner}

# --- Instructions to Run the API ---
# To run this API, save it as 08_api.py and execute the following command in your terminal:
# uvicorn 08_api:app --reload --host 0.0.0.0 --port 8000
# Then, open your browser to http://localhost:8000/docs for the interactive API documentation (Swagger UI).