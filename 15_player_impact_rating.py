import pandas as pd
import os
import numpy as np
import optuna

# Constants
DATA_DIR = "/Users/ashishmishra/ipl-analytics"
OUTPUT_DIR = "/Users/ashishmishra/ipl-analytics"

def calculate_batting_impact(player_stats, context, params):
    """
    Calculates a single impact score for a batting performance.
    """
    runs = player_stats.get('runs', 0)
    balls_faced = player_stats.get('balls_faced', 0)
    is_out = player_stats.get('is_out', 0)
    boundary_runs = player_stats.get('boundary_runs', 0) # New stat
    opponent_strength = player_stats.get('opponent_strength', 1500) # Default to 1500

    if balls_faced == 0: return 0.0

    strike_rate = (runs / balls_faced) * 100
    boundary_percentage = (boundary_runs / runs) * 100 if runs > 0 else 0

    # Incorporate opponent strength: higher opponent strength means higher impact for same performance
    # Normalize opponent strength around 1500 (base rating)
    strength_multiplier = opponent_strength / 1500.0

    # Apply phase multipliers
    phase_multiplier = 1.0
    # Assuming context['inning'] and context['over'] are available from delivery data
    # For now, we'll use a simplified approach based on match_df's first inning
    # This will need to be refined when processing all deliveries
    if 'inning' in context and 'over' in context:
        if context['over'] <= 6: # Powerplay
            phase_multiplier = params['powerplay_multiplier_bat']
        elif context['over'] >= 16: # Death overs
            phase_multiplier = params['death_over_multiplier_bat']

    # Use optimized parameters
    impact = (runs * params['run_weight'] + 
              strike_rate * params['strike_rate_weight'] + 
              boundary_percentage * params['boundary_weight'] - 
              (is_out * params['is_out_penalty']) + 
              (1 - is_out) * params['not_out_bonus']) * strength_multiplier * params['batting_opponent_strength_multiplier'] * phase_multiplier
    return impact

def calculate_bowling_impact(player_stats, context, params, maiden_overs):
    """
    Calculates a single impact score for a bowling performance.
    """
    runs_conceded = player_stats.get('runs_conceded', 0)
    balls_bowled = player_stats.get('balls_bowled', 0)
    wickets_taken = player_stats.get('wickets_taken', 0)
    dot_balls = player_stats.get('dot_balls', 0) # New stat
    opponent_strength = player_stats.get('opponent_strength', 1500) # Default to 1500

    if balls_bowled == 0: return 0.0

    economy_rate = (runs_conceded / balls_bowled) * 6
    dot_ball_percentage = (dot_balls / balls_bowled) * 100

    # Incorporate opponent strength: higher opponent strength means higher impact for restricting them
    # Normalize opponent strength around 1500 (base rating)
    strength_multiplier = opponent_strength / 1500.0

    # Apply phase multipliers
    phase_multiplier = 1.0
    # Assuming context['inning'] and context['over'] are available from delivery data
    # For now, we'll use a simplified approach based on match_df's first inning
    # This will need to be refined when processing all deliveries
    if 'inning' in context and 'over' in context:
        if context['over'] <= 6: # Powerplay
            phase_multiplier = params['powerplay_multiplier_bowl']
        elif context['over'] >= 16: # Death overs
            phase_multiplier = params['death_over_multiplier_bowl']

    # Use optimized parameters
    impact = (wickets_taken * params['wicket_weight'] + 
              dot_ball_percentage * params['dot_ball_weight'] - 
              economy_rate * params['economy_weight'] +
              maiden_overs * params['maiden_over_bonus']) * strength_multiplier * params['bowling_opponent_strength_multiplier'] * phase_multiplier
    return impact

def objective(trial, df, batting_ratings, bowling_ratings):
    # Define hyperparameters to be optimized
    # Batting weights
    run_weight = trial.suggest_float("run_weight", 0.5, 2.0)
    strike_rate_weight = trial.suggest_float("strike_rate_weight", 0.0, 0.5)
    is_out_penalty = trial.suggest_float("is_out_penalty", 0.0, 10.0)
    boundary_weight = trial.suggest_float("boundary_weight", 0.0, 0.5) # New parameter
    batting_opponent_strength_multiplier = trial.suggest_float("batting_opponent_strength_multiplier", 0.5, 1.5)

    # Bowling weights
    wicket_weight = trial.suggest_float("wicket_weight", 10.0, 50.0)
    economy_weight = trial.suggest_float("economy_weight", 0.0, 5.0)
    dot_ball_weight = trial.suggest_float("dot_ball_weight", 0.0, 0.5) # New parameter
    bowling_opponent_strength_multiplier = trial.suggest_float("bowling_opponent_strength_multiplier", 0.5, 1.5)

    # Phase multipliers (new parameters)
    powerplay_multiplier_bat = trial.suggest_float("powerplay_multiplier_bat", 0.8, 1.2)
    death_over_multiplier_bat = trial.suggest_float("death_over_multiplier_bat", 0.8, 1.2)
    powerplay_multiplier_bowl = trial.suggest_float("powerplay_multiplier_bowl", 0.8, 1.2)
    death_over_multiplier_bowl = trial.suggest_float("death_over_multiplier_bowl", 0.8, 1.2)

    # New impact features
    not_out_bonus = trial.suggest_float("not_out_bonus", 0.0, 5.0)
    maiden_over_bonus = trial.suggest_float("maiden_over_bonus", 0.0, 5.0)

    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    match_data_for_lgbm = []

    all_match_ids = df['match_id'].unique()

    for match_id in all_match_ids:
        match_df = df[df['match_id'] == match_id].copy()
        
        match_impact_scores, actual_margin_of_victory = process_match(match_df, trial.params, batting_ratings, bowling_ratings)

        team_impacts = {}
        for _, delivery in match_df.iterrows():
            batting_team = delivery['batting_team']
            bowling_team = delivery['bowling_team']

            if batting_team not in team_impacts: team_impacts[batting_team] = {'batting': 0, 'bowling': 0}
            if bowling_team not in team_impacts: team_impacts[bowling_team] = {'batting': 0, 'bowling': 0}

            batter = delivery['batter']
            bowler = delivery['bowler']

            if batter in match_impact_scores: team_impacts[batting_team]['batting'] += match_impact_scores[batter]['batting_impact']
            if bowler in match_impact_scores: team_impacts[bowling_team]['bowling'] += match_impact_scores[bowler]['bowling_impact']

        if len(team_impacts) == 2:
            team1_name = match_df['batting_team'].iloc[0]
            
            team2_df = match_df[match_df['batting_team'] != team1_name]
            if not team2_df.empty:
                team2_name = team2_df['batting_team'].iloc[0]
            else:
                continue

            team1_batting_impact = team_impacts[team1_name]['batting']
            team1_bowling_impact = team_impacts[team1_name]['bowling']
            team2_batting_impact = team_impacts[team2_name]['batting']
            team2_bowling_impact = team_impacts[team2_name]['bowling']

            match_data_for_lgbm.append({
                'team1_batting_impact': team1_batting_impact,
                'team1_bowling_impact': team1_bowling_impact,
                'team2_batting_impact': team2_batting_impact,
                'team2_bowling_impact': team2_bowling_impact,
                'actual_margin_of_victory': actual_margin_of_victory
            })

    if not match_data_for_lgbm:
        return float('inf')

    lgbm_df = pd.DataFrame(match_data_for_lgbm)

    X = lgbm_df[['team1_batting_impact', 'team1_bowling_impact', 'team2_batting_impact', 'team2_bowling_impact']]
    y = lgbm_df['actual_margin_of_victory']

    # Train a simple LightGBM Regressor
    model = lgb.LGBMRegressor(objective='regression_l1', metric='mae', n_estimators=100, learning_rate=0.1, random_state=42, device='gpu')
    model.fit(X, y)

    # Predict and calculate MSE
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    return mse

def process_match(match_df, params, batting_ratings, bowling_ratings):
    """
    Processes a single match to calculate impact scores for all players.
    """
    player_match_stats = {}

    for player in pd.concat([match_df['batter'], match_df['bowler']]).unique():
        player_match_stats[player] = {
            'batsman_runs': 0,
            'balls_faced': 0,
            'is_wicket_batter': 0,
            'boundary_runs': 0,
            'total_runs_conceded': 0,
            'balls_bowled': 0,
            'wickets_taken': 0,
            'dot_balls': 0,
            'maiden_overs': 0, # New stat
            'deliveries': []
        }

    for _, delivery in match_df.iterrows():
        batter = delivery['batter']
        bowler = delivery['bowler']
        runs_off_bat = delivery['batsman_runs']
        total_runs = delivery['total_runs']
        is_wicket_delivery = 1 if delivery['is_wicket'] == 1 else 0

        player_match_stats[batter]['batsman_runs'] += runs_off_bat
        player_match_stats[batter]['balls_faced'] += 1
        if runs_off_bat == 4 or runs_off_bat == 6:
            player_match_stats[batter]['boundary_runs'] += runs_off_bat
        if is_wicket_delivery == 1 and delivery['player_dismissed'] == batter:
            player_match_stats[batter]['is_wicket_batter'] = 1

        player_match_stats[bowler]['total_runs_conceded'] += total_runs
        player_match_stats[bowler]['balls_bowled'] += 1
        if total_runs == 0:
            player_match_stats[bowler]['dot_balls'] += 1
        dismissal_kind = delivery.get('dismissal_kind')
        if is_wicket_delivery == 1 and dismissal_kind not in ['run out', 'retired hurt', 'obstructing the field']:
            player_match_stats[bowler]['wickets_taken'] += 1

        # Track runs per over for maiden over calculation
        over_key = (delivery['match_id'], delivery['inning'], delivery['over'], bowler)
        if 'overs_data' not in player_match_stats[bowler]:
            player_match_stats[bowler]['overs_data'] = {}
        if over_key not in player_match_stats[bowler]['overs_data']:
            player_match_stats[bowler]['overs_data'][over_key] = {'runs_conceded_this_over': 0, 'balls_this_over': 0}
        player_match_stats[bowler]['overs_data'][over_key]['runs_conceded_this_over'] += total_runs
        player_match_stats[bowler]['overs_data'][over_key]['balls_this_over'] += 1

        player_match_stats[batter]['deliveries'].append(delivery)
        player_match_stats[bowler]['deliveries'].append(delivery)

    # Calculate maiden overs after processing all deliveries for the match
    for player, stats in player_match_stats.items():
        if 'overs_data' in stats:
            for over_key, over_data in stats['overs_data'].items():
                if over_data['balls_this_over'] == 6 and over_data['runs_conceded_this_over'] == 0:
                    player_match_stats[player]['maiden_overs'] += 1

    match_impact_scores = {}
    context = {
        'match_id': match_df['match_id'].iloc[0],
        'inning': match_df['inning'].iloc[0],
        'team_total': match_df['total_runs'].sum()
    }

    for player, stats in player_match_stats.items():
        batting_impact = 0
        bowling_impact = 0

        if stats['balls_faced'] > 0:
            opponent_bowlers = [d['bowler'] for d in stats['deliveries'] if d['batter'] == player]
            avg_opponent_bowler_strength = np.mean([bowling_ratings.get(b, 1500) for b in opponent_bowlers]) if opponent_bowlers else 1500

            batting_impact = calculate_batting_impact({
                'runs': stats['batsman_runs'],
                'balls_faced': stats['balls_faced'],
                'is_out': stats['is_wicket_batter'],
                'boundary_runs': stats['boundary_runs'],
                'opponent_strength': avg_opponent_bowler_strength
            }, context, params)

        if stats['balls_bowled'] > 0:
            opponent_batsmen = [d['batter'] for d in stats['deliveries'] if d['bowler'] == player]
            avg_opponent_batsman_strength = np.mean([batting_ratings.get(b, 1500) for b in opponent_batsmen]) if opponent_batsmen else 1500

            bowling_impact = calculate_bowling_impact({
                'runs_conceded': stats['total_runs_conceded'],
                'balls_bowled': stats['balls_bowled'],
                'wickets_taken': stats['wickets_taken'],
                'dot_balls': stats['dot_balls'],
                'maiden_overs': stats['maiden_overs'], # Pass maiden_overs
                'opponent_strength': avg_opponent_batsman_strength
            }, context, params, stats['maiden_overs']) # Pass maiden_overs as a separate argument
        
        match_impact_scores[player] = {
            'batting_impact': batting_impact,
            'bowling_impact': bowling_impact
        }

    team1_name = match_df['batting_team'].iloc[0]
    
    team2_df = match_df[match_df['batting_team'] != team1_name]
    if not team2_df.empty:
        team2_name = team2_df['batting_team'].iloc[0]
        team2_runs = team2_df['total_runs'].sum()
    else:
        team2_name = None
        team2_runs = 0

    team1_runs = match_df[match_df['batting_team'] == team1_name]['total_runs'].sum()

    margin_of_victory = team1_runs - team2_runs

    return match_impact_scores, margin_of_victory

def main():
    """
    Main function to run the player impact rating calculation and Optuna optimization.
    """
    print("Starting Player Impact Rating calculation and Optuna optimization...")

    try:
        file_path = os.path.join(DATA_DIR, 'ipl_data_enriched.csv')
        df = pd.read_csv(file_path)
        print("Successfully loaded ipl_data_enriched.csv")

        batting_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_batting_ratings_optimized.csv'))
        bowling_ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'glicko_bowling_ratings_optimized.csv'))
        
        latest_batting_ratings = batting_ratings_df.groupby('player')['mu'].last().to_dict()
        latest_bowling_ratings = bowling_ratings_df.groupby('player')['mu'].last().to_dict()
        
        print("Successfully loaded Glicko ratings.")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e}")
        return

    df_filtered = df.copy()
    print(f"\nUsing all available data. Total deliveries: {len(df_filtered)}")

    print("\nStarting Optuna study...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df_filtered, latest_batting_ratings, latest_bowling_ratings), n_trials=1000)

    print("\nOptuna study finished.")
    print(f"Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()
