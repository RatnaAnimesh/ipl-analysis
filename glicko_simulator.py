import pandas as pd
import numpy as np
import os

# --- Glicko-2 Constants and Helper Functions ---
RATING_DEFAULT = 1500.0
RD_DEFAULT = 350.0
SIGMA_DEFAULT = 0.06

def g(phi):
    return 1 / np.sqrt(1 + (3 * phi**2) / (np.pi**2))

def E(r, r_j, phi_j):
    return 1 / (1 + np.exp(-g(phi_j) * (r - r_j)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def update_rating_simplified(player_rating, player_rd, opponent_data, sigma_fixed=SIGMA_DEFAULT):
    mu = (player_rating - RATING_DEFAULT) / RD_DEFAULT
    phi = player_rd / RD_DEFAULT

    v_sum = 0
    delta_sum = 0
    for opp_r, opp_rd, outcome in opponent_data:
        opp_mu = (opp_r - RATING_DEFAULT) / RD_DEFAULT
        opp_phi = opp_rd / RD_DEFAULT
        g_opp_phi = g(opp_phi)
        E_val = E(mu, opp_mu, opp_phi)
        v_sum += (g_opp_phi**2) * E_val * (1 - E_val)
        delta_sum += g_opp_phi * (outcome - E_val)
    
    v = 1 / v_sum if v_sum > 0 else 1
    delta = v * delta_sum

    new_phi = 1 / np.sqrt(1 / (phi**2 + sigma_fixed**2) + 1 / v)
    new_mu = mu + new_phi * delta_sum
    new_rating = new_mu * RD_DEFAULT + RATING_DEFAULT
    new_rd = new_phi * RD_DEFAULT

    return new_rating, new_rd

def calculate_batsman_performance_score(batter_df, bowling_ratings, params):
    runs_scored = batter_df['batsman_runs'].sum()
    balls_faced = len(batter_df)
    if balls_faced == 0: return 0.0

    total_adjusted_runs = 0
    for _, ball in batter_df.iterrows():
        bowler = ball['bowler']
        bowler_rating = bowling_ratings.get(bowler, {'mu': 1500})['mu']
        runs_this_ball = ball['batsman_runs']
        opponent_multiplier = max(0.5, bowler_rating / 1500.0)
        total_adjusted_runs += (runs_this_ball * opponent_multiplier)

    strike_rate = (runs_scored / balls_faced) * 100
    base_score = (total_adjusted_runs / params['batsman_norm_runs']) * (strike_rate / params['batsman_norm_sr'])

    is_not_out = batter_df['is_wicket'].sum() == 0
    not_out_bonus = params['batsman_not_out_bonus'] if is_not_out and runs_scored > 20 else 0
    
    final_score = base_score + not_out_bonus
    return max(0, min(1, final_score))

def calculate_bowler_performance_score(bowler_df, batting_ratings, params):
    runs_conceded = bowler_df['total_runs'].sum()
    balls_bowled = len(bowler_df)
    if balls_bowled == 0: return 0.0

    economy_rate = (runs_conceded / balls_bowled) * 6
    base_economy_score = max(0, 1 - (economy_rate / params['bowler_economy_norm']))

    total_wicket_value = 0
    wickets_df = bowler_df[bowler_df['is_wicket'] == 1]
    
    for _, wicket in wickets_df.iterrows():
        dismissed_batter = wicket['batter']
        batter_rating = batting_ratings.get(dismissed_batter, {'mu': 1500})['mu']
        opponent_multiplier = max(0.5, batter_rating / 1500.0)
        over = wicket['over']
        phase_multiplier = 1.0
        if over >= 16: phase_multiplier = params['bowler_death_over_multiplier']
        elif over <= 6: phase_multiplier = params['bowler_powerplay_multiplier']
        total_wicket_value += (params['bowler_wicket_base_value'] * opponent_multiplier * phase_multiplier)

    final_score = (params['bowler_economy_weight'] * base_economy_score) + ((1 - params['bowler_economy_weight']) * total_wicket_value)
    return max(0, min(1, final_score))

def run_glicko_simulation(params, ipl_df):
    all_players = set(ipl_df['batter'].unique()) | set(ipl_df['bowler'].unique())
    
    batting_ratings = {p: {'mu': RATING_DEFAULT, 'phi': RD_DEFAULT} for p in all_players}
    bowling_ratings = {p: {'mu': RATING_DEFAULT, 'phi': RD_DEFAULT} for p in all_players}

    total_squared_error = 0
    innings_count = 0

    for match_id, match_df in ipl_df.groupby('match_id'):
        # --- Score Prediction and Loss Calculation (New Logic) ---
        for innings in [1, 2]:
            innings_df = match_df[match_df['inning'] == innings]
            if innings_df.empty:
                continue

            # Correctly identify players in the innings
            batting_team_players = set(innings_df['batter'].unique())
            bowling_team_players = set(innings_df['bowler'].unique())

            batting_strength = sum(batting_ratings[p]['mu'] for p in batting_team_players if p in batting_ratings)
            bowling_strength = sum(bowling_ratings[p]['mu'] for p in bowling_team_players if p in bowling_ratings)

            predicted_score = 160 + 0.05 * (batting_strength - bowling_strength)
            actual_score = innings_df['total_runs'].sum()

            total_squared_error += (predicted_score - actual_score)**2
            innings_count += 1

        # --- Rating updates proceed as before ---
        batting_match_data = {}
        for batter, batter_df in match_df.groupby('batter'):
            outcome = calculate_batsman_performance_score(batter_df, bowling_ratings, params)
            opp_data = [(bowling_ratings[b]['mu'], bowling_ratings[b]['phi'], outcome) for b in batter_df['bowler'].unique() if b in bowling_ratings]
            if opp_data: batting_match_data[batter] = opp_data

        bowling_match_data = {}
        for bowler, bowler_df in match_df.groupby('bowler'):
            outcome = calculate_bowler_performance_score(bowler_df, batting_ratings, params)
            opp_data = [(batting_ratings[b]['mu'], batting_ratings[b]['phi'], outcome) for b in bowler_df['batter'].unique() if b in batting_ratings]
            if opp_data: bowling_match_data[bowler] = opp_data

        for player, opp_data in batting_match_data.items():
            new_rating, new_rd = update_rating_simplified(batting_ratings[player]['mu'], batting_ratings[player]['phi'], opp_data)
            batting_ratings[player]['mu'] = new_rating
            batting_ratings[player]['phi'] = new_rd

        for player, opp_data in bowling_match_data.items():
            new_rating, new_rd = update_rating_simplified(bowling_ratings[player]['mu'], bowling_ratings[player]['phi'], opp_data)
            bowling_ratings[player]['mu'] = new_rating
            bowling_ratings[player]['phi'] = new_rd

    return total_squared_error / innings_count if innings_count > 0 else float('inf')

def generate_and_save_ratings(params, ipl_df, output_dir):
    all_players = set(ipl_df['batter'].unique()) | set(ipl_df['bowler'].unique())
    
    batting_ratings = {p: {'mu': RATING_DEFAULT, 'phi': RD_DEFAULT} for p in all_players}
    bowling_ratings = {p: {'mu': RATING_DEFAULT, 'phi': RD_DEFAULT} for p in all_players}

    batting_history = []
    bowling_history = []

    for match_id, match_df in ipl_df.groupby('match_id'):
        date = match_df['date'].iloc[0]

        batting_match_data = {}
        for batter, batter_df in match_df.groupby('batter'):
            outcome = calculate_batsman_performance_score(batter_df, bowling_ratings, params)
            opp_data = [(bowling_ratings[b]['mu'], bowling_ratings[b]['phi'], outcome) for b in batter_df['bowler'].unique() if b in bowling_ratings]
            if opp_data: batting_match_data[batter] = opp_data

        bowling_match_data = {}
        for bowler, bowler_df in match_df.groupby('bowler'):
            outcome = calculate_bowler_performance_score(bowler_df, batting_ratings, params)
            opp_data = [(batting_ratings[b]['mu'], batting_ratings[b]['phi'], outcome) for b in bowler_df['batter'].unique() if b in batting_ratings]
            if opp_data: bowling_match_data[bowler] = opp_data

        for player, opp_data in batting_match_data.items():
            new_rating, new_rd = update_rating_simplified(batting_ratings[player]['mu'], batting_ratings[player]['phi'], opp_data)
            batting_ratings[player]['mu'] = new_rating
            batting_ratings[player]['phi'] = new_rd

        for player, opp_data in bowling_match_data.items():
            new_rating, new_rd = update_rating_simplified(bowling_ratings[player]['mu'], bowling_ratings[player]['phi'], opp_data)
            bowling_ratings[player]['mu'] = new_rating
            bowling_ratings[player]['phi'] = new_rd

        for player in all_players:
            batting_history.append([date, match_id, player, batting_ratings[player]['mu'], batting_ratings[player]['phi']])
            bowling_history.append([date, match_id, player, bowling_ratings[player]['mu'], bowling_ratings[player]['phi']])

    batting_history_df = pd.DataFrame(batting_history, columns=['date', 'match_id', 'player', 'mu', 'phi'])
    bowling_history_df = pd.DataFrame(bowling_history, columns=['date', 'match_id', 'player', 'mu', 'phi'])

    batting_history_df.to_csv(os.path.join(output_dir, 'glicko_batting_ratings_optimized.csv'), index=False)
    bowling_history_df.to_csv(os.path.join(output_dir, 'glicko_bowling_ratings_optimized.csv'), index=False)
    print("Successfully generated and saved optimized ratings.")
