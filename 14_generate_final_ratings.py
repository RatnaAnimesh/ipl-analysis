import pandas as pd
import os
from glicko_simulator import generate_and_save_ratings

# --- 1. Load Data ---
DATA_DIR = "/Users/ashishmishra/ipl-analytics"
ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_enriched.csv'))
ipl_df['date'] = pd.to_datetime(ipl_df['date'])
ipl_df.sort_values(by='date', inplace=True)

# --- 2. Define Best Parameters ---
# These are the optimal parameters found by the optimization script.
best_params = {
    'batsman_norm_runs': 25.002199582651414,
    'batsman_norm_sr': 147.17480577238206,
    'batsman_not_out_bonus': 0.26364535303020564,
    'bowler_economy_norm': 10.447291934213858,
    'bowler_wicket_base_value': 0.1006074550324688,
    'bowler_death_over_multiplier': 1.3668964687699492,
    'bowler_powerplay_multiplier': 1.042664224054015,
    'bowler_economy_weight': 0.6516456047697832
}

# --- 3. Generate and Save Final Ratings ---
if __name__ == "__main__":
    print("Generating final ratings using the best parameters found...")
    generate_and_save_ratings(best_params, ipl_df, DATA_DIR)
    print("\nFinal ratings have been saved to:")
    print(f"- {os.path.join(DATA_DIR, 'glicko_batting_ratings_optimized.csv')}")
    print(f"- {os.path.join(DATA_DIR, 'glicko_bowling_ratings_optimized.csv')}")
