import optuna
import pandas as pd
import os
from glicko_simulator import run_glicko_simulation

# --- 1. Load Data ---
# Load the data once to be passed to each trial, avoiding repeated reads.
DATA_DIR = "/Users/ashishmishra/ipl-analytics"
ipl_df = pd.read_csv(os.path.join(DATA_DIR, 'ipl_data_enriched.csv'))
ipl_df['date'] = pd.to_datetime(ipl_df['date'])
ipl_df.sort_values(by='date', inplace=True)

# --- 2. Define the Objective Function for Optuna ---
def objective(trial):
    """
    This function is called by Optuna for each trial.
    It suggests a set of parameters, runs the simulation, and returns the loss.
    """
    params = {
        # Batsman parameters
        'batsman_norm_runs': trial.suggest_float('batsman_norm_runs', 25.0, 55.0),
        'batsman_norm_sr': trial.suggest_float('batsman_norm_sr', 120.0, 180.0),
        'batsman_not_out_bonus': trial.suggest_float('batsman_not_out_bonus', 0.05, 0.3),
        
        # Bowler parameters
        'bowler_economy_norm': trial.suggest_float('bowler_economy_norm', 7.0, 11.0),
        'bowler_wicket_base_value': trial.suggest_float('bowler_wicket_base_value', 0.1, 0.4),
        'bowler_death_over_multiplier': trial.suggest_float('bowler_death_over_multiplier', 1.1, 1.5),
        'bowler_powerplay_multiplier': trial.suggest_float('bowler_powerplay_multiplier', 1.0, 1.3),
        'bowler_economy_weight': trial.suggest_float('bowler_economy_weight', 0.3, 0.7),
    }

    # Run the simulation with the suggested parameters
    loss = run_glicko_simulation(params, ipl_df)

    # Print progress
    trial_number = trial.number
    print(f"Trial {trial_number}: Loss = {loss:.4f}")

    return loss

# --- 3. Run the Optimization ---
if __name__ == "__main__":
    print("Starting ROUND 2 of hyperparameter optimization...")
    print("Using best parameters from the previous run as a starting point.")

    # Best parameters from the first run to warm-start the new study
    best_params_from_round_1 = {
        'batsman_norm_runs': 30.70696337550536,
        'batsman_norm_sr': 146.58500994440132,
        'batsman_not_out_bonus': 0.2670894255290553,
        'bowler_economy_norm': 10.315696808995524,
        'bowler_wicket_base_value': 0.16330745397232857,
        'bowler_death_over_multiplier': 1.4455306990962742,
        'bowler_powerplay_multiplier': 1.1445525397787464,
        'bowler_economy_weight': 0.6214791465748212
    }

    # Create a study object and specify the direction is to minimize the objective.
    study = optuna.create_study(direction='minimize')

    # Enqueue the best trial from the previous run to warm-start the optimization
    study.enqueue_trial(best_params_from_round_1)
    
    # Start the optimization for a longer duration.
    study.optimize(objective, n_trials=2000)

    # --- 4. Print Results ---
    print("\nOptimization Round 2 finished.")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial achieved a loss of: {best_trial.value:.4f}")

    print("\nBest parameters found:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
