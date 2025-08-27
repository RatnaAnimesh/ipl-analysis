
import pandas as pd
import os
import numpy as np

# Constants
DATA_DIR = "/Users/ashishmishra/ipl-analytics"
OUTPUT_DIR = "/Users/ashishmishra/ipl-analytics"

def calculate_batting_impact(player_stats, context):
    """
    Calculates a single impact score for a batting performance.
    Placeholder implementation.
    """
    # Simple placeholder: runs scored
    return player_stats.get('runs', 0)

def calculate_bowling_impact(player_stats, context):
    """
    Calculates a single impact score for a bowling performance.
    Placeholder implementation.
    """
    # Simple placeholder: wickets * 20
    return player_stats.get('wickets', 0) * 20

def process_match(match_df):
    """
    Processes a single match to calculate impact scores for all players.
    """
    match_impact_scores = {}
    
    # Dummy context for now
    context = {
        'phase': 'middle', # Example context
        'match_importance': 1 
    }

    for _, delivery in match_df.iterrows():
        batsman = delivery['batsman']
        bowler = delivery['bowler']

        # This is a simplified aggregation. A real implementation would need to
        # aggregate player stats for the whole match before scoring.
        # For now, we'll just assign a dummy score per delivery for demonstration.

        batting_stats = {'runs': delivery['runs_off_bat']}
        bowling_stats = {'wickets': 1 if delivery['wicket_type'] != 'run out' and pd.notna(delivery['wicket_type']) else 0}

        if batsman not in match_impact_scores:
            match_impact_scores[batsman] = {'batting_impact': 0, 'bowling_impact': 0}
        if bowler not in match_impact_scores:
            match_impact_scores[bowler] = {'batting_impact': 0, 'bowling_impact': 0}

        match_impact_scores[batsman]['batting_impact'] += calculate_batting_impact(batting_stats, context)
        match_impact_scores[bowler]['bowling_impact'] += calculate_bowling_impact(bowling_stats, context)

    return match_impact_scores

def main():
    """
    Main function to run the player impact rating calculation.
    """
    print("Starting Player Impact Rating calculation...")

    # Load data
    try:
        file_path = os.path.join(DATA_DIR, 'ipl_data_enriched.csv')
        df = pd.read_csv(file_path)
        print("Successfully loaded ipl_data_enriched.csv")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        return

    # For demonstration, we'll just process the first match
    first_match_id = df['match_id'].iloc[0]
    first_match_df = df[df['match_id'] == first_match_id]
    
    print(f"\nProcessing first match (ID: {first_match_id})...")
    
    match_scores = process_match(first_match_df.copy())

    print("\n--- Player Impact Scores for Match ---")
    # Sort players by total impact
    sorted_players = sorted(match_scores.items(), key=lambda item: item[1]['batting_impact'] + item[1]['bowling_impact'], reverse=True)
    
    for player, scores in sorted_players[:10]: # Print top 10 for brevity
        print(f"Player: {player:<20} | Batting Impact: {scores['batting_impact']:.2f} | Bowling Impact: {scores['bowling_impact']:.2f}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
