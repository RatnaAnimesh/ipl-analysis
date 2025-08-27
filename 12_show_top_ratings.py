import pandas as pd
import os

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Define file paths
BATTING_RATINGS_FILE = os.path.join(DATA_DIR, 'glicko_batting_ratings_optimized.csv')
BOWLING_RATINGS_FILE = os.path.join(DATA_DIR, 'glicko_bowling_ratings_optimized.csv')

def display_top_100(ratings_file, rating_type):
    """
    Reads a ratings file, finds the peak rating for each player,
    and prints the top 100 all-time ratings.
    """
    if not os.path.exists(ratings_file):
        print(f"Error: Ratings file not found at {ratings_file}")
        print("Please run '04_glicko_rating_system.py' first to generate the ratings.")
        return

    print(f"--- Top 100 All-Time Peak {rating_type} Ratings ---")
    print("=" * 50)

    # Load the ratings history
    df = pd.read_csv(ratings_file)

    # Find the peak rating for each player
    # We group by player and find the index of the maximum 'mu' (rating)
    peak_ratings_idx = df.groupby('player')['mu'].idxmax()
    peak_ratings_df = df.loc[peak_ratings_idx]

    # Sort by the peak rating in descending order
    top_100 = peak_ratings_df.sort_values('mu', ascending=False).head(100)

    # Select and rename columns for display
    top_100_display = top_100[['player', 'mu', 'phi', 'date']].copy()
    top_100_display.rename(columns={
        'mu': 'Peak Rating',
        'phi': 'RD (at peak)',
        'date': 'Date of Peak'
    }, inplace=True)

    # Format the rating to be more readable
    top_100_display['Peak Rating'] = top_100_display['Peak Rating'].round(2)
    top_100_display['RD (at peak)'] = top_100_display['RD (at peak)'].round(2)


    # Reset index for clean printing
    top_100_display.reset_index(drop=True, inplace=True)
    top_100_display.index += 1 # Start index from 1 for ranking

    print(top_100_display.to_string())
    print("\n" * 2)


if __name__ == "__main__":
    display_top_100(BATTING_RATINGS_FILE, "Batting")
    display_top_100(BOWLING_RATINGS_FILE, "Bowling")
