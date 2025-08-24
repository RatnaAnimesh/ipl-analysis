import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

# Define the absolute path to the data directory
DATA_DIR = "/Users/ashishmishra/ipl-analytics"

# Load the datasets
try:
    deliveries_df = pd.read_csv(os.path.join(DATA_DIR, 'deliveries.csv'))
    matches_df = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the data files are in the '{DATA_DIR}' directory.")
    exit()

# Display the first few rows of each dataframe
print("Deliveries DataFrame Head:")
print(deliveries_df.head())
print("\n" + "="*50 + "\n")

print("Matches DataFrame Head:")
print(matches_df.head())
print("\n" + "="*50 + "\n")

# Display concise summary of each dataframe
print("Deliveries DataFrame Info:")
deliveries_df.info()
print("\n" + "="*50 + "\n")

print("Matches DataFrame Info:")
matches_df.info()