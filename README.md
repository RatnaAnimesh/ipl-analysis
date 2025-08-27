# 🏏 IPL Player Ranking & Performance Analytics System

## Project Overview

This project aims to build a professional-grade IPL player ranking and performance analytics system using the comprehensive IPL dataset (2008-2020) from Kaggle. The goal is to create a dynamic, explainable, and scalable platform that can be utilized by IPL teams, analysts, and broadcasters for in-depth player and match analysis.

## Features

This system encompasses several key phases, from raw data ingestion to interactive visualization and predictive modeling:

1.  **Data Ingestion & Cleaning:**
    *   Loading and initial cleaning of IPL Kaggle datasets (ball-by-ball, match info).
    *   Handling missing values, inconsistencies, and unifying player/team identifiers.

2.  **Feature Engineering:**
    *   Enriching the dataset with contextual features (e.g., match phase: Powerplay, Middle, Death).
    *   Calculating player-specific performance metrics (e.g., strike rate, economy rate).

3.  **Player Rating Model (Glicko-2):**
    *   Implementation of a simplified Glicko-2 rating system to dynamically rank players based on their batting and bowling performances.
    *   Ratings update after each match, reflecting player form and strength.

4.  **Machine Learning-Based Player Impact Score:**
    *   A LightGBM regression model predicts runs scored per ball.
    *   Player impact is quantified by analyzing the residuals (actual runs vs. predicted runs), indicating performance beyond expectation.

5.  **Predictive Analytics:**
    *   **Player Performance Forecasting:** LightGBM models predict a player's expected runs/wickets in upcoming matches based on historical data, Glicko ratings, and impact scores.
    *   **Match Outcome Prediction:** A LightGBM classification model predicts match winners based on aggregated team Glicko ratings, team form, and match context.

6.  **RESTful API (FastAPI):**
    *   A robust API built with FastAPI to expose player ratings, impact scores, and predictions, serving as the backend for any future frontend application.

7.  **Explainability and Reporting:**
    *   Utilizes SHAP (SHapley Additive exPlanations) to provide insights into model predictions and feature importance.
    *   Generates text-based reports summarizing key player metrics and model explanations.

## Project Architecture

The project is structured into modular Python scripts, each handling a specific phase of the data pipeline and modeling. This ensures maintainability, reusability, and clarity.

```
ipl-analytics/
├── 01_data_ingestion.py        # Loads raw data, initial exploration
├── 02_data_cleaning.py         # Cleans and merges raw data
├── 03_feature_engineering.py   # Creates new features for modeling
├── 04_glicko_rating_system.py  # Implements Glicko-2 rating system
├── 05_player_impact_model.py   # Trains ML model for player impact
├── 06_predictive_analytics.py  # Trains models for player/match prediction
├── 08_api.py                   # FastAPI application for API endpoints
├── 09_explainability_reporting.py # Generates model explanations and reports
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── ipl_data_cleaned.csv        # Output of 02_data_cleaning.py
├── ipl_data_enriched.csv       # Output of 03_feature_engineering.py
├── glicko_batting_ratings.csv  # Output of 04_glicko_rating_system.py
├── glicko_bowling_ratings.csv  # Output of 04_glicko_rating_system.py
├── player_impact_batter.csv    # Output of 05_player_impact_model.py
├── player_impact_bowler.csv    # Output of 05_player_impact_model.py
├── player_impact_model.pkl     # Saved ML model from 05_player_impact_model.py
├── batter_performance_model.pkl # Saved ML model from 06_predictive_analytics.py
├── bowler_performance_model.pkl # Saved ML model from 06_predictive_analytics.py
└── match_outcome_model.pkl     # Saved ML model from 06_predictive_analytics.py
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Kaggle API key configured (for data download)

### 1. Clone the Repository

```bash
git clone https://github.com/RatnaAnimesh/ipl-analysis.git
cd ipl-analysis
```

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install pandas numpy scikit-learn lightgbm fastapi uvicorn[standard] shap joblib
```

### 3. Download the Dataset

This project uses the "IPL Complete Dataset 2008-2020" from Kaggle. Ensure your Kaggle API is configured (usually by placing `kaggle.json` in `~/.kaggle/`).

```bash
kaggle datasets download -d patrickb1912/ipl-complete-dataset-20082020
unzip ipl-complete-dataset-20082020.zip -d .
rm ipl-complete-dataset-20082020.zip
```

## How to Run

Execute the Python scripts in the following order to generate the necessary data and models. Ensure you are in the `ipl-analysis` directory.

1.  **Data Ingestion & Cleaning:**
    ```bash
    python 01_data_ingestion.py
    python 02_data_cleaning.py
    ```

2.  **Feature Engineering:**
    ```bash
    python 03_feature_engineering.py
    ```

3.  **Player Rating Model (Glicko-2):**
    ```bash
    python 04_glicko_rating_system.py
    ```

4.  **Machine Learning-Based Player Impact Score:**
    ```bash
    python 05_player_impact_model.py
    ```

5.  **Predictive Analytics:**
    ```bash
    python 06_predictive_analytics.py
    ```

6.  **Explainability and Reporting:**
    ```bash
    python 09_explainability_reporting.py
    ```

### Running the RESTful API

After running all the above scripts, you can launch the FastAPI application:

```bash
uvicorn 08_api:app --reload --host 0.0.0.0 --port 8000
```

Then, open your browser to `http://localhost:8000/docs` for the interactive API documentation (Swagger UI).

## Future Enhancements

*   **Dedicated Frontend:** Develop a dedicated frontend using React/Angular/Vue.js to consume the API and provide a rich, interactive user experience.
*   **More Sophisticated Models:** Explore advanced ML techniques for player impact, win probability, and injury risk modeling.
*   **Real-time Data Integration:** Integrate with live match data sources.
*   **Deployment:** Deploy the API to a cloud platform (e.g., AWS, GCP, Azure).
*   **Comprehensive Documentation:** Expand user and technical manuals.

---
