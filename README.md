# IPL Player Rating and Predictive Analytics System

This project aims to develop a robust, context-aware, and predictive player rating and analytics system for the Indian Premier League (IPL). The journey has involved iterative development, rigorous testing, and a willingness to pivot approaches based on empirical results.

## Predictive Analytics

Beyond player ratings, this project includes a predictive analytics component that forecasts player performance and match outcomes. This is achieved through a suite of LightGBM models that leverage the player impact ratings and other features.

The predictive models include:
*   **Batter Performance Model:** Predicts the number of runs a batter will score.
*   **Bowler Performance Model:** Predicts the number of wickets a bowler will take.
*   **Match Outcome Model:** Predicts the winner of a match.

These models are trained on historical data and can be used to generate predictions for future matches.

## Player Impact Rating: The Core of the System

Learning from the limitations of a Glicko-2 based system, we are now pivoting to a more direct, interpretable, and robust approach: the **Player Impact Rating**.

*   **Core Idea:** Instead of a relative rating system, we will calculate a direct "Impact Score" for each player's performance in every match. This score will quantify their contribution to the team's success.

*   **Optimization Objective:** The parameters for calculating these impact scores will be optimized using `optuna` to directly predict the **margin of victory** in a match. This ensures a strong link between individual performance and team outcome.

*   **Rating Calculation:** A player's overall rating will be a **time-decayed moving average** of their individual match impact scores. This design inherently addresses the "Legacy Bias" by giving more weight to recent performances and allowing for dynamic shifts in player standing over time.

This new approach promises to deliver a more intuitive, accurate, and actionable player rating system, providing a stronger foundation for further quantitative analysis in cricket.

## Technical Notes

### Memory Optimization

During development, we encountered significant memory issues, leading to "Killed: 9" errors. These issues were caused by a combination of factors, including:

*   **Concurrent Processes:** Running other memory-intensive processes (like the `football-ai-analysis` project) at the same time.
*   **Inefficient Data Loading:** The default behavior of `pandas.read_csv` can be very memory-intensive, especially for datasets with a large number of columns.

To address these issues, we have implemented the following memory optimization techniques:

*   **Sequential Execution:** It is recommended to run the `ipl-analytics` scripts sequentially, without other heavy processes running in the background.
*   **Explicit Data Types:** We now explicitly define the data types for each column when loading data with `pandas.read_csv`. This prevents pandas from having to infer the data types, which significantly reduces memory usage.

### GPU Training

This project supports GPU-accelerated training for the LightGBM models. To enable GPU training, set the `device` parameter to `'gpu'` in the LightGBM model configuration.

**Note:** GPU training can have a higher memory overhead than CPU training, especially on machines with limited VRAM. If you encounter memory issues with GPU training, it is recommended to switch back to CPU training.