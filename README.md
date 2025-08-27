# IPL Player Rating System: From Glicko-2 to Player Impact

This project aims to develop a robust, context-aware, and predictive player rating system for the Indian Premier League (IPL). The journey has involved iterative development, rigorous testing, and a willingness to pivot approaches based on empirical results.

## Initial Approach: Glicko-2 Based Rating System

Our initial endeavor focused on adapting the Glicko-2 rating system, typically used for 1v1 competitive games, to the complex, team-based environment of cricket.

*   **Core Idea:** Players' ratings would update match-by-match based on their performance.
*   **Performance Metrics:** Early iterations used simple metrics (e.g., win/loss based on strike rate), which proved inadequate due to rating inertia and sensitivity to outliers. This led to the development of more sophisticated, context-aware scoring functions for both batsmen and bowlers.
*   **Parameter Optimization:** To avoid manual tuning, we employed an `optuna`-based Bayesian optimization framework. This framework was designed to learn the optimal parameters for our scoring functions by minimizing a defined loss function.

## Challenges and Iterations: Learning from Failure

The path to a reliable rating system has been fraught with challenges, leading to significant insights and necessary pivots.

### Iteration 1: Win Prediction (LogLoss)

*   **Objective:** The first loss function aimed to optimize parameters by minimizing the LogLoss of predicting match winners.
*   **Outcome:** This approach proved too noisy and susceptible to local minima. A notable issue, dubbed the "**Kohli Anomaly**," emerged: despite Virat Kohli's undeniable status as an IPL legend, his rating remained surprisingly low and his peak was incorrectly identified in early seasons. This indicated the model was not effectively capturing sustained, high-level performance.

### Iteration 2: Score Prediction (Mean Squared Error)

*   **Objective:** Recognizing the limitations of win prediction, we shifted to a loss function based on predicting each team's final score (Mean Squared Error). This was hypothesized to be more directly tied to individual player performance.
*   **Outcome:** A 2000-trial `optuna` run was successfully completed. While this iteration improved the model's ability to predict scores, it unfortunately did **not** resolve the core issues with player rankings. The "Kohli Anomaly" persisted, and a broader "**Legacy Bias**" became apparent: players from early IPL seasons (with less data) often held disproportionately high and sticky ratings, while modern-day greats were undervalued. The bowling ratings, in particular, remained largely uninformative. This indicated that even with optimized parameters, the Glicko-2 framework, as applied, was not suitable for generating intuitive and credible IPL player rankings.

## Current Approach: Player Impact Rating

Learning from the limitations of the Glicko-2 based system, we are now pivoting to a more direct, interpretable, and robust approach: the **Player Impact Rating**.

*   **Core Idea:** Instead of a relative rating system, we will calculate a direct "Impact Score" for each player's performance in every match. This score will quantify their contribution to the team's success.

*   **Optimization Objective:** The parameters for calculating these impact scores will be optimized using `optuna` to directly predict the **margin of victory** in a match. This ensures a strong link between individual performance and team outcome.

*   **Rating Calculation:** A player's overall rating will be a **time-decayed moving average** of their individual match impact scores. This design inherently addresses the "Legacy Bias" by giving more weight to recent performances and allowing for dynamic shifts in player standing over time.

This new approach promises to deliver a more intuitive, accurate, and actionable player rating system, providing a stronger foundation for further quantitative analysis in cricket.
