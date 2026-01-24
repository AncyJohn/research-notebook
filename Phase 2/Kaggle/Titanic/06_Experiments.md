# Titanic – Experiment Summary

## Baseline
- Logistic Regression
- Simple preprocessing + imputation + scaling
- CV accuracy: 0.8268

## Feature Engineering
- FamilySize
- IsAlone
- Title
- CV improvement: +0.0116

## Model Comparison
- Logistic Regression
- Random Forest
- Gradient Boosting
- Logistic Regression performed best

## Submission
- Public leaderboard score: 0.77272
- CV vs LB gap: 0.0657 (6%)

## Key Learnings
- Validation discipline matters more than model complexity
- Strong features can outperform complex models
- Tree-based models can overfit small tabular datasets