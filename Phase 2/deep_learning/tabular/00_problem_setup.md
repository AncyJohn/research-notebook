# Deep Learning for Tabular Data – Titanic

## Motivation
Classical machine learning models often perform well on tabular datasets.
This project explores whether a neural network can outperform logistic regression
on the Titanic dataset, and under what conditions.

## Research Question
Can a simple multilayer perceptron (MLP) improve generalization performance
compared to logistic regression on a small tabular dataset?

## Baseline Reference
- Model: Logistic Regression
- Validation: Stratified 5-fold CV
- Performance: Refer to Kaggle Titanic Phase

## Evaluation Criteria
- Primary metric: Accuracy
- Comparison based on cross-validation, not leaderboard score

## Constraints
- No excessive feature engineering
- Same input features for all models
- Focus on generalization, not maximum score

## Expected Outcome
Neural networks may match or slightly improve performance,
but may also exhibit higher variance due to dataset size.