# Titanic: Machine Learning from Disaster

This project applies a disciplined, research-oriented approach to the Titanic Kaggle competition.

## Approach
- Clean baseline model
- Proper validation using stratified cross-validation
- Controlled feature engineering
- Model comparison without leaderboard chasing

## Final Model
- Logistic Regression
- Feature set: baseline + FamilySize, IsAlone, Title
- Validation: 5-fold stratified CV

## Results
- Mean CV accuracy: 0.8384
- Public leaderboard score: 0.7727

## Key Takeaways
- Validation stability > leaderboard score
- Feature quality can outweigh model complexity
- Reproducible pipelines reduce leakage and bias