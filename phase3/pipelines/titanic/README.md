Titanic Survival Predictor: Stable Baseline Pipeline

🚀 Project Overview
This project implements a professional machine learning pipeline to predict passenger survival on the Titanic. The system has been refactored from a non-linear notebook into a Modular, Reproducible, and Production-Ready architecture designed to eliminate data leakage and "hidden state" bugs.

📊 Final Evaluation Results
The model was verified using a stratified validation set. The following results represent the performance of the "Best Brain" weights preserved by the Early Stopping system.
Core Metrics:
Final Accuracy: 0.8045 (80.45%)
Macro F1-Score: 0.78
Weighted Avg F1: 0.80
Detailed Classification Report:
Class	Precision	Recall	F1-Score	Support
0 (Victims)	0.80	0.90	0.85	110
1 (Survivors)	0.80	0.65	0.72	69
The model demonstrates high reliability, particularly in identifying Class 0 with 90% recall, ensuring a robust baseline for safety-critical analysis.

🛠️ Pipeline Architecture
The project follows a "Stateless" design where code, configuration, and data artifacts are strictly separated:
config.yaml: Centralized source of truth for all hyperparameters (LR: 1e-3, Batch Size: 256, Seed: 42) and I/O paths.
train.py: Automated training engine. Features include deterministic GPU seeding, leakage-free scaling, and automated training history logging.
evaluate.py: Independent auditor script. Reconstructs the validation environment and verifies the saved artifacts.
PreprocessingDataset: A custom PyTorch Dataset class that handles missing values and categorical encoding using training-set statistics to prevent snooping.

📦 Generated Artifacts
Every successful run produces the following deployment-ready files:
best_model.pt: Optimized PyTorch state dictionary.
train_stats.joblib: Encrypted preprocessing metadata (means/mappings) required for consistent inference.
loss_curve.png: Diagnostic visualization of the Training vs. Validation loss.

💻 How to Run
Ensure your environment has the required libraries (torch, yaml, joblib, sklearn, pandas).
Train the Model:
python train.py

Verify Results:
python evaluate.py

Project Complete. Now we have a high-integrity repository that serves as a template for future predictive modeling.