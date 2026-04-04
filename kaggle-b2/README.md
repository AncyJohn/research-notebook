# Kaggle-B2: Deep Learning Research Framework

A reproducible, professional-grade framework for competing in Kaggle tabular, NLP, and CV competitions. This repository is structured to beat classical ML by implementing Deep Learning best practices like Entity Embeddings, Residual Connections, and Stratified K-Fold Cross-Validation.

## 📂 Project Structure
```text
kaggle-b2/
├── shared/                 # Core reusable logic
│   ├── configs/            # Experiment hyperparameter schemas
│   ├── training/           # Model architectures, Trainers, and Datasets
│   └── utils/              # Reproducibility (seeding) and Logging
├── competitions/           # Competition-specific notebooks/scripts
│   └── titanic-redux/      # Baseline Deep Learning for Titanic
├── artifacts/              # Local storage for weights (.pth) and logs
├── setup.py                # Package installation script
└── requirements.txt        # Environment dependencies

🚀 Getting Started
1. Environment Setup
Install the shared module as an editable package to enable clean imports:
bash
pip install -r requirements.txt
pip install -e .
Use code with caution.

2. The Research Workflow
This framework follows a 4-phase discipline test:
Configuration: Defined in shared/configs/, ensuring all hyperparameters are tracked.
Preprocessing: Utilizes Entity Embeddings for categorical features and StandardScaler for numerical stability.
Cross-Validation: 5-Fold Stratified CV with Out-of-Fold (OOF) prediction tracking to ensure the CV score is a reliable proxy for the leaderboard.
Inference: Ensemble averaging across all fold models to reduce variance.
🧠 Model Philosophy (Tabular DL)
To "cleanly" beat classical ML (XGBoost/Random Forest), this framework implements:
Entity Embeddings: Learning multi-dimensional vectors for categories (Title, Pclass, Sex).
Residual Connections: Preventing gradient vanishing on small datasets.
Label Smoothing & Weight Decay: Advanced regularisation to prevent overfitting on small-n data.
🛠️ Usage on Kaggle
To use the shared library on Kaggle:
Upload the shared/ folder as a Private Dataset.
Install it in your notebook: !pip install /kaggle/input/your-dataset-name/.
Import via: from shared.training import DLTrainer.
