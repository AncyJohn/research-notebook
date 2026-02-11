# Titanic ML Pipeline: Professional Baseline

A modular, production-ready PyTorch pipeline for stratified survival prediction. This architecture prioritizes **strict state isolation** and **deterministic reproducibility** to eliminate "hidden state" bugs and data leakage.

## 🚀 Quick Start
1. **Clone & Setup**:
   ```bash
   git clone 'https://github.com/AncyJohn/research-notebook.git'
   cd phase3/pipelines/titanic
   pip install -r requirements.txt

2. Execute Pipelin:
    python run.py

📊 Final Evaluation Results
The model was verified using a stratified validation set. The following results represent the performance of the "Best Brain" weights preserved by the Early Stopping system.
Metric	Value
Accuracy	80.45%
Macro F1-Score	0.78
Weighted F1	0.80
Class 0 (Victims) Recall	0.90
Audited via stratified validation using the evaluate.py independent auditor.

🏗️ Architecture & Artifacts
The system follows a "Stateless" design where hyperparameters and I/O paths are managed via config.yaml.
Orchestration: run.py (Executes training followed by evaluation).
Models: artifacts/models/mlp.pt (Full-state dictionary including optimizer momentum).
Metadata: artifacts/models/train_stats.joblib (Leakage-free scaling parameters).
Reporting: artifacts/metrics/metrics.json and artifacts/plots/loss_curve.png.

🧠 Engineering Reflection
Transitioning from a non-linear notebook to this Modular Pipeline Architecture represents a shift from "experimental scripting" to "machine learning engineering."
By implementing Full-State Checkpointing, the system ensures that training resumes are mathematically seamless, preserving optimizer velocity and historical best loss. The separation of the PreprocessingDataset logic ensures that training statistics are strictly applied during inference, guaranteeing the model's integrity against data snooping. This repository now serves as a high-integrity template for scalable predictive modeling.

Status: Project Complete | Environment: Cross-Platform (Win/Linux/Colab)