import torch
import numpy as np
import random
import os
import json

def seed_everything(seed=42):
    """Sets all seeds to ensure reproducible results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # Standard for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
    print(f"🚀 Random seed set to: {seed}")

def log_metrics(metrics, artifact_dir, filename="metrics.json"):
    """
    Logs a dictionary of metrics to a JSON file.
    Usage: log_metrics(fold_results, "artifacts/v1", filename=f"fold_{i}.json")
    """
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, filename)
    
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics logged to {path}")