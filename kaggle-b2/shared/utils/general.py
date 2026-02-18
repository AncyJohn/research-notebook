import torch
import numpy as np
import random
import os
import json

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def log_metrics(metrics, artifact_dir):
    path = os.path.join(artifact_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics logged to {path}")
