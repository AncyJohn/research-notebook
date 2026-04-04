from dataclasses import dataclass
from typing import Any, Dict
import torch

@dataclass
class BaseConfig:
    # Experiment Metadata
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training Hyperparameters
    epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Pathing (Passed in at runtime)
    artifacts_path: str = "artifacts"
    
    def to_dict(self) -> Dict[str, Any]:
        # Professional touch: Convert dataclass to dict for JSON logging
        return self.__dict__