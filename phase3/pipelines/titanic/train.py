import os
import random
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- REPRODUCIBILITY ---
def set_reproducibility(seed):
    # 1. Standard Python & OS level randomness
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. The MATH Libraries
    # Modern NumPy (Generator-based) 
    np.random.seed(seed) 
    # PyTorch (CPU and all GPUs) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 3. The Hardware Level. Forcing Deterministic GPU Algorithms (Crucial for 2026)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # New: Force error if deterministic op isn't possible
    torch.use_deterministic_algorithms(True)

# --- DATA CLASSES ---
class PreprocessingDataset(Dataset):
    def __init__(self, df, target_col=None, drop_cols=None, stats=None):
        """
        df: Input DataFrame
        target_col: Name of the label column
        drop_cols: List of non-predictive columns to remove (e.g., ['PassengerId', 'Name'])
        stats: Dictionary containing training means, stds, and mappings
        """
        # 1. DROP UNWANTED COLUMNS
        df = df.copy()
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # 2. SEPARATE TARGET & IDENTIFY TYPES
        if target_col and target_col in df.columns:
            self.target = torch.tensor(df[target_col].values, dtype=torch.float32)
            data_df = df.drop(columns=[target_col])
        else:
            self.target = None
            data_df = df
        
        # Automatic Column Type Identification
        num_cols = data_df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = data_df.select_dtypes(include=['object', 'category']).columns
        num_tensor = torch.tensor(data_df[num_cols].values, dtype=torch.float32)
        cat_data = data_df[cat_cols].astype(str).values.tolist()

        # 3. LEAKAGE-FREE SCALING & MAPPING
        if stats is None:
            # Training Mode: Calculate Stats
            self.means = torch.nanmean(num_tensor, dim=0) # calculate mean
            # Replace NANs with mean 
            temp_num = torch.where(torch.isnan(num_tensor), self.means, num_tensor)
            self.stds = torch.std(temp_num, dim=0) # calculate std
            self.stds[self.stds == 0] = 1.0 # to avoid division by 0 during standardisation process
            self.cat_mappings = self._build_mappings(cat_data)
            self.stats = {'means': self.means, 'stds': self.stds, 'mappings': self.cat_mappings}
        else:
            # Validation Mode: Reuse Train Stats
            self.means, self.stds, self.cat_mappings = stats['means'], stats['stds'], stats['mappings']
            self.stats = stats

        # 4. APPLY TRANSFORMATIONS
        # Impute and Standardize
        num_imputed = torch.where(torch.isnan(num_tensor), self.means, num_tensor)
        self.num_final = (num_imputed - self.means) / self.stds
        # Encode Categories
        self.cat_final = self._encode(cat_data)
        # 5. CONSOLIDATE INTO ONE TENSOR
        self.X = torch.cat([self.num_final, self.cat_final], dim=1)

    def _build_mappings(self, data):
        if not data: return []
        mappings = []
        for col_idx in range(len(data[0])):
            unique_vals = sorted(list(set(row[col_idx] for row in data)))
            # BEST PRACTICE: Map known categories starting at index 0, 
            # and use len(unique_vals) as the 'Unknown' bucket.
            mapping = {val: i for i, val in enumerate(unique_vals)}
            mapping['__unknown__'] = len(unique_vals)
            mappings.append(mapping)
        return mappings

    def _encode(self, data):
        if not data: return torch.empty((len(self.num_final), 0))
        # Safer: Use the dedicated '__unknown__' index for unseen data
        encoded = [[self.cat_mappings[i].get(val, self.cat_mappings[i]['__unknown__']) 
                   for i, val in enumerate(row)] for row in data]
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.target[idx]) if self.target is not None else self.X[idx]

# --- MODEL ---
class BaselineNet(nn.Module): # The physical structure of the brain (the neurons)
    def __init__(self, input_size, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Stabilizes inputs to the next layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    # The thought process (data goes in, a decision comes out).
    def forward(self, x): return self.net(x.float()) # Force float32 for safety

class EarlyStopping:
    def __init__(self, patience, path):
        self.patience, self.path, self.counter, self.best_loss, self.early_stop = patience, path, 0, None, False
    def __call__(self, val_loss, model):
        # print(f"DEBUG: val_loss={val_loss:.4f}, best_loss={self.best_loss}")
        if self.best_loss is None or val_loss < self.best_loss:
            print(f"✅ Validation loss decreased ({self.best_loss} --> {val_loss:.4f}). Saving model...")
            self.best_loss = val_loss
            torch.save(model.state_dict(), 
                       "phase3/pipelines/titanic/artifacts/models/mlp.pt")
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️ No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            '''Saves model when validation loss decreases.'''
            if self.counter >= self.patience: self.early_stop = True


# --- MAIN ENGINE ---
def main():
    # 1. Load Configuration
    with open("config.yaml", "r") as f: 
        config = yaml.safe_load(f)
    
    set_reproducibility(config['train_params']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Preparation
    print(f"Reading data from: {config['data']['raw_path']}")
    df = pd.read_csv(config['data']['raw_path'])
    
    # Corrected: Stratify ensures the class balance is kept across splits
    train_df, val_df = train_test_split(
        df, 
        test_size=config['train_params']['test_size'], 
        stratify=df[config['data']['target']], 
        random_state=config['train_params']['seed']
    )
    
    # 3. Process Datasets
    train_ds = PreprocessingDataset(train_df, config['data']['target'], config['data']['drop_columns'])
    val_ds = PreprocessingDataset(val_df, config['data']['target'], config['data']['drop_columns'], stats=train_ds.stats)
    
    print(f"💾 Saving stats to {config['outputs']['stats_path']}...")
    joblib.dump(train_ds.stats, config['outputs']['stats_path'])

    train_loader = DataLoader(train_ds, batch_size=config['train_params']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train_params']['batch_size'])

    # 4. Model Setup
    # Fixed: train_ds.X.shape[1] is perfect for dynamic input sizing
    model = BaselineNet(
        train_ds.X.shape[1], 
        config['model_params']['hidden_dim'], 
        config['model_params']['dropout']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['train_params']['lr'], 
        weight_decay=config['train_params']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['train_params']['scheduler_factor'], 
        patience=config['train_params']['scheduler_patience']
    )
    
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(config['train_params']['patience'], config['outputs']['model_path'])

    # 5. Training Loop
    t_history, v_history = [], []
    print(f"🚀 Starting training on {device}...")
    
    for epoch in range(config['train_params']['epochs']):
        # --- TRAINING ---
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
        
        # --- VALIDATION ---
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += criterion(model(x).view(-1), y).item() * x.size(0)
        
        epoch_t = t_loss/len(train_ds)
        epoch_v = v_loss/len(val_ds)
        t_history.append(epoch_t)
        v_history.append(epoch_v)
        
        # Step the scheduler and the early stopper
        scheduler.step(epoch_v)
        early_stopper(epoch_v, model)
        
        print(f"Epoch {epoch+1:02d}: Train Loss {epoch_t:.4f} | Val Loss {epoch_v:.4f}")
        
        if early_stopper.early_stop:
            print(f"🛑 Early stopping triggered. Best Val Loss: {early_stopper.best_loss:.4f}")
            break

    # 6. Save Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_history, label='Train Loss')
    plt.plot(v_history, label='Val Loss')
    plt.title('Baseline Model History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(config['outputs']['plot_path'])
    plt.close()
    print(f"📊 Plot saved to {config['outputs']['plot_path']}")

if __name__ == "__main__":
    main()
