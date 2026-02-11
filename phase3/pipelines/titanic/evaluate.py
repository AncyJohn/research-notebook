import torch
import yaml
import joblib
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split  # Added this import
from sklearn.metrics import accuracy_score, classification_report
from train import PreprocessingDataset, BaselineNet 

def evaluate():
    with open("config.yaml", "r") as f: 
        config = yaml.safe_load(f)

    Path(config['outputs']['metrics_path']).parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    df = pd.read_csv(config['data']['raw_path'])
    
    # We recreate the exact same split using the SEED from config
    _, val_df = train_test_split(
        df, 
        test_size=config['train_params']['test_size'], 
        stratify=df[config['data']['target']], 
        random_state=config['train_params']['seed']
    )
    
    # Load the preprocessing stats saved during training
    stats = joblib.load(config['outputs']['stats_path'])
    
    # Initialize dataset with the SAVED stats to ensure no leakage
    val_ds = PreprocessingDataset(
        val_df, 
        target_col=config['data']['target'], 
        drop_cols=config['data']['drop_columns'], 
        stats=stats
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=config['train_params']['batch_size'],
        shuffle=False
    )

    # 2. Load Model Architecture and Weights
    input_size = val_ds.X.shape[1]
    model = BaselineNet(
        input_size=input_size, 
        hidden_dim=config['model_params']['hidden_dim'], 
        dropout=config['model_params']['dropout']
    ).to(device)
    
    checkpoint = torch.load(config['outputs']['model_path'], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict']) # Access the weights inside the dict
    model.eval()

    # 3. Inference & Metric Calculation
    criterion = torch.nn.BCEWithLogitsLoss() # Use the same loss as training
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).float()
            logits = model(x).view(-1)

            # Calculate running loss
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)

            # Binary predictions
            probs = torch.sigmoid(logits)
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate final metrics
    val_loss = running_loss / len(val_ds)
    val_acc = accuracy_score(all_labels, all_preds)

    # 4. Save Metrics to JSON
    metrics = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss)
    }
    
    target_path = config['outputs']['metrics_path'] # Define it here to use in the print below
    with open(target_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation complete. Metrics saved to {target_path}")
    print(f"Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f}")
    
if __name__ == "__main__":
    evaluate()
