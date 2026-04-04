import torch
import torch.nn as nn
import copy
import numpy as np
import os  # Required for path checking and directory creation
from tqdm.auto import tqdm
from ..utils.seed import log_metrics

class DLTrainer:
    def __init__(self, model, config, criterion=None, optimizer=None, scheduler=None):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = scheduler
        self.best_loss = float('inf')
        self.best_model_wts = None
        self.history = {"train_loss": [], "val_loss": []}

    def load_checkpoint(self, fold_idx):
        """Phase 4: Restart Safety. Checks for existing best model."""
        path = f"{self.config.artifacts_path}/model_fold_{fold_idx}.pth"
        if os.path.exists(path):
            print(f"--- ♻️ Found Checkpoint for Fold {fold_idx}. Loading... ---")
            self.model.load_state_dict(torch.load(path, map_location=self.config.device))
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            return True
        return False

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for x_cat, x_cont, targets in dataloader:
            x_cat, x_cont, targets = x_cat.to(self.config.device), x_cont.to(self.config.device), targets.to(self.config.device)
            self.optimizer.zero_grad()
            outputs = self.model(x_cat, x_cont)
            loss = self.criterion(outputs, targets.view_as(outputs))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * x_cat.size(0)
        return running_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        for x_cat, x_cont, targets in dataloader:
            x_cat, x_cont, targets = x_cat.to(self.config.device), x_cont.to(self.config.device), targets.to(self.config.device)
            outputs = self.model(x_cat, x_cont)
            loss = self.criterion(outputs, targets.view_as(outputs))
            running_loss += loss.item() * x_cat.size(0)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
        return running_loss / len(dataloader.dataset), np.concatenate(all_probs)

    def fit(self, train_loader, val_loader, fold_idx=0):
        # 1. CHECK FOR CHECKPOINT FIRST (Restart Safety)
        if self.load_checkpoint(fold_idx):
            val_loss, oof_preds = self.evaluate(val_loader)
            print(f"Resume successful. Best Val Loss: {val_loss:.4f}")
            return oof_preds

        # 2. START NORMAL TRAINING IF NO CHECKPOINT
        self.best_loss = float('inf') 
        patience_counter = 0
        best_oof_preds = None
        main_pbar = tqdm(range(self.config.epochs), desc=f"Fold {fold_idx}", leave=False)
        
        for epoch in main_pbar:
            train_loss = self.train_one_epoch(train_loader)
            val_loss, current_oof_preds = self.evaluate(val_loader)
            
            if self.scheduler: self.scheduler.step(val_loss)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                best_oof_preds = current_oof_preds
                patience_counter = 0
                # Auto-save best version immediately (Phase 4 Discipline)
                torch.save(self.best_model_wts, f"{self.config.artifacts_path}/model_fold_{fold_idx}.pth")
            else:
                patience_counter += 1

            main_pbar.set_postfix({"V-Loss": f"{val_loss:.4f}", "Best": f"{self.best_loss:.4f}"})
            if patience_counter >= self.config.early_stopping_patience: break
        
        self.model.load_state_dict(self.best_model_wts)
        log_metrics({"best_val_loss": self.best_loss}, self.config.artifacts_path, filename=f"fold_{fold_idx}_metrics.json")
        return best_oof_preds

    @torch.no_grad()
    def predict(self, dataloader):
        """Standard Inference Method for Leaderboard Submissions"""
        self.model.eval()
        all_probs = []
        for x_cat, x_cont in dataloader:
            x_cat, x_cont = x_cat.to(self.config.device), x_cont.to(self.config.device)
            outputs = self.model(x_cat, x_cont)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
        return np.concatenate(all_probs)