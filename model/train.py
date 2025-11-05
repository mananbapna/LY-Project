#!/usr/bin/env python3
"""
Train script (updated):
- Smaller MLP ([64,32]) with dropout=0.4
- L2 weight decay
- Early stopping (patience)
- LR scheduler (ReduceLROnPlateau)
- Saves: outputs/best_model.pth, outputs/scaler.joblib, outputs/loss_curve.png, outputs/metrics.json
Run as: python3 -m model.train
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model.net import MLP
from evaluate import classification_metrics

# Config
TRAIN_CSV = "processed/train.csv"
TEST_CSV = "processed/test.csv"
OUT_DIR = "outputs"
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8  # early stopping patience on validation AUC
VAL_FRAC = 0.2
RANDOM_STATE = 42

def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_csv(path):
    return pd.read_csv(path, index_col=0)

def prepare_data(train_csv=TRAIN_CSV, val_frac=VAL_FRAC, random_state=RANDOM_STATE):
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} not found. Run split_dataset.py first.")
    df = load_csv(train_csv)
    if 'churn' not in df.columns:
        raise ValueError("train.csv must contain 'churn' column.")
    X = df.drop(columns=['churn']).values
    y = df['churn'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac,
                                                      random_state=random_state, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, scaler

def train_and_evaluate(train_csv=TRAIN_CSV, test_csv=TEST_CSV, output_dir=OUT_DIR,
                       epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                       weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_val, y_train, y_val, scaler = prepare_data(train_csv)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print("Scaler saved to", os.path.join(output_dir, 'scaler.joblib'))

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = get_device()
    print("Using device:", device)

    model = MLP(input_dim=X_train.shape[1], hidden=[64,32], dropout=0.4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)


    best_auc = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # small input noise augmentation to regularize (only if training)
            xb_noise = xb + torch.randn_like(xb) * 0.01
            logits = model(xb_noise)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss_total = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                val_loss_total += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                trues.append(yb.cpu().numpy())
        val_loss = val_loss_total / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else 0.0
        except Exception:
            auc = 0.0
        history['val_auc'].append(float(auc))

        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_auc={auc:.4f}")

        # scheduler step (use val AUC)
        scheduler.step(auc)

        # early stopping & save best model by AUC
        if auc > best_auc + 1e-6:
            best_auc = auc
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model (AUC={best_auc:.4f}) to {os.path.join(output_dir, 'best_model.pth')}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no_improve={no_improve})")
                break

    # Save loss curve
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train vs Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    print("Saved loss curve to", os.path.join(output_dir, 'loss_curve.png'))

    # Final evaluation on test set (if present)
    metrics = {}
    if os.path.exists(test_csv):
        test_df = load_csv(test_csv)
        if 'churn' in test_df.columns:
            X_test = test_df.drop(columns=['churn']).values
            y_test = test_df['churn'].values
            X_test = scaler.transform(X_test)
            # load best model
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"{best_model_path} not found.")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X_test.astype('float32')).to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
            metrics = classification_metrics(y_test, probs, threshold=0.5)
            # also add ROC-AUC check if present
            try:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc'] = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test))>1 else 0.0
            except Exception:
                metrics['roc_auc'] = metrics.get('roc_auc', 0.0)

            # write metrics
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            print("Test metrics:", metrics)
        else:
            print(f"Test csv found at {test_csv} but no 'churn' column. Skipping final evaluation.")
    else:
        print(f"No test csv at {test_csv}; skipping final evaluation.")

if __name__ == "__main__":
    train_and_evaluate()
