#!/usr/bin/env python3
"""
split_dataset.py
- Read processed/processed.csv
- Stratified train/test split (80:20)
- Save processed/train.csv and processed/test.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "processed/processed.csv"
OUT_DIR = "processed"
TRAIN_PATH = os.path.join(OUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUT_DIR, "test.csv")

def split(in_path=IN_PATH, out_dir=OUT_DIR, test_size=0.2, random_state=42):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"{in_path} not found. Run preprocess.py first.")
    df = pd.read_csv(in_path, index_col=0)
    if 'churn' not in df.columns:
        raise ValueError("Processed dataset must contain 'churn' column.")
    X = df.drop(columns=['churn'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    os.makedirs(out_dir, exist_ok=True)
    train = X_train.copy()
    train['churn'] = y_train
    test = X_test.copy()
    test['churn'] = y_test
    train.to_csv(TRAIN_PATH, index=True)
    test.to_csv(TEST_PATH, index=True)
    print(f"Saved train -> {TRAIN_PATH} shape={train.shape}")
    print(f"Saved test  -> {TEST_PATH} shape={test.shape}")

if __name__ == "__main__":
    split()
