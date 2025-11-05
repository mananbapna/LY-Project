#!/usr/bin/env python3
"""
Run custom test cases through your saved models.

Produces:
 - tests/custom_tests.csv        (the created test cases)
 - outputs/custom_test_results.csv  (predictions + metadata)
Prints a table to stdout.

Behavior:
 - Loads features from processed/train.csv header (so it matches your pipeline)
 - Uses outputs/scaler.joblib to scale features
 - Loads MLP from outputs/best_model.pth (model/net.MLP must exist)
 - If outputs/xgb_calibrated.joblib exists, uses it for an XGBoost calibrated prob as well
"""

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import torch

# Files/paths
PROJECT = Path.cwd()
TESTS_DIR = PROJECT / "tests"
TESTS_DIR.mkdir(exist_ok=True)
TEST_CSV = TESTS_DIR / "custom_tests.csv"
OUTPUT_CSV = PROJECT / "outputs" / "custom_test_results.csv"
SCALER_PATH = PROJECT / "outputs" / "scaler.joblib"
MODEL_PATH = PROJECT / "outputs" / "best_model.pth"
XGB_CAL_PATH = PROJECT / "outputs" / "xgb_calibrated.joblib"  # optional

# threshold for 'intervene' decision
THRESHOLD = 0.5

# get device
def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# load feature names from processed/train.csv (drop churn)
def get_feature_names():
    proc_train = PROJECT / "processed" / "train.csv"
    proc_proc = PROJECT / "processed" / "processed.csv"
    if proc_train.exists():
        df = pd.read_csv(proc_train, index_col=0)
    elif proc_proc.exists():
        df = pd.read_csv(proc_proc, index_col=0)
    else:
        raise FileNotFoundError("processed/train.csv or processed/processed.csv not found. Run preprocessing & split.")
    if 'churn' not in df.columns:
        raise ValueError("'churn' not found in processed data")
    feat_names = [c for c in df.columns if c != 'churn']
    return feat_names

# Create custom test cases (you can edit these after file is written)
def create_sample_tests(feature_names):
    # Example mapping assumptions (edit if your encodings differ)
    # typical columns we observed: avg_score, studied_credits, num_of_prev_attempts, highest_education, gender, region, ...
    # We will try to set values for known columns; for others use median or reasonable defaults.
    # Build 5 example students (IDs will be 1000001..)
    default_vals = {}
    # derive some sensible defaults by reading processed.csv medians if available
    try:
        df_proc = pd.read_csv("processed/processed.csv", index_col=0)
        for c in feature_names:
            if pd.api.types.is_numeric_dtype(df_proc[c]):
                default_vals[c] = float(df_proc[c].median())
            else:
                default_vals[c] = df_proc[c].mode().iloc[0]
    except Exception:
        # fallback simple defaults
        for c in feature_names:
            default_vals[c] = 0.0

    # now define a few cases, customizing commonly-seen columns
    cases = []
    # Case 1: high score, low attempts (low risk)
    c1 = default_vals.copy()
    if 'avg_score' in c1: c1['avg_score'] = 85.0
    if 'studied_credits' in c1: c1['studied_credits'] = 60
    if 'num_of_prev_attempts' in c1: c1['num_of_prev_attempts'] = 0
    if 'highest_education' in c1: c1['highest_education'] = 2
    if 'gender' in c1: c1['gender'] = 1
    if 'region' in c1: c1['region'] = 5
    cases.append(c1)

    # Case 2: moderate score, one previous attempt (medium risk)
    c2 = default_vals.copy()
    if 'avg_score' in c2: c2['avg_score'] = 66.0
    if 'studied_credits' in c2: c2['studied_credits'] = 60
    if 'num_of_prev_attempts' in c2: c2['num_of_prev_attempts'] = 1
    if 'highest_education' in c2: c2['highest_education'] = 1
    if 'gender' in c2: c2['gender'] = 0
    if 'region' in c2: c2['region'] = 3
    cases.append(c2)

    # Case 3: low score, many previous attempts (high risk)
    c3 = default_vals.copy()
    if 'avg_score' in c3: c3['avg_score'] = 45.0
    if 'studied_credits' in c3: c3['studied_credits'] = 120
    if 'num_of_prev_attempts' in c3: c3['num_of_prev_attempts'] = 3
    if 'highest_education' in c3: c3['highest_education'] = 0
    if 'gender' in c3: c3['gender'] = 1
    if 'region' in c3: c3['region'] = 0
    cases.append(c3)

    # Case 4: borderline but recent high credits (mixed risk)
    c4 = default_vals.copy()
    if 'avg_score' in c4: c4['avg_score'] = 70.0
    if 'studied_credits' in c4: c4['studied_credits'] = 240
    if 'num_of_prev_attempts' in c4: c4['num_of_prev_attempts'] = 0
    if 'highest_education' in c4: c4['highest_education'] = 3
    if 'gender' in c4: c4['gender'] = 0
    if 'region' in c4: c4['region'] = 8
    cases.append(c4)

    # Case 5: Missing/edge case (use defaults)
    c5 = default_vals.copy()
    cases.append(c5)

    # build DataFrame preserving feature order
    df_cases = pd.DataFrame(cases, columns=feature_names)
    # create an index id_student for these test cases
    base_id = 1000001
    df_cases.index = [str(base_id + i) for i in range(len(df_cases))]
    return df_cases

# load NN model and infer
def infer_with_nn(X_array):
    # expects numpy array of shape (n, d)
    device = get_device()
    # import model class
    from model.net import MLP
    # determine input dim
    input_dim = X_array.shape[1]
    model = MLP(input_dim=input_dim, hidden=[64,32], dropout=0.4).to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train and save model first.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_array.astype('float32')).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return probs

# attempt to use calibrated xgboost if exists
def infer_with_xgb_calibrated(X_df):
    if not XGB_CAL_PATH.exists():
        return None
    cal = joblib.load(XGB_CAL_PATH)
    # expects a scikit-learn like calibrated classifier with predict_proba
    probs = cal.predict_proba(X_df)[:,1]
    return probs

def main():
    # feature names
    feat_names = get_feature_names()
    print("Feature names detected:", feat_names)

    # create sample test cases and save
    df_cases = create_sample_tests(feat_names)
    df_cases.to_csv(TEST_CSV)
    print("Wrote custom tests to", TEST_CSV)

    # load scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"{SCALER_PATH} not found. Run training to create scaler.joblib")
    scaler = joblib.load(SCALER_PATH)

    # prepare arrays for NN
    X_raw = df_cases.copy()
    # If any non-numeric columns exist, try to coerce to numeric (assumes preprocessing encoded them)
    for c in X_raw.columns:
        if X_raw[c].dtype == object:
            try:
                X_raw[c] = pd.to_numeric(X_raw[c])
            except Exception:
                raise ValueError(f"Column {c} is non-numeric in custom test cases. Preprocessing expects numeric inputs.")
    X_scaled = scaler.transform(X_raw.values)

    # NN inference
    nn_probs = infer_with_nn(X_scaled)

    # XGBoost calibrated inference (optional)
    xgb_probs = None
    try:
        xgb_probs = infer_with_xgb_calibrated(X_raw)
    except Exception as e:
        print("Warning: could not run calibrated XGBoost inference:", e)
        xgb_probs = None

    # assemble results
    out = df_cases.copy()
    out['nn_prob'] = nn_probs
    if xgb_probs is not None:
        out['xgb_prob'] = xgb_probs
    out['recommend_intervene_nn'] = (out['nn_prob'] >= THRESHOLD).astype(int)
    if xgb_probs is not None:
        out['recommend_intervene_xgb'] = (out['xgb_prob'] >= THRESHOLD).astype(int)

    # pretty print
    pd.set_option('display.max_columns', 50)
    print("\nCustom test cases with predictions:\n")
    print(out)

    # save outputs
    out_dir = OUTPUT_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV)
    print("\nSaved results to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
