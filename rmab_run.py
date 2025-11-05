#!/usr/bin/env python3
"""
rmab_run.py
- Load processed/test.csv
- Load trained model (outputs/best_model.pth) and scaler
- Infer churn probabilities (risk_prob)
- Run RMAB approximate Whittle simulation
- Save outputs/rmab_simulation.json
"""
import os
import json
import pandas as pd
import torch
import joblib

from model.net import MLP
from model.rmab_whittle import select_top_k_by_index, simulate_policy

PROCESSED_TEST = "processed/test.csv"
MODEL_PATH = "outputs/best_model.pth"
SCALER_PATH = "outputs/scaler.joblib"
OUTPUT_JSON = "outputs/rmab_simulation.json"

def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def infer_risk_probs(test_csv=PROCESSED_TEST, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"{test_csv} not found. Run split_dataset.py first.")
    df = pd.read_csv(test_csv, index_col=0)
    if 'churn' not in df.columns:
        raise ValueError("test csv must contain 'churn' column.")
    X = df.drop(columns=['churn']).values
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        print("Scaler not found; using raw features")
    device = get_device()
    model = MLP(input_dim=X.shape[1], hidden=[128,64], dropout=0.2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"{model_path} not found. Train model first.")
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.astype('float32')).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    df['risk_prob'] = probs
    return df

def main():
    df = infer_risk_probs()
    print("Sample risk_prob:\n", df[['risk_prob','churn']].head())
    budget = 50
    chosen, as_df = select_top_k_by_index(df, belief_col='risk_prob', k=budget,
                                          intervention_effect=0.25, horizon=10, discount=0.95, cost=1.0)
    print(f"Selected {len(chosen)} arms for intervention (top-{budget})")

    sim_hist = simulate_policy(df, rounds=20, budget=budget, belief_col='risk_prob',
                               intervention_effect=0.25, horizon=10, discount=0.95, cost=1.0, seed=42)

    os.makedirs('outputs', exist_ok=True)
    out = {
        'config': {
            'budget': int(budget),
            'intervention_effect': 0.25,
            'horizon': 10,
            'discount': 0.95
        },
        'sim_history': sim_hist.to_dict(orient='records')
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved RMAB simulation to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
