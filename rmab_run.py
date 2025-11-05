#!/usr/bin/env python3
"""
RMAB run script (fixed for multiple rows per student)
- Aggregates predictions by student id (mean risk_prob, max churn)
- Runs greedy RMAB simulation on unique students
- Saves outputs/rmab_simulation.json
"""
import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import torch

ROUNDS = 10
TOP_K = None           # if None will use 5% of population
EFFICACY = 0.6         # intervention reduces risk by this fraction (0.6 => risk*(1-0.6))
SEED = 42

from model.net import MLP

def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def infer_risk_probs(test_csv="processed/test.csv",
                     scaler_path="outputs/scaler.joblib",
                     model_path="outputs/best_model.pth"):
    # load test set
    if not Path(test_csv).exists():
        raise FileNotFoundError(f"{test_csv} not found. Run training and data split first.")
    df = pd.read_csv(test_csv, index_col=0)
    if 'churn' not in df.columns:
        raise ValueError(f"{test_csv} must contain 'churn' column for evaluation/simulation.")
    # features and ids
    X = df.drop(columns=['churn'])
    ids = X.index.astype(str)
    # load scaler
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"{scaler_path} not found. Run training first.")
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X.values)
    # build model and load checkpoint
    input_dim = Xs.shape[1]
    device = get_device()
    model = MLP(input_dim=input_dim, hidden=[64,32], dropout=0.4).to(device)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found. Run training first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # batch inference
    batch = 2048
    probs = []
    with torch.no_grad():
        for i in range(0, Xs.shape[0], batch):
            xb = torch.from_numpy(Xs[i:i+batch].astype('float32')).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(p)
    probs = np.concatenate(probs)
    # create dataframe with possibly multiple rows per student
    out = pd.DataFrame({
        'id_student': ids,
        'risk_prob': probs,
        'churn': df['churn'].astype(int).values
    }).set_index('id_student')
    # Aggregate to student-level: mean risk_prob, max churn
    agg = out.groupby(out.index).agg({'risk_prob': 'mean', 'churn': 'max'})
    agg.index = agg.index.astype(str)
    return agg

def run_rmab_sim(df_probs,
                 rounds=ROUNDS,
                 top_k=None,
                 efficacy=EFFICACY,
                 random_state=SEED):
    """
    df_probs: DataFrame index=id_student, columns=['risk_prob','churn'] (unique per student)
    returns dict with 'sim_history' and 'selected_history'
    """
    np.random.seed(random_state)
    n = len(df_probs)
    if top_k is None:
        top_k = max(1, int(0.05 * n))   # default 5% per round
    working_prob = df_probs['risk_prob'].astype(float).copy()
    intervened = set()
    sim_history = []
    selected_history = []

    initial_expected = float(working_prob.mean())
    sim_history.append({'round': 0, 'selected_count': 0, 'expected_churn_rate': initial_expected})

    for r in range(1, rounds+1):
        # select top-k non-intervened ids
        candidates = [i for i in working_prob.index if i not in intervened]
        if not candidates:
            break
        cand_probs = working_prob.loc[candidates]
        top_ids = cand_probs.sort_values(ascending=False).head(top_k).index.tolist()
        # apply intervention
        for sid in top_ids:
            oldp_series = working_prob.loc[[sid]]
            # since we aggregated to unique students, this should be scalar; keep robust:
            if isinstance(oldp_series, pd.Series) and oldp_series.shape == ():
                oldp = float(oldp_series)
            else:
                oldp = float(oldp_series.values[0])
            newp = oldp * (1.0 - efficacy)
            working_prob.at[sid] = newp
            intervened.add(sid)
        exp_churn = float(working_prob.mean())
        sim_history.append({
            'round': r,
            'selected_count': len(top_ids),
            'expected_churn_rate': exp_churn,
            'mean_risk_pre': float(cand_probs.sort_values(ascending=False).head(top_k).mean()),
            'mean_risk_post': float(working_prob.loc[top_ids].mean())
        })
        selected_history.append({'round': r, 'selected_ids': top_ids})
    out = {'sim_history': sim_history, 'selected_history': selected_history, 'params': {'rounds': rounds, 'top_k': top_k, 'efficacy': efficacy}}
    return out

def main():
    print("Running RMAB simulation...")
    probs = infer_risk_probs()
    print("Inferred risk probabilities for", len(probs), "unique students.")
    sim = run_rmab_sim(probs, rounds=ROUNDS, top_k=TOP_K, efficacy=EFFICACY, random_state=SEED)
    out_path = Path("outputs/rmab_simulation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sim, open(out_path, "w"), indent=2)
    print("Saved RMAB simulation to", out_path)
    for row in sim['sim_history']:
        print(f"round {row['round']:2d} selected={row.get('selected_count',0):3d} expected_churn={row['expected_churn_rate']:.4f}")

if __name__ == "__main__":
    main()
