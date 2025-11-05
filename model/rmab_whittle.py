import numpy as np
import pandas as pd

def approx_whittle_index(belief, intervention_effect=0.25, horizon=10, discount=0.95, cost=1.0):
    p0 = float(belief)
    p1 = max(0.0, p0 * (1 - intervention_effect))
    delta0 = p0 - p1
    alpha = 0.4
    cumulative = 0.0
    curr_reduction = delta0
    for t in range(horizon):
        cumulative += (discount ** t) * curr_reduction
        curr_reduction = curr_reduction * (1 - alpha)
    return cumulative / cost

def compute_indices(beliefs, intervention_effect=0.25, horizon=10, discount=0.95, cost=1.0):
    beliefs = np.asarray(beliefs, dtype=float)
    vec = np.vectorize(lambda b: approx_whittle_index(b, intervention_effect, horizon, discount, cost))
    return vec(beliefs)

def select_top_k_by_index(df, belief_col='risk_prob', k=10, intervention_effect=0.25,
                          horizon=10, discount=0.95, cost=1.0):
    if belief_col not in df.columns:
        raise ValueError(f"{belief_col} not found in dataframe")
    beliefs = df[belief_col].values
    indices = compute_indices(beliefs, intervention_effect, horizon, discount, cost)
    df = df.copy()
    df['_rmab_index'] = indices
    chosen = df.nlargest(k, '_rmab_index')
    return chosen.index.tolist(), df

def simulate_policy(df, rounds=10, budget=10, belief_col='risk_prob', intervention_effect=0.25,
                    horizon=10, discount=0.95, cost=1.0, seed=42):
    rng = np.random.RandomState(seed)
    df = df.copy()
    history = []
    if belief_col not in df.columns:
        if 'churn' in df.columns:
            df[belief_col] = df['churn'].astype(float)
        else:
            df[belief_col] = 0.0
    for t in range(rounds):
        chosen_idx, _ = select_top_k_by_index(df, belief_col=belief_col, k=budget,
                                              intervention_effect=intervention_effect,
                                              horizon=horizon, discount=discount, cost=cost)
        df.loc[chosen_idx, belief_col] = np.maximum(0.0, df.loc[chosen_idx, belief_col] * (1 - intervention_effect))
        drift = 0.01
        non_chosen = df.index.difference(chosen_idx)
        df.loc[non_chosen, belief_col] = np.clip(df.loc[non_chosen, belief_col] + drift, 0.0, 1.0)
        df['churn_sim'] = rng.binomial(1, df[belief_col].clip(0,1))
        churn_rate = df['churn_sim'].mean()
        history.append({'round': t, 'churn_rate': float(churn_rate), 'avg_risk': float(df[belief_col].mean())})
    return pd.DataFrame(history)
