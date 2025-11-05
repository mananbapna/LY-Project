import numpy as np
import pandas as pd

def simple_priority_policy(df, score_col='risk_score', budget=10):
    return df.sort_values(score_col, ascending=False).head(budget)

def simulate_rmab(df, score_col='risk_score', budget=10, rounds=10, intervention_effect=0.2):
    history = []
    df = df.copy()
    if score_col not in df.columns:
        if 'churn' in df.columns:
            df[score_col] = df['churn'].astype(float)
        else:
            df[score_col] = 0.0
    for t in range(rounds):
        selected = simple_priority_policy(df, score_col=score_col, budget=budget)
        df.loc[df.index.isin(selected.index), score_col] = np.maximum(0, df.loc[df.index.isin(selected.index), score_col] - intervention_effect)
        df['churn_sim'] = np.random.binomial(1, df[score_col].clip(0,1))
        churn_rate = df['churn_sim'].mean()
        history.append({'round': t, 'churn_rate': float(churn_rate), 'avg_score': float(df[score_col].mean())})
    return pd.DataFrame(history)
