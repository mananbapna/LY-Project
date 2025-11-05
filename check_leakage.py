# check_leakage.py (fixed)
import pandas as pd, numpy as np
df = pd.read_csv("processed/processed.csv", index_col=0)
print("Loaded processed/processed.csv shape:", df.shape)
if 'churn' not in df.columns:
    raise SystemExit("No 'churn' column found")

# numeric correlations
corr = df.select_dtypes(include=[np.number]).corr()['churn'].abs().sort_values(ascending=False)
print("\nTop numeric correlations with churn:\n", corr.head(20))

# show columns where values often equal churn (string equality)
print("\nColumns that match churn label often (string equality check):")
for c in df.columns:
    if c == 'churn': 
        continue
    try:
        frac_eq = (df[c].astype(str) == df['churn'].astype(str)).mean()
        if frac_eq > 0.5:
            print(f"  {c}: matches churn in {frac_eq*100:.1f}% of rows")
    except Exception:
        pass

# top correlated numeric features sample
top_feats = [x for x in corr.index.tolist() if x != 'churn'][0:10]
print("\nTop correlated numeric features (sample rows):", top_feats)
print(df[['churn'] + top_feats].head(30))

# candidate categorical / low-card
cand = [c for c in df.columns if (df[c].dtype == object) or (df[c].nunique() < 50)]
print("\nCandidate categorical / low-card columns to inspect:")
for c in cand:
    if c == 'churn': 
        continue
    print(f"\n-- {c} (nunique={df[c].nunique()}):")
    print(df[[c,'churn']].groupby(c)['churn'].agg(['count','sum','mean']).sort_values('mean', ascending=False).head(20))
