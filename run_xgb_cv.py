# run_xgb_cv.py
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from evaluate import classification_metrics

df = pd.read_csv("processed/processed.csv", index_col=0)
X = df.drop(columns=['churn']).values
y = df['churn'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = []
for train_idx, val_idx in skf.split(X, y):
    Xtr, Xv = X[train_idx], X[val_idx]
    ytr, yv = y[train_idx], y[val_idx]
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xv, label=yv)
    params = {'objective':'binary:logistic', 'eval_metric':'auc', 'verbosity':0}
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dval,'val')], early_stopping_rounds=10)
    probs = bst.predict(dval)
    metrics.append(classification_metrics(yv, probs))
keys = metrics[0].keys()
agg = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
print("5-fold CV mean metrics:", agg)
