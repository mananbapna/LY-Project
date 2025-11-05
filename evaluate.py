#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def smape(y_true, y_pred, eps=1e-6):
    """
    SMAPE between y_true (0/1) and y_pred (probabilities or predictions).
    Returns percentage.
    """
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    diff = np.abs(y_pred - y_true)
    return 10.0 * np.mean(2.0 * diff / denom)

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    metrics['smape'] = float(smape(y_true, y_prob))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        metrics['roc_auc'] = 0.0
    return metrics
