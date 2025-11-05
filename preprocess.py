#!/usr/bin/env python3
"""
preprocess.py
- Load raw CSVs from data/
- Merge & engineer features
- Save processed/processed.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"
OUT_DIR = "processed"
OUT_PATH = os.path.join(OUT_DIR, "processed.csv")

def safe_read(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None

def load_raw(data_dir=DATA_DIR):
    expected = {
        "vle": "vle.csv",
        "studentVle": "studentVle.csv",
        "studentRegistration": "studentRegistration.csv",
        "studentInfo": "studentInfo.csv",
        "studentAssessment": "studentAssessment.csv",
        "courses": "courses.csv",
        "assessments": "assessments.csv"
    }
    dfs = {}
    for k, fname in expected.items():
        path = os.path.join(data_dir, fname)
        df = safe_read(path)
        if df is not None:
            dfs[k] = df
            print(f"Loaded {fname} shape={df.shape}")
        else:
            print(f"Missing (optional) {fname}")
    return dfs

def engineer_features(dfs):
    # Base dataframe
    if "studentInfo" in dfs:
        base = dfs["studentInfo"].copy()
    elif "studentRegistration" in dfs:
        base = dfs["studentRegistration"].copy()
    else:
        raise FileNotFoundError("studentInfo.csv or studentRegistration.csv required in data/")

    # normalize id column name
    id_col = None
    for c in ["id_student", "student_id", "id"]:
        if c in base.columns:
            id_col = c
            break
    if id_col is None:
        id_col = base.columns[0]
    if id_col != "id_student":
        base = base.rename(columns={id_col: "id_student"})

    # merge registration info if present
    if "studentRegistration" in dfs:
        reg = dfs["studentRegistration"].copy()
        if 'id_student' not in reg.columns and 'student_id' in reg.columns:
            reg = reg.rename(columns={'student_id': 'id_student'})
        if 'id_student' in reg.columns:
            base = base.merge(reg, on='id_student', how='left')

    # aggregate VLE interactions
    if "studentVle" in dfs:
        sv = dfs["studentVle"].copy()
        if 'id_student' not in sv.columns and 'student_id' in sv.columns:
            sv = sv.rename(columns={'student_id': 'id_student'})
        if 'sum_click' in sv.columns:
            agg = sv.groupby('id_student')['sum_click'].sum().rename('vle_interactions').reset_index()
        else:
            agg = sv.groupby('id_student').size().rename('vle_interactions').reset_index()
        base = base.merge(agg, on='id_student', how='left')
    elif "vle" in dfs:
        v = dfs["vle"].copy()
        if 'id_student' not in v.columns and 'student_id' in v.columns:
            v = v.rename(columns={'student_id':'id_student'})
        agg = v.groupby('id_student').size().rename('vle_interactions').reset_index()
        base = base.merge(agg, on='id_student', how='left')

    # average assessment score
    if "studentAssessment" in dfs:
        sa = dfs["studentAssessment"].copy()
        if 'id_student' not in sa.columns and 'student_id' in sa.columns:
            sa = sa.rename(columns={'student_id':'id_student'})
        if 'score' in sa.columns:
            agg2 = sa.groupby('id_student')['score'].mean().rename('avg_score').reset_index()
            base = base.merge(agg2, on='id_student', how='left')

    # fill numerical missing values with median
    num_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        base[c] = base[c].fillna(base[c].median())

    # categorical columns to encode if present
    cat_candidates = ['gender','final_result','education','course_id','region','country']
    cat_cols = [c for c in cat_candidates if c in base.columns]

    # create churn label
    if 'final_result' in base.columns:
        base['churn'] = base['final_result'].astype(str).str.lower().apply(
            lambda x: 1 if any(tok in x for tok in ['withdrawn','dropout','fail','absent']) else 0
        )
    else:
        if 'avg_score' in base.columns:
            q = base['avg_score'].quantile(0.25)
            base['churn'] = (base['avg_score'] < q).astype(int)
        elif 'vle_interactions' in base.columns:
            q = base['vle_interactions'].quantile(0.25)
            base['churn'] = (base['vle_interactions'] < q).astype(int)
        else:
            base['churn'] = 0

    # label encode categorical features
    for c in cat_cols:
        base[c] = base[c].fillna('NA').astype(str)
        le = LabelEncoder()
        base[c] = le.fit_transform(base[c])

    # final selection: numeric + encoded categorical + churn
    keep_cols = list(set(num_cols + cat_cols + ['churn']))
    keep_cols = [c for c in keep_cols if c in base.columns]
    processed = base[keep_cols].copy()

    # ensure index contains id_student if present
    if 'id_student' in base.columns:
        processed.index = base['id_student']

    return processed

def save_processed(df, out_dir=OUT_DIR, out_path=OUT_PATH):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=True)
    print(f"Saved processed data to {out_path} shape={df.shape}")

def main():
    dfs = load_raw(DATA_DIR)
    processed = engineer_features(dfs)
    save_processed(processed)

if __name__ == "__main__":
    main()
