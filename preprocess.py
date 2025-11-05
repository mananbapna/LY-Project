#!/usr/bin/env python3
"""
Robust preprocess.py - normalize ID dtypes to string and skip LFS pointer files
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
        df = pd.read_csv(path)
        return df
    except Exception as e:
        # Try reading without header (fallback)
        try:
            df = pd.read_csv(path, header=None)
            print(f"Read {path} with header=None (shape={df.shape})")
            return df
        except Exception as e2:
            print(f"Could not read {path}: {e} / {e2}")
            return None

def is_lfs_pointer(df):
    """
    Detect if file is a Git LFS pointer file (text like 'version https://git-lfs.github.com/spec/v1').
    These are not the actual data and should be skipped.
    """
    if df is None:
        return False
    if df.shape[1] == 1:
        first = str(df.iloc[0,0]).lower()
        if 'version https://git-lfs.github.com/spec/v1' in first or 'oid sha256' in first:
            return True
    return False

def find_id_col(df):
    """Return column name that likely represents a student id, or None."""
    if df is None:
        return None
    cols = list(df.columns)
    # prefer exact matches
    for c in ['id_student','student_id','studentid','id','userid','user_id','id_student_id']:
        if c in cols:
            return c
    # look for substring matches
    for c in cols:
        lc = str(c).lower()
        if 'student' in lc or (lc == 'id') or ('id' in lc and 'date' not in lc):
            return c
    # if there is only one column, return it (fallback)
    if len(cols) == 1:
        return cols[0]
    # no good candidate
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
        if os.path.exists(path):
            df = safe_read(path)
            if df is not None:
                print(f"Loaded {fname} shape={df.shape} columns={list(df.columns)[:10]}")
                dfs[k] = df
            else:
                print(f"Failed to load {fname}")
        else:
            print(f"Missing (optional) {fname}")
    return dfs

def to_str_id(df, col):
    """Return DataFrame with column col converted to string, if present."""
    if df is None or col is None or col not in df.columns:
        return df
    df2 = df.copy()
    # convert to string, strip spaces
    df2[col] = df2[col].astype(str).str.strip()
    return df2

def engineer_features(dfs):
    # Base dataframe selection
    if "studentInfo" in dfs and dfs["studentInfo"] is not None:
        base = dfs["studentInfo"].copy()
    elif "studentRegistration" in dfs and dfs["studentRegistration"] is not None:
        base = dfs["studentRegistration"].copy()
    else:
        raise FileNotFoundError("studentInfo.csv or studentRegistration.csv required in data/")

    # find id in base and normalize
    base_id_col = find_id_col(base)
    if base_id_col:
        base = to_str_id(base, base_id_col)
        if base_id_col != 'id_student':
            base = base.rename(columns={base_id_col: 'id_student'})
    else:
        print("Warning: could not detect id column in base dataframe; using index as id.")
        base = base.reset_index().rename(columns={'index':'id_student'})
        base['id_student'] = base['id_student'].astype(str)

    # merge registration info if present
    if "studentRegistration" in dfs and dfs["studentRegistration"] is not None:
        reg = dfs["studentRegistration"].copy()
        reg_id = find_id_col(reg)
        if reg_id:
            reg = to_str_id(reg, reg_id)
            if reg_id != 'id_student':
                reg = reg.rename(columns={reg_id: 'id_student'})
            # ensure both id columns are string type
            reg['id_student'] = reg['id_student'].astype(str)
            base['id_student'] = base['id_student'].astype(str)
            # merge if id present on both
            if 'id_student' in reg.columns and 'id_student' in base.columns:
                base = base.merge(reg, on='id_student', how='left')
                print("Merged studentRegistration on id_student")
            else:
                print("Skipping merge with studentRegistration: id_student missing in one side.")
        else:
            print("Skipping studentRegistration merge: could not find id column in studentRegistration.")

    # VLE aggregation - robust handling and skip LFS pointer
    if "studentVle" in dfs and dfs["studentVle"] is not None:
        sv = dfs["studentVle"].copy()
        if is_lfs_pointer(sv):
            print("studentVle.csv appears to be a Git LFS pointer file. Skipping VLE aggregation.")
        else:
            sv_id = find_id_col(sv)
            if sv_id:
                sv = to_str_id(sv, sv_id)
                if sv_id != 'id_student':
                    sv = sv.rename(columns={sv_id: 'id_student'})
                sv['id_student'] = sv['id_student'].astype(str)
                if 'sum_click' in sv.columns:
                    agg = sv.groupby('id_student')['sum_click'].sum().rename('vle_interactions').reset_index()
                else:
                    numeric_cols = sv.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        agg = sv.groupby('id_student')[numeric_cols[0]].sum().rename('vle_interactions').reset_index()
                    else:
                        agg = sv.groupby('id_student').size().rename('vle_interactions').reset_index()
                # ensure base id is string
                base['id_student'] = base['id_student'].astype(str)
                agg['id_student'] = agg['id_student'].astype(str)
                base = base.merge(agg, on='id_student', how='left')
                print("Merged studentVle aggregation on id_student")
            else:
                print("studentVle present but no id column detected; skipping VLE aggregation.")
    elif "vle" in dfs and dfs["vle"] is not None:
        v = dfs["vle"].copy()
        if is_lfs_pointer(v):
            print("vle.csv appears to be a Git LFS pointer file. Skipping.")
        else:
            v_id = find_id_col(v)
            if v_id:
                v = to_str_id(v, v_id)
                if v_id != 'id_student':
                    v = v.rename(columns={v_id: 'id_student'})
                agg = v.groupby('id_student').size().rename('vle_interactions').reset_index()
                base['id_student'] = base['id_student'].astype(str)
                agg['id_student'] = agg['id_student'].astype(str)
                base = base.merge(agg, on='id_student', how='left')
                print("Merged vle aggregation on id_student")
            else:
                print("vle present but no id column detected; skipping VLE aggregation.")
    else:
        print("No studentVle/vle file found; continuing without VLE features.")

    # assessments average
    if "studentAssessment" in dfs and dfs["studentAssessment"] is not None:
        sa = dfs["studentAssessment"].copy()
        sa_id = find_id_col(sa)
        if sa_id:
            sa = to_str_id(sa, sa_id)
            if sa_id != 'id_student':
                sa = sa.rename(columns={sa_id: 'id_student'})
            if 'score' in sa.columns:
                agg2 = sa.groupby('id_student')['score'].mean().rename('avg_score').reset_index()
                base['id_student'] = base['id_student'].astype(str)
                agg2['id_student'] = agg2['id_student'].astype(str)
                base = base.merge(agg2, on='id_student', how='left')
                print("Merged studentAssessment avg_score on id_student")
        else:
            print("studentAssessment present but no id column detected; skipping assessment aggregation.")

    # fill numeric missing values
    num_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        try:
            base[c] = base[c].fillna(base[c].median())
        except Exception:
            base[c] = base[c].fillna(0)

    # categorical columns to encode if present
    cat_candidates = ['gender','final_result','highest_education','education','course_id','region','country']
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
            print("No final_result/avg_score/vle_interactions available - churn set to 0 (all safe).")

    # label encode categorical features
    for c in cat_cols:
        try:
            base[c] = base[c].fillna('NA').astype(str)
            le = LabelEncoder()
            base[c] = le.fit_transform(base[c])
        except Exception as e:
            print(f"Warning: failed to encode column {c}: {e}")

    # final selection: numeric + encoded categorical + churn
    keep_cols = list(set(num_cols + cat_cols + ['churn']))
    keep_cols = [c for c in keep_cols if c in base.columns]
    processed = base[keep_cols].copy()

    # ensure index contains id_student if present in original base
    if 'id_student' in base.columns:
        processed.index = base['id_student'].astype(str)

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
