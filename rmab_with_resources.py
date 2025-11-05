#!/usr/bin/env python3
"""
rmab_with_resources.py
- Uses train.csv as default input.
- If nn_prob is missing, builds it from label column or a heuristic using numeric columns.
- Keeps target-picks, cooldown, adaptive budget logic from previous version.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd
import sys
import numpy as np

# ---------------- Config ----------------
INPUT_CSV = Path("processed/train.csv")                # <-- now uses train.csv
OUTPUT_JSON = Path("outputs/rmab_with_resources.json")
OUTPUT_RESULTS_CSV = Path("outputs/rmab_with_resources_results.csv")

PROB_COL = "nn_prob"
POSSIBLE_ID_COLS = ["id_student", "id", "student_id", "student", "idnumber", "Unnamed: 0"]

ROUNDS = 10

# Selection control
ALLOW_MULTIPLE_PER_STUDENT_IN_SAME_ROUND = False
MAX_SELECTION_PER_ROUND = None  # None -> budget controls
COOLDOWN_ROUNDS = 2             # avoid reselecting same student for this many rounds

# Resources and efficacy (fractional reduction of current risk)
RESOURCES = {
    "sms":         {"efficacy": 0.10},
    "assignment":  {"efficacy": 0.25},
    "mentor_call": {"efficacy": 0.50},
}

# Buckets and costs
BUCKET_RANGES = {
    "high": {"min": 0.7, "max": 1.0},
    "mid":  {"min": 0.4, "max": 0.7},
    "low":  {"min": 0.0, "max": 0.4},
}
BUCKET_COST = {"low": 1.0, "mid": 5.0, "high": 10.0}
BUCKET_RESOURCE_PRIORITY = {
    "high": ["mentor_call", "assignment", "sms"],
    "mid":  ["assignment", "sms"],
    "low":  ["sms"]
}

# Budget strategy
BUDGET_FRACTION = 0.010           # fraction of total potential cost
MIN_BUDGET_PER_ROUND = None     # override floor (None -> uses cheapest cost)
TARGET_PICKS_PER_ROUND = None   # if set, ensures budget >= target * min_cost
# ----------------------------------------

MIN_REDUCTION = 1e-6


def detect_id_col(df: pd.DataFrame) -> str:
    for c in POSSIBLE_ID_COLS:
        if c in df.columns:
            return c
    lowered = {col.lower(): col for col in df.columns}
    for c in POSSIBLE_ID_COLS:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise KeyError(f"couldn't find any id column. Try one of: {POSSIBLE_ID_COLS}")


def bucket_of(p: float) -> str:
    for name, rng in BUCKET_RANGES.items():
        if (p >= rng["min"]) and (p <= rng["max"]):
            return name
    return "low"


def synthesize_probabilities(df: pd.DataFrame, id_col: str) -> pd.Series:
    """
    Create a probability column when none exists:
    1) If a binary label exists (target/churn/label/is_dropout), use it as soft probs.
    2) Otherwise, use heuristic from numeric columns: prefer 'avg_score' if present,
       otherwise normalize numeric columns and produce composite risk (higher means more risk).
    """
    # 1) find label column
    possible_labels = ["target", "churn", "label", "is_dropout", "dropout"]
    for lbl in possible_labels:
        if lbl in df.columns:
            vals = df[lbl].dropna().unique()
            if set(np.unique(vals)).issubset({0, 1}) or set(np.unique(vals)).issubset({0.0, 1.0}):
                # map 1 -> high risk prob ~0.9, 0 -> low risk prob ~0.1, add tiny noise
                probs = df[lbl].fillna(0).astype(float).map(lambda x: 0.9 if x >= 0.5 else 0.1)
                # add small jitter to avoid ties
                probs = probs + np.random.normal(0, 0.02, size=len(probs))
                probs = np.clip(probs, 0.01, 0.99)
                print(f"Built probabilities from label column '{lbl}' (binary -> soft prob).")
                return probs

    # 2) heuristic using numeric columns
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        # fallback random small probs
        print("No numeric columns found — assigning random probabilities (uniform small).")
        return pd.Series(np.clip(np.random.rand(len(df)) * 0.5 + 0.25, 0.01, 0.99), index=df.index)

    # prefer avg_score: higher score -> lower risk
    if "avg_score" in numeric.columns:
        vals = numeric["avg_score"].fillna(numeric["avg_score"].mean()).astype(float).values
        # invert and normalize
        inv = np.max(vals) - vals
        if np.ptp(inv) == 0:
            probs = np.full(len(vals), 0.5)
        else:
            probs = (inv - inv.min()) / (inv.max() - inv.min())
    else:
        # composite of normalized numeric columns
        arr = numeric.fillna(numeric.mean()).astype(float).values
        # normalize each column to [0,1]
        colmins = arr.min(axis=0)
        colmaxs = arr.max(axis=0)
        denom = np.where(colmaxs - colmins == 0, 1.0, colmaxs - colmins)
        norm = (arr - colmins) / denom
        # assume larger values more risk (if not, we can't know)—so use mean
        probs = norm.mean(axis=1)

    # scale to sensible [0.01, 0.99]
    probs = np.clip(probs * 1.0, 0.01, 0.99)
    print("Synthesized probabilities from numeric features (heuristic).")
    return pd.Series(probs, index=df.index)


def compute_adaptive_budget(df: pd.DataFrame, prob_col: str) -> float:
    bucket_counts = {"low": 0, "mid": 0, "high": 0}
    for p in df[prob_col]:
        try:
            b = bucket_of(float(p))
        except Exception:
            b = "low"
        bucket_counts[b] += 1

    min_cost = min(BUCKET_COST.values())
    total_cost = (
        bucket_counts["low"] * BUCKET_COST["low"] +
        bucket_counts["mid"] * BUCKET_COST["mid"] +
        bucket_counts["high"] * BUCKET_COST["high"]
    )

    rounds_safe = max(1, ROUNDS)
    per_round_calc = (BUDGET_FRACTION * total_cost) / rounds_safe

    floor = MIN_BUDGET_PER_ROUND if MIN_BUDGET_PER_ROUND is not None else min_cost
    per_round = max(per_round_calc, floor)

    if TARGET_PICKS_PER_ROUND is not None and TARGET_PICKS_PER_ROUND > 0:
        per_round = max(per_round, float(TARGET_PICKS_PER_ROUND) * min_cost)

    print("=== budget debug ===")
    print("bucket_counts:", bucket_counts)
    print(f"total potential cost: {total_cost:.2f}")
    print(f"computed per_round (BUDGET_FRACTION path): {per_round_calc:.2f}")
    print(f"applied floor: {floor:.2f}")
    if TARGET_PICKS_PER_ROUND:
        print(f"target picks requested: {TARGET_PICKS_PER_ROUND} -> ensuring budget >= {TARGET_PICKS_PER_ROUND * min_cost:.2f}")
    print(f"final per_round budget: {per_round:.2f}")
    print("====================")

    return float(per_round)


def select_candidates_greedy(
    df: pd.DataFrame,
    id_col: str,
    prob_col: str,
    budget: float,
    allow_multiple_per_student: bool = False,
    max_selection: int | None = None,
    excluded_ids: set | None = None
) -> Tuple[List[Dict[str, Any]], float]:
    excluded_ids = set() if excluded_ids is None else set(excluded_ids)

    dfc = df[[id_col, prob_col]].copy()
    dfc[prob_col] = pd.to_numeric(dfc[prob_col], errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    dfc[id_col] = dfc[id_col].astype(str)

    candidates = []
    for _, row in dfc.iterrows():
        sid = str(row[id_col])
        if sid in excluded_ids:
            continue
        p = float(row[prob_col])
        if p <= 0.0:
            continue
        b = bucket_of(p)
        cost = BUCKET_COST[b]
        allowed_resources = BUCKET_RESOURCE_PRIORITY.get(b, ["sms"])
        for res in allowed_resources:
            eff = float(RESOURCES[res]["efficacy"])
            reduction = p * eff
            if reduction < MIN_REDUCTION:
                continue
            score = reduction / cost
            tie = random.random() * 1e-6
            risk_post = max(0.0, p - reduction)
            candidates.append({
                "id_student": sid,
                "resource": res,
                "bucket": b,
                "cost": cost,
                "efficacy": eff,
                "reduction": reduction,
                "score": score,
                "risk_pre": p,
                "risk_post": risk_post,
                "tie": tie
            })

    if not candidates:
        return [], float(budget)

    candidates.sort(key=lambda c: (c["score"], c["tie"]), reverse=True)

    picks = []
    remaining_budget = float(budget)
    selected_students = set()

    for c in candidates:
        if max_selection is not None and len(picks) >= max_selection:
            break
        if c["cost"] > remaining_budget:
            continue
        if (not allow_multiple_per_student) and (c["id_student"] in selected_students):
            continue
        picks.append(c)
        remaining_budget -= c["cost"]
        selected_students.add(c["id_student"])
        if remaining_budget <= 1e-9:
            break

    return picks, remaining_budget


def run_simulation(df_preds: pd.DataFrame, id_col: str, prob_col: str, rounds: int):
    budget_per_round = compute_adaptive_budget(df_preds, prob_col)
    print(f"Using per-round budget ≈ {budget_per_round:.2f}")

    df = df_preds.copy()
    df[id_col] = df[id_col].astype(str)
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    sim_history, selected_history = [], []
    sim_history.append({
        "round": 0,
        "selected_count": 0,
        "expected_churn_rate": float(df[prob_col].mean())
    })

    cooldown_remaining = defaultdict(int)

    for r in range(1, rounds + 1):
        excluded = {sid for sid, left in cooldown_remaining.items() if left > 0}

        picks, rem_budget = select_candidates_greedy(
            df=df,
            id_col=id_col,
            prob_col=prob_col,
            budget=budget_per_round,
            allow_multiple_per_student=ALLOW_MULTIPLE_PER_STUDENT_IN_SAME_ROUND,
            max_selection=MAX_SELECTION_PER_ROUND,
            excluded_ids=excluded
        )

        id_to_updates = defaultdict(list)
        for p in picks:
            id_to_updates[p["id_student"]].append(p)

        for sid, updates in id_to_updates.items():
            mask = df[id_col].astype(str) == sid
            if not mask.any():
                continue
            for up in updates:
                curp = float(df.loc[mask, prob_col].iloc[0])
                newp = max(0.0, curp - curp * up["efficacy"])
                df.loc[mask, prob_col] = newp

            if COOLDOWN_ROUNDS > 0:
                cooldown_remaining[sid] = COOLDOWN_ROUNDS

        for sid in list(cooldown_remaining.keys()):
            if cooldown_remaining[sid] > 0:
                cooldown_remaining[sid] -= 1
            if cooldown_remaining[sid] <= 0:
                del cooldown_remaining[sid]

        mean_risk_pre = float(sum(p["risk_pre"] for p in picks) / len(picks)) if picks else None
        mean_risk_post = float(sum(float(df.loc[df[id_col].astype(str) == p["id_student"], prob_col].iloc[0]) for p in picks) / len(picks)) if picks else None

        sim_history.append({
            "round": r,
            "selected_count": len(picks),
            "expected_churn_rate": float(df[prob_col].mean()),
            **({"mean_risk_pre": mean_risk_pre, "mean_risk_post": mean_risk_post} if picks else {})
        })

        selected_history.append({
            "round": r,
            "selected": [
                {
                    "id_student": p["id_student"],
                    "resource": p["resource"],
                    "bucket": p["bucket"],
                    "cost": p["cost"],
                    "risk_pre": p["risk_pre"],
                    "risk_post": float(df.loc[df[id_col].astype(str) == p["id_student"], prob_col].iloc[0]),
                    "reduction": p["reduction"]
                } for p in picks
            ]
        })

    out = {
        "sim_history": sim_history,
        "selected_history": selected_history,
        "params": {
            "rounds": rounds,
            "budget_fraction": BUDGET_FRACTION,
            "min_budget_per_round": MIN_BUDGET_PER_ROUND,
            "target_picks_per_round": TARGET_PICKS_PER_ROUND,
            "cooldown_rounds": COOLDOWN_ROUNDS,
            "bucket_cost": BUCKET_COST
        }
    }
    return out, df


def main():
    random.seed(0)
    if not INPUT_CSV.exists():
        print(f"ERROR: input not found: {INPUT_CSV.resolve()}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded CSV: {INPUT_CSV}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    try:
        id_col = detect_id_col(df)
    except KeyError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    print("Using id column:", id_col)

    if PROB_COL not in df.columns:
        print(f"Probability column '{PROB_COL}' not found — synthesizing probabilities.")
        df[PROB_COL] = synthesize_probabilities(df, id_col)
    else:
        # ensure 0..1 floats
        df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    sim_out, final_df = run_simulation(df, id_col=id_col, prob_col=PROB_COL, rounds=ROUNDS)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as f:
        json.dump(sim_out, f, indent=2)
    print(f"Wrote {OUTPUT_JSON}")

    res_df = final_df[[id_col, PROB_COL]].copy()
    res_df["bucket"] = res_df[PROB_COL].apply(bucket_of)
    res_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"Wrote {OUTPUT_RESULTS_CSV}")

    first = sim_out["sim_history"][0]
    last = sim_out["sim_history"][-1]
    print("\nSIMULATION SUMMARY (first + last):")
    print(json.dumps([first, last], indent=2))


if __name__ == "__main__":
    main()
