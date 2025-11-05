# debug_select.py — robust debug + greedy selector for resource-aware RMAB
from pathlib import Path
import pandas as pd
import sys
from collections import Counter

CSV = Path("outputs/custom_test_results.csv")

if not CSV.exists():
    print(f"ERROR: {CSV} not found. Run inference/test to produce this file first.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(CSV)
print("Loaded CSV:", CSV)
print("Columns:", list(df.columns))
print("Number of rows:", len(df))

# --- id & prob detection with fallback to Unnamed: 0 ---
id_candidates = ["id_student", "id", "student_id", "student", "idnumber", "Unnamed: 0"]
prob_candidates = ["nn_prob", "prob", "score", "risk", "risk_prob", "recommend_intervene_nn"]

def find_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    for c in cols:
        low = c.lower()
        for cand in candidates:
            if cand in low:
                return c
    return None

ID_COL = find_col(id_candidates, df.columns)
PROB_COL = find_col(prob_candidates, df.columns)

if ID_COL is None:
    print("ERROR: couldn't find any id column. Try one of:", id_candidates, file=sys.stderr)
    sys.exit(1)

if PROB_COL is None:
    print("ERROR: couldn't find any probability column. Try one of:", prob_candidates, file=sys.stderr)
    sys.exit(1)

print(f"Using ID column: '{ID_COL}'")
print(f"Using probability column: '{PROB_COL}'")

# Normalize/protect probability column
df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors="coerce").fillna(0.0).clip(0,1)
df[ID_COL] = df[ID_COL].astype(str)

# Resource definitions (match your rmab_with_resources.py — adjust if different)
RESOURCES = {
    "sms": {"cost": 1.0, "efficacy": 0.10},
    "assignment": {"cost": 5.0, "efficacy": 0.25},
    "mentor_call": {"cost": 20.0, "efficacy": 0.50},
}

# Buckets and resource priorities (adjust as you need)
BUCKETS = {
    "high": {"min": 0.70, "max": 1.00},
    "mid": {"min": 0.40, "max": 0.70},
    "low": {"min": 0.00, "max": 0.40}
}
BUCKET_RESOURCE_PRIORITY = {
    "high": ["mentor_call", "assignment", "sms"],
    "mid": ["assignment", "sms"],
    "low": ["sms"]
}

def bucket_of(p):
    for name, rng in BUCKETS.items():
        if rng["min"] <= p <= rng["max"]:
            return name
    return "low"

def expected_reduction(p, resource):
    # reduction in risk after applying resource: here p * efficacy (you can change)
    eff = RESOURCES[resource]["efficacy"]
    return p * eff

# Build candidate pairs: one entry per (student, resource)
candidates = []
for _, row in df[[ID_COL, PROB_COL]].iterrows():
    sid = str(row[ID_COL])
    p = float(row[PROB_COL])
    b = bucket_of(p)
    # For each bucket choose allowed resources (priority order), but include all as candidates
    for r in BUCKET_RESOURCE_PRIORITY[b]:
        cost = RESOURCES[r]["cost"]
        reduction = expected_reduction(p, r)
        # "value per cost" score for greedy knapsack
        score = reduction / cost if cost > 0 else 0
        candidates.append({
            "id": sid,
            "bucket": b,
            "resource": r,
            "cost": cost,
            "reduction": reduction,
            "score": score,
            "prob": p
        })

print("Built candidate pairs:", len(candidates), " (students * allowed resources)")

# GREEDY SELECTOR: pick best candidate pairs until budget exhausted (no per-student limit)
def greedy_select(candidates, budget):
    picked = []
    remaining_budget = float(budget)
    # sort by score then absolute reduction (desc)
    sorted_c = sorted(candidates, key=lambda x: (x["score"], x["reduction"]), reverse=True)
    for c in sorted_c:
        if c["cost"] <= remaining_budget:
            picked.append(c)
            remaining_budget -= c["cost"]
    return picked, remaining_budget

# run example selection with a budget similar to your run
BUDGET_PER_ROUND = 1000.0
picked, rem = greedy_select(candidates, BUDGET_PER_ROUND)

print(f"\nWith budget_per_round={BUDGET_PER_ROUND} you can pick {len(picked)} candidate actions; remaining budget={rem:.2f}")
print("\nTop 10 selected actions (sample):")
for i, c in enumerate(picked[:10], 1):
    print(f"{i}. id={c['id']}, prob={c['prob']:.4f}, bucket={c['bucket']}, resource={c['resource']}, cost={c['cost']}, reduction={c['reduction']:.4f}, score={c['score']:.6f}")

# Stats
print("\nSelected resources counts:", Counter([c['resource'] for c in picked]))
print("Unique students selected:", len({c['id'] for c in picked}))
