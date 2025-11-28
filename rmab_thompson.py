#!/usr/bin/env python3
"""
rmab_thompson.py

- Runs a TRUE learning agent using Thompson Sampling over multiple intervention types.
- Learns per-resource efficacy over time and logs detailed per-round metrics.
- Output: outputs/rmab_thompson_learning.json
"""

import json
import random
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import sys

# ---------------- Config ----------------
INPUT_CSV = Path("processed/train.csv")
OUTPUT_JSON = Path("outputs/rmab_thompson_learning.json")

PROB_COL = "nn_prob"
POSSIBLE_ID_COLS = ["id_student", "id", "student_id"]

ROUNDS = 20  # number of simulated weeks / rounds

# 1. The Hidden Truth (Environment – not known to the agent)
TRUE_EFFICACY = {
    "sms":         0.10,
    "assignment":  0.25,
    "mentor_call": 0.50,
}

# 2. Known Costs (visible to the agent)
RESOURCE_COSTS = {
    "sms": 1.0,
    "assignment": 5.0,
    "mentor_call": 10.0,
}

BUDGET_PER_ROUND = 500.0
# ----------------------------------------


class ThompsonSampler:
    def __init__(self, resource_names):
        self.resources = resource_names
        # Alpha=1, Beta=1 => uniform prior Beta(1,1)
        self.params = {r: {'alpha': 1.0, 'beta': 1.0} for r in resource_names}

    def sample_efficacy(self, resource_name):
        a = self.params[resource_name]['alpha']
        b = self.params[resource_name]['beta']
        return np.random.beta(a, b)

    def update(self, resource_name, success: bool):
        if success:
            self.params[resource_name]['alpha'] += 1.0
        else:
            self.params[resource_name]['beta'] += 1.0

    def get_belief(self, resource_name):
        """Mean of Beta(a,b) = a / (a + b)."""
        a = self.params[resource_name]['alpha']
        b = self.params[resource_name]['beta']
        return a / (a + b)


def synthesize_probabilities(df, id_col):
    """Fallback if nn_prob not present."""
    if "avg_score" in df.columns:
        vals = df["avg_score"].fillna(df["avg_score"].mean()).astype(float).values
        inv = np.max(vals) - vals
        if np.ptp(inv) > 0:
            probs = (inv - inv.min()) / (inv.max() - inv.min())
        else:
            probs = np.full(len(vals), 0.5)
        return pd.Series(np.clip(probs, 0.01, 0.99), index=df.index)
    # Completely synthetic
    return pd.Series(np.random.rand(len(df)), index=df.index)


def detect_id_col(df):
    for c in POSSIBLE_ID_COLS:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]


def run_learning_simulation(df: pd.DataFrame,
                            id_col: str,
                            prob_col: str,
                            rounds: int):
    """
    Runs the TS-based RMAB simulation and logs per-round metrics:
      - successes, optimal_successes
      - regret and cumulative_regret
      - action_accuracy (how often the chosen resource matches optimal resource)
      - method_counts, total_cost
      - agent_beliefs per resource
    """
    agent = ThompsonSampler(list(RESOURCE_COSTS.keys()))
    history = []

    cumulative_regret = 0.0
    cumulative_successes = 0
    cumulative_optimal_successes = 0.0

    print(f"Starting Simulation (₹ {BUDGET_PER_ROUND} / round equivalent budget)...")

    for r in range(1, rounds + 1):
        # 1. SAMPLE from current belief distributions
        sampled_efficacies = {
            res: agent.sample_efficacy(res) for res in RESOURCE_COSTS.keys()
        }

        # 2. PLAN: build all candidate (student, resource) interventions with an ROI score
        candidates = []
        for row_idx, row in df.iterrows():
            sid = row[id_col]
            risk = float(row[prob_col])

            for res, cost in RESOURCE_COSTS.items():
                sampled_eff = sampled_efficacies[res]
                estimated_reduction = risk * sampled_eff
                score = estimated_reduction / cost

                candidates.append({
                    "row_idx": row_idx,   # index in df
                    "student_id": sid,
                    "risk": risk,
                    "resource": res,
                    "cost": cost,
                    "score": score,
                })

        # Sort by ROI descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 3. ACT under budget constraint
        selected = []
        spent = 0.0
        students_picked_this_round = set()

        correct_actions = 0
        total_actions = 0

        for cand in candidates:
            sid = cand["student_id"]
            cost = cand["cost"]

            # one resource per student per round
            if sid in students_picked_this_round:
                continue
            # budget constraint
            if spent + cost > BUDGET_PER_ROUND:
                continue

            # compute optimal resource under TRUE efficacy for this student
            risk = cand["risk"]
            best_true_res = None
            best_true_score = -1.0
            for res2, cost2 in RESOURCE_COSTS.items():
                true_score = risk * TRUE_EFFICACY[res2] / cost2
                if true_score > best_true_score:
                    best_true_score = true_score
                    best_true_res = res2

            if cand["resource"] == best_true_res:
                correct_actions += 1
            total_actions += 1

            selected.append(cand)
            spent += cost
            students_picked_this_round.add(sid)

        method_counts = Counter([x["resource"] for x in selected])

        # 4. FEEDBACK + LEARNING
        successes = 0
        optimal_successes = 0.0

        for item in selected:
            res = item["resource"]
            row_idx = item["row_idx"]
            true_eff = TRUE_EFFICACY[res]

            # Bernoulli outcome according to the hidden true efficacy
            is_success = 1 if random.random() < true_eff else 0

            # update TS posterior
            agent.update(res, bool(is_success))

            if is_success:
                successes += 1
                # reduce risk for this student (simple multiplicative model)
                current_risk = float(df.at[row_idx, prob_col])
                df.at[row_idx, prob_col] = max(0.0, current_risk * 0.8)

            # what an "oracle" that knows TRUE_EFFICACY would expect
            optimal_successes += true_eff

        cumulative_successes += successes
        cumulative_optimal_successes += optimal_successes

        # 5. METRICS: regret, beliefs, action accuracy
        regret = optimal_successes - successes
        cumulative_regret += regret

        beliefs_snapshot = {
            res: round(agent.get_belief(res), 3) for res in RESOURCE_COSTS.keys()
        }

        success_rate = successes / len(selected) if selected else 0.0
        action_accuracy = (
            correct_actions / total_actions if total_actions > 0 else 0.0
        )

        print(
            f"Round {r}: {dict(method_counts)} | "
            f"Succ={successes} (rate={success_rate:.2f}) | "
            f"Regret={regret:.2f} (cum={cumulative_regret:.2f}) | "
            f"ActionAcc={action_accuracy:.2f} | "
            f"Beliefs={beliefs_snapshot}"
        )

        history.append({
            "round": r,
            "total_selected": len(selected),
            "total_cost": float(spent),
            "method_counts": dict(method_counts),
            "successes": successes,
            "success_rate": success_rate,
            "optimal_successes": optimal_successes,
            "regret": regret,
            "cumulative_regret": cumulative_regret,
            "cumulative_successes": cumulative_successes,
            "cumulative_optimal_successes": cumulative_optimal_successes,
            "action_accuracy": action_accuracy,
            "agent_beliefs": beliefs_snapshot,
        })

    return history


def main():
    if not INPUT_CSV.exists():
        print("Error: processed/train.csv not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    id_col = detect_id_col(df)

    if PROB_COL not in df.columns:
        df[PROB_COL] = synthesize_probabilities(df, id_col)

    # keep default integer index; we track row_idx explicitly
    history = run_learning_simulation(df, id_col, PROB_COL, ROUNDS)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved full simulation log to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
