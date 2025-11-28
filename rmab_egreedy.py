#!/usr/bin/env python3
"""
rmab_egreedy.py

Bandit-style RMAB with an ε-greedy learning agent over intervention types.
Environment + data loading is same as rmab_thompson.py so results are comparable.
"""

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import sys

# ---------- Config ----------
INPUT_CSV = Path("processed/train.csv")
LEARNING_JSON = Path("outputs/rmab_egreedy_learning.json")
EVAL_JSON     = Path("outputs/rmab_egreedy_eval.json")

PROB_COL = "nn_prob"
POSSIBLE_ID_COLS = ["id_student", "id", "student_id"]

ROUNDS = 20

# Hidden true efficacies (same as Thompson version)
TRUE_EFFICACY = {
    "sms":         0.10,
    "assignment":  0.25,
    "mentor_call": 0.50,
}

RESOURCE_COSTS = {
    "sms": 1.0,
    "assignment": 5.0,
    "mentor_call": 10.0
}

BUDGET_PER_ROUND = 500.0

# ε for ε-greedy
EPSILON = 0.1
# -----------------------------


class EpsilonGreedyAgent:
    """
    Maintains Q estimates (mean success probability) for each resource.
    With prob ε -> explore (random resource)
    With prob 1-ε -> exploit (argmax Q)
    Updates Q via incremental mean.
    """
    def __init__(self, resources, epsilon=0.1):
        self.resources = list(resources)
        self.epsilon = epsilon
        self.counts = {r: 0 for r in self.resources}
        self.q_values = {r: 0.0 for r in self.resources}  # estimated success prob

    def select_resource(self):
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.resources)
        # Exploitation
        # break ties randomly
        max_q = max(self.q_values.values())
        best = [r for r, q in self.q_values.items() if q == max_q]
        return random.choice(best)

    def update(self, resource_name, reward):
        """
        reward is 1 for success, 0 for failure (Bernoulli).
        """
        self.counts[resource_name] += 1
        n = self.counts[resource_name]
        old_q = self.q_values[resource_name]
        new_q = old_q + (reward - old_q) / n
        self.q_values[resource_name] = new_q

    def get_beliefs(self):
        """
        For comparison with Thompson, we treat q_values as "beliefs".
        """
        return dict(self.q_values)


def detect_id_col(df):
    for c in POSSIBLE_ID_COLS:
        if c in df.columns:
            return c
    return df.columns[0]


def synthesize_probabilities(df, id_col):
    if "avg_score" in df.columns:
        vals = df["avg_score"].fillna(df["avg_score"].mean()).astype(float).values
        inv = np.max(vals) - vals
        probs = (inv - inv.min()) / (inv.max() - inv.min()) if np.ptp(inv) > 0 else np.full(len(vals), 0.5)
        return pd.Series(np.clip(probs, 0.01, 0.99), index=df.index)
    return pd.Series(np.random.rand(len(df)), index=df.index)


def optimal_resource_for_student():
    """
    For evaluation: best single resource in expectation is just argmax(TRUE_EFFICACY).
    (If you wanted a more complex student-specific optimal policy, adapt here.)
    """
    return max(TRUE_EFFICACY.items(), key=lambda x: x[1])[0]


def run_egreedy_simulation(df, id_col, prob_col, rounds):
    agent = EpsilonGreedyAgent(RESOURCE_COSTS.keys(), epsilon=EPSILON)
    history = []

    cumulative_successes = 0
    cumulative_optimal_successes = 0.0
    cumulative_regret = 0.0
    total_cost_spent = 0.0
    total_actions = 0
    correct_action_count = 0

    print(f"Starting ε-greedy Simulation (budget ₹ {BUDGET_PER_ROUND} / round)...")

    for r in range(1, rounds + 1):
        round_selected = []
        round_spent = 0.0
        round_successes = 0
        round_optimal_successes = 0.0
        round_correct_action = 0

        students = df.sample(frac=1.0, random_state=r)  # shuffle each round

        for idx, row in students.iterrows():
            sid = str(row[id_col])
            risk = float(row[prob_col])

            # pick a resource using ε-greedy policy
            resource = agent.select_resource()
            cost = RESOURCE_COSTS[resource]

            if round_spent + cost > BUDGET_PER_ROUND:
                continue

            round_spent += cost
            total_cost_spent += cost
            total_actions += 1
            round_selected.append((sid, resource))

            # environment feedback
            true_eff = TRUE_EFFICACY[resource]
            success = 1 if random.random() < true_eff else 0
            round_successes += success
            cumulative_successes += success

            # update agent
            agent.update(resource, success)

            # approximate "risk reduction" not explicitly used in eval here
            # but you could apply it to df[prob_col] if desired

            # optimal agent baseline: always use best resource
            best_res = optimal_resource_for_student()
            best_eff = TRUE_EFFICACY[best_res]
            # expected success under optimal: risk * best_eff
            round_optimal_successes += best_eff  # simplified baseline
            cumulative_optimal_successes += best_eff

            if resource == best_res:
                round_correct_action += 1
                correct_action_count += 1

        # per-round regret
        round_regret = round_optimal_successes - round_successes
        cumulative_regret += round_regret

        # beliefs snapshot
        beliefs_snapshot = agent.get_beliefs()

        method_counts = Counter([res for _, res in round_selected])

        round_total_actions = len(round_selected)
        round_success_rate = round_successes / round_total_actions if round_total_actions > 0 else 0.0

        print(
            f"Round {r}: used {dict(method_counts)}, "
            f"succ={round_successes}, opt_succ≈{round_optimal_successes:.2f}, "
            f"regret≈{round_regret:.2f}, beliefs={ {k: round(v,3) for k,v in beliefs_snapshot.items()} }"
        )

        history.append({
            "round": r,
            "total_selected": round_total_actions,
            "method_counts": dict(method_counts),
            "round_successes": round_successes,
            "round_optimal_successes": round_optimal_successes,
            "round_regret": round_regret,
            "cumulative_successes": cumulative_successes,
            "cumulative_optimal_successes": cumulative_optimal_successes,
            "cumulative_regret": cumulative_regret,
            "success_rate": round_success_rate,
            "agent_beliefs": beliefs_snapshot,
            "round_spent": round_spent,
        })

    # evaluation metrics (same style as Thompson)
    avg_regret_per_round = cumulative_regret / rounds
    policy_efficiency = cumulative_successes / cumulative_optimal_successes if cumulative_optimal_successes > 0 else 0.0
    avg_success_rate = np.mean([h["success_rate"] for h in history])
    avg_action_accuracy = correct_action_count / total_actions if total_actions > 0 else 0.0
    reward_per_cost = cumulative_successes / total_cost_spent if total_cost_spent > 0 else 0.0

    eval_summary = {
        "rounds": rounds,
        "final_cumulative_successes": cumulative_successes,
        "final_cumulative_optimal_successes": cumulative_optimal_successes,
        "final_cumulative_regret": cumulative_regret,
        "avg_regret_per_round": avg_regret_per_round,
        "policy_efficiency": policy_efficiency,
        "avg_success_rate": avg_success_rate,
        "avg_action_accuracy": avg_action_accuracy,
        "total_cost": total_cost_spent,
        "reward_per_cost": reward_per_cost,
        "epsilon": EPSILON,
    }

    return history, eval_summary


def main():
    if not INPUT_CSV.exists():
        print("Error: processed/train.csv not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    id_col = detect_id_col(df)

    if PROB_COL not in df.columns:
        df[PROB_COL] = synthesize_probabilities(df, id_col)

    df.set_index(id_col, inplace=True, drop=False)

    history, eval_summary = run_egreedy_simulation(df, id_col, PROB_COL, ROUNDS)

    LEARNING_JSON.parent.mkdir(exist_ok=True)
    with open(LEARNING_JSON, "w") as f:
        json.dump(history, f, indent=2)

    with open(EVAL_JSON, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSaved ε-greedy learning log to {LEARNING_JSON}")
    print(f"Saved ε-greedy evaluation summary to {EVAL_JSON}")
    print("\nε-greedy Evaluation Summary:")
    for k, v in eval_summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
