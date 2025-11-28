#!/usr/bin/env python3
"""
rmab_egreedy_eval.py

Reads outputs/rmab_egreedy_learning.json and computes evaluation metrics:
  - cumulative regret and average regret per round
  - average success rate
  - average action accuracy
  - reward per cost

Run after: python3 rmab_egreedy.py
"""

import json
from pathlib import Path
import statistics as stats

HISTORY_JSON = Path("outputs/rmab_egreedy_learning.json")
EVAL_JSON = Path("outputs/rmab_egreedy_eval.json")

# Must match the values in rmab_egreedy.py (for reference / consistency)
TRUE_EFFICACY = {
    "sms":         0.10,
    "assignment":  0.25,
    "mentor_call": 0.50,
}

RESOURCE_COSTS = {
    "sms": 1.0,
    "assignment": 5.0,
    "mentor_call": 10.0,
}


def main():
    if not HISTORY_JSON.exists():
        print(f"Error: {HISTORY_JSON} not found. Run rmab_egreedy.py first.")
        return

    with open(HISTORY_JSON, "r") as f:
        history = json.load(f)

    if not history:
        print("History is empty; nothing to evaluate.")
        return

    n_rounds = len(history)

    # We assume the logging in rmab_egreedy.py stores these keys
    last = history[-1]
    final_cum_successes = last.get("cumulative_successes", 0.0)
    final_cum_optimal = last.get("cumulative_optimal_successes", 0.0)
    final_cum_regret = last.get("cumulative_regret", 0.0)

    # epsilon may be stored per round; if so use the last one, else None
    epsilon = last.get("epsilon", None)

    # ---- Regret ----
    avg_regret_per_round = final_cum_regret / n_rounds if n_rounds > 0 else 0.0

    # ---- Policy efficiency ----
    policy_efficiency = (
        final_cum_successes / final_cum_optimal if final_cum_optimal > 0 else 0.0
    )

    # ---- Success metrics ----
    success_rates = [h.get("success_rate", 0.0) for h in history]
    avg_success_rate = stats.mean(success_rates) if success_rates else 0.0

    # ---- Action accuracy ----
    action_accuracies = [h.get("action_accuracy", 0.0) for h in history]
    avg_action_accuracy = (
        stats.mean(action_accuracies) if action_accuracies else 0.0
    )

    # ---- Cost & reward-per-cost ----
    total_cost = sum(h.get("total_cost", 0.0) for h in history)
    reward_per_cost = (
        final_cum_successes / total_cost if total_cost > 0 else 0.0
    )

    print("\n=== Regret & Reward Metrics (ε-greedy) ===")
    print(f"Rounds                         : {n_rounds}")
    print(f"Final cumulative successes     : {final_cum_successes:.2f}")
    print(f"Final cumulative optimal       : {final_cum_optimal:.2f}")
    print(f"Final cumulative regret        : {final_cum_regret:.2f}")
    print(f"Avg regret / round             : {avg_regret_per_round:.3f}")
    print(f"Policy efficiency (succ/opt)   : {policy_efficiency:.3f}")

    print("\n=== Rates & Accuracy ===")
    print(f"Average success rate           : {avg_success_rate:.6f}")
    print(f"Average action accuracy        : {avg_action_accuracy:.6f}")

    print("\n=== Cost Efficiency ===")
    print(f"Total cost spent               : {total_cost:.2f}")
    print(f"Reward per cost                : {reward_per_cost:.4f}")

    if epsilon is not None:
        print(f"\nEpsilon (exploration rate)     : {epsilon}")

    # Save compact JSON summary (same style as Thompson eval)
    summary = {
        "rounds": n_rounds,
        "final_cumulative_successes": final_cum_successes,
        "final_cumulative_optimal_successes": final_cum_optimal,
        "final_cumulative_regret": final_cum_regret,
        "avg_regret_per_round": avg_regret_per_round,
        "policy_efficiency": policy_efficiency,
        "avg_success_rate": avg_success_rate,
        "avg_action_accuracy": avg_action_accuracy,
        "total_cost": total_cost,
        "reward_per_cost": reward_per_cost,
        "epsilon": epsilon,
    }

    EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation summary to {EVAL_JSON}")


if __name__ == "__main__":
    main()
