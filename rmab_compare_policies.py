#!/usr/bin/env python3
"""
rmab_compare_policies.py

Compares evaluation summaries of:
  - Thompson Sampling  (outputs/rmab_thompson_eval.json)
  - ε-greedy           (outputs/rmab_egreedy_eval.json)

Run after:
  python3 rmab_thompson_eval.py
  python3 rmab_egreedy_eval.py
"""

import json
from pathlib import Path

THOMPSON_EVAL = Path("outputs/rmab_thompson_eval.json")
EGREEDY_EVAL = Path("outputs/rmab_egreedy_eval.json")


def load_eval(path, name):
    if not path.exists():
        print(f"Error: {path} not found. Run the corresponding *_eval.py first for {name}.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def fmt(val):
    """Pretty formatting for ints/floats/others."""
    if isinstance(val, float):
        # Higher precision for rates, regret etc.
        return f"{val:.6f}".rstrip("0").rstrip(".")
    if isinstance(val, int):
        return str(val)
    return str(val)


def main():
    th = load_eval(THOMPSON_EVAL, "Thompson")
    eg = load_eval(EGREEDY_EVAL, "ε-greedy")

    if th is None or eg is None:
        return

    print("\n=== Policy Comparison: Thompson vs ε-greedy ===\n")

    # Metrics we want to compare (key, label)
    metric_keys = [
        ("rounds", "Rounds"),
        ("final_cumulative_successes", "Final cumulative successes"),
        ("final_cumulative_optimal_successes", "Final cumulative optimal"),
        ("final_cumulative_regret", "Final cumulative regret"),
        ("avg_regret_per_round", "Avg regret per round"),
        ("policy_efficiency", "Policy efficiency (succ/opt)"),
        ("avg_success_rate", "Average success rate"),
        ("avg_action_accuracy", "Average action accuracy"),
        ("total_cost", "Total cost spent"),
        ("reward_per_cost", "Reward per cost"),
    ]

    # Header
    print(f"{'Metric':35s} | {'Thompson':>15s} | {'ε-greedy':>15s}")
    print("-" * 73)

    for key, label in metric_keys:
        th_val = th.get(key, None)
        eg_val = eg.get(key, None)

        # Skip if both missing
        if th_val is None and eg_val is None:
            continue

        th_s = fmt(th_val) if th_val is not None else "-"
        eg_s = fmt(eg_val) if eg_val is not None else "-"

        print(f"{label:35s} | {th_s:>15s} | {eg_s:>15s}")

    # Show epsilon of epsilon-greedy if available
    epsilon = eg.get("epsilon", None)
    if epsilon is not None:
        print("\nε-greedy exploration parameter:")
        print(f"  epsilon: {fmt(epsilon)}")

    print("\nNote:")
    print("  - Lower regret and higher policy_efficiency are better.")
    print("  - Higher reward_per_cost means more successes per unit cost.")
    print("  - Higher avg_action_accuracy means the policy chose closer to the optimal action more often.")


if __name__ == "__main__":
    main()
