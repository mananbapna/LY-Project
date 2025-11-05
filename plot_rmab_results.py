#!/usr/bin/env python3
"""
plot_rmab_results.py

Reads RMAB simulation results from outputs/rmab_with_resources.json
and plots key metrics: expected churn rate and selected count per round.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_JSON = Path("outputs/rmab_with_resources.json")

def plot_rmab_results():
    if not INPUT_JSON.exists():
        print(f"Error: {INPUT_JSON.resolve()} not found. Run rmab_with_resources.py first.")
        return

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    sim = data["sim_history"]
    rounds = [r["round"] for r in sim]
    churn_rate = [r["expected_churn_rate"] for r in sim]
    selected_count = [r.get("selected_count", 0) for r in sim]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Plot expected churn rate ---
    ax1.plot(rounds, churn_rate, marker="o", linewidth=2, color="tab:blue", label="Expected Churn Rate")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Expected Churn Rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Plot selected count on second axis ---
    ax2 = ax1.twinx()
    ax2.bar(rounds, selected_count, alpha=0.3, color="tab:orange", label="Students Selected")
    ax2.set_ylabel("Selected Students", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # --- Title and legend ---
    title_params = data.get("params", {})
    title = f"RMAB Simulation — Rounds: {title_params.get('rounds', len(rounds))}"
    plt.title(title, fontsize=14, pad=12)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_rmab_results()
