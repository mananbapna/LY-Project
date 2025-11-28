#!/usr/bin/env python3
"""
rmab_plot_policies.py

Visual comparison of:
  - Thompson Sampling    (outputs/rmab_thompson_learning.json)
  - ε-greedy             (outputs/rmab_egreedy_learning.json)

Plots:
  - cumulative regret vs round
  - success rate vs round
  - action accuracy vs round (if available)

Saves:
  - outputs/rmab_compare_regret.png
  - outputs/rmab_compare_success_rate.png
  - outputs/rmab_compare_action_accuracy.png   (if applicable)

Run after:
  python3 rmab_thompson.py
  python3 rmab_egreedy_learning.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

THOMPSON_HISTORY = Path("outputs/rmab_thompson_learning.json")
EGREEDY_HISTORY = Path("outputs/rmab_egreedy_learning.json")
OUT_DIR = Path("outputs")


def load_history(path, name):
    if not path.exists():
        print(f"Error: {path} not found. Run the corresponding learning script first for {name}.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def extract_series(history):
    """
    From a list of round dicts, extract per-round series:
      - rounds
      - cumulative_regret
      - success_rate
      - action_accuracy (if present)
    """
    rounds = []
    cum_regret = []
    success_rate = []
    action_accuracy = []

    for h in history:
        rounds.append(h.get("round"))
        cum_regret.append(h.get("cumulative_regret", 0.0))
        success_rate.append(h.get("success_rate", 0.0))
        action_accuracy.append(h.get("action_accuracy"))

    return {
        "rounds": rounds,
        "cum_regret": cum_regret,
        "success_rate": success_rate,
        "action_accuracy": action_accuracy,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    th_hist = load_history(THOMPSON_HISTORY, "Thompson")
    eg_hist = load_history(EGREEDY_HISTORY, "ε-greedy")

    if th_hist is None or eg_hist is None:
        return

    th_series = extract_series(th_hist)
    eg_series = extract_series(eg_hist)

    r_th = th_series["rounds"]
    r_eg = eg_series["rounds"]

    # 1) Cumulative Regret vs Round
    plt.figure()
    plt.plot(r_th, th_series["cum_regret"], marker="o", label="Thompson")
    plt.plot(r_eg, eg_series["cum_regret"], marker="o", label="ε-greedy")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret vs Round")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    regret_path = OUT_DIR / "rmab_compare_regret.png"
    plt.savefig(regret_path, dpi=300)
    print(f"Saved: {regret_path}")

    # 2) Success Rate vs Round
    plt.figure()
    plt.plot(r_th, th_series["success_rate"], marker="o", label="Thompson")
    plt.plot(r_eg, eg_series["success_rate"], marker="o", label="ε-greedy")
    plt.xlabel("Round")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Round")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    sr_path = OUT_DIR / "rmab_compare_success_rate.png"
    plt.savefig(sr_path, dpi=300)
    print(f"Saved: {sr_path}")

    # 3) Action Accuracy vs Round (if present)
    has_th_acc = any(a is not None for a in th_series["action_accuracy"])
    has_eg_acc = any(a is not None for a in eg_series["action_accuracy"])

    if has_th_acc or has_eg_acc:
        plt.figure()
        if has_th_acc:
            plt.plot(
                r_th,
                [a if a is not None else 0.0 for a in th_series["action_accuracy"]],
                marker="o",
                label="Thompson",
            )
        if has_eg_acc:
            plt.plot(
                r_eg,
                [a if a is not None else 0.0 for a in eg_series["action_accuracy"]],
                marker="o",
                label="ε-greedy",
            )
        plt.xlabel("Round")
        plt.ylabel("Action Accuracy")
        plt.title("Action Accuracy vs Round")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        acc_path = OUT_DIR / "rmab_compare_action_accuracy.png"
        plt.savefig(acc_path, dpi=300)
        print(f"Saved: {acc_path}")

    # Show all figures interactively
    plt.show()


if __name__ == "__main__":
    main()
