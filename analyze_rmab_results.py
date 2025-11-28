#!/usr/bin/env python3
"""
analyze_rmab_results.py

Reads the Thompson RMAB logs and:
- Plots regret per round & cumulative regret
- Plots success rate per round
- Plots belief convergence per resource (sms / assignment / mentor_call)
Saves all plots into outputs/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

LEARNING_JSON = Path("outputs/rmab_thompson_learning.json")
EVAL_JSON     = Path("outputs/rmab_thompson_eval.json")
OUT_DIR       = Path("outputs")

def load_learning_history():
    if not LEARNING_JSON.exists():
        raise FileNotFoundError(f"{LEARNING_JSON} not found. Run rmab_thompson.py first.")
    with open(LEARNING_JSON, "r") as f:
        history = json.load(f)
    return history

def load_eval_summary():
    if not EVAL_JSON.exists():
        print(f"Warning: {EVAL_JSON} not found. Will only plot per-round metrics.")
        return None
    with open(EVAL_JSON, "r") as f:
        return json.load(f)

def plot_regret(history):
    rounds = [h["round"] for h in history]
    # Be tolerant with key names
    round_regret = []
    cum_regret = []

    running = 0.0
    for h in history:
        reg = h.get("round_regret", h.get("regret", 0.0))
        running = h.get("cumulative_regret", running + reg)
        round_regret.append(reg)
        cum_regret.append(running)

    # Per-round regret
    plt.figure()
    plt.plot(rounds, round_regret, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Regret (this round)")
    plt.title("Per-Round Regret (Thompson RMAB)")
    plt.grid(True)
    out_path = OUT_DIR / "thompson_regret_per_round.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Cumulative regret
    plt.figure()
    plt.plot(rounds, cum_regret, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret over Rounds (Thompson RMAB)")
    plt.grid(True)
    out_path = OUT_DIR / "thompson_cumulative_regret.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_success_rate(history):
    rounds = [h["round"] for h in history]
    success_rates = []

    for h in history:
        if "success_rate" in h:
            success_rates.append(h["success_rate"])
        else:
            succ = h.get("round_successes", 0)
            total = h.get("total_selected", max(1, succ))
            success_rates.append(succ / total)

    plt.figure()
    plt.plot(rounds, success_rates, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Success Rate")
    plt.title("Round-wise Success Rate (Thompson RMAB)")
    plt.grid(True)
    out_path = OUT_DIR / "thompson_success_rate.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_belief_convergence(history):
    rounds = [h["round"] for h in history]
    resources = set()
    for h in history:
        if "agent_beliefs" in h:
            resources.update(h["agent_beliefs"].keys())

    resources = sorted(resources)
    beliefs = {r: [] for r in resources}

    for h in history:
        b = h.get("agent_beliefs", {})
        for r in resources:
            beliefs[r].append(b.get(r, np.nan))

    plt.figure()
    for r in resources:
        plt.plot(rounds, beliefs[r], marker="o", label=r)
    plt.xlabel("Round")
    plt.ylabel("Belief: E[efficacy]")
    plt.title("Belief Convergence per Resource (Thompson RMAB)")
    plt.legend()
    plt.grid(True)
    out_path = OUT_DIR / "thompson_belief_convergence.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    OUT_DIR.mkdir(exist_ok=True)
    history = load_learning_history()
    summary = load_eval_summary()

    if summary is not None:
        print("=== Evaluation Summary (from rmab_thompson_eval.json) ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=========================================================")

    plot_regret(history)
    plot_success_rate(history)
    plot_belief_convergence(history)

if __name__ == "__main__":
    main()
