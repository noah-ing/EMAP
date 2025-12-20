#!/usr/bin/env python3
"""
EMAP Complete Experiment Suite

Runs all experiments across 4 budget regimes with 3 seeds each.
Robust design: each experiment saves immediately, can resume from interruption.

Budget Regimes:
- TIGHT (2000 tokens): Tests minimal resource allocation
- MEDIUM (5000 tokens): Moderate constraint
- LOOSE (10000 tokens): Relaxed constraint
- UNCONSTRAINED (50000 tokens): Effectively unlimited

For PhD-level rigor:
- 12 generations per experiment (enough for convergence)
- 10 population size (sufficient genetic diversity)
- 12% sample per generation (20 tasks), 25% final eval (41 tasks)
- 3 seeds for statistical validity (42, 43, 44)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Experiment matrix
BUDGETS = [2000, 5000, 10000, 50000]
SEEDS = [42, 43, 44]

# Experiment parameters (PhD rigor)
GENERATIONS = 12
POPULATION = 10
SAMPLE_FRACTION = 0.12  # ~20 tasks per eval
FINAL_SAMPLE = 0.25     # ~41 tasks for final eval


def check_completed(budget: int, seed: int) -> bool:
    """Check if experiment already completed."""
    result_file = Path(f"experiments/results/focused/evolution_budget{budget}_seed{seed}.json")
    if not result_file.exists():
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        return data.get("completed", False)
    except:
        return False


def run_single(budget: int, seed: int) -> dict:
    """Run a single experiment using subprocess for isolation."""
    cmd = [
        sys.executable, "experiments/run_single_experiment.py",
        "--budget", str(budget),
        "--seed", str(seed),
        "--generations", str(GENERATIONS),
        "--population", str(POPULATION),
        "--sample", str(SAMPLE_FRACTION),
        "--final-sample", str(FINAL_SAMPLE),
    ]

    print(f"\n{'='*60}")
    print(f"RUNNING: Budget={budget}, Seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Experiment failed with return code {result.returncode}")
        return {"success": False, "time": elapsed}

    return {"success": True, "time": elapsed}


def load_results() -> dict:
    """Load all completed results."""
    results = {}
    for budget in BUDGETS:
        results[budget] = []
        for seed in SEEDS:
            result_file = Path(f"experiments/results/focused/evolution_budget{budget}_seed{seed}.json")
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("completed", False):
                    results[budget].append(data)
    return results


def print_summary(results: dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    regime_names = {2000: "TIGHT", 5000: "MEDIUM", 10000: "LOOSE", 50000: "UNCONSTRAINED"}

    print(f"\n{'Regime':<15} {'Budget':>8} {'Fitness':>12} {'Agents':>10} {'Edges':>10} {'N':>5}")
    print("-" * 70)

    summary_data = {}
    for budget in BUDGETS:
        regime = regime_names.get(budget, str(budget))
        data = results.get(budget, [])

        if data:
            fitnesses = [d["final_best_fitness"] for d in data]
            agents = [d["avg_agents_per_gen"][-1] for d in data]
            edges = [d["avg_edges_per_gen"][-1] for d in data]

            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            mean_agents = np.mean(agents)
            mean_edges = np.mean(edges)

            print(f"{regime:<15} {budget:>8} {mean_fit:.3f}±{std_fit:.3f} {mean_agents:>10.1f} {mean_edges:>10.1f} {len(data):>5}")

            summary_data[regime] = {
                "budget": budget,
                "mean_fitness": mean_fit,
                "std_fitness": std_fit,
                "mean_agents": mean_agents,
                "mean_edges": mean_edges,
                "n": len(data),
            }
        else:
            print(f"{regime:<15} {budget:>8} {'PENDING':>12} {'-':>10} {'-':>10} {0:>5}")

    print("-" * 70)

    # Save summary
    summary_file = Path("experiments/results/focused/summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "generations": GENERATIONS,
                "population": POPULATION,
                "sample_fraction": SAMPLE_FRACTION,
                "final_sample": FINAL_SAMPLE,
                "seeds": SEEDS,
            },
            "results": summary_data,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


def main():
    print("=" * 70)
    print("EMAP COMPLETE EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Budgets: {BUDGETS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {len(BUDGETS) * len(SEEDS)}")
    print(f"Config: {GENERATIONS} gen, {POPULATION} pop, {SAMPLE_FRACTION*100:.0f}% sample")
    print("=" * 70)

    # Check what's already done
    completed = []
    pending = []
    for budget in BUDGETS:
        for seed in SEEDS:
            if check_completed(budget, seed):
                completed.append((budget, seed))
            else:
                pending.append((budget, seed))

    print(f"\nCompleted: {len(completed)}/{len(BUDGETS)*len(SEEDS)}")
    print(f"Pending: {len(pending)}")

    if completed:
        print("\nAlready completed:")
        for budget, seed in completed:
            print(f"  - Budget {budget}, Seed {seed}")

    if pending:
        print("\nTo run:")
        for budget, seed in pending:
            print(f"  - Budget {budget}, Seed {seed}")

    # Run pending experiments
    total_start = time.time()
    for i, (budget, seed) in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] Running experiment...")
        run_single(budget, seed)

    total_time = time.time() - total_start

    # Print final summary
    results = load_results()
    print_summary(results)

    print(f"\nTotal runtime: {total_time/3600:.2f} hours")


if __name__ == "__main__":
    main()
