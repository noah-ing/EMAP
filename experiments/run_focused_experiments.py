#!/usr/bin/env python3
"""
EMAP Focused Experiments

Runs complete experiments across 4 budget regimes for statistical comparison.
Optimized for ~2-4 hour completion with meaningful results.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emap.genome.representation import MultiAgentGenome
from emap.genome.operators import initialize_population, mutate, crossover
from emap.evolution.selection import tournament_select, elitist_selection, compute_diversity
from emap.evolution.integrated_eval import IntegratedEvaluator
from emap.agents.executor import OpenAIBackend
from emap.benchmarks.humaneval import load_humaneval
from emap.visualization.dashboard import create_dashboard_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiments/focused_experiment.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Budget regimes (core research question)
BUDGETS = {
    "tight": 2000,
    "medium": 5000,
    "loose": 10000,
    "unconstrained": 50000,
}

# Focused config for reliable completion
CONFIG = {
    "population_size": 8,
    "generations": 10,
    "tournament_size": 2,
    "elite_count": 1,
    "mutation_rate": 0.4,
    "crossover_rate": 0.5,
    "sample_fraction": 0.10,  # 10% = ~16 tasks per eval
}

SEEDS = [42, 43, 44]


def run_single_experiment(tasks, budget, seed, config, backend, output_dir, viz_path=None):
    """Run a single evolution experiment."""
    rng = np.random.default_rng(seed)

    evaluator = IntegratedEvaluator(backend=backend, sample_fraction=config["sample_fraction"])
    population = initialize_population(size=config["population_size"], include_baselines=True, rng=rng)

    viz_callback = None
    if viz_path:
        viz_callback = create_dashboard_callback(Path(viz_path))

    history = {
        "budget": budget,
        "seed": seed,
        "config": config,
        "best_fitness_per_gen": [],
        "avg_fitness_per_gen": [],
        "diversity_per_gen": [],
        "avg_agents_per_gen": [],
        "avg_edges_per_gen": [],
        "total_tokens": 0,
        "total_api_calls": 0,
    }

    best_genome = None
    best_fitness = -1

    for gen in range(config["generations"]):
        gen_start = time.time()

        fitness_scores = []
        gen_tokens = 0

        for genome in population:
            try:
                result = evaluator.evaluate(genome, tasks, budget, sample=True)
                fitness_scores.append(result.fitness)
                gen_tokens += sum(tr.tokens_used for tr in result.task_results)
            except Exception as e:
                logger.warning(f"Eval error: {e}")
                fitness_scores.append(0.0)

        history["total_tokens"] += gen_tokens
        history["total_api_calls"] += len(population) * max(1, int(len(tasks) * config["sample_fraction"]))

        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_genome = population[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = gen_best_genome

        diversity = compute_diversity(population)
        avg_agents = np.mean([g.num_agents for g in population])
        avg_edges = np.mean([g.num_edges for g in population])

        history["best_fitness_per_gen"].append(gen_best_fitness)
        history["avg_fitness_per_gen"].append(float(np.mean(fitness_scores)))
        history["diversity_per_gen"].append(diversity)
        history["avg_agents_per_gen"].append(float(avg_agents))
        history["avg_edges_per_gen"].append(float(avg_edges))

        gen_time = time.time() - gen_start

        if viz_callback:
            viz_callback({
                "generation": gen,
                "max_generation": config["generations"],
                "generations": list(range(gen + 1)),
                "best_fitness_per_gen": history["best_fitness_per_gen"],
                "avg_fitness_per_gen": history["avg_fitness_per_gen"],
                "diversity_per_gen": history["diversity_per_gen"],
                "avg_agents_per_gen": history["avg_agents_per_gen"],
                "avg_edges_per_gen": history["avg_edges_per_gen"],
                "population_fitness": fitness_scores,
                "best_genome": best_genome.to_dict() if best_genome else None,
                "budget": budget,
                "seed": seed,
                "total_tokens": history["total_tokens"],
                "total_api_calls": history["total_api_calls"],
            })

        logger.info(f"Gen {gen:2d} | Best: {gen_best_fitness:.3f} | Avg: {np.mean(fitness_scores):.3f} | "
                   f"Div: {diversity:.2f} | Agents: {avg_agents:.1f} | Edges: {avg_edges:.1f} | Time: {gen_time:.1f}s")

        # Selection and reproduction
        pop_with_fitness = list(zip(population, fitness_scores))
        elites = list(elitist_selection(pop_with_fitness, config["elite_count"]))
        next_population = elites.copy()

        while len(next_population) < config["population_size"]:
            parent1 = tournament_select(pop_with_fitness, config["tournament_size"], rng)
            parent2 = tournament_select(pop_with_fitness, config["tournament_size"], rng)

            if rng.random() < config["crossover_rate"]:
                children = crossover(parent1, parent2, rng)
                child = children[0]
            else:
                child = parent1.copy()

            if rng.random() < config["mutation_rate"]:
                child = mutate(child, rng=rng)

            next_population.append(child)

        population = next_population[:config["population_size"]]

    # Final evaluation on larger sample
    logger.info("Running final evaluation on 20% sample...")
    final_evaluator = IntegratedEvaluator(backend=backend, sample_fraction=0.2)
    final_result = final_evaluator.evaluate(best_genome, tasks, budget, sample=True)

    history["final_best_fitness"] = final_result.fitness
    history["final_passed"] = sum(1 for tr in final_result.task_results if tr.success)
    history["final_total"] = len(final_result.task_results)
    history["best_genome"] = best_genome.to_dict() if best_genome else None

    result_file = output_dir / f"evolution_budget{budget}_seed{seed}.json"
    with open(result_file, "w") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    output_dir = Path("experiments/results/focused")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("EMAP FOCUSED EXPERIMENTS")
    logger.info(f"Budgets: {list(BUDGETS.keys())}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Config: pop={CONFIG['population_size']}, gen={CONFIG['generations']}, sample={CONFIG['sample_fraction']}")
    logger.info("="*60)

    tasks = load_humaneval()
    logger.info(f"Loaded {len(tasks)} HumanEval tasks")

    backend = OpenAIBackend(model="gpt-4o-mini")

    all_results = {}
    start_time = time.time()

    for regime_name, budget in BUDGETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"REGIME: {regime_name.upper()} (Budget: {budget} tokens)")
        logger.info(f"{'='*60}")

        regime_results = []

        for seed in SEEDS:
            logger.info(f"\n--- {regime_name.upper()} / Seed {seed} ---")

            viz_path = f"experiments/viz/{regime_name}_seed{seed}.png"

            result = run_single_experiment(
                tasks=tasks,
                budget=budget,
                seed=seed,
                config=CONFIG,
                backend=backend,
                output_dir=output_dir,
                viz_path=viz_path,
            )

            regime_results.append(result)
            logger.info(f">>> Completed: fitness={result['final_best_fitness']:.3f}, "
                       f"passed={result['final_passed']}/{result['final_total']}")

        all_results[regime_name] = regime_results

        # Regime summary
        final_fitnesses = [r["final_best_fitness"] for r in regime_results]
        avg_agents = [r["avg_agents_per_gen"][-1] for r in regime_results]
        logger.info(f"\n{regime_name.upper()} SUMMARY:")
        logger.info(f"  Fitness: {np.mean(final_fitnesses):.3f} ± {np.std(final_fitnesses):.3f}")
        logger.info(f"  Agents:  {np.mean(avg_agents):.1f} ± {np.std(avg_agents):.1f}")

    total_time = time.time() - start_time

    # Cross-regime comparison
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-REGIME COMPARISON")
    logger.info(f"{'='*60}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_hours": total_time / 3600,
        "config": CONFIG,
        "results": {}
    }

    for regime, results in all_results.items():
        fitnesses = [r["final_best_fitness"] for r in results]
        agents = [r["avg_agents_per_gen"][-1] for r in results]
        edges = [r["avg_edges_per_gen"][-1] for r in results]
        tokens = [r["total_tokens"] for r in results]

        summary["results"][regime] = {
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "mean_agents": float(np.mean(agents)),
            "std_agents": float(np.std(agents)),
            "mean_edges": float(np.mean(edges)),
            "mean_tokens": float(np.mean(tokens)),
            "budget": BUDGETS[regime],
        }

        logger.info(f"{regime:15s} | Fitness: {np.mean(fitnesses):.3f}±{np.std(fitnesses):.3f} | "
                   f"Agents: {np.mean(agents):.1f} | Edges: {np.mean(edges):.1f}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_time/3600:.2f} hours ({total_time:.0f}s)")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{'='*60}")

    return summary


if __name__ == "__main__":
    main()
