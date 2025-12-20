#!/usr/bin/env python3
"""
EMAP Full Experiment Runner

This script runs the complete set of experiments for the EMAP paper:
1. Evolution under 4 budget regimes (Tight, Medium, Loose, Unconstrained)
2. 5 random seeds per regime for statistical validity
3. Cross-benchmark transfer evaluation

Usage:
    # Set your API key first
    export OPENAI_API_KEY="sk-..."

    # Run pilot experiment (1 regime, 1 seed)
    python experiments/run_experiments.py --pilot

    # Run full experiments
    python experiments/run_experiments.py --full

    # Run specific regime
    python experiments/run_experiments.py --budget 2000 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emap.genome.representation import MultiAgentGenome
from emap.genome.operators import initialize_population, mutate, crossover
from emap.evolution.selection import tournament_select, elitist_selection, compute_diversity
from emap.evolution.fitness import Task
from emap.evolution.integrated_eval import IntegratedEvaluator
from emap.agents.executor import OpenAIBackend
from emap.benchmarks.humaneval import load_humaneval, _create_placeholder_tasks
from emap.visualization.dashboard import EvolutionDashboard, create_dashboard_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiments/experiment.log')
    ]
)
logger = logging.getLogger(__name__)


# Budget regimes from the paper
BUDGET_REGIMES = {
    "tight": 2000,
    "medium": 5000,
    "loose": 10000,
    "unconstrained": 50000,  # "Unlimited" but still capped for safety
}

# Default evolution parameters (from paper)
DEFAULT_CONFIG = {
    "population_size": 20,
    "generations": 50,
    "tournament_size": 3,
    "elite_count": 2,
    "mutation_rate": 0.3,
    "crossover_rate": 0.5,
    "sample_fraction": 0.2,  # Use 20% of tasks during evolution
}


def load_tasks() -> List[Task]:
    """Load benchmark tasks."""
    # Try to load HumanEval, fall back to placeholders
    try:
        tasks = load_humaneval()
        if tasks and len(tasks) > 5:  # More than just placeholders
            logger.info(f"Loaded {len(tasks)} HumanEval tasks")
            return tasks
    except Exception as e:
        logger.warning(f"Could not load HumanEval: {e}")

    # Use placeholder tasks
    tasks = _create_placeholder_tasks()
    logger.info(f"Using {len(tasks)} placeholder tasks")
    return tasks


def run_evolution(
    tasks: List[Task],
    budget: int,
    seed: int,
    config: dict,
    backend: OpenAIBackend,
    output_dir: Path,
    viz_callback: callable = None,
) -> dict:
    """
    Run a single evolution experiment.

    Returns dict with evolution history and best genome.
    """
    rng = np.random.default_rng(seed)

    # Create evaluator
    evaluator = IntegratedEvaluator(
        backend=backend,
        sample_fraction=config["sample_fraction"],
    )

    # Initialize population
    logger.info(f"Initializing population of {config['population_size']} genomes...")
    population = initialize_population(
        size=config["population_size"],
        include_baselines=True,
        rng=rng,
    )

    # Evolution history
    history = {
        "budget": budget,
        "seed": seed,
        "config": config,
        "generations": [],
        "best_fitness_per_gen": [],
        "avg_fitness_per_gen": [],
        "diversity_per_gen": [],
        "avg_agents_per_gen": [],
        "avg_edges_per_gen": [],
        "total_api_calls": 0,
        "total_tokens": 0,
    }

    best_genome = None
    best_fitness = -1

    # Evolution loop
    for gen in range(config["generations"]):
        gen_start = time.time()

        # Evaluate population
        fitness_scores = []
        gen_tokens = 0

        for genome in population:
            try:
                result = evaluator.evaluate(genome, tasks, budget, sample=True)
                fitness_scores.append(result.fitness)
                gen_tokens += sum(tr.tokens_used for tr in result.task_results)
            except Exception as e:
                logger.warning(f"Evaluation error for {genome.id[:8]}: {e}")
                fitness_scores.append(0.0)

        history["total_tokens"] += gen_tokens
        history["total_api_calls"] += len(population) * max(1, int(len(tasks) * config["sample_fraction"]))

        # Track best
        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_genome = population[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = gen_best_genome

        # Compute stats
        diversity = compute_diversity(population)
        avg_agents = np.mean([g.num_agents for g in population])
        avg_edges = np.mean([g.num_edges for g in population])

        history["generations"].append(gen)
        history["best_fitness_per_gen"].append(gen_best_fitness)
        history["avg_fitness_per_gen"].append(float(np.mean(fitness_scores)))
        history["diversity_per_gen"].append(diversity)
        history["avg_agents_per_gen"].append(float(avg_agents))
        history["avg_edges_per_gen"].append(float(avg_edges))

        gen_time = time.time() - gen_start

        # Update visualization callback if provided
        if viz_callback:
            viz_callback({
                "generation": gen,
                "max_generation": config["generations"],
                "generations": history["generations"],
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

        # Log progress
        if gen % 5 == 0 or gen == config["generations"] - 1:
            logger.info(
                f"Gen {gen:3d} | Best: {gen_best_fitness:.3f} | "
                f"Avg: {np.mean(fitness_scores):.3f} | "
                f"Div: {diversity:.2f} | "
                f"Agents: {avg_agents:.1f} | "
                f"Time: {gen_time:.1f}s"
            )

        # Selection and reproduction
        pop_with_fitness = list(zip(population, fitness_scores))

        # Elitism
        elites = list(elitist_selection(pop_with_fitness, config["elite_count"]))

        # Create next generation
        next_population = elites.copy()

        while len(next_population) < config["population_size"]:
            # Tournament selection
            parent1 = tournament_select(pop_with_fitness, config["tournament_size"], rng)
            parent2 = tournament_select(pop_with_fitness, config["tournament_size"], rng)

            # Crossover
            if rng.random() < config["crossover_rate"]:
                children = crossover(parent1, parent2, rng)
                child = children[0]
            else:
                child = parent1.copy()

            # Mutation
            if rng.random() < config["mutation_rate"]:
                child = mutate(child, rng=rng)

            next_population.append(child)

        population = next_population[:config["population_size"]]

    # Final evaluation on full task set
    logger.info("Running final evaluation on full task set...")
    final_fitness_scores = []
    for genome in population:
        try:
            result = evaluator.evaluate(genome, tasks, budget, sample=False)
            final_fitness_scores.append(result.fitness)
        except Exception as e:
            logger.warning(f"Final eval error for {genome.id[:8]}: {e}")
            final_fitness_scores.append(0.0)

    # Get actual best after full evaluation
    final_best_idx = int(np.argmax(final_fitness_scores))
    best_genome = population[final_best_idx]
    best_fitness = final_fitness_scores[final_best_idx]

    history["final_best_fitness"] = best_fitness
    history["final_avg_fitness"] = float(np.mean(final_fitness_scores))
    history["best_genome"] = best_genome.to_dict()

    # Save results
    result_file = output_dir / f"evolution_budget{budget}_seed{seed}.json"
    with open(result_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Results saved to {result_file}")

    return history


def run_pilot_experiment(output_dir: Path, viz_callback: callable = None) -> dict:
    """Run a quick pilot experiment to validate the pipeline."""
    logger.info("="*60)
    logger.info("PILOT EXPERIMENT")
    logger.info("="*60)

    # Smaller config for pilot
    pilot_config = {
        **DEFAULT_CONFIG,
        "population_size": 10,
        "generations": 5,
        "sample_fraction": 0.5,  # Use more tasks for validation
    }

    tasks = load_tasks()
    backend = OpenAIBackend(model="gpt-4o-mini")

    result = run_evolution(
        tasks=tasks,
        budget=2000,  # Tight budget
        seed=42,
        config=pilot_config,
        backend=backend,
        output_dir=output_dir,
        viz_callback=viz_callback,
    )

    logger.info(f"\nPilot complete! Best fitness: {result['final_best_fitness']:.3f}")
    logger.info(f"Total API calls: {result['total_api_calls']}")
    logger.info(f"Total tokens: {result['total_tokens']}")

    return result


def run_full_experiments(output_dir: Path, seeds: List[int] = None, viz_callback: callable = None) -> Dict[str, List[dict]]:
    """Run full experiments across all budget regimes."""
    logger.info("="*60)
    logger.info("FULL EXPERIMENTS")
    logger.info("="*60)

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]  # 5 seeds

    tasks = load_tasks()
    backend = OpenAIBackend(model="gpt-4o-mini")

    all_results = {}

    for regime_name, budget in BUDGET_REGIMES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"REGIME: {regime_name.upper()} (Budget: {budget} tokens)")
        logger.info(f"{'='*60}")

        regime_results = []

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")

            result = run_evolution(
                tasks=tasks,
                budget=budget,
                seed=seed,
                config=DEFAULT_CONFIG,
                backend=backend,
                output_dir=output_dir,
                viz_callback=viz_callback,
            )

            regime_results.append(result)

        all_results[regime_name] = regime_results

        # Compute regime summary
        final_fitnesses = [r["final_best_fitness"] for r in regime_results]
        logger.info(f"\n{regime_name.upper()} Summary:")
        logger.info(f"  Mean best fitness: {np.mean(final_fitnesses):.3f} ± {np.std(final_fitnesses):.3f}")
        logger.info(f"  Best across seeds: {np.max(final_fitnesses):.3f}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "regimes": list(BUDGET_REGIMES.keys()),
        "seeds": seeds,
        "results": {
            regime: {
                "mean_fitness": float(np.mean([r["final_best_fitness"] for r in results])),
                "std_fitness": float(np.std([r["final_best_fitness"] for r in results])),
                "mean_agents": float(np.mean([r["avg_agents_per_gen"][-1] for r in results])),
                "total_tokens": sum(r["total_tokens"] for r in results),
            }
            for regime, results in all_results.items()
        }
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="EMAP Experiment Runner")
    parser.add_argument("--pilot", action="store_true", help="Run pilot experiment")
    parser.add_argument("--full", action="store_true", help="Run full experiments")
    parser.add_argument("--budget", type=int, help="Specific budget to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--viz", type=str, help="Enable live visualization, save to specified PNG path")

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set! Please set it first:")
        logger.error("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup visualization callback if requested
    viz_callback = None
    if args.viz:
        viz_path = Path(args.viz)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        viz_callback = create_dashboard_callback(viz_path)
        logger.info(f"Visualization enabled: {viz_path}")

    if args.pilot:
        run_pilot_experiment(output_dir, viz_callback=viz_callback)
    elif args.full:
        run_full_experiments(output_dir, seeds=args.seeds if len(args.seeds) > 1 else None, viz_callback=viz_callback)
    elif args.budget:
        # Run specific budget
        tasks = load_tasks()
        backend = OpenAIBackend(model="gpt-4o-mini")

        for seed in args.seeds:
            run_evolution(
                tasks=tasks,
                budget=args.budget,
                seed=seed,
                config=DEFAULT_CONFIG,
                backend=backend,
                output_dir=output_dir,
                viz_callback=viz_callback,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
