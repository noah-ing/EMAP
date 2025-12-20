#!/usr/bin/env python3
"""
EMAP Single Experiment Runner

Runs a single evolution experiment with specified budget and seed.
Designed for robustness - can be called repeatedly for different configurations.

Usage:
    python run_single_experiment.py --budget 2000 --seed 42
    python run_single_experiment.py --budget 5000 --seed 42 --generations 15
"""

from __future__ import annotations

import argparse
import json
import logging
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


def setup_logging(budget: int, seed: int):
    """Setup logging for this specific experiment."""
    log_file = Path(f"experiments/logs/budget{budget}_seed{seed}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger(__name__)


def run_experiment(budget: int, seed: int, generations: int = 12, population_size: int = 10,
                   sample_fraction: float = 0.12, final_sample_fraction: float = 0.25):
    """
    Run a single evolution experiment.

    Args:
        budget: Token budget constraint
        seed: Random seed for reproducibility
        generations: Number of generations to evolve
        population_size: Size of the population
        sample_fraction: Fraction of HumanEval to sample per evaluation
        final_sample_fraction: Fraction for final evaluation (larger for accuracy)
    """
    logger = setup_logging(budget, seed)

    # Configuration
    config = {
        "budget": budget,
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "tournament_size": 2,
        "elite_count": 2,
        "mutation_rate": 0.4,
        "crossover_rate": 0.5,
        "sample_fraction": sample_fraction,
        "final_sample_fraction": final_sample_fraction,
    }

    logger.info("=" * 60)
    logger.info(f"EMAP EXPERIMENT: Budget={budget}, Seed={seed}")
    logger.info(f"Config: pop={population_size}, gen={generations}, sample={sample_fraction}")
    logger.info("=" * 60)

    # Setup
    output_dir = Path("experiments/results/focused")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"evolution_budget{budget}_seed{seed}.json"

    # Check if already completed
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
        if existing.get("completed", False):
            logger.info(f"Experiment already completed: {result_file}")
            return existing

    # Initialize
    rng = np.random.default_rng(seed)
    tasks = load_humaneval()
    logger.info(f"Loaded {len(tasks)} HumanEval tasks")

    backend = OpenAIBackend(model="gpt-4o-mini")
    evaluator = IntegratedEvaluator(backend=backend, sample_fraction=sample_fraction)
    population = initialize_population(size=population_size, include_baselines=True, rng=rng)

    # Visualization
    regime_name = {2000: "tight", 5000: "medium", 10000: "loose", 50000: "unconstrained"}.get(budget, f"budget{budget}")
    viz_path = Path(f"experiments/viz/{regime_name}_seed{seed}.png")
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    viz_callback = create_dashboard_callback(viz_path)

    # History tracking
    history = {
        "budget": budget,
        "seed": seed,
        "config": config,
        "start_time": datetime.now().isoformat(),
        "best_fitness_per_gen": [],
        "avg_fitness_per_gen": [],
        "diversity_per_gen": [],
        "avg_agents_per_gen": [],
        "avg_edges_per_gen": [],
        "best_genome_per_gen": [],
        "total_tokens": 0,
        "total_api_calls": 0,
        "completed": False,
    }

    best_genome = None
    best_fitness = -1
    start_time = time.time()

    try:
        for gen in range(generations):
            gen_start = time.time()

            # Evaluate population
            fitness_scores = []
            gen_tokens = 0

            for i, genome in enumerate(population):
                try:
                    result = evaluator.evaluate(genome, tasks, budget, sample=True)
                    fitness_scores.append(result.fitness)
                    gen_tokens += sum(tr.tokens_used for tr in result.task_results)
                except Exception as e:
                    logger.warning(f"Eval error for genome {i}: {e}")
                    fitness_scores.append(0.0)

                # Rate limit protection: small delay between genome evaluations
                time.sleep(1.0)

            history["total_tokens"] += gen_tokens
            history["total_api_calls"] += len(population) * max(1, int(len(tasks) * sample_fraction))

            # Track best
            gen_best_idx = int(np.argmax(fitness_scores))
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_best_genome = population[gen_best_idx]

            # Update best genome: prefer higher fitness, or more complex genome if tied
            should_update = (
                gen_best_fitness > best_fitness or
                (gen_best_fitness == best_fitness and best_genome is not None and
                 gen_best_genome.num_agents > best_genome.num_agents)
            )
            if should_update or best_genome is None:
                best_fitness = gen_best_fitness
                best_genome = gen_best_genome

            # Compute stats
            diversity = compute_diversity(population)
            avg_agents = float(np.mean([g.num_agents for g in population]))
            avg_edges = float(np.mean([g.num_edges for g in population]))

            # Record
            history["best_fitness_per_gen"].append(float(gen_best_fitness))
            history["avg_fitness_per_gen"].append(float(np.mean(fitness_scores)))
            history["diversity_per_gen"].append(float(diversity))
            history["avg_agents_per_gen"].append(avg_agents)
            history["avg_edges_per_gen"].append(avg_edges)
            history["best_genome_per_gen"].append(gen_best_genome.to_dict())

            gen_time = time.time() - gen_start

            # Update visualization
            viz_callback({
                "generation": gen,
                "max_generation": generations,
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

            logger.info(f"Gen {gen:2d}/{generations} | Best: {gen_best_fitness:.3f} | "
                       f"Avg: {np.mean(fitness_scores):.3f} | Div: {diversity:.2f} | "
                       f"Agents: {avg_agents:.1f} | Edges: {avg_edges:.1f} | Time: {gen_time:.1f}s")

            # Save checkpoint after each generation
            with open(result_file, "w") as f:
                json.dump(history, f, indent=2)

            # Selection and reproduction
            if gen < generations - 1:  # Skip on last generation
                pop_with_fitness = list(zip(population, fitness_scores))
                elites = list(elitist_selection(pop_with_fitness, config["elite_count"]))
                next_population = elites.copy()

                while len(next_population) < population_size:
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

                population = next_population[:population_size]

        # Final evaluation on larger sample for accuracy
        logger.info(f"\nRunning final evaluation on {final_sample_fraction*100:.0f}% sample...")
        final_evaluator = IntegratedEvaluator(backend=backend, sample_fraction=final_sample_fraction)
        final_result = final_evaluator.evaluate(best_genome, tasks, budget, sample=True)

        history["final_best_fitness"] = float(final_result.fitness)
        history["final_passed"] = sum(1 for tr in final_result.task_results if tr.success)
        history["final_total"] = len(final_result.task_results)
        history["best_genome"] = best_genome.to_dict() if best_genome else None
        history["completed"] = True
        history["end_time"] = datetime.now().isoformat()
        history["total_time_seconds"] = time.time() - start_time

        # Save final results
        with open(result_file, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"EXPERIMENT COMPLETE")
        logger.info(f"Final fitness: {history['final_best_fitness']:.3f} ({history['final_passed']}/{history['final_total']})")
        logger.info(f"Best genome: {best_genome.num_agents} agents, {best_genome.num_edges} edges")
        logger.info(f"Total time: {history['total_time_seconds']/60:.1f} minutes")
        logger.info(f"Results saved to: {result_file}")
        logger.info("=" * 60)

        return history

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        history["error"] = str(e)
        history["end_time"] = datetime.now().isoformat()
        with open(result_file, "w") as f:
            json.dump(history, f, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run single EMAP evolution experiment")
    parser.add_argument("--budget", type=int, required=True, help="Token budget (2000, 5000, 10000, 50000)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--generations", type=int, default=12, help="Number of generations")
    parser.add_argument("--population", type=int, default=10, help="Population size")
    parser.add_argument("--sample", type=float, default=0.12, help="Sample fraction per eval")
    parser.add_argument("--final-sample", type=float, default=0.25, help="Final eval sample fraction")

    args = parser.parse_args()

    run_experiment(
        budget=args.budget,
        seed=args.seed,
        generations=args.generations,
        population_size=args.population,
        sample_fraction=args.sample,
        final_sample_fraction=args.final_sample,
    )


if __name__ == "__main__":
    main()
