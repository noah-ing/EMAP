#!/usr/bin/env python3
"""
EMAP Full Overnight Experiments

Runs experiments across 4 budget regimes with 3 seeds each.
Designed to complete in ~12-20 hours.
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
        logging.FileHandler('experiments/full_experiment.log')
    ]
)
logger = logging.getLogger(__name__)

# Budget regimes
BUDGETS = {
    "tight": 2000,
    "medium": 5000,
    "loose": 10000,
    "unconstrained": 50000,
}

# Overnight config (reduced for ~12-20 hour runtime)
CONFIG = {
    "population_size": 10,  # Reduced from 20
    "generations": 20,      # Reduced from 50
    "tournament_size": 2,
    "elite_count": 1,
    "mutation_rate": 0.4,
    "crossover_rate": 0.5,
    "sample_fraction": 0.05,  # 5% = ~8 tasks per eval
}

SEEDS = [42, 43, 44]  # 3 seeds for statistical validity


def run_single_experiment(tasks, budget, seed, config, backend, output_dir, viz_path=None):
    """Run a single evolution experiment."""
    rng = np.random.default_rng(seed)
    
    evaluator = IntegratedEvaluator(backend=backend, sample_fraction=config["sample_fraction"])
    population = initialize_population(size=config["population_size"], include_baselines=True, rng=rng)
    
    # Setup visualization if provided
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
        "total_tokens": 0,
        "total_api_calls": 0,
    }
    
    best_genome = None
    best_fitness = -1
    
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
                logger.warning(f"Eval error: {e}")
                fitness_scores.append(0.0)
        
        history["total_tokens"] += gen_tokens
        history["total_api_calls"] += len(population) * max(1, int(len(tasks) * config["sample_fraction"]))
        
        # Track best
        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = population[gen_best_idx]
        
        # Record stats
        diversity = compute_diversity(population)
        avg_agents = np.mean([g.num_agents for g in population])
        avg_edges = np.mean([g.num_edges for g in population])
        
        history["best_fitness_per_gen"].append(gen_best_fitness)
        history["avg_fitness_per_gen"].append(float(np.mean(fitness_scores)))
        history["diversity_per_gen"].append(diversity)
        history["avg_agents_per_gen"].append(float(avg_agents))
        
        gen_time = time.time() - gen_start
        
        # Update visualization
        if viz_callback:
            viz_callback({
                "generation": gen,
                "max_generation": config["generations"],
                "generations": list(range(gen + 1)),
                "best_fitness_per_gen": history["best_fitness_per_gen"],
                "avg_fitness_per_gen": history["avg_fitness_per_gen"],
                "diversity_per_gen": history["diversity_per_gen"],
                "avg_agents_per_gen": history["avg_agents_per_gen"],
                "avg_edges_per_gen": [avg_edges],
                "best_genome": best_genome.to_dict() if best_genome else None,
                "budget": budget,
                "seed": seed,
                "total_tokens": history["total_tokens"],
                "total_api_calls": history["total_api_calls"],
            })
        
        logger.info(f"Gen {gen:2d} | Best: {gen_best_fitness:.3f} | Avg: {np.mean(fitness_scores):.3f} | "
                   f"Div: {diversity:.2f} | Agents: {avg_agents:.1f} | Time: {gen_time:.1f}s")
        
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
    
    history["final_best_fitness"] = best_fitness
    history["best_genome"] = best_genome.to_dict() if best_genome else None
    
    # Save results
    result_file = output_dir / f"evolution_budget{budget}_seed{seed}.json"
    with open(result_file, "w") as f:
        json.dump(history, f, indent=2)
    
    return history


def main():
    output_dir = Path("experiments/results/full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("EMAP FULL OVERNIGHT EXPERIMENTS")
    logger.info(f"Budgets: {list(BUDGETS.keys())}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*60)
    
    tasks = load_humaneval()
    logger.info(f"Loaded {len(tasks)} HumanEval tasks")
    
    backend = OpenAIBackend(model="gpt-4o-mini")
    
    all_results = {}
    start_time = time.time()
    
    for regime_name, budget in BUDGETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"REGIME: {regime_name.upper()} (Budget: {budget})")
        logger.info(f"{'='*60}")
        
        regime_results = []
        
        for seed in SEEDS:
            logger.info(f"\n--- Budget {budget}, Seed {seed} ---")
            
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
            logger.info(f"Completed: Best fitness = {result['final_best_fitness']:.3f}")
        
        all_results[regime_name] = regime_results
        
        # Regime summary
        final_fitnesses = [r["final_best_fitness"] for r in regime_results]
        logger.info(f"\n{regime_name.upper()} SUMMARY:")
        logger.info(f"  Mean: {np.mean(final_fitnesses):.3f} ± {np.std(final_fitnesses):.3f}")
        logger.info(f"  Best: {np.max(final_fitnesses):.3f}")
    
    total_time = time.time() - start_time
    
    # Final summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_hours": total_time / 3600,
        "config": CONFIG,
        "results": {
            regime: {
                "mean_fitness": float(np.mean([r["final_best_fitness"] for r in results])),
                "std_fitness": float(np.std([r["final_best_fitness"] for r in results])),
                "mean_agents": float(np.mean([r["avg_agents_per_gen"][-1] for r in results])),
            }
            for regime, results in all_results.items()
        }
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_time/3600:.1f} hours")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
