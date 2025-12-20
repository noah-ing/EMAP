#!/usr/bin/env python3
"""Quick demo with live visualization - smaller config for faster feedback."""
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from emap.genome.representation import MultiAgentGenome
from emap.genome.operators import initialize_population, mutate, crossover
from emap.evolution.selection import tournament_select, elitist_selection, compute_diversity
from emap.evolution.integrated_eval import IntegratedEvaluator
from emap.agents.executor import OpenAIBackend
from emap.benchmarks.humaneval import load_humaneval
from emap.visualization.dashboard import create_dashboard_callback
import time

# Fast config for demo
CONFIG = {
    "population_size": 8,
    "generations": 15,
    "tournament_size": 2,
    "elite_count": 1,
    "mutation_rate": 0.4,
    "crossover_rate": 0.5,
    "sample_fraction": 0.05,  # 5% = ~8 tasks per genome
}

def main():
    print("Loading HumanEval...")
    tasks = load_humaneval()
    print(f"Loaded {len(tasks)} tasks, sampling {int(len(tasks) * CONFIG['sample_fraction'])} per eval")
    
    backend = OpenAIBackend(model="gpt-4o-mini")
    evaluator = IntegratedEvaluator(backend=backend, sample_fraction=CONFIG["sample_fraction"])
    viz_callback = create_dashboard_callback(Path("experiments/viz/live.png"))
    
    rng = np.random.default_rng(42)
    population = initialize_population(CONFIG["population_size"], include_baselines=True, rng=rng)
    
    history = {"generations": [], "best_fitness_per_gen": [], "avg_fitness_per_gen": [],
               "diversity_per_gen": [], "avg_agents_per_gen": [], "avg_edges_per_gen": [],
               "total_tokens": 0, "total_api_calls": 0}
    
    best_genome, best_fitness = None, -1
    
    for gen in range(CONFIG["generations"]):
        gen_start = time.time()
        fitness_scores = []
        
        for genome in population:
            try:
                result = evaluator.evaluate(genome, tasks, 5000, sample=True)
                fitness_scores.append(result.fitness)
                history["total_tokens"] += sum(tr.tokens_used for tr in result.task_results)
            except Exception as e:
                print(f"  Error: {e}")
                fitness_scores.append(0.0)
        
        history["total_api_calls"] += len(population) * max(1, int(len(tasks) * CONFIG["sample_fraction"]))
        
        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best_fitness = fitness_scores[gen_best_idx]
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = population[gen_best_idx]
        
        diversity = compute_diversity(population)
        avg_agents = np.mean([g.num_agents for g in population])
        avg_edges = np.mean([g.num_edges for g in population])
        
        history["generations"].append(gen)
        history["best_fitness_per_gen"].append(gen_best_fitness)
        history["avg_fitness_per_gen"].append(float(np.mean(fitness_scores)))
        history["diversity_per_gen"].append(diversity)
        history["avg_agents_per_gen"].append(float(avg_agents))
        history["avg_edges_per_gen"].append(float(avg_edges))
        
        # Update visualization
        viz_callback({
            "generation": gen, "max_generation": CONFIG["generations"],
            "generations": history["generations"],
            "best_fitness_per_gen": history["best_fitness_per_gen"],
            "avg_fitness_per_gen": history["avg_fitness_per_gen"],
            "diversity_per_gen": history["diversity_per_gen"],
            "avg_agents_per_gen": history["avg_agents_per_gen"],
            "avg_edges_per_gen": history["avg_edges_per_gen"],
            "population_fitness": fitness_scores,
            "best_genome": best_genome.to_dict() if best_genome else None,
            "budget": 5000, "seed": 42,
            "total_tokens": history["total_tokens"],
            "total_api_calls": history["total_api_calls"],
        })
        
        gen_time = time.time() - gen_start
        print(f"Gen {gen:2d} | Best: {gen_best_fitness:.3f} | Avg: {np.mean(fitness_scores):.3f} | "
              f"Div: {diversity:.2f} | Agents: {avg_agents:.1f} | Time: {gen_time:.1f}s | PNG updated!")
        
        # Selection & reproduction
        pop_with_fitness = list(zip(population, fitness_scores))
        elites = list(elitist_selection(pop_with_fitness, CONFIG["elite_count"]))
        next_pop = elites.copy()
        
        while len(next_pop) < CONFIG["population_size"]:
            p1 = tournament_select(pop_with_fitness, CONFIG["tournament_size"], rng)
            p2 = tournament_select(pop_with_fitness, CONFIG["tournament_size"], rng)
            child = crossover(p1, p2, rng)[0] if rng.random() < CONFIG["crossover_rate"] else p1.copy()
            if rng.random() < CONFIG["mutation_rate"]:
                child = mutate(child, rng=rng)
            next_pop.append(child)
        
        population = next_pop[:CONFIG["population_size"]]
    
    print(f"\nDone! Final best: {best_fitness:.3f}")
    print(f"Best genome: {best_genome}")

if __name__ == "__main__":
    main()
