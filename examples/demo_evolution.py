#!/usr/bin/env python3
"""
EMAP Demo: Resource-Constrained Evolution of Multi-Agent Architectures

This script demonstrates the core EMAP functionality:
1. Initialize a population of diverse multi-agent architectures
2. Evaluate fitness under strict token budgets
3. Evolve architectures through selection, mutation, and crossover
4. Track how architectures adapt to resource constraints

Run with:
    python examples/demo_evolution.py [--generations 10] [--budget 500]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

# EMAP imports
from emap.genome.representation import (
    MultiAgentGenome,
    AgentRole,
    create_single_agent,
    create_pipeline,
    create_star,
    create_random_genome,
)
from emap.genome.operators import (
    mutate,
    crossover,
    initialize_population,
)
from emap.evolution.selection import (
    tournament_select,
    elitist_selection,
    compute_diversity,
)
from emap.agents.executor import MockBackend
from emap.evolution.fitness import Task
from emap.evolution.integrated_eval import IntegratedEvaluator


def create_demo_tasks() -> list[Task]:
    """Create simple tasks for demo."""
    return [
        Task(
            id="demo/add",
            prompt="def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n",
            entry_point="add",
            test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
            canonical_solution="    return a + b",
        ),
        Task(
            id="demo/double",
            prompt="def double(n: int) -> int:\n    \"\"\"Return n doubled.\"\"\"\n",
            entry_point="double",
            test_code="assert double(5) == 10\nassert double(0) == 0\nassert double(-3) == -6",
            canonical_solution="    return n * 2",
        ),
        Task(
            id="demo/is_even",
            prompt="def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even.\"\"\"\n",
            entry_point="is_even",
            test_code="assert is_even(4) == True\nassert is_even(7) == False\nassert is_even(0) == True",
            canonical_solution="    return n % 2 == 0",
        ),
        Task(
            id="demo/max_of_three",
            prompt="def max_of_three(a: int, b: int, c: int) -> int:\n    \"\"\"Return the maximum of three numbers.\"\"\"\n",
            entry_point="max_of_three",
            test_code="assert max_of_three(1, 2, 3) == 3\nassert max_of_three(5, 5, 5) == 5\nassert max_of_three(-1, -2, -3) == -1",
            canonical_solution="    return max(a, b, c)",
        ),
        Task(
            id="demo/reverse_string",
            prompt="def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed string.\"\"\"\n",
            entry_point="reverse_string",
            test_code="assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
            canonical_solution="    return s[::-1]",
        ),
    ]


class SmartMockBackend(MockBackend):
    """
    A smarter mock backend that simulates realistic LLM behavior.
    
    Key behaviors:
    - Larger architectures use more tokens
    - Sometimes generates correct code, sometimes not
    - Respects token limits
    """
    
    def __init__(self, success_rate: float = 0.3, tokens_per_response: int = 50):
        super().__init__()
        self.success_rate = success_rate
        self.tokens_per_response = tokens_per_response
        self.rng = np.random.default_rng()
        
        # Canonical solutions for demo tasks
        self.solutions = {
            "add": "    return a + b",
            "double": "    return n * 2",
            "is_even": "    return n % 2 == 0",
            "max_of_three": "    return max(a, b, c)",
            "reverse_string": "    return s[::-1]",
        }
    
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Generate mock response with realistic token usage."""
        self.call_count += 1
        
        # Determine token usage (some variance)
        base_tokens = self.tokens_per_response
        variance = int(base_tokens * 0.3)
        tokens_used = min(
            max_tokens,
            base_tokens + self.rng.integers(-variance, variance + 1),
        )
        
        # Try to extract entry point from user message
        entry_point = None
        for func_name in self.solutions.keys():
            if func_name in user_message:
                entry_point = func_name
                break
        
        # Generate response
        if entry_point and self.rng.random() < self.success_rate:
            # Generate correct solution
            response = self.solutions[entry_point]
        else:
            # Generate plausible but possibly wrong code
            response = "    return None  # TODO: implement"
        
        return response, tokens_used


def print_genome_summary(genome: MultiAgentGenome, prefix: str = "") -> None:
    """Print a summary of a genome."""
    roles = [a.role.value for a in genome.agents]
    print(f"{prefix}ID: {genome.id[:8]}, Agents: {len(genome.agents)}, "
          f"Edges: {genome.num_edges}, Roles: {roles}")


def print_generation_summary(
    generation: int,
    population: list[MultiAgentGenome],
    fitness_scores: list[float],
    budget: int,
) -> None:
    """Print summary of a generation."""
    best_fitness = max(fitness_scores)
    avg_fitness = np.mean(fitness_scores)
    diversity = compute_diversity(population)
    
    # Find best genome
    best_idx = np.argmax(fitness_scores)
    best_genome = population[best_idx]
    
    print(f"\n{'='*60}")
    print(f"Generation {generation}")
    print(f"{'='*60}")
    print(f"Budget: {budget} tokens per task")
    print(f"Population size: {len(population)}")
    print(f"Best fitness: {best_fitness:.3f}")
    print(f"Avg fitness: {avg_fitness:.3f}")
    print(f"Diversity: {diversity:.3f}")
    print(f"\nBest architecture:")
    print_genome_summary(best_genome, "  ")
    print()


def run_evolution(
    population_size: int = 20,
    generations: int = 10,
    budget: int = 500,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.5,
    elitism_count: int = 2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run evolution experiment.
    
    Args:
        population_size: Number of genomes in population
        generations: Number of generations to evolve
        budget: Token budget per task
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        elitism_count: Number of best genomes to preserve
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evolution results
    """
    rng = np.random.default_rng(seed)
    
    # Create tasks
    tasks = create_demo_tasks()
    
    # Create evaluator with mock backend
    # Higher success rate for demo purposes - real experiments would use actual LLM
    backend = SmartMockBackend(success_rate=0.7, tokens_per_response=60)
    evaluator = IntegratedEvaluator(
        backend=backend,
        sample_fraction=1.0,  # Use all tasks for demo
    )
    
    # Initialize population
    if verbose:
        print("Initializing population...")
    population = initialize_population(
        size=population_size,
        include_baselines=True,
        rng=rng,
    )
    
    # Evolution history
    history = {
        "generations": [],
        "best_fitness": [],
        "avg_fitness": [],
        "diversity": [],
        "best_genome": [],
    }
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for genome in population:
            result = evaluator.evaluate(genome, tasks, budget, sample=False)
            fitness_scores.append(result.fitness)
        
        # Record history
        best_idx = int(np.argmax(fitness_scores))
        history["generations"].append(gen)
        history["best_fitness"].append(max(fitness_scores))
        history["avg_fitness"].append(float(np.mean(fitness_scores)))
        history["diversity"].append(compute_diversity(population))
        history["best_genome"].append(population[best_idx].to_dict())
        
        if verbose:
            print_generation_summary(gen, population, fitness_scores, budget)
        
        # Create population with fitness for selection
        pop_with_fitness = list(zip(population, fitness_scores))
        
        # Selection
        elite = elitist_selection(pop_with_fitness, elitism_count)
        
        # Create next generation
        next_population = list(elite)
        
        while len(next_population) < population_size:
            # Tournament selection
            parent1 = tournament_select(pop_with_fitness, 3, rng)
            parent2 = tournament_select(pop_with_fitness, 3, rng)
            
            # Crossover
            if rng.random() < crossover_rate:
                children = crossover(parent1, parent2, rng)
                child = children[0]
            else:
                child = parent1.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                child = mutate(child, rng=rng)
            
            next_population.append(child)
        
        population = next_population[:population_size]
    
    # Final evaluation
    if verbose:
        print("\n" + "="*60)
        print("EVOLUTION COMPLETE")
        print("="*60)
        print(f"\nFinal best fitness: {history['best_fitness'][-1]:.3f}")
        print(f"Improvement: {history['best_fitness'][-1] - history['best_fitness'][0]:.3f}")
        
        print("\nTop architectures in final population:")
        final_fitness = []
        for genome in population:
            result = evaluator.evaluate(genome, tasks, budget, sample=False)
            final_fitness.append((genome, result.fitness))
        
        final_fitness.sort(key=lambda x: x[1], reverse=True)
        for i, (genome, fitness) in enumerate(final_fitness[:5]):
            print(f"\n  {i+1}. Fitness: {fitness:.3f}")
            print_genome_summary(genome, "     ")
    
    return {
        "history": history,
        "final_population": [g.to_dict() for g in population],
        "config": {
            "population_size": population_size,
            "generations": generations,
            "budget": budget,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
        },
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EMAP Demo: Resource-Constrained Evolution"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=10,
        help="Number of generations (default: 10)",
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=20,
        help="Population size (default: 20)",
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=500,
        help="Token budget per task (default: 500)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  EMAP: Evolutionary Multi-Agent Programming Systems")
    print("  Resource-Constrained Architecture Evolution Demo")
    print("="*60)
    
    start_time = time.time()
    
    results = run_evolution(
        population_size=args.population,
        generations=args.generations,
        budget=args.budget,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
