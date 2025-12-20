"""
Evolution Loop: Main Evolutionary Algorithm

This module implements the main evolution loop that:
1. Initializes population
2. Evaluates fitness under budget constraints
3. Applies selection, crossover, mutation
4. Tracks lineage and metrics
5. Returns best evolved architectures

The loop implements a steady-state genetic algorithm with:
- Tournament selection for parent choice
- Elitism to preserve top performers
- Hard budget constraints in fitness evaluation

Key Parameters:
- population_size: Number of genomes in population
- generations: Number of evolution iterations
- budget: Token budget for fitness evaluation (hard constraint)
- mutation_rate: Probability of mutation per offspring
- crossover_rate: Probability of crossover vs. cloning
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

from emap.genome.representation import MultiAgentGenome
from emap.genome.operators import (
    mutate,
    crossover,
    initialize_population,
)
from emap.evolution.fitness import (
    FitnessEvaluator,
    Task,
    EvaluationResult,
)
from emap.evolution.selection import (
    tournament_select,
    elitist_selection,
    select_parents,
    compute_diversity,
)


logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    diversity: float
    best_genome_id: str
    population_size: int
    elapsed_seconds: float
    
    # Structural stats
    avg_agents: float
    avg_edges: float
    role_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "min_fitness": self.min_fitness,
            "diversity": self.diversity,
            "best_genome_id": self.best_genome_id,
            "population_size": self.population_size,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_agents": self.avg_agents,
            "avg_edges": self.avg_edges,
            "role_distribution": self.role_distribution,
        }


@dataclass
class EvolutionResult:
    """Complete result of evolution run."""
    budget: int
    seed: int
    generations: int
    best_genome: MultiAgentGenome
    best_fitness: float
    generation_stats: List[GenerationStats]
    final_population: List[Tuple[MultiAgentGenome, float]]
    total_seconds: float
    config: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "budget": self.budget,
            "seed": self.seed,
            "generations": self.generations,
            "best_fitness": self.best_fitness,
            "best_genome": self.best_genome.to_dict(),
            "generation_stats": [s.to_dict() for s in self.generation_stats],
            "total_seconds": self.total_seconds,
            "config": self.config,
        }
    
    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    population_size: int = 20
    generations: int = 50
    tournament_size: int = 3
    elite_count: int = 2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    budget: int = 5000  # Token budget per task
    sample_fraction: float = 0.2  # Fraction of benchmark for fitness
    min_diversity: float = 0.2  # Minimum population diversity
    seed: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "tournament_size": self.tournament_size,
            "elite_count": self.elite_count,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "budget": self.budget,
            "sample_fraction": self.sample_fraction,
            "min_diversity": self.min_diversity,
            "seed": self.seed,
        }


def compute_generation_stats(
    generation: int,
    population: List[Tuple[MultiAgentGenome, float]],
    elapsed: float,
) -> GenerationStats:
    """Compute statistics for current generation."""
    fitnesses = [f for _, f in population]
    genomes = [g for g, _ in population]
    
    # Role distribution
    role_counts: Dict[str, int] = {}
    for genome in genomes:
        for role, count in genome.role_distribution.items():
            role_counts[role.value] = role_counts.get(role.value, 0) + count
    
    best_idx = np.argmax(fitnesses)
    
    return GenerationStats(
        generation=generation,
        best_fitness=max(fitnesses),
        mean_fitness=float(np.mean(fitnesses)),
        std_fitness=float(np.std(fitnesses)),
        min_fitness=min(fitnesses),
        diversity=compute_diversity(genomes),
        best_genome_id=genomes[best_idx].id,
        population_size=len(population),
        elapsed_seconds=elapsed,
        avg_agents=float(np.mean([g.num_agents for g in genomes])),
        avg_edges=float(np.mean([g.num_edges for g in genomes])),
        role_distribution=role_counts,
    )


def evolve(
    tasks: List[Task],
    config: Optional[EvolutionConfig] = None,
    evaluator: Optional[FitnessEvaluator] = None,
    callback: Optional[callable] = None,
) -> EvolutionResult:
    """
    Run evolutionary optimization of multi-agent architectures.
    
    Args:
        tasks: Programming tasks for fitness evaluation
        config: Evolution configuration
        evaluator: Fitness evaluator (created if not provided)
        callback: Optional callback(generation, stats) called each generation
    
    Returns:
        EvolutionResult with best genome and evolution history
    """
    if config is None:
        config = EvolutionConfig()
    
    # Set up random number generator
    rng = np.random.default_rng(config.seed)
    
    # Set up evaluator
    if evaluator is None:
        evaluator = FitnessEvaluator(sample_fraction=config.sample_fraction)
    
    logger.info(f"Starting evolution: {config.generations} generations, budget={config.budget}")
    
    start_time = time.time()
    
    # Initialize population
    population = initialize_population(
        size=config.population_size,
        include_baselines=True,
        rng=rng,
    )
    
    # Evaluate initial population
    scored_population: List[Tuple[MultiAgentGenome, float]] = []
    for genome in population:
        result = evaluator.evaluate(genome, tasks, config.budget)
        scored_population.append((genome, result.fitness))
    
    # Track stats
    all_stats: List[GenerationStats] = []
    gen_start = time.time()
    stats = compute_generation_stats(0, scored_population, time.time() - gen_start)
    all_stats.append(stats)
    
    logger.info(f"Gen 0: best={stats.best_fitness:.3f}, mean={stats.mean_fitness:.3f}, diversity={stats.diversity:.2f}")
    
    if callback:
        callback(0, stats)
    
    # Evolution loop
    for gen in range(1, config.generations + 1):
        gen_start = time.time()
        
        # Elitism: preserve top performers
        elites = elitist_selection(scored_population, config.elite_count)
        
        # Generate offspring
        offspring: List[MultiAgentGenome] = list(elites)
        
        while len(offspring) < config.population_size:
            # Select parents
            parent1 = tournament_select(scored_population, config.tournament_size, rng)
            
            if rng.random() < config.crossover_rate:
                # Crossover
                parent2 = tournament_select(scored_population, config.tournament_size, rng)
                child1, child2 = crossover(parent1, parent2, rng)
                offspring.append(child1)
                if len(offspring) < config.population_size:
                    offspring.append(child2)
            else:
                # Clone
                offspring.append(parent1.copy())
        
        # Mutation
        mutated_offspring = []
        for genome in offspring:
            if rng.random() < config.mutation_rate:
                genome = mutate(genome, rng)
            mutated_offspring.append(genome)
        
        # Evaluate offspring
        scored_population = []
        for genome in mutated_offspring[:config.population_size]:
            result = evaluator.evaluate(genome, tasks, config.budget)
            scored_population.append((genome, result.fitness))
        
        # Compute stats
        stats = compute_generation_stats(gen, scored_population, time.time() - gen_start)
        all_stats.append(stats)
        
        # Log progress
        if gen % 5 == 0 or gen == config.generations:
            logger.info(
                f"Gen {gen}: best={stats.best_fitness:.3f}, "
                f"mean={stats.mean_fitness:.3f}, "
                f"diversity={stats.diversity:.2f}, "
                f"avg_agents={stats.avg_agents:.1f}"
            )
        
        if callback:
            callback(gen, stats)
    
    # Get best genome
    best_genome, best_fitness = max(scored_population, key=lambda x: x[1])
    
    total_time = time.time() - start_time
    logger.info(f"Evolution complete: best_fitness={best_fitness:.3f}, time={total_time:.1f}s")
    
    return EvolutionResult(
        budget=config.budget,
        seed=config.seed or -1,
        generations=config.generations,
        best_genome=best_genome,
        best_fitness=best_fitness,
        generation_stats=all_stats,
        final_population=scored_population,
        total_seconds=total_time,
        config=config.to_dict(),
    )


def evolve_under_budget_regimes(
    tasks: List[Task],
    budgets: List[int],
    seeds: List[int],
    config: Optional[EvolutionConfig] = None,
    output_dir: Optional[Path] = None,
) -> Dict[int, List[EvolutionResult]]:
    """
    Run evolution under multiple budget regimes.
    
    This is the main experiment function that tests H1-H4.
    
    Args:
        tasks: Programming tasks
        budgets: List of token budgets to test
        seeds: Random seeds for reproducibility
        config: Base configuration (budget will be overridden)
        output_dir: Optional directory to save results
    
    Returns:
        Dict mapping budget -> list of EvolutionResult (one per seed)
    """
    if config is None:
        config = EvolutionConfig()
    
    results: Dict[int, List[EvolutionResult]] = {b: [] for b in budgets}
    
    for budget in budgets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Budget regime: {budget} tokens")
        logger.info(f"{'='*60}\n")
        
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            
            run_config = EvolutionConfig(
                **{**config.to_dict(), "budget": budget, "seed": seed}
            )
            
            result = evolve(tasks, run_config)
            results[budget].append(result)
            
            # Save if output directory provided
            if output_dir:
                filename = f"evolution_budget{budget}_seed{seed}.json"
                result.save(output_dir / filename)
    
    return results
