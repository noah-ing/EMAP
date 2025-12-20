"""
Selection Operators for Evolutionary Algorithm

This module implements selection strategies for choosing parent genomes
and maintaining the population across generations.

Selection Strategies:
- Tournament selection: Select k individuals, pick best
- Roulette wheel: Probability proportional to fitness
- Elitism: Always keep top performers

The selection pressure should be tuned to balance:
- Exploitation: Favoring high-fitness individuals
- Exploration: Maintaining population diversity
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from emap.genome.representation import MultiAgentGenome


def tournament_select(
    population: List[Tuple[MultiAgentGenome, float]],
    tournament_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> MultiAgentGenome:
    """
    Tournament selection: randomly sample k individuals, return best.
    
    Args:
        population: List of (genome, fitness) tuples
        tournament_size: Number of individuals in tournament
        rng: Random number generator
    
    Returns:
        Selected genome (copy)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample tournament participants
    indices = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    participants = [population[i] for i in indices]
    
    # Select best
    winner = max(participants, key=lambda x: x[1])
    return winner[0].copy()


def roulette_select(
    population: List[Tuple[MultiAgentGenome, float]],
    rng: Optional[np.random.Generator] = None,
) -> MultiAgentGenome:
    """
    Roulette wheel selection: probability proportional to fitness.
    
    Args:
        population: List of (genome, fitness) tuples
        rng: Random number generator
    
    Returns:
        Selected genome (copy)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    fitnesses = np.array([f for _, f in population])
    
    # Handle all-zero case
    if fitnesses.sum() == 0:
        idx = rng.integers(0, len(population))
        return population[idx][0].copy()
    
    # Normalize to probabilities
    probs = fitnesses / fitnesses.sum()
    
    # Select
    idx = rng.choice(len(population), p=probs)
    return population[idx][0].copy()


def elitist_selection(
    population: List[Tuple[MultiAgentGenome, float]],
    n_elite: int = 2,
) -> List[MultiAgentGenome]:
    """
    Elitist selection: return top n individuals.
    
    Args:
        population: List of (genome, fitness) tuples
        n_elite: Number of elite individuals to preserve
    
    Returns:
        List of top genomes (copies)
    """
    # Sort by fitness descending
    sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
    
    # Return top n
    return [g.copy() for g, _ in sorted_pop[:n_elite]]


def select_parents(
    population: List[Tuple[MultiAgentGenome, float]],
    n_parents: int,
    tournament_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> List[MultiAgentGenome]:
    """
    Select n parents for reproduction using tournament selection.
    
    Args:
        population: List of (genome, fitness) tuples
        n_parents: Number of parents to select
        tournament_size: Tournament size for selection
        rng: Random number generator
    
    Returns:
        List of selected parent genomes
    """
    if rng is None:
        rng = np.random.default_rng()
    
    parents = []
    for _ in range(n_parents):
        parent = tournament_select(population, tournament_size, rng)
        parents.append(parent)
    
    return parents


def compute_diversity(population: List[MultiAgentGenome]) -> float:
    """
    Compute population diversity based on structural signatures.
    
    Higher diversity means more unique architectures.
    Returns value in [0, 1] where 1 = all unique structures.
    """
    if not population:
        return 0.0
    
    signatures = [g.structural_signature() for g in population]
    unique = len(set(signatures))
    
    return unique / len(population)


def diversity_maintenance(
    population: List[Tuple[MultiAgentGenome, float]],
    min_diversity: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[MultiAgentGenome, float]]:
    """
    Ensure minimum diversity by replacing duplicates with random genomes.
    
    Args:
        population: Current population with fitness scores
        min_diversity: Minimum required diversity ratio
        rng: Random number generator
    
    Returns:
        Population with diversity maintained
    """
    if rng is None:
        rng = np.random.default_rng()
    
    genomes = [g for g, _ in population]
    current_diversity = compute_diversity(genomes)
    
    if current_diversity >= min_diversity:
        return population
    
    # Find duplicate signatures
    from emap.genome.representation import create_random_genome
    
    seen_signatures = set()
    new_population = []
    
    for genome, fitness in population:
        sig = genome.structural_signature()
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            new_population.append((genome, fitness))
        else:
            # Replace with random genome
            random_genome = create_random_genome(rng=rng)
            new_population.append((random_genome, 0.0))  # Fitness will be computed
    
    return new_population
