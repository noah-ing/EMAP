"""Tests for evolution components."""

import pytest
import numpy as np

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
    mutate_add_agent,
    mutate_remove_agent,
    mutate_rewire_topology,
    initialize_population,
)
from emap.evolution.selection import (
    tournament_select,
    elitist_selection,
    compute_diversity,
)


class TestMutationOperators:
    """Tests for mutation operators."""
    
    def test_mutate_add_agent(self):
        """Test adding agent mutation."""
        rng = np.random.default_rng(42)
        genome = create_single_agent()
        
        mutated = mutate_add_agent(genome, rng)
        
        assert mutated.num_agents == 2
        assert mutated.id != genome.id
        assert mutated.is_valid()
    
    def test_mutate_add_agent_max_cap(self):
        """Test that add_agent respects max agents."""
        rng = np.random.default_rng(42)
        genome = create_pipeline([AgentRole.PLANNER, AgentRole.CODER, 
                                   AgentRole.REVIEWER, AgentRole.TESTER, AgentRole.DEBUGGER])
        
        assert genome.num_agents == 5
        
        mutated = mutate_add_agent(genome, rng)
        
        # Should not add beyond 5
        assert mutated.num_agents == 5
    
    def test_mutate_remove_agent(self):
        """Test removing agent mutation."""
        rng = np.random.default_rng(42)
        genome = create_pipeline([AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER])
        
        mutated = mutate_remove_agent(genome, rng)
        
        assert mutated.num_agents == 2
        assert mutated.is_valid()
    
    def test_mutate_remove_agent_min_cap(self):
        """Test that remove_agent respects min agents."""
        rng = np.random.default_rng(42)
        genome = create_single_agent()
        
        mutated = mutate_remove_agent(genome, rng)
        
        # Should not remove last agent
        assert mutated.num_agents == 1
    
    def test_mutate_rewire_topology(self):
        """Test topology rewiring mutation."""
        rng = np.random.default_rng(42)
        genome = create_pipeline([AgentRole.CODER, AgentRole.REVIEWER])
        original_edges = genome.num_edges
        
        mutated = mutate_rewire_topology(genome, rng)
        
        assert mutated.is_valid()
        # Edges may have changed
        assert mutated.num_edges != original_edges or mutated.num_edges == original_edges
    
    def test_mutate_preserves_validity(self):
        """Test that mutation always produces valid genomes."""
        rng = np.random.default_rng(42)
        genome = create_random_genome(rng=rng)
        
        for _ in range(20):
            mutated = mutate(genome, rng)
            assert mutated.is_valid(), f"Mutation produced invalid genome"
            genome = mutated


class TestCrossoverOperators:
    """Tests for crossover operators."""
    
    def test_crossover_produces_two_children(self):
        """Test that crossover produces two offspring."""
        rng = np.random.default_rng(42)
        parent1 = create_pipeline([AgentRole.PLANNER, AgentRole.CODER])
        parent2 = create_star(AgentRole.ARCHITECT, [AgentRole.CODER, AgentRole.TESTER])
        
        child1, child2 = crossover(parent1, parent2, rng)
        
        assert child1.id != child2.id
        assert child1.is_valid()
        assert child2.is_valid()
    
    def test_crossover_tracks_lineage(self):
        """Test that children track parent lineage."""
        rng = np.random.default_rng(42)
        parent1 = create_single_agent()
        parent2 = create_pipeline([AgentRole.CODER, AgentRole.REVIEWER])
        
        child1, child2 = crossover(parent1, parent2, rng)
        
        assert parent1.id in child1.parent_ids
        assert parent2.id in child1.parent_ids


class TestSelection:
    """Tests for selection operators."""
    
    def test_tournament_select(self):
        """Test tournament selection."""
        rng = np.random.default_rng(42)
        
        # Create population with known fitnesses
        population = [
            (create_single_agent(), 0.1),
            (create_single_agent(), 0.9),  # Best
            (create_single_agent(), 0.5),
        ]
        
        # Run many selections - best should be selected most often
        selections = [tournament_select(population, 2, rng) for _ in range(100)]
        
        # All selections should be valid genomes
        assert all(s.is_valid() for s in selections)
    
    def test_elitist_selection(self):
        """Test elitist selection returns top performers."""
        population = [
            (create_single_agent(), 0.3),
            (create_single_agent(), 0.9),
            (create_single_agent(), 0.7),
            (create_single_agent(), 0.1),
        ]
        
        elites = elitist_selection(population, n_elite=2)
        
        assert len(elites) == 2
        # Elites should be copies (new IDs)
        original_ids = {g.id for g, _ in population}
        assert all(e.id not in original_ids for e in elites)
    
    def test_compute_diversity(self):
        """Test diversity computation."""
        # All same structure
        same_genomes = [create_single_agent() for _ in range(5)]
        diversity_same = compute_diversity(same_genomes)
        
        # All different structures
        diff_genomes = [
            create_single_agent(),
            create_pipeline([AgentRole.CODER, AgentRole.REVIEWER]),
            create_star(AgentRole.ARCHITECT, [AgentRole.CODER]),
        ]
        diversity_diff = compute_diversity(diff_genomes)
        
        # Different structures should have higher diversity
        assert diversity_diff > diversity_same


class TestPopulationInitialization:
    """Tests for population initialization."""
    
    def test_initialize_population_size(self):
        """Test population has correct size."""
        rng = np.random.default_rng(42)
        population = initialize_population(size=10, include_baselines=True, rng=rng)
        
        assert len(population) == 10
    
    def test_initialize_population_includes_baselines(self):
        """Test population includes baseline architectures."""
        rng = np.random.default_rng(42)
        population = initialize_population(size=10, include_baselines=True, rng=rng)
        
        # Should have diverse structures
        signatures = {g.structural_signature() for g in population}
        assert len(signatures) > 1  # Not all identical
    
    def test_initialize_population_all_valid(self):
        """Test all initialized genomes are valid."""
        rng = np.random.default_rng(42)
        population = initialize_population(size=20, rng=rng)
        
        assert all(g.is_valid() for g in population)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
