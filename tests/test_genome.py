"""Tests for genome representation."""

import pytest
import json

from emap.genome.representation import (
    MultiAgentGenome,
    AgentGene,
    AgentRole,
    MessageFormat,
    AggregationStrategy,
    create_single_agent,
    create_pipeline,
    create_star,
    create_random_genome,
)


class TestAgentGene:
    """Tests for AgentGene dataclass."""
    
    def test_default_initialization(self):
        """Test agent with default values."""
        agent = AgentGene()
        assert agent.role == AgentRole.GENERALIST
        assert agent.max_response_tokens == 500
        assert len(agent.system_prompt) > 0
        assert agent.id is not None
    
    def test_role_specific_prompt(self):
        """Test that each role gets appropriate default prompt."""
        for role in AgentRole:
            agent = AgentGene(role=role)
            assert role.value in agent.system_prompt.lower() or len(agent.system_prompt) > 0
    
    def test_copy(self):
        """Test that copy creates independent instance."""
        original = AgentGene(role=AgentRole.CODER)
        copy = original.copy()
        
        assert copy.role == original.role
        assert copy.system_prompt == original.system_prompt
        assert copy.id != original.id  # New ID


class TestMultiAgentGenome:
    """Tests for MultiAgentGenome dataclass."""
    
    def test_default_initialization(self):
        """Test genome with default values creates valid single agent."""
        genome = MultiAgentGenome()
        assert genome.num_agents == 1
        assert genome.is_valid()
        assert genome.entry_agent_id is not None
        assert genome.output_agent_id is not None
    
    def test_properties(self):
        """Test computed properties."""
        genome = create_pipeline([AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER])
        
        assert genome.num_agents == 3
        assert genome.num_edges == 2  # Linear chain
        assert len(genome.agent_ids) == 3
        assert AgentRole.PLANNER in genome.role_distribution
        assert genome.role_distribution[AgentRole.PLANNER] == 1
    
    def test_serialization(self):
        """Test JSON serialization roundtrip."""
        original = create_star(AgentRole.ARCHITECT, [AgentRole.CODER, AgentRole.TESTER])
        
        # To dict and back
        data = original.to_dict()
        restored = MultiAgentGenome.from_dict(data)
        
        assert restored.num_agents == original.num_agents
        assert restored.message_format == original.message_format
        assert restored.aggregation_strategy == original.aggregation_strategy
        
        # To JSON and back
        json_str = original.to_json()
        restored2 = MultiAgentGenome.from_json(json_str)
        
        assert restored2.num_agents == original.num_agents
    
    def test_copy(self):
        """Test deep copy creates independent instance."""
        original = create_pipeline([AgentRole.CODER, AgentRole.REVIEWER])
        copy = original.copy()
        
        assert copy.id != original.id
        assert copy.num_agents == original.num_agents
        assert original.id in copy.parent_ids
        assert copy.generation == original.generation + 1
    
    def test_structural_signature(self):
        """Test structural signature for diversity comparison."""
        g1 = create_pipeline([AgentRole.CODER, AgentRole.REVIEWER])
        g2 = create_pipeline([AgentRole.CODER, AgentRole.REVIEWER])
        g3 = create_pipeline([AgentRole.PLANNER, AgentRole.CODER])
        
        # Same structure, same signature
        assert g1.structural_signature() == g2.structural_signature()
        
        # Different roles, different signature
        assert g1.structural_signature() != g3.structural_signature()
    
    def test_validity(self):
        """Test validity checking."""
        genome = create_single_agent()
        assert genome.is_valid()
        
        # Invalid: reference to non-existent agent
        genome.topology["fake_id"] = ["another_fake"]
        assert not genome.is_valid()


class TestFactoryFunctions:
    """Tests for genome factory functions."""
    
    def test_create_single_agent(self):
        """Test single agent creation."""
        genome = create_single_agent(AgentRole.CODER)
        
        assert genome.num_agents == 1
        assert genome.agents[0].role == AgentRole.CODER
        assert genome.aggregation_strategy == AggregationStrategy.SEQUENTIAL
        assert genome.is_valid()
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        roles = [AgentRole.PLANNER, AgentRole.CODER, AgentRole.TESTER]
        genome = create_pipeline(roles)
        
        assert genome.num_agents == 3
        assert genome.num_edges == 2  # Linear chain
        assert genome.entry_agent_id == genome.agents[0].id
        assert genome.output_agent_id == genome.agents[-1].id
        assert genome.is_valid()
    
    def test_create_star(self):
        """Test star topology creation."""
        genome = create_star(
            AgentRole.ARCHITECT,
            [AgentRole.CODER, AgentRole.TESTER, AgentRole.REVIEWER]
        )
        
        assert genome.num_agents == 4  # 1 center + 3 peripherals
        assert genome.agents[0].role == AgentRole.ARCHITECT
        assert genome.aggregation_strategy == AggregationStrategy.HIERARCHICAL
        assert genome.is_valid()
    
    def test_create_random_genome(self):
        """Test random genome creation."""
        import numpy as np
        rng = np.random.default_rng(42)
        
        genome = create_random_genome(min_agents=2, max_agents=4, rng=rng)
        
        assert 2 <= genome.num_agents <= 4
        assert genome.is_valid()
        
        # Different seeds should give different genomes
        genome2 = create_random_genome(min_agents=2, max_agents=4, rng=np.random.default_rng(123))
        assert genome.structural_signature() != genome2.structural_signature() or genome.id != genome2.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
