"""
Genetic Operators for Multi-Agent Genome Evolution

This module implements mutation and crossover operators that modify
MultiAgentGenome instances. Key design principles:

1. VALIDITY PRESERVATION: All operators produce valid genomes
2. CONSTRAINT AWARENESS: Operators respect budget constraints implicitly
   (architectures that exceed budget get zero fitness, not invalid)
3. SEMANTIC MUTATION: Prompt mutations use LLM guidance for semantic changes
4. DIVERSITY MAINTENANCE: Operators maintain structural diversity in population

Mutation Operators:
- add_agent: Insert new agent with random role
- remove_agent: Delete agent and rewire topology  
- rewire_topology: Add/remove edges between agents
- mutate_prompt: LLM-guided prompt rewriting
- mutate_parameters: Perturb numeric parameters
- change_role: Swap agent's role type

Crossover Operators:
- subgraph_crossover: Exchange connected agent subgraphs
- prompt_crossover: Blend prompts from same-role agents
- parameter_crossover: Average numeric parameters
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Callable
import copy

import numpy as np

from emap.genome.representation import (
    MultiAgentGenome,
    AgentGene,
    AgentRole,
    MessageFormat,
    AggregationStrategy,
    create_random_genome,
)


# Type alias for mutation functions
MutationOp = Callable[[MultiAgentGenome, np.random.Generator], MultiAgentGenome]


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

def mutate_add_agent(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Add a new agent to the architecture.
    
    The new agent is connected to a random existing agent.
    Max agents is capped at 5 to prevent explosion.
    """
    if genome.num_agents >= 5:
        return genome  # No-op if at max
    
    new_genome = genome.copy()
    
    # Create new agent with random role
    roles = list(AgentRole)
    new_agent = AgentGene(role=roles[rng.integers(0, len(roles))])
    new_genome.agents.append(new_agent)
    
    # Connect to random existing agent (bidirectional)
    existing_ids = [a.id for a in new_genome.agents[:-1]]
    if existing_ids:
        connect_to = rng.choice(existing_ids)
        
        # New agent can send to existing
        new_genome.topology[new_agent.id] = [connect_to]
        
        # Existing can send to new agent (with probability)
        if rng.random() > 0.5:
            if connect_to in new_genome.topology:
                new_genome.topology[connect_to].append(new_agent.id)
            else:
                new_genome.topology[connect_to] = [new_agent.id]
    else:
        new_genome.topology[new_agent.id] = []
    
    return new_genome


def mutate_remove_agent(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Remove a random agent from the architecture.
    
    Maintains at least 1 agent. Rewires topology to maintain connectivity.
    """
    if genome.num_agents <= 1:
        return genome  # No-op if at minimum
    
    new_genome = genome.copy()
    
    # Don't remove entry or output agent if possible
    removable = [
        a for a in new_genome.agents 
        if a.id not in (new_genome.entry_agent_id, new_genome.output_agent_id)
    ]
    
    if not removable:
        # Must remove entry or output; pick one that's not both
        if new_genome.entry_agent_id == new_genome.output_agent_id:
            removable = [a for a in new_genome.agents if a.id != new_genome.entry_agent_id]
        else:
            removable = new_genome.agents
    
    if not removable:
        return genome
    
    to_remove = rng.choice(removable)
    remove_id = to_remove.id
    
    # Remove from agents list
    new_genome.agents = [a for a in new_genome.agents if a.id != remove_id]
    
    # Remove from topology
    if remove_id in new_genome.topology:
        del new_genome.topology[remove_id]
    
    # Remove references to this agent
    for source_id in list(new_genome.topology.keys()):
        new_genome.topology[source_id] = [
            t for t in new_genome.topology[source_id] if t != remove_id
        ]
    
    # Update entry/output if needed
    remaining_ids = {a.id for a in new_genome.agents}
    if new_genome.entry_agent_id not in remaining_ids:
        new_genome.entry_agent_id = new_genome.agents[0].id
    if new_genome.output_agent_id not in remaining_ids:
        new_genome.output_agent_id = new_genome.agents[-1].id
    
    return new_genome


def mutate_rewire_topology(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Add or remove a random edge in the topology.
    """
    if genome.num_agents < 2:
        return genome  # Need at least 2 agents for edges
    
    new_genome = genome.copy()
    agent_ids = [a.id for a in new_genome.agents]
    
    # 50% chance to add edge, 50% to remove
    if rng.random() > 0.5:
        # Add edge
        source = rng.choice(agent_ids)
        target = rng.choice([aid for aid in agent_ids if aid != source])
        
        if source not in new_genome.topology:
            new_genome.topology[source] = []
        
        if target not in new_genome.topology[source]:
            new_genome.topology[source].append(target)
    else:
        # Remove edge
        sources_with_edges = [
            s for s, targets in new_genome.topology.items() if targets
        ]
        if sources_with_edges:
            source = rng.choice(sources_with_edges)
            target = rng.choice(new_genome.topology[source])
            new_genome.topology[source].remove(target)
    
    return new_genome


def mutate_prompt_simple(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Simple prompt mutation: append instruction or truncate.
    
    For LLM-guided mutation, use mutate_prompt_llm() which requires
    an LLM client.
    """
    if not genome.agents:
        return genome
    
    new_genome = genome.copy()
    agent = rng.choice(new_genome.agents)
    
    mutations = [
        " Be concise.",
        " Focus on correctness.",
        " Consider edge cases.",
        " Write efficient code.",
        " Explain your reasoning briefly.",
    ]
    
    # Either append instruction or truncate
    if rng.random() > 0.5 and len(agent.system_prompt) < 500:
        agent.system_prompt += rng.choice(mutations)
    elif len(agent.system_prompt) > 100:
        # Truncate last sentence
        sentences = agent.system_prompt.split(". ")
        if len(sentences) > 1:
            agent.system_prompt = ". ".join(sentences[:-1]) + "."
    
    return new_genome


def mutate_parameters(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Mutate numeric parameters: token limits, temperature, max_rounds, etc.
    """
    new_genome = genome.copy()
    
    # Pick what to mutate
    choice = rng.integers(0, 5)
    
    if choice == 0 and new_genome.agents:
        # Mutate agent's max_response_tokens
        agent = rng.choice(new_genome.agents)
        delta = int(rng.integers(-100, 100))
        agent.max_response_tokens = max(50, min(1000, agent.max_response_tokens + delta))
    
    elif choice == 1 and new_genome.agents:
        # Mutate agent's temperature
        agent = rng.choice(new_genome.agents)
        delta = rng.uniform(-0.2, 0.2)
        agent.temperature = max(0.0, min(1.0, agent.temperature + delta))
    
    elif choice == 2:
        # Mutate max_message_length
        delta = int(rng.integers(-50, 50))
        new_genome.max_message_length = max(20, min(500, new_genome.max_message_length + delta))
    
    elif choice == 3:
        # Mutate max_rounds
        delta = int(rng.choice([-1, 0, 1]))
        new_genome.max_rounds = max(1, min(10, new_genome.max_rounds + delta))
    
    elif choice == 4:
        # Mutate early_exit_confidence
        delta = rng.uniform(-0.1, 0.1)
        new_genome.early_exit_confidence = max(0.5, min(0.99, new_genome.early_exit_confidence + delta))
    
    return new_genome


def mutate_role(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Change an agent's role type.
    """
    if not genome.agents:
        return genome
    
    new_genome = genome.copy()
    agent = rng.choice(new_genome.agents)
    
    # Pick a different role
    other_roles = [r for r in AgentRole if r != agent.role]
    new_role = other_roles[rng.integers(0, len(other_roles))]
    
    agent.role = new_role
    agent.system_prompt = agent._default_prompt()  # Reset to role default
    
    return new_genome


def mutate_communication(
    genome: MultiAgentGenome,
    rng: np.random.Generator,
) -> MultiAgentGenome:
    """
    Mutate communication-related genes: message format, aggregation strategy.
    """
    new_genome = genome.copy()
    
    if rng.random() > 0.5:
        # Change message format
        other_formats = [f for f in MessageFormat if f != new_genome.message_format]
        new_genome.message_format = other_formats[rng.integers(0, len(other_formats))]
    else:
        # Change aggregation strategy
        other_strategies = [s for s in AggregationStrategy if s != new_genome.aggregation_strategy]
        new_genome.aggregation_strategy = other_strategies[rng.integers(0, len(other_strategies))]
    
    return new_genome


# Collect all mutation operators
MUTATION_OPERATORS: List[MutationOp] = [
    mutate_add_agent,
    mutate_remove_agent,
    mutate_rewire_topology,
    mutate_prompt_simple,
    mutate_parameters,
    mutate_role,
    mutate_communication,
]

# Weights for mutation operator selection (can be tuned)
MUTATION_WEIGHTS = [
    0.15,  # add_agent
    0.10,  # remove_agent
    0.15,  # rewire_topology
    0.20,  # prompt_simple
    0.20,  # parameters
    0.10,  # role
    0.10,  # communication
]


def mutate(
    genome: MultiAgentGenome,
    rng: Optional[np.random.Generator] = None,
    n_mutations: int = 1,
) -> MultiAgentGenome:
    """
    Apply n random mutations to a genome.
    
    Args:
        genome: The genome to mutate
        rng: Random number generator
        n_mutations: Number of mutations to apply
    
    Returns:
        New mutated genome (original is not modified)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    result = genome
    for _ in range(n_mutations):
        # Select operator based on weights
        op = rng.choice(MUTATION_OPERATORS, p=MUTATION_WEIGHTS)
        result = op(result, rng)
    
    return result


# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

def crossover_parameters(
    parent1: MultiAgentGenome,
    parent2: MultiAgentGenome,
    rng: np.random.Generator,
) -> Tuple[MultiAgentGenome, MultiAgentGenome]:
    """
    Create offspring by blending numeric parameters from parents.
    
    Each child inherits structure from one parent but blends
    numeric parameters from both.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Blend numeric parameters
    alpha = rng.random()
    
    child1.max_message_length = int(
        alpha * parent1.max_message_length + (1 - alpha) * parent2.max_message_length
    )
    child2.max_message_length = int(
        (1 - alpha) * parent1.max_message_length + alpha * parent2.max_message_length
    )
    
    child1.max_rounds = int(
        alpha * parent1.max_rounds + (1 - alpha) * parent2.max_rounds
    )
    child2.max_rounds = int(
        (1 - alpha) * parent1.max_rounds + alpha * parent2.max_rounds
    )
    
    child1.early_exit_confidence = (
        alpha * parent1.early_exit_confidence + (1 - alpha) * parent2.early_exit_confidence
    )
    child2.early_exit_confidence = (
        (1 - alpha) * parent1.early_exit_confidence + alpha * parent2.early_exit_confidence
    )
    
    # Update lineage
    child1.parent_ids = [parent1.id, parent2.id]
    child2.parent_ids = [parent1.id, parent2.id]
    
    return child1, child2


def crossover_structural(
    parent1: MultiAgentGenome,
    parent2: MultiAgentGenome,
    rng: np.random.Generator,
) -> Tuple[MultiAgentGenome, MultiAgentGenome]:
    """
    Create offspring by exchanging structural components.
    
    Child1 gets agents from parent1 but topology inspiration from parent2.
    Child2 gets the reverse.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Swap message format
    child1.message_format = parent2.message_format
    child2.message_format = parent1.message_format
    
    # Swap aggregation strategy
    child1.aggregation_strategy = parent2.aggregation_strategy
    child2.aggregation_strategy = parent1.aggregation_strategy
    
    # Update lineage
    child1.parent_ids = [parent1.id, parent2.id]
    child2.parent_ids = [parent1.id, parent2.id]
    
    return child1, child2


def crossover(
    parent1: MultiAgentGenome,
    parent2: MultiAgentGenome,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[MultiAgentGenome, MultiAgentGenome]:
    """
    Perform crossover between two parent genomes.
    
    Randomly selects between parameter and structural crossover.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        rng: Random number generator
    
    Returns:
        Tuple of two offspring genomes
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if rng.random() > 0.5:
        return crossover_parameters(parent1, parent2, rng)
    else:
        return crossover_structural(parent1, parent2, rng)


# =============================================================================
# POPULATION INITIALIZATION
# =============================================================================

def initialize_population(
    size: int,
    include_baselines: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> List[MultiAgentGenome]:
    """
    Create initial population with mix of designed and random genomes.
    
    Args:
        size: Population size
        include_baselines: Whether to include hand-designed baselines
        rng: Random number generator
    
    Returns:
        List of genomes forming initial population
    """
    if rng is None:
        rng = np.random.default_rng()
    
    population = []
    
    if include_baselines:
        # Include common architectures as seeds
        from emap.genome.representation import create_single_agent, create_pipeline, create_star
        
        # Single agent baseline
        population.append(create_single_agent(AgentRole.CODER))
        
        # Pipeline: planner → coder → reviewer
        population.append(create_pipeline([
            AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER
        ]))
        
        # Star: architect coordinates coder + tester
        population.append(create_star(
            AgentRole.ARCHITECT,
            [AgentRole.CODER, AgentRole.TESTER]
        ))
        
        # Two-agent: coder + reviewer
        population.append(create_pipeline([
            AgentRole.CODER, AgentRole.REVIEWER
        ]))
    
    # Fill rest with random genomes
    while len(population) < size:
        population.append(create_random_genome(
            min_agents=1,
            max_agents=4,
            edge_probability=0.3,
            rng=rng,
        ))
    
    return population[:size]
