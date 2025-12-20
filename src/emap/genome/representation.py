"""
Multi-Agent Genome Representation

This module defines the evolvable genome structure for multi-agent programming
architectures. The genome encodes:
- Structural genes: number of agents, topology, roles
- Behavioral genes: prompts, tool access, response limits
- Communication genes: message format, compression
- Meta genes: early stopping, max rounds

The design follows the principle that constraint should shape architecture,
not just behavior. Hard budget limits during evolution create genuine
selective pressure for efficiency.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import copy

import numpy as np


class AgentRole(str, Enum):
    """
    Predefined agent roles that can emerge or be assigned.
    
    We start with common software engineering roles but evolution
    may discover novel combinations or specializations.
    """
    PLANNER = "planner"       # Breaks down tasks, creates plans
    CODER = "coder"           # Writes code implementations
    REVIEWER = "reviewer"     # Reviews code for errors
    DEBUGGER = "debugger"     # Fixes failing tests
    TESTER = "tester"         # Writes test cases
    ARCHITECT = "architect"   # Designs overall structure
    GENERALIST = "generalist" # Can do multiple tasks


class MessageFormat(str, Enum):
    """Communication format between agents."""
    FREEFORM = "freeform"      # Natural language, no structure
    STRUCTURED = "structured"  # JSON-like structured messages
    MINIMAL = "minimal"        # Compressed, essential info only


class AggregationStrategy(str, Enum):
    """How to combine outputs from multiple agents."""
    SEQUENTIAL = "sequential"    # Pipeline: agent1 → agent2 → ... → output
    VOTING = "voting"            # Majority vote among agents
    HIERARCHICAL = "hierarchical"  # Leader agent synthesizes others
    BEST_OF_N = "best_of_n"      # Select best output based on confidence


@dataclass
class AgentGene:
    """
    Specification for a single agent within the multi-agent system.
    
    Each agent has a role, system prompt, and operational parameters
    that can be mutated during evolution.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: AgentRole = AgentRole.GENERALIST
    system_prompt: str = ""
    max_response_tokens: int = 500
    tools: List[str] = field(default_factory=list)
    temperature: float = 0.7
    
    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = self._default_prompt()
    
    def _default_prompt(self) -> str:
        """Generate default prompt based on role."""
        prompts = {
            AgentRole.PLANNER: (
                "You are a planning agent. Break down programming tasks into "
                "clear, actionable steps. Output a numbered plan."
            ),
            AgentRole.CODER: (
                "You are a coding agent. Write clean, correct Python code "
                "that solves the given task. Include necessary imports."
            ),
            AgentRole.REVIEWER: (
                "You are a code review agent. Analyze code for bugs, edge cases, "
                "and improvements. Be specific about issues found."
            ),
            AgentRole.DEBUGGER: (
                "You are a debugging agent. Given failing code and error messages, "
                "identify and fix the bugs. Explain your fixes."
            ),
            AgentRole.TESTER: (
                "You are a testing agent. Write comprehensive test cases "
                "that cover edge cases and validate correctness."
            ),
            AgentRole.ARCHITECT: (
                "You are an architecture agent. Design the overall structure "
                "and approach before implementation begins."
            ),
            AgentRole.GENERALIST: (
                "You are a programming assistant. Solve the given task "
                "by writing correct, efficient Python code."
            ),
        }
        return prompts.get(self.role, prompts[AgentRole.GENERALIST])
    
    def mutate_prompt(self, mutation_instruction: str) -> None:
        """
        Apply LLM-guided mutation to the system prompt.
        
        In practice, this calls an LLM with the mutation instruction
        to rewrite the prompt. Here we just mark it for external handling.
        """
        # Actual mutation happens in operators.py using LLM
        pass
    
    def copy(self) -> AgentGene:
        """Create a deep copy of this agent gene."""
        return AgentGene(
            id=str(uuid.uuid4())[:8],  # New ID for copy
            role=self.role,
            system_prompt=self.system_prompt,
            max_response_tokens=self.max_response_tokens,
            tools=list(self.tools),
            temperature=self.temperature,
        )


@dataclass 
class MultiAgentGenome:
    """
    Complete genome for a multi-agent programming architecture.
    
    This is the unit of evolution: populations contain MultiAgentGenomes,
    selection operates on fitness derived from this genome, and genetic
    operators (mutation, crossover) produce new genomes.
    
    The genome encodes:
    1. STRUCTURE: How many agents, what roles, how connected
    2. BEHAVIOR: What each agent does (prompts, parameters)
    3. COMMUNICATION: How agents exchange information
    4. META: When to stop, how to aggregate outputs
    
    Design Principles:
    - Genomes must be valid (connected graph, at least one agent)
    - Genomes must be serializable (for logging and reproduction)
    - Genomes should support efficient comparison (for diversity)
    """
    
    # Unique identifier
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Structural genes
    agents: List[AgentGene] = field(default_factory=list)
    topology: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> [connected_agent_ids]
    
    # Communication genes  
    message_format: MessageFormat = MessageFormat.STRUCTURED
    max_message_length: int = 200  # tokens per inter-agent message
    
    # Aggregation genes
    aggregation_strategy: AggregationStrategy = AggregationStrategy.SEQUENTIAL
    entry_agent_id: Optional[str] = None  # Which agent receives initial input
    output_agent_id: Optional[str] = None  # Which agent produces final output
    
    # Meta genes
    max_rounds: int = 3  # Maximum communication rounds
    early_exit_confidence: float = 0.9  # Confidence threshold for early stopping
    
    # Lineage tracking
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    
    def __post_init__(self):
        """Ensure genome validity after initialization."""
        if not self.agents:
            # Create a minimal single-agent genome
            agent = AgentGene(role=AgentRole.CODER)
            self.agents = [agent]
            self.topology = {agent.id: []}
            self.entry_agent_id = agent.id
            self.output_agent_id = agent.id
    
    @property
    def num_agents(self) -> int:
        """Number of agents in this architecture."""
        return len(self.agents)
    
    @property
    def num_edges(self) -> int:
        """Number of communication edges."""
        return sum(len(targets) for targets in self.topology.values())
    
    @property
    def agent_ids(self) -> Set[str]:
        """Set of all agent IDs."""
        return {a.id for a in self.agents}
    
    @property
    def role_distribution(self) -> Dict[AgentRole, int]:
        """Count of each role type."""
        dist = {}
        for agent in self.agents:
            dist[agent.role] = dist.get(agent.role, 0) + 1
        return dist
    
    @property 
    def total_prompt_tokens(self) -> int:
        """Approximate total prompt tokens (rough estimate)."""
        # Rough estimate: ~0.75 tokens per character
        return int(sum(len(a.system_prompt) * 0.75 for a in self.agents))
    
    def get_agent(self, agent_id: str) -> Optional[AgentGene]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def is_valid(self) -> bool:
        """Check if genome represents a valid architecture."""
        if not self.agents:
            return False
        
        # All topology references must be valid
        agent_ids = self.agent_ids
        for source, targets in self.topology.items():
            if source not in agent_ids:
                return False
            for target in targets:
                if target not in agent_ids:
                    return False
        
        # Entry and output agents must exist
        if self.entry_agent_id and self.entry_agent_id not in agent_ids:
            return False
        if self.output_agent_id and self.output_agent_id not in agent_ids:
            return False
            
        return True
    
    def copy(self) -> MultiAgentGenome:
        """Create a deep copy of this genome."""
        new_genome = MultiAgentGenome(
            id=str(uuid.uuid4())[:8],
            agents=[a.copy() for a in self.agents],
            topology=copy.deepcopy(self.topology),
            message_format=self.message_format,
            max_message_length=self.max_message_length,
            aggregation_strategy=self.aggregation_strategy,
            entry_agent_id=self.entry_agent_id,
            output_agent_id=self.output_agent_id,
            max_rounds=self.max_rounds,
            early_exit_confidence=self.early_exit_confidence,
            parent_ids=[self.id],
            generation=self.generation + 1,
        )
        
        # Remap agent IDs in new genome
        id_map = {old.id: new.id for old, new in zip(self.agents, new_genome.agents)}
        new_genome.topology = {
            id_map.get(k, k): [id_map.get(v, v) for v in vs]
            for k, vs in new_genome.topology.items()
        }
        if new_genome.entry_agent_id:
            new_genome.entry_agent_id = id_map.get(
                new_genome.entry_agent_id, new_genome.entry_agent_id
            )
        if new_genome.output_agent_id:
            new_genome.output_agent_id = id_map.get(
                new_genome.output_agent_id, new_genome.output_agent_id
            )
        
        return new_genome
    
    def to_dict(self) -> dict:
        """Serialize genome to dictionary."""
        return {
            "id": self.id,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role.value,
                    "system_prompt": a.system_prompt,
                    "max_response_tokens": a.max_response_tokens,
                    "tools": a.tools,
                    "temperature": a.temperature,
                }
                for a in self.agents
            ],
            "topology": self.topology,
            "message_format": self.message_format.value,
            "max_message_length": self.max_message_length,
            "aggregation_strategy": self.aggregation_strategy.value,
            "entry_agent_id": self.entry_agent_id,
            "output_agent_id": self.output_agent_id,
            "max_rounds": self.max_rounds,
            "early_exit_confidence": self.early_exit_confidence,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> MultiAgentGenome:
        """Deserialize genome from dictionary."""
        agents = [
            AgentGene(
                id=a["id"],
                role=AgentRole(a["role"]),
                system_prompt=a["system_prompt"],
                max_response_tokens=a["max_response_tokens"],
                tools=a.get("tools", []),
                temperature=a.get("temperature", 0.7),
            )
            for a in data["agents"]
        ]
        
        return cls(
            id=data["id"],
            agents=agents,
            topology=data["topology"],
            message_format=MessageFormat(data["message_format"]),
            max_message_length=data["max_message_length"],
            aggregation_strategy=AggregationStrategy(data["aggregation_strategy"]),
            entry_agent_id=data.get("entry_agent_id"),
            output_agent_id=data.get("output_agent_id"),
            max_rounds=data["max_rounds"],
            early_exit_confidence=data["early_exit_confidence"],
            parent_ids=data.get("parent_ids", []),
            generation=data.get("generation", 0),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> MultiAgentGenome:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def structural_signature(self) -> Tuple:
        """
        Generate a hashable structural signature for diversity comparison.
        
        Two genomes with the same signature have the same structure
        (agents, roles, topology shape) but may differ in prompts/params.
        """
        roles = tuple(sorted(a.role.value for a in self.agents))
        edge_count = self.num_edges
        return (self.num_agents, roles, edge_count, self.aggregation_strategy.value)
    
    def __repr__(self) -> str:
        roles = [a.role.value for a in self.agents]
        return f"Genome({self.id}, agents={roles}, edges={self.num_edges})"


# Factory functions for common architectures

def create_single_agent(role: AgentRole = AgentRole.CODER) -> MultiAgentGenome:
    """Create a minimal single-agent genome."""
    agent = AgentGene(role=role)
    return MultiAgentGenome(
        agents=[agent],
        topology={agent.id: []},
        entry_agent_id=agent.id,
        output_agent_id=agent.id,
        aggregation_strategy=AggregationStrategy.SEQUENTIAL,
    )


def create_pipeline(roles: List[AgentRole]) -> MultiAgentGenome:
    """Create a sequential pipeline of agents."""
    agents = [AgentGene(role=role) for role in roles]
    
    # Linear chain topology: a1 → a2 → a3 → ...
    topology = {}
    for i, agent in enumerate(agents):
        if i < len(agents) - 1:
            topology[agent.id] = [agents[i + 1].id]
        else:
            topology[agent.id] = []
    
    return MultiAgentGenome(
        agents=agents,
        topology=topology,
        entry_agent_id=agents[0].id,
        output_agent_id=agents[-1].id,
        aggregation_strategy=AggregationStrategy.SEQUENTIAL,
    )


def create_star(center_role: AgentRole, peripheral_roles: List[AgentRole]) -> MultiAgentGenome:
    """Create a star topology with central agent coordinating peripherals."""
    center = AgentGene(role=center_role)
    peripherals = [AgentGene(role=role) for role in peripheral_roles]
    
    all_agents = [center] + peripherals
    
    # Star topology: center connects to all peripherals
    topology = {center.id: [p.id for p in peripherals]}
    for p in peripherals:
        topology[p.id] = [center.id]  # Peripherals can respond to center
    
    return MultiAgentGenome(
        agents=all_agents,
        topology=topology,
        entry_agent_id=center.id,
        output_agent_id=center.id,
        aggregation_strategy=AggregationStrategy.HIERARCHICAL,
    )


def create_random_genome(
    min_agents: int = 1,
    max_agents: int = 5,
    edge_probability: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> MultiAgentGenome:
    """Create a random genome for population initialization."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Random number of agents
    n_agents = rng.integers(min_agents, max_agents + 1)
    
    # Random roles
    roles = list(AgentRole)
    agents = [
        AgentGene(role=roles[rng.integers(0, len(roles))])
        for _ in range(n_agents)
    ]
    
    # Random topology (ensure connectivity)
    topology = {a.id: [] for a in agents}
    for i, source in enumerate(agents):
        for j, target in enumerate(agents):
            if i != j and rng.random() < edge_probability:
                topology[source.id].append(target.id)
    
    # Random parameters
    message_formats = list(MessageFormat)
    aggregation_strategies = list(AggregationStrategy)
    
    return MultiAgentGenome(
        agents=agents,
        topology=topology,
        message_format=message_formats[rng.integers(0, len(message_formats))],
        max_message_length=int(rng.integers(50, 300)),
        aggregation_strategy=aggregation_strategies[rng.integers(0, len(aggregation_strategies))],
        entry_agent_id=agents[0].id,
        output_agent_id=agents[-1].id if rng.random() > 0.5 else agents[0].id,
        max_rounds=int(rng.integers(1, 5)),
        early_exit_confidence=float(rng.uniform(0.7, 0.95)),
    )
