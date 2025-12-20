"""
EMAP: Evolution under Multi-Agent Pressure

Resource-Constrained Evolution of Multi-Agent Programming Architectures
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

from emap.genome.representation import MultiAgentGenome, AgentRole
from emap.evolution.loop import evolve

__all__ = [
    "MultiAgentGenome",
    "AgentRole", 
    "evolve",
]
