"""
EMAP Visualization Module

Real-time PNG dashboard for evolution monitoring.
"""

from .dashboard import (
    EvolutionDashboard,
    EvolutionState,
    create_dashboard_callback,
)
from .themes import (
    Theme,
    DARK_THEME,
    ROLE_COLORS,
    get_role_color,
    fitness_to_color,
)

__all__ = [
    "EvolutionDashboard",
    "EvolutionState",
    "create_dashboard_callback",
    "Theme",
    "DARK_THEME",
    "ROLE_COLORS",
    "get_role_color",
    "fitness_to_color",
]
