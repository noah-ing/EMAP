"""
Color themes and styling for EMAP visualization.

Dark theme with neon accents - publication-quality aesthetics.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    """Color theme configuration."""

    # Backgrounds
    bg_primary: str = "#0d1117"      # Main background (GitHub dark)
    bg_secondary: str = "#161b22"    # Panel backgrounds
    bg_tertiary: str = "#21262d"     # Elevated elements

    # Text
    text_primary: str = "#e6edf3"    # Main text
    text_secondary: str = "#8b949e"  # Muted text
    text_accent: str = "#58a6ff"     # Highlighted text

    # Accents
    accent_blue: str = "#58a6ff"     # Primary accent
    accent_green: str = "#3fb950"    # Success / fitness
    accent_yellow: str = "#d29922"   # Warning
    accent_red: str = "#f85149"      # Error / low values
    accent_purple: str = "#a371f7"   # Special
    accent_cyan: str = "#39d9d9"     # Alternative accent

    # Grid and borders
    grid_color: str = "#30363d"
    border_color: str = "#30363d"

    # Chart colors
    line_best: str = "#3fb950"       # Best fitness line
    line_avg: str = "#58a6ff"        # Average fitness line
    fill_alpha: float = 0.2          # Fill transparency

    # Heatmap gradient (low to high)
    heatmap_low: str = "#161b22"
    heatmap_mid: str = "#1f6feb"
    heatmap_high: str = "#3fb950"


# Role-specific colors for topology visualization
ROLE_COLORS: Dict[str, str] = {
    "planner": "#a371f7",     # Purple
    "coder": "#58a6ff",       # Blue
    "reviewer": "#3fb950",    # Green
    "debugger": "#f85149",    # Red
    "tester": "#d29922",      # Orange
    "architect": "#39d9d9",   # Cyan
    "generalist": "#8b949e",  # Gray
}


# Default theme instance
DARK_THEME = Theme()


def get_role_color(role: str) -> str:
    """Get color for agent role."""
    return ROLE_COLORS.get(role.lower(), ROLE_COLORS["generalist"])


def fitness_to_color(fitness: float, theme: Theme = DARK_THEME) -> str:
    """Convert fitness value (0-1) to color gradient."""
    if fitness < 0.5:
        return theme.accent_red
    elif fitness < 0.75:
        return theme.accent_yellow
    elif fitness < 0.9:
        return theme.accent_blue
    else:
        return theme.accent_green
