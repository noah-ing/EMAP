"""
Main EMAP Evolution Dashboard - Real-time PNG Visualization.

Generates publication-quality PNG files showing evolution progress.
Updates in real-time as evolution runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .themes import Theme, DARK_THEME
from .components import (
    render_topology,
    render_fitness_curve,
    render_status_panel,
    render_population_heatmap,
    render_structural_metrics,
    render_diversity_timeline,
    render_structure_timeline,
    render_token_bar,
    render_header,
)


@dataclass
class EvolutionState:
    """Snapshot of evolution state for visualization."""

    generation: int = 0
    max_generation: int = 50

    # Metrics history
    generations: List[int] = field(default_factory=list)
    best_fitness_per_gen: List[float] = field(default_factory=list)
    avg_fitness_per_gen: List[float] = field(default_factory=list)
    diversity_per_gen: List[float] = field(default_factory=list)
    avg_agents_per_gen: List[float] = field(default_factory=list)
    avg_edges_per_gen: List[float] = field(default_factory=list)

    # Current population
    population_fitness: List[float] = field(default_factory=list)

    # Best genome info
    best_genome: Optional[Dict] = None

    # Run metadata
    budget: int = 0
    seed: int = 42
    total_tokens: int = 0
    total_api_calls: int = 0

    @classmethod
    def from_json_file(cls, path: Path) -> "EvolutionState":
        """Load state from experiment results JSON."""
        with open(path) as f:
            data = json.load(f)

        state = cls(
            generation=len(data.get("generations", [])) - 1,
            max_generation=data.get("config", {}).get("generations", 50),
            generations=data.get("generations", []),
            best_fitness_per_gen=data.get("best_fitness_per_gen", []),
            avg_fitness_per_gen=data.get("avg_fitness_per_gen", []),
            diversity_per_gen=data.get("diversity_per_gen", []),
            avg_agents_per_gen=data.get("avg_agents_per_gen", []),
            avg_edges_per_gen=data.get("avg_edges_per_gen", []),
            budget=data.get("budget", 0),
            seed=data.get("seed", 42),
            total_tokens=data.get("total_tokens", 0),
            total_api_calls=data.get("total_api_calls", 0),
            best_genome=data.get("best_genome"),
        )

        return state

    @property
    def current_best_fitness(self) -> float:
        """Get current best fitness."""
        return self.best_fitness_per_gen[-1] if self.best_fitness_per_gen else 0.0

    @property
    def current_avg_fitness(self) -> float:
        """Get current average fitness."""
        return self.avg_fitness_per_gen[-1] if self.avg_fitness_per_gen else 0.0

    @property
    def current_diversity(self) -> float:
        """Get current diversity."""
        return self.diversity_per_gen[-1] if self.diversity_per_gen else 0.0

    @property
    def current_avg_agents(self) -> float:
        """Get current average agents."""
        return self.avg_agents_per_gen[-1] if self.avg_agents_per_gen else 0.0

    @property
    def current_avg_edges(self) -> float:
        """Get current average edges."""
        return self.avg_edges_per_gen[-1] if self.avg_edges_per_gen else 0.0


class EvolutionDashboard:
    """
    Real-time PNG dashboard for EMAP evolution visualization.

    Generates a single PNG file that updates every generation,
    showing all key metrics in a beautiful dark-themed layout.
    """

    def __init__(
        self,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
        dpi: int = 100,
        theme: Theme = DARK_THEME,
    ):
        """
        Initialize the dashboard.

        Args:
            output_path: Path to save PNG file
            width: Image width in pixels
            height: Image height in pixels
            dpi: Dots per inch
            theme: Color theme
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.dpi = dpi
        self.theme = theme

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def render(self, state: EvolutionState) -> Path:
        """
        Render the dashboard to PNG.

        Args:
            state: Current evolution state

        Returns:
            Path to saved PNG file
        """
        # Create figure with dark background
        fig = plt.figure(
            figsize=(self.width / self.dpi, self.height / self.dpi),
            dpi=self.dpi,
            facecolor=self.theme.bg_primary,
        )

        # Create grid layout - 6-panel design
        # Layout:
        # [              HEADER                    ]
        # [ TOPOLOGY     | STATUS                  ]
        # [ FITNESS      | DIVERSITY | STRUCTURE   ]
        # [              TOKEN BAR                 ]

        gs = gridspec.GridSpec(
            4, 3,
            figure=fig,
            height_ratios=[0.06, 0.38, 0.48, 0.08],
            width_ratios=[0.40, 0.30, 0.30],
            hspace=0.12,
            wspace=0.08,
            left=0.04,
            right=0.96,
            top=0.97,
            bottom=0.03,
        )

        # Header (spans all columns)
        ax_header = fig.add_subplot(gs[0, :])
        render_header(
            ax_header,
            state.generation,
            state.max_generation,
            state.budget,
            self.theme,
        )

        # Topology panel (top-left, spans 1 col)
        ax_topology = fig.add_subplot(gs[1, 0])
        if state.best_genome:
            render_topology(
                ax_topology,
                state.best_genome.get("agents", []),
                state.best_genome.get("topology", {}),
                self.theme,
            )
        else:
            render_topology(ax_topology, [], {}, self.theme)

        # Status panel (top-right, spans 2 cols)
        ax_status = fig.add_subplot(gs[1, 1:])
        status_stats = {
            "budget": state.budget,
            "seed": state.seed,
            "generation": state.generation,
            "max_gen": state.max_generation,
            "best_fitness": state.current_best_fitness,
            "avg_fitness": state.current_avg_fitness,
            "diversity": state.current_diversity,
            "avg_agents": state.current_avg_agents,
            "avg_edges": state.current_avg_edges,
        }
        render_status_panel(ax_status, status_stats, self.theme)

        # Fitness curve (bottom-left)
        ax_fitness = fig.add_subplot(gs[2, 0])
        render_fitness_curve(
            ax_fitness,
            state.generations,
            state.best_fitness_per_gen,
            state.avg_fitness_per_gen,
            self.theme,
        )

        # Diversity timeline (bottom-middle)
        ax_diversity = fig.add_subplot(gs[2, 1])
        render_diversity_timeline(
            ax_diversity,
            state.generations,
            state.diversity_per_gen,
            self.theme,
        )

        # Structure timeline (bottom-right)
        ax_structure = fig.add_subplot(gs[2, 2])
        render_structure_timeline(
            ax_structure,
            state.generations,
            state.avg_agents_per_gen,
            state.avg_edges_per_gen,
            self.theme,
        )

        # Token bar (bottom, spans all columns)
        ax_tokens = fig.add_subplot(gs[3, :])
        render_token_bar(
            ax_tokens,
            state.total_tokens,
            state.total_api_calls,
            self.theme,
        )

        # Save figure
        fig.savefig(
            self.output_path,
            dpi=self.dpi,
            facecolor=self.theme.bg_primary,
            edgecolor='none',
        )
        plt.close(fig)

        return self.output_path

    def render_from_json(self, json_path: Path) -> Path:
        """
        Convenience method to render directly from JSON results file.

        Args:
            json_path: Path to experiment results JSON

        Returns:
            Path to saved PNG file
        """
        state = EvolutionState.from_json_file(json_path)
        return self.render(state)


def create_dashboard_callback(
    output_path: Path,
    theme: Theme = DARK_THEME,
) -> callable:
    """
    Factory function to create a dashboard callback for the evolution loop.

    Usage in evolution loop:
        callback = create_dashboard_callback(Path("viz/evolution.png"))
        # In loop:
        callback(state_dict)

    Args:
        output_path: Where to save PNG
        theme: Color theme

    Returns:
        Callback function that accepts state dict
    """
    dashboard = EvolutionDashboard(output_path, theme=theme)

    def callback(state_dict: Dict[str, Any]) -> None:
        """Update dashboard with current state."""
        state = EvolutionState(
            generation=state_dict.get("generation", 0),
            max_generation=state_dict.get("max_generation", 50),
            generations=state_dict.get("generations", []),
            best_fitness_per_gen=state_dict.get("best_fitness_per_gen", []),
            avg_fitness_per_gen=state_dict.get("avg_fitness_per_gen", []),
            diversity_per_gen=state_dict.get("diversity_per_gen", []),
            avg_agents_per_gen=state_dict.get("avg_agents_per_gen", []),
            avg_edges_per_gen=state_dict.get("avg_edges_per_gen", []),
            population_fitness=state_dict.get("population_fitness", []),
            best_genome=state_dict.get("best_genome"),
            budget=state_dict.get("budget", 0),
            seed=state_dict.get("seed", 42),
            total_tokens=state_dict.get("total_tokens", 0),
            total_api_calls=state_dict.get("total_api_calls", 0),
        )
        dashboard.render(state)

    return callback


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m emap.visualization.dashboard <json_file> [output.png]")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("evolution_dashboard.png")

    dashboard = EvolutionDashboard(output_path)
    result_path = dashboard.render_from_json(json_path)
    print(f"Dashboard saved to: {result_path}")
