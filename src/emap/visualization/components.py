"""
Polished visualization components for the EMAP dashboard.

Dark theme with neon accents - publication-quality aesthetics.
"""

from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as path_effects
import numpy as np

from .themes import Theme, DARK_THEME, get_role_color, fitness_to_color


def _add_panel_border(ax: plt.Axes, theme: Theme) -> None:
    """Add a subtle border to a panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(theme.border_color)
        spine.set_linewidth(1)


def _style_title(ax: plt.Axes, title: str, theme: Theme) -> None:
    """Add styled title to panel."""
    ax.set_title(
        title,
        color=theme.accent_blue,
        fontsize=11,
        fontweight='bold',
        loc='left',
        pad=8,
        fontfamily='monospace',
    )


def render_topology(
    ax: plt.Axes,
    agents: List[Dict],
    topology: Dict[str, List[str]],
    theme: Theme = DARK_THEME,
    title: str = "BEST GENOME",
) -> None:
    """Render the multi-agent topology as a network graph."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)

    if not agents:
        ax.text(0.5, 0.5, "SINGLE AGENT", ha='center', va='center',
                color=theme.text_secondary, fontsize=14, fontweight='bold',
                fontfamily='monospace', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Calculate positions
    n = len(agents)
    if n == 1:
        positions = {agents[0]['id']: (0.5, 0.5)}
    elif n == 2:
        positions = {
            agents[0]['id']: (0.3, 0.5),
            agents[1]['id']: (0.7, 0.5),
        }
    else:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi/2
        radius = 0.32
        positions = {
            agent['id']: (0.5 + radius * np.cos(angles[i]),
                         0.5 + radius * np.sin(angles[i]))
            for i, agent in enumerate(agents)
        }

    # Draw edges with glow effect
    for source_id, targets in topology.items():
        if source_id not in positions:
            continue
        for target_id in targets:
            if target_id not in positions:
                continue
            x1, y1 = positions[source_id]
            x2, y2 = positions[target_id]

            # Glow layer
            ax.annotate(
                '', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=theme.accent_cyan,
                    alpha=0.3,
                    lw=6,
                    connectionstyle="arc3,rad=0.15",
                ),
            )
            # Main arrow
            ax.annotate(
                '', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=theme.accent_blue,
                    alpha=0.9,
                    lw=2,
                    connectionstyle="arc3,rad=0.15",
                ),
            )

    # Draw nodes with glow
    node_radius = 0.1 if n <= 3 else 0.08
    for agent in agents:
        x, y = positions[agent['id']]
        role = agent.get('role', 'generalist')
        color = get_role_color(role)

        # Glow circle
        glow = Circle((x, y), node_radius * 1.4, facecolor=color,
                      alpha=0.15, edgecolor='none')
        ax.add_patch(glow)

        # Main circle
        circle = Circle((x, y), node_radius, facecolor=color,
                        edgecolor=theme.text_primary, linewidth=2, alpha=0.95)
        ax.add_patch(circle)

        # Role label with outline
        label = role[:4].upper()
        txt = ax.text(x, y, label, ha='center', va='center',
                      color=theme.bg_primary, fontsize=9, fontweight='bold',
                      fontfamily='monospace')
        txt.set_path_effects([
            path_effects.withStroke(linewidth=0, foreground=theme.bg_primary)
        ])

    # Legend (compact)
    unique_roles = list(set(a.get('role', 'generalist') for a in agents))
    if len(unique_roles) > 1:
        legend_text = " | ".join([f"{r[:4].upper()}" for r in unique_roles])
        ax.text(0.5, 0.05, legend_text, ha='center', va='bottom',
                color=theme.text_secondary, fontsize=8, fontfamily='monospace',
                transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])


def render_fitness_curve(
    ax: plt.Axes,
    generations: List[int],
    best_fitness: List[float],
    avg_fitness: List[float],
    theme: Theme = DARK_THEME,
    title: str = "FITNESS",
) -> None:
    """Render fitness progression curves with glow effects."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)

    if not generations or not best_fitness:
        ax.text(0.5, 0.5, "AWAITING DATA...", ha='center', va='center',
                color=theme.text_secondary, fontsize=12, fontfamily='monospace',
                transform=ax.transAxes)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Fill under best curve
    ax.fill_between(generations, 0, best_fitness, color=theme.accent_green,
                    alpha=0.1)

    # Glow for best line
    ax.plot(generations, best_fitness, color=theme.accent_green,
            linewidth=6, alpha=0.2)
    ax.plot(generations, best_fitness, color=theme.accent_green,
            linewidth=3, alpha=0.4)
    # Main best line
    ax.plot(generations, best_fitness, color=theme.accent_green,
            linewidth=2, label='Best', marker='o', markersize=2)

    # Avg line (subtler)
    ax.plot(generations, avg_fitness, color=theme.accent_blue,
            linewidth=1.5, label='Avg', linestyle='--', alpha=0.7)

    # Current fitness highlight
    if best_fitness:
        current = best_fitness[-1]
        ax.axhline(y=current, color=theme.accent_green, linestyle=':',
                   alpha=0.3, linewidth=1)
        ax.text(0.98, current + 0.03, f'{current:.2f}', ha='right', va='bottom',
                color=theme.accent_green, fontsize=10, fontweight='bold',
                fontfamily='monospace', transform=ax.get_yaxis_transform())

    # Styling
    ax.set_xlabel('Generation', color=theme.text_secondary, fontsize=9,
                  fontfamily='monospace')
    ax.set_ylabel('Fitness', color=theme.text_secondary, fontsize=9,
                  fontfamily='monospace')
    ax.tick_params(colors=theme.text_secondary, labelsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, max(generations) if generations else 50)

    # Grid
    ax.grid(True, color=theme.grid_color, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='lower right', facecolor=theme.bg_tertiary,
              edgecolor=theme.border_color, labelcolor=theme.text_secondary,
              fontsize=8, framealpha=0.9)


def render_status_panel(
    ax: plt.Axes,
    stats: Dict,
    theme: Theme = DARK_THEME,
    title: str = "STATUS",
) -> None:
    """Render the status panel with key metrics."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)
    ax.set_xticks([])
    ax.set_yticks([])

    # Large fitness display
    best_fitness = stats.get('best_fitness', 0)
    fitness_color = fitness_to_color(best_fitness, theme)

    ax.text(0.5, 0.78, f"{best_fitness:.1%}", ha='center', va='center',
            color=fitness_color, fontsize=36, fontweight='bold',
            fontfamily='monospace', transform=ax.transAxes)
    ax.text(0.5, 0.62, "BEST FITNESS", ha='center', va='center',
            color=theme.text_secondary, fontsize=9,
            fontfamily='monospace', transform=ax.transAxes)

    # Divider line (use plot instead of axhline for transform support)
    ax.plot([0.1, 0.9], [0.55, 0.55], color=theme.border_color, linewidth=1,
            transform=ax.transAxes)

    # Stats grid
    stats_data = [
        ("BUDGET", f"{stats.get('budget', 0):,}", theme.text_primary),
        ("GEN", f"{stats.get('generation', 0)}/{stats.get('max_gen', 50)}", theme.accent_blue),
        ("AVG FIT", f"{stats.get('avg_fitness', 0):.1%}", theme.text_primary),
        ("DIVERSITY", f"{stats.get('diversity', 0):.2f}", theme.accent_purple),
        ("AGENTS", f"{stats.get('avg_agents', 0):.1f}", theme.text_secondary),
        ("EDGES", f"{stats.get('avg_edges', 0):.1f}", theme.text_secondary),
    ]

    # Render in 2 columns
    for i, (label, value, color) in enumerate(stats_data):
        col = i % 2
        row = i // 2
        x = 0.25 + col * 0.5
        y = 0.42 - row * 0.15

        ax.text(x, y, value, ha='center', va='center',
                color=color, fontsize=12, fontweight='bold',
                fontfamily='monospace', transform=ax.transAxes)
        ax.text(x, y - 0.06, label, ha='center', va='center',
                color=theme.text_secondary, fontsize=7,
                fontfamily='monospace', transform=ax.transAxes)


def render_diversity_timeline(
    ax: plt.Axes,
    generations: List[int],
    diversity: List[float],
    theme: Theme = DARK_THEME,
    title: str = "DIVERSITY",
) -> None:
    """Render diversity over time."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)

    if not generations or not diversity:
        ax.text(0.5, 0.5, "...", ha='center', va='center',
                color=theme.text_secondary, fontsize=12,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Area fill
    ax.fill_between(generations, 0, diversity, color=theme.accent_purple, alpha=0.3)
    ax.plot(generations, diversity, color=theme.accent_purple, linewidth=2)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(generations) if generations else 50)
    ax.tick_params(colors=theme.text_secondary, labelsize=7)
    ax.grid(True, color=theme.grid_color, alpha=0.2)


def render_structure_timeline(
    ax: plt.Axes,
    generations: List[int],
    avg_agents: List[float],
    avg_edges: List[float],
    theme: Theme = DARK_THEME,
    title: str = "STRUCTURE",
) -> None:
    """Render structural metrics over time."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)

    if not generations:
        ax.text(0.5, 0.5, "...", ha='center', va='center',
                color=theme.text_secondary, fontsize=12,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.plot(generations, avg_agents, color=theme.accent_blue, linewidth=2,
            label='Agents', marker='', alpha=0.9)
    ax.plot(generations, avg_edges, color=theme.accent_cyan, linewidth=2,
            label='Edges', linestyle='--', alpha=0.9)

    ax.set_ylim(0, max(max(avg_agents or [1]), max(avg_edges or [1])) * 1.2)
    ax.set_xlim(0, max(generations) if generations else 50)
    ax.tick_params(colors=theme.text_secondary, labelsize=7)
    ax.grid(True, color=theme.grid_color, alpha=0.2)
    ax.legend(loc='upper right', fontsize=7, facecolor=theme.bg_tertiary,
              edgecolor=theme.border_color, labelcolor=theme.text_secondary)


def render_token_bar(
    ax: plt.Axes,
    tokens_used: int,
    api_calls: int,
    theme: Theme = DARK_THEME,
) -> None:
    """Render a token/API call status bar."""
    ax.set_facecolor(theme.bg_primary)
    ax.axis('off')

    text = f"TOKENS: {tokens_used:,}  |  API CALLS: {api_calls:,}"
    ax.text(0.5, 0.5, text, ha='center', va='center',
            color=theme.text_secondary, fontsize=10, fontfamily='monospace',
            fontweight='bold', transform=ax.transAxes)


def render_header(
    ax: plt.Axes,
    generation: int,
    max_gen: int,
    budget: int = 0,
    theme: Theme = DARK_THEME,
) -> None:
    """Render the dashboard header."""
    ax.set_facecolor(theme.bg_primary)
    ax.axis('off')

    # Title with glow effect
    title_text = ax.text(0.02, 0.5, "EMAP",
                         ha='left', va='center',
                         color=theme.accent_blue, fontsize=22, fontweight='bold',
                         fontfamily='monospace', transform=ax.transAxes)
    title_text.set_path_effects([
        path_effects.withStroke(linewidth=4, foreground=theme.accent_blue + "40")
    ])

    ax.text(0.12, 0.5, "Evolution Monitor",
            ha='left', va='center',
            color=theme.text_secondary, fontsize=14,
            fontfamily='monospace', transform=ax.transAxes)

    # Budget badge
    if budget:
        budget_str = f"{budget:,}"
        ax.text(0.65, 0.5, f"BUDGET: {budget_str}",
                ha='center', va='center',
                color=theme.accent_yellow, fontsize=11, fontweight='bold',
                fontfamily='monospace', transform=ax.transAxes)

    # Generation badge with progress
    progress = generation / max_gen if max_gen > 0 else 0
    gen_color = theme.accent_green if progress >= 0.9 else theme.accent_blue

    ax.text(0.92, 0.5, f"GEN {generation}/{max_gen}",
            ha='right', va='center',
            color=gen_color, fontsize=14, fontweight='bold',
            fontfamily='monospace', transform=ax.transAxes)


def render_population_heatmap(
    ax: plt.Axes,
    fitness_scores: List[float],
    theme: Theme = DARK_THEME,
    title: str = "POPULATION",
) -> None:
    """Render population fitness as a bar chart."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)

    if not fitness_scores:
        ax.text(0.5, 0.5, "...", ha='center', va='center',
                color=theme.text_secondary, fontsize=12,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    n = len(fitness_scores)
    x = range(n)
    colors = [fitness_to_color(f, theme) for f in fitness_scores]

    ax.bar(x, fitness_scores, color=colors, alpha=0.8, width=0.8)

    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.axhline(y=1.0, color=theme.accent_green, linestyle=':', alpha=0.3)
    ax.tick_params(colors=theme.text_secondary, labelsize=7)
    ax.set_xticks([])


def render_structural_metrics(
    ax: plt.Axes,
    avg_agents: float,
    avg_edges: float,
    diversity: float,
    theme: Theme = DARK_THEME,
    title: str = "METRICS",
) -> None:
    """Render horizontal progress bars for structural metrics."""
    ax.set_facecolor(theme.bg_secondary)
    _style_title(ax, title, theme)
    _add_panel_border(ax, theme)

    metrics = [
        ("AGENTS", avg_agents, 5.0, theme.accent_blue),
        ("EDGES", avg_edges, 5.0, theme.accent_cyan),
        ("DIVERS", diversity, 1.0, theme.accent_purple),
    ]

    bar_height = 0.2
    y_positions = [0.7, 0.45, 0.2]

    for y, (label, value, max_val, color) in zip(y_positions, metrics):
        # Background track
        bg_rect = FancyBboxPatch(
            (0.15, y - bar_height/2), 0.7, bar_height,
            boxstyle="round,pad=0.01",
            facecolor=theme.bg_tertiary,
            edgecolor=theme.border_color,
            transform=ax.transAxes,
        )
        ax.add_patch(bg_rect)

        # Value bar
        width = min(value / max_val, 1.0) * 0.7
        value_rect = FancyBboxPatch(
            (0.15, y - bar_height/2), width, bar_height,
            boxstyle="round,pad=0.01",
            facecolor=color,
            alpha=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(value_rect)

        # Label
        ax.text(0.12, y, label, ha='right', va='center',
                color=theme.text_secondary, fontsize=8,
                fontfamily='monospace', transform=ax.transAxes)

        # Value
        ax.text(0.88, y, f"{value:.1f}", ha='left', va='center',
                color=theme.text_primary, fontsize=9, fontweight='bold',
                fontfamily='monospace', transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
