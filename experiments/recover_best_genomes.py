#!/usr/bin/env python3
"""
Recovery script to fix experiments affected by the best_genome tracking bug.

The bug: When multiple genomes achieved 100% fitness, the first one (often a simple
single-agent baseline) was kept instead of more complex evolved architectures.

This script:
1. Finds the most complex genome from best_genome_per_gen data
2. Re-runs final evaluation on the correct genome
3. Updates the result file with corrected data
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emap.genome.representation import MultiAgentGenome
from emap.evolution.integrated_eval import IntegratedEvaluator
from emap.agents.executor import OpenAIBackend
from emap.benchmarks.humaneval import load_humaneval

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_best_genome(result_data: dict) -> tuple[dict, int]:
    """
    Find the best genome from per-generation data.
    Prefers: highest fitness, then most agents, then most recent generation.
    """
    best_gen_genomes = result_data.get("best_genome_per_gen", [])
    best_fitnesses = result_data.get("best_fitness_per_gen", [])

    if not best_gen_genomes:
        return result_data.get("best_genome"), -1

    best_idx = 0
    best_score = (-1, 0, 0)  # (fitness, agents, gen)

    for i, genome_dict in enumerate(best_gen_genomes):
        fitness = best_fitnesses[i] if i < len(best_fitnesses) else 0
        agents = len(genome_dict.get("agents", []))
        score = (fitness, agents, i)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_gen_genomes[best_idx], best_idx


def recover_experiment(result_file: Path, backend, tasks, dry_run: bool = False):
    """Recover a single experiment's best genome and re-evaluate."""
    logger.info(f"Processing: {result_file.name}")

    with open(result_file) as f:
        data = json.load(f)

    if not data.get("completed", False):
        logger.info(f"  Skipping (not completed)")
        return None

    # Find the correct best genome
    best_genome_dict, best_gen = find_best_genome(data)
    current_best = data.get("best_genome", {})

    current_agents = len(current_best.get("agents", []))
    correct_agents = len(best_genome_dict.get("agents", []))

    logger.info(f"  Current best: {current_agents} agents")
    logger.info(f"  Correct best (gen {best_gen}): {correct_agents} agents")

    if current_agents >= correct_agents:
        logger.info(f"  No recovery needed")
        return None

    if dry_run:
        logger.info(f"  [DRY RUN] Would re-evaluate with {correct_agents}-agent genome")
        return {"needs_recovery": True, "file": str(result_file)}

    # Re-evaluate with correct genome
    logger.info(f"  Re-evaluating with correct genome...")
    genome = MultiAgentGenome.from_dict(best_genome_dict)
    budget = data.get("budget", 5000)

    evaluator = IntegratedEvaluator(backend=backend, sample_fraction=0.25)
    result = evaluator.evaluate(genome, tasks, budget, sample=True)

    # Update data
    data["best_genome"] = best_genome_dict
    data["final_best_fitness"] = result.fitness
    data["final_passed"] = sum(1 for tr in result.task_results if tr.success)
    data["final_total"] = len(result.task_results)
    data["recovered"] = True
    data["recovery_gen"] = best_gen

    # Save
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"  Recovered: fitness={result.fitness:.3f} ({data['final_passed']}/{data['final_total']})")

    return {
        "file": str(result_file),
        "old_agents": current_agents,
        "new_agents": correct_agents,
        "fitness": result.fitness,
        "passed": data["final_passed"],
        "total": data["final_total"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't make changes")
    args = parser.parse_args()

    results_dir = Path("experiments/results/focused")
    result_files = sorted(results_dir.glob("evolution_*.json"))

    if not result_files:
        logger.error("No result files found")
        return

    logger.info(f"Found {len(result_files)} result files")

    if not args.dry_run:
        tasks = load_humaneval()
        backend = OpenAIBackend(model="gpt-4o-mini")
    else:
        tasks = None
        backend = None

    recovered = []
    for rf in result_files:
        result = recover_experiment(rf, backend, tasks, dry_run=args.dry_run)
        if result:
            recovered.append(result)

    logger.info(f"\nRecovery complete: {len(recovered)} experiments updated")
    for r in recovered:
        if args.dry_run:
            logger.info(f"  Would recover: {r['file']}")
        else:
            logger.info(f"  {r['file']}: {r['old_agents']} -> {r['new_agents']} agents, fitness={r['fitness']:.3f}")


if __name__ == "__main__":
    main()
