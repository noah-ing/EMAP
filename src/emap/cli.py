"""
Command-line interface for EMAP.

Usage:
    emap-evolve --budget 2000 --generations 50 --seed 42
    emap-evolve --config experiments/configs/tight_budget.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from emap.evolution.loop import evolve, EvolutionConfig
from emap.benchmarks.humaneval import load_humaneval


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EMAP: Evolution under Multi-Agent Pressure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--budget", type=int, default=5000,
        help="Token budget per task (hard constraint)"
    )
    parser.add_argument(
        "--population", type=int, default=20,
        help="Population size"
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of generations"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--sample-fraction", type=float, default=0.2,
        help="Fraction of benchmark to use for fitness (faster evolution)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("EMAP: Evolution under Multi-Agent Pressure")
    logger.info(f"Budget: {args.budget} tokens")
    logger.info(f"Population: {args.population}")
    logger.info(f"Generations: {args.generations}")
    
    # Load benchmark
    logger.info("Loading HumanEval benchmark...")
    tasks = load_humaneval()
    logger.info(f"Loaded {len(tasks)} tasks")
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        budget=args.budget,
        seed=args.seed,
        sample_fraction=args.sample_fraction,
    )
    
    # Run evolution
    result = evolve(tasks, config)
    
    # Save results
    output_file = args.output / f"evolution_budget{args.budget}_seed{args.seed or 'none'}.json"
    result.save(output_file)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"Best Fitness: {result.best_fitness:.3f}")
    print(f"Best Genome: {result.best_genome}")
    print(f"Total Time: {result.total_seconds:.1f}s")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
