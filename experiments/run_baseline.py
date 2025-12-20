#!/usr/bin/env python3
"""
EMAP Baseline Evaluation

Evaluates single-agent (coder-only) performance on full HumanEval.
This establishes the baseline that multi-agent evolution should beat.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emap.genome.representation import AgentRole, create_single_agent, create_pipeline
from emap.evolution.integrated_eval import IntegratedEvaluator
from emap.agents.executor import OpenAIBackend
from emap.benchmarks.humaneval import load_humaneval


def run_baseline():
    """Run baseline evaluation on full HumanEval."""
    print("Loading HumanEval...")
    tasks = load_humaneval()
    print(f"Loaded {len(tasks)} tasks")

    backend = OpenAIBackend(model="gpt-4o-mini")
    evaluator = IntegratedEvaluator(backend=backend, sample_fraction=1.0)

    # Test different baseline architectures
    baselines = {
        "single_coder": create_single_agent(AgentRole.CODER),
        "coder_reviewer": create_pipeline([AgentRole.CODER, AgentRole.REVIEWER]),
        "planner_coder": create_pipeline([AgentRole.PLANNER, AgentRole.CODER]),
        "planner_coder_reviewer": create_pipeline([
            AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER
        ]),
    }

    results = {}

    for name, genome in baselines.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Agents: {genome.num_agents}, Edges: {genome.num_edges}")
        print(f"{'='*60}")

        # Sample 20% for faster evaluation
        start = time.time()
        result = evaluator.evaluate(genome, tasks, budget=5000, sample=True)
        elapsed = time.time() - start

        results[name] = {
            "fitness": result.fitness,
            "passed": sum(1 for tr in result.task_results if tr.success),
            "total": len(result.task_results),
            "tokens_used": sum(tr.tokens_used for tr in result.task_results),
            "time_seconds": elapsed,
            "agents": genome.num_agents,
            "edges": genome.num_edges,
        }

        print(f"Fitness: {result.fitness:.3f}")
        print(f"Passed: {results[name]['passed']}/{results[name]['total']}")
        print(f"Tokens: {results[name]['tokens_used']}")
        print(f"Time: {elapsed:.1f}s")

    # Save results
    output = {
        "benchmark": "HumanEval",
        "total_tasks": len(tasks),
        "sample_fraction": 0.2,
        "budget": 5000,
        "model": "gpt-4o-mini",
        "baselines": results,
    }

    output_file = Path(__file__).parent / "results" / "baseline_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["fitness"]):
        print(f"{name:25s} | Fitness: {r['fitness']:.3f} | "
              f"Pass: {r['passed']}/{r['total']} | Tokens: {r['tokens_used']}")

    print(f"\nResults saved to {output_file}")
    return results


if __name__ == "__main__":
    run_baseline()
