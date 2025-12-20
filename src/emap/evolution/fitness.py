"""
Fitness Evaluation with Token Budgets

This module implements the core fitness function that evaluates multi-agent
architectures under hard resource constraints. Key design:

1. HARD CONSTRAINT: Architectures exceeding budget get ZERO fitness
   - This creates genuine evolutionary pressure for efficiency
   - Different from soft Pareto approaches that trade off cost

2. TOKEN TRACKING: Every API call is tracked for accurate budget enforcement
   - Uses tiktoken for prompt/completion token counting
   - Includes inter-agent communication in budget

3. TASK SAMPLING: Fitness computed on random subset for efficiency
   - Full evaluation only for final/best architectures
   - Reduces computational cost during evolution

The fitness function signature:
    fitness(genome, benchmark, budget) -> float in [0, 1]

Where:
- genome: MultiAgentGenome specifying the architecture
- benchmark: List of programming tasks
- budget: Token limit (hard constraint)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time

import tiktoken

from emap.genome.representation import MultiAgentGenome


@dataclass
class TaskResult:
    """Result of running architecture on a single task."""
    task_id: str
    success: bool  # Did it pass tests?
    tokens_used: int  # Total tokens consumed
    time_seconds: float  # Wall clock time
    output: str  # Generated code
    error: Optional[str] = None  # Error message if failed
    

@dataclass
class EvaluationResult:
    """Complete evaluation result for a genome."""
    genome_id: str
    budget: int
    task_results: List[TaskResult] = field(default_factory=list)
    
    @property
    def fitness(self) -> float:
        """Compute fitness: fraction of tasks passed."""
        if not self.task_results:
            return 0.0
        passed = sum(1 for r in self.task_results if r.success)
        return passed / len(self.task_results)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used across all tasks."""
        return sum(r.tokens_used for r in self.task_results)
    
    @property
    def avg_tokens_per_task(self) -> float:
        """Average tokens per task."""
        if not self.task_results:
            return 0.0
        return self.total_tokens / len(self.task_results)
    
    @property
    def budget_exceeded(self) -> bool:
        """Did any task exceed the per-task budget?"""
        task_budget = self.budget  # budget is per-task
        return any(r.tokens_used > task_budget for r in self.task_results)
    
    @property
    def success_rate(self) -> float:
        """Alias for fitness."""
        return self.fitness
    
    @property
    def tokens_on_failures(self) -> int:
        """Tokens spent on failed tasks (expensive failure metric)."""
        return sum(r.tokens_used for r in self.task_results if not r.success)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "genome_id": self.genome_id,
            "budget": self.budget,
            "fitness": self.fitness,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_task": self.avg_tokens_per_task,
            "success_rate": self.success_rate,
            "tokens_on_failures": self.tokens_on_failures,
            "num_tasks": len(self.task_results),
            "tasks": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "tokens_used": r.tokens_used,
                    "time_seconds": r.time_seconds,
                }
                for r in self.task_results
            ],
        }


class TokenCounter:
    """
    Accurate token counting using tiktoken.
    
    Tracks tokens for:
    - Prompts (input tokens)
    - Completions (output tokens)
    - Inter-agent messages
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize with encoding for specified model."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def add_prompt(self, text: str) -> int:
        """Add prompt tokens to running total."""
        tokens = self.count(text)
        self._prompt_tokens += tokens
        self._total_tokens += tokens
        return tokens
    
    def add_completion(self, text: str) -> int:
        """Add completion tokens to running total."""
        tokens = self.count(text)
        self._completion_tokens += tokens
        self._total_tokens += tokens
        return tokens
    
    @property
    def total(self) -> int:
        """Total tokens used."""
        return self._total_tokens
    
    @property
    def prompt_tokens(self) -> int:
        """Prompt tokens used."""
        return self._prompt_tokens
    
    @property
    def completion_tokens(self) -> int:
        """Completion tokens used."""
        return self._completion_tokens
    
    def reset(self) -> None:
        """Reset counters."""
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
    
    def check_budget(self, budget: int) -> bool:
        """Check if still within budget."""
        return self._total_tokens <= budget


@dataclass
class Task:
    """Programming task for evaluation."""
    id: str
    prompt: str  # Task description
    entry_point: str  # Function name to implement
    test_code: str  # Test cases to run
    canonical_solution: Optional[str] = None  # Reference solution


class FitnessEvaluator:
    """
    Evaluates genome fitness under resource constraints.
    
    The evaluator:
    1. Instantiates the multi-agent architecture from genome
    2. Runs it on sampled tasks from benchmark
    3. Tracks token usage with hard budget enforcement
    4. Returns fitness score based on task success rate
    
    If budget is exceeded on any task, that task counts as failed
    (not zero overall fitness, but the task itself fails).
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        sample_fraction: float = 0.2,
        timeout_seconds: float = 60.0,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: LLM model to use
            sample_fraction: Fraction of benchmark to use for fitness
            timeout_seconds: Max time per task
        """
        self.model = model
        self.sample_fraction = sample_fraction
        self.timeout_seconds = timeout_seconds
        self.token_counter = TokenCounter(model)
    
    def evaluate(
        self,
        genome: MultiAgentGenome,
        tasks: List[Task],
        budget: int,
        sample: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate genome on tasks under budget constraint.
        
        Args:
            genome: Architecture to evaluate
            tasks: List of programming tasks
            budget: Token budget per task (hard constraint)
            sample: Whether to sample tasks (True for evolution, False for final eval)
        
        Returns:
            EvaluationResult with fitness and metrics
        """
        # Sample tasks if requested
        if sample and self.sample_fraction < 1.0:
            import random
            n_sample = max(1, int(len(tasks) * self.sample_fraction))
            eval_tasks = random.sample(tasks, n_sample)
        else:
            eval_tasks = tasks
        
        results = []
        for task in eval_tasks:
            result = self._evaluate_task(genome, task, budget)
            results.append(result)
        
        return EvaluationResult(
            genome_id=genome.id,
            budget=budget,
            task_results=results,
        )
    
    def _evaluate_task(
        self,
        genome: MultiAgentGenome,
        task: Task,
        budget: int,
    ) -> TaskResult:
        """
        Evaluate genome on a single task.
        
        This is a placeholder implementation. The real implementation
        would instantiate the multi-agent system and run it.
        """
        start_time = time.time()
        self.token_counter.reset()
        
        try:
            # TODO: Replace with actual multi-agent execution
            # For now, simulate with placeholder
            output, tokens = self._simulate_execution(genome, task, budget)
            
            # Check if budget exceeded
            if tokens > budget:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    tokens_used=tokens,
                    time_seconds=time.time() - start_time,
                    output="",
                    error=f"Budget exceeded: {tokens} > {budget}",
                )
            
            # Run tests to check correctness
            success = self._run_tests(output, task)
            
            return TaskResult(
                task_id=task.id,
                success=success,
                tokens_used=tokens,
                time_seconds=time.time() - start_time,
                output=output,
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                tokens_used=self.token_counter.total,
                time_seconds=time.time() - start_time,
                output="",
                error=str(e),
            )
    
    def _simulate_execution(
        self,
        genome: MultiAgentGenome,
        task: Task,
        budget: int,
    ) -> Tuple[str, int]:
        """
        Placeholder: Simulate multi-agent execution.
        
        In the real implementation, this would:
        1. Instantiate agents from genome
        2. Route task through topology
        3. Track all token usage
        4. Return final output
        """
        import random
        
        # Simulate token usage based on architecture complexity
        base_tokens = 200
        per_agent_tokens = 150 * genome.num_agents
        per_edge_tokens = 50 * genome.num_edges
        per_round_tokens = 100 * genome.max_rounds
        
        estimated_tokens = base_tokens + per_agent_tokens + per_edge_tokens + per_round_tokens
        
        # Add some randomness
        actual_tokens = int(estimated_tokens * random.uniform(0.8, 1.2))
        
        # Simulate success probability (placeholder)
        # In reality, this would run actual code generation and tests
        success_base = 0.3  # Single agent baseline
        agent_bonus = 0.1 * min(genome.num_agents - 1, 2)  # Diminishing returns
        success_prob = min(0.9, success_base + agent_bonus)
        
        # Simulate output
        output = f"def {task.entry_point}():\n    # Simulated output\n    pass"
        
        return output, actual_tokens
    
    def _run_tests(self, output: str, task: Task) -> bool:
        """
        Run test cases on generated code.
        
        Placeholder implementation. Real version would:
        1. Create sandbox environment
        2. Execute generated code
        3. Run test cases
        4. Return pass/fail
        """
        # Placeholder: random success based on output quality
        import random
        
        if not output or "pass" in output:
            return random.random() < 0.1  # Low success for placeholder
        
        return random.random() < 0.5


def compute_fitness(
    genome: MultiAgentGenome,
    tasks: List[Task],
    budget: int,
    evaluator: Optional[FitnessEvaluator] = None,
) -> float:
    """
    Convenience function to compute fitness.
    
    Args:
        genome: Architecture to evaluate
        tasks: Programming tasks
        budget: Token budget per task
        evaluator: Optional evaluator instance
    
    Returns:
        Fitness score in [0, 1]
    """
    if evaluator is None:
        evaluator = FitnessEvaluator()
    
    result = evaluator.evaluate(genome, tasks, budget)
    return result.fitness
