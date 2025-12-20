"""
Integrated fitness evaluation using actual LLM execution.

This module connects the evolution loop to the agent executor and sandbox,
providing real fitness evaluation under token budgets.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from emap.genome.representation import MultiAgentGenome
from emap.agents.executor import (
    MultiAgentExecutor,
    LLMBackend,
    MockBackend,
    ExecutionResult,
    ExecutionStatus,
)
from emap.benchmarks.sandbox import execute_code, ExecutionOutcome
from emap.evolution.fitness import (
    Task,
    TaskResult,
    EvaluationResult,
    TokenCounter,
)


def extract_code_from_response(response: str, entry_point: str) -> str:
    """
    Extract Python code from LLM response.
    
    LLMs often wrap code in markdown blocks or add explanations.
    This function extracts the actual code.
    """
    # Try to find code in markdown code block
    code_block_pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the largest match (likely the main code)
        return max(matches, key=len).strip()
    
    # Look for the function definition
    func_pattern = rf"(def {entry_point}\s*\(.*?\):.*?)(?=\ndef |\nclass |\Z)"
    match = re.search(func_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If all else fails, return the raw response (may cause syntax errors)
    return response.strip()


def build_coding_prompt(task: Task) -> str:
    """
    Build a prompt for code generation task.
    
    This formats the task prompt to elicit clean code output.
    """
    prompt = f"""Complete the following Python function. Output ONLY the function code, no explanations.

{task.prompt}

Complete the implementation:"""
    return prompt


@dataclass
class IntegratedEvaluator:
    """
    Evaluator that uses actual LLM execution with sandbox testing.
    
    This is the real evaluator used during evolution, connecting:
    - Genome -> MultiAgentExecutor -> LLM calls
    - LLM output -> Code extraction -> Sandbox execution
    - Sandbox result -> Task pass/fail -> Fitness
    """
    
    backend: LLMBackend
    sample_fraction: float = 0.2
    timeout_seconds: float = 10.0
    _token_counter: TokenCounter = field(default_factory=lambda: TokenCounter())
    
    def __post_init__(self):
        self.executor = MultiAgentExecutor(
            backend=self.backend,
            default_budget=2000,
        )
    
    async def evaluate_async(
        self,
        genome: MultiAgentGenome,
        tasks: List[Task],
        budget: int,
        sample: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate genome on tasks asynchronously.
        
        Args:
            genome: Architecture to evaluate
            tasks: List of tasks
            budget: Token budget per task
            sample: Whether to sample tasks
            
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
            result = await self._evaluate_task_async(genome, task, budget)
            results.append(result)
            # Rate limit protection - 2 seconds between tasks
            await asyncio.sleep(2.0)
        
        return EvaluationResult(
            genome_id=genome.id,
            budget=budget,
            task_results=results,
        )
    
    def evaluate(
        self,
        genome: MultiAgentGenome,
        tasks: List[Task],
        budget: int,
        sample: bool = True,
    ) -> EvaluationResult:
        """Synchronous wrapper for evaluate_async."""
        return asyncio.run(self.evaluate_async(genome, tasks, budget, sample))
    
    async def _evaluate_task_async(
        self,
        genome: MultiAgentGenome,
        task: Task,
        budget: int,
    ) -> TaskResult:
        """Evaluate genome on a single task."""
        start_time = time.time()
        
        try:
            # Build the coding prompt
            prompt = build_coding_prompt(task)
            
            # Execute the multi-agent system
            exec_result = await self.executor.execute(
                genome=genome,
                task=prompt,
                token_budget=budget,
            )
            
            # Check budget
            if exec_result.status == ExecutionStatus.BUDGET_EXCEEDED:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    tokens_used=exec_result.total_tokens_used,
                    time_seconds=time.time() - start_time,
                    output="",
                    error=f"Budget exceeded: {exec_result.total_tokens_used} > {budget}",
                )
            
            # Extract code from response
            code = extract_code_from_response(
                exec_result.final_output,
                task.entry_point,
            )
            
            # Combine with prompt (signature + implementation)
            full_code = task.prompt + "\n" + code
            
            # Run in sandbox
            sandbox_result = execute_code(
                code=full_code,
                test_code=task.test_code,
                timeout=self.timeout_seconds,
                check_safety=True,
            )
            
            return TaskResult(
                task_id=task.id,
                success=sandbox_result.success,
                tokens_used=exec_result.total_tokens_used,
                time_seconds=time.time() - start_time,
                output=code,
                error=sandbox_result.error_message if not sandbox_result.success else None,
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                tokens_used=0,
                time_seconds=time.time() - start_time,
                output="",
                error=str(e),
            )


async def evaluate_population_async(
    genomes: List[MultiAgentGenome],
    tasks: List[Task],
    budget: int,
    backend: LLMBackend,
    sample: bool = True,
    sample_fraction: float = 0.2,
    max_concurrent: int = 5,
) -> List[EvaluationResult]:
    """
    Evaluate entire population asynchronously.
    
    Args:
        genomes: Population to evaluate
        tasks: Benchmark tasks
        budget: Token budget per task
        backend: LLM backend
        sample: Whether to sample tasks
        sample_fraction: Fraction of tasks to sample
        max_concurrent: Max concurrent genome evaluations
        
    Returns:
        List of EvaluationResults
    """
    evaluator = IntegratedEvaluator(
        backend=backend,
        sample_fraction=sample_fraction,
    )
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_one(genome: MultiAgentGenome) -> EvaluationResult:
        async with semaphore:
            return await evaluator.evaluate_async(genome, tasks, budget, sample)
    
    results = await asyncio.gather(
        *[evaluate_one(g) for g in genomes],
        return_exceptions=True,
    )
    
    # Handle exceptions
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append(EvaluationResult(
                genome_id=genomes[i].id,
                budget=budget,
                task_results=[TaskResult(
                    task_id="error",
                    success=False,
                    tokens_used=0,
                    time_seconds=0,
                    output="",
                    error=str(result),
                )],
            ))
        else:
            processed.append(result)
    
    return processed


def quick_fitness_check(
    genome: MultiAgentGenome,
    tasks: List[Task],
    budget: int,
    n_tasks: int = 5,
) -> float:
    """
    Quick fitness check using mock backend.
    
    Useful for testing evolution loop without API calls.
    """
    backend = MockBackend(
        response_template="def {entry_point}():\n    # Implementation\n    return None"
    )
    
    evaluator = IntegratedEvaluator(
        backend=backend,
        sample_fraction=1.0,
    )
    
    # Take subset of tasks
    eval_tasks = tasks[:n_tasks]
    
    result = evaluator.evaluate(genome, eval_tasks, budget, sample=False)
    return result.fitness
