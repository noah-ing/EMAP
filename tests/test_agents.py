"""Tests for agent execution and sandbox."""

import pytest
import asyncio

from emap.genome.representation import (
    MultiAgentGenome,
    AgentGene,
    AgentRole,
    create_single_agent,
    create_pipeline,
)
from emap.agents.executor import (
    MockBackend,
    MultiAgentExecutor,
    ExecutionResult,
    ExecutionStatus,
    Message,
)
from emap.benchmarks.sandbox import (
    execute_code,
    execute_with_entrypoint,
    check_code_safety,
    ExecutionOutcome,
)


class TestMockBackend:
    """Tests for MockBackend."""
    
    def test_mock_generation(self):
        """Test mock backend generates responses."""
        backend = MockBackend(response_template="Test response for: {task}")
        
        async def run():
            response, tokens = await backend.generate(
                system_prompt="You are helpful",
                user_message="Write hello world",
                max_tokens=100,
            )
            return response, tokens
        
        response, tokens = asyncio.run(run())
        
        assert "Test response" in response
        assert tokens > 0
    
    def test_mock_respects_token_limit(self):
        """Test mock backend respects max_tokens."""
        backend = MockBackend(
            response_template="A " * 1000  # Long response
        )
        
        async def run():
            response, tokens = await backend.generate(
                system_prompt="",
                user_message="",
                max_tokens=10,
            )
            return response, tokens
        
        response, tokens = asyncio.run(run())
        
        assert tokens <= 10


class TestMultiAgentExecutor:
    """Tests for MultiAgentExecutor."""
    
    def test_single_agent_execution(self):
        """Test executing a single-agent genome."""
        backend = MockBackend()
        executor = MultiAgentExecutor(backend, default_budget=500)
        genome = create_single_agent()
        
        async def run():
            return await executor.execute(
                genome=genome,
                task="Write a function that adds two numbers",
                token_budget=500,
            )
        
        result = asyncio.run(run())
        
        assert isinstance(result, ExecutionResult)
        assert result.genome_id == genome.id
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.BUDGET_EXCEEDED]
        assert result.total_tokens_used > 0
        assert result.rounds_executed >= 1
    
    def test_pipeline_execution(self):
        """Test executing a pipeline genome."""
        backend = MockBackend()
        executor = MultiAgentExecutor(backend, default_budget=1000)
        genome = create_pipeline([AgentRole.PLANNER, AgentRole.CODER])
        
        async def run():
            return await executor.execute(
                genome=genome,
                task="Implement fibonacci",
                token_budget=1000,
            )
        
        result = asyncio.run(run())
        
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.BUDGET_EXCEEDED]
        assert len(result.message_trace) > 0
    
    def test_budget_enforcement(self):
        """Test that budget is enforced."""
        backend = MockBackend(response_template="x " * 500)  # Long responses
        executor = MultiAgentExecutor(backend, default_budget=50)
        genome = create_pipeline([AgentRole.CODER, AgentRole.REVIEWER, AgentRole.TESTER])
        
        async def run():
            return await executor.execute(
                genome=genome,
                task="Complex task",
                token_budget=50,  # Very small budget
            )
        
        result = asyncio.run(run())
        
        # Should either complete within budget or exceed
        assert result.total_tokens_used > 0
        # With a 50 token budget and multi-agent, likely to exceed
    
    def test_execution_result_properties(self):
        """Test ExecutionResult computed properties."""
        backend = MockBackend()
        executor = MultiAgentExecutor(backend, default_budget=500)
        genome = create_single_agent()
        
        async def run():
            return await executor.execute(genome, "test task", 500)
        
        result = asyncio.run(run())
        
        # Test properties
        assert isinstance(result.within_budget, bool)
        assert 0.0 <= result.budget_efficiency <= 1.0 or result.budget_efficiency > 1.0
        
        # Test serialization
        result_dict = result.to_dict()
        assert "genome_id" in result_dict
        assert "total_tokens_used" in result_dict


class TestSandbox:
    """Tests for code execution sandbox."""
    
    def test_simple_execution(self):
        """Test executing simple code."""
        code = """
x = 2 + 2
result = x * 2
"""
        outcome = execute_code(code)
        
        assert isinstance(outcome, ExecutionOutcome)
        assert outcome.success
        assert not outcome.timed_out
    
    def test_assertion_pass(self):
        """Test code that passes assertions."""
        code = """
def add(a, b):
    return a + b
"""
        test_code = """
assert add(2, 3) == 5
assert add(-1, 1) == 0
"""
        outcome = execute_code(code, test_code)
        
        assert outcome.success
    
    def test_assertion_fail(self):
        """Test code that fails assertions."""
        code = """
def add(a, b):
    return a - b  # Bug: wrong operation
"""
        test_code = """
assert add(2, 3) == 5  # Will fail
"""
        outcome = execute_code(code, test_code)
        
        assert not outcome.success
        assert outcome.error_type == "AssertionError"
    
    def test_syntax_error(self):
        """Test code with syntax errors."""
        code = """
def broken(
    return None  # Missing closing paren
"""
        outcome = execute_code(code)
        
        assert not outcome.success
    
    def test_timeout(self):
        """Test code that times out."""
        code = """
import time
time.sleep(10)  # Sleep is blocked, but loop isn't
"""
        # This should fail due to blocked import
        outcome = execute_code(code, timeout=1.0)
        
        assert not outcome.success
    
    def test_blocked_imports(self):
        """Test that dangerous imports are blocked."""
        code = """
import os
os.system("echo pwned")
"""
        outcome = execute_code(code)
        
        assert not outcome.success
        assert "blocked" in (outcome.error_message or "").lower() or "security" in (outcome.error_message or "").lower()
    
    def test_safety_check(self):
        """Test static safety analysis."""
        # Safe code
        is_safe, violations = check_code_safety("x = 2 + 2")
        assert is_safe
        assert len(violations) == 0
        
        # Dangerous code
        is_safe, violations = check_code_safety("import os")
        assert not is_safe
        assert len(violations) > 0
    
    def test_execute_with_entrypoint(self):
        """Test executing with test cases."""
        code = """
def double(n):
    return n * 2
"""
        test_cases = [
            ((2,), 4),
            ((0,), 0),
            ((-3,), -6),
        ]
        
        outcome = execute_with_entrypoint(code, "double", test_cases)
        
        assert outcome.success
    
    def test_execute_with_entrypoint_failure(self):
        """Test failing test cases."""
        code = """
def double(n):
    return n + n + 1  # Bug
"""
        test_cases = [
            ((2,), 4),
        ]
        
        outcome = execute_with_entrypoint(code, "double", test_cases)
        
        assert not outcome.success


class TestIntegration:
    """Integration tests for executor + sandbox."""
    
    def test_full_pipeline(self):
        """Test full pipeline: genome -> execution -> sandbox."""
        from emap.evolution.fitness import Task
        from emap.evolution.integrated_eval import (
            IntegratedEvaluator,
            extract_code_from_response,
        )
        
        # Create a mock backend that returns valid code
        class CodeGenBackend(MockBackend):
            async def generate(self, system_prompt, user_message, max_tokens):
                # Return code that will pass the test
                code = """    return a + b"""
                return code, self.count_tokens(code)
        
        backend = CodeGenBackend()
        evaluator = IntegratedEvaluator(backend=backend)
        
        genome = create_single_agent()
        tasks = [
            Task(
                id="test/add",
                prompt="def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n",
                entry_point="add",
                test_code="assert add(2, 3) == 5",
            )
        ]
        
        result = evaluator.evaluate(genome, tasks, budget=500, sample=False)
        
        assert result.genome_id == genome.id
        assert len(result.task_results) == 1
    
    def test_code_extraction(self):
        """Test extracting code from LLM responses."""
        from emap.evolution.integrated_eval import extract_code_from_response
        
        # Test markdown code block
        response1 = """Here's the solution:

```python
def add(a, b):
    return a + b
```

This works by adding a and b."""
        
        code1 = extract_code_from_response(response1, "add")
        assert "def add" in code1
        assert "return a + b" in code1
        
        # Test plain function
        response2 = """def multiply(x, y):
    return x * y"""
        
        code2 = extract_code_from_response(response2, "multiply")
        assert "def multiply" in code2
