"""
Multi-agent execution engine with hard token budget enforcement.

This module is the heart of the EMAP system. It executes multi-agent 
architectures while enforcing strict token budgets - the key innovation
that enables "scarcity breeds efficiency" evolution.

Key design decisions:
1. Token budgets are HARD limits, not soft suggestions
2. Agents that exceed budgets have their responses truncated
3. All token usage is tracked for fitness calculation
4. Message passing follows the genome's topology
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

import tiktoken

from emap.genome.representation import (
    MultiAgentGenome,
    AgentGene,
    MessageFormat,
    AggregationStrategy,
)


class ExecutionStatus(Enum):
    """Status of an execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class Message:
    """A message passed between agents."""
    sender_id: str
    receiver_id: str
    content: str
    tokens: int
    round_number: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Serialize message."""
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "tokens": self.tokens,
            "round": self.round_number,
        }


@dataclass
class AgentState:
    """Runtime state for a single agent."""
    agent: AgentGene
    messages_received: list[Message] = field(default_factory=list)
    messages_sent: list[Message] = field(default_factory=list)
    tokens_used: int = 0
    last_response: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING


@dataclass
class ExecutionResult:
    """Result of executing a multi-agent genome on a task."""
    genome_id: str
    task_id: str
    final_output: str
    total_tokens_used: int
    token_budget: int
    execution_time_ms: float
    status: ExecutionStatus
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    message_trace: list[Message] = field(default_factory=list)
    rounds_executed: int = 0
    early_exit: bool = False
    error_message: Optional[str] = None
    
    @property
    def within_budget(self) -> bool:
        """Check if execution stayed within token budget."""
        return self.total_tokens_used <= self.token_budget
    
    @property
    def budget_efficiency(self) -> float:
        """Fraction of budget used (lower is more efficient)."""
        if self.token_budget == 0:
            return 1.0
        return min(1.0, self.total_tokens_used / self.token_budget)
    
    def to_dict(self) -> dict:
        """Serialize for logging/analysis."""
        return {
            "genome_id": self.genome_id,
            "task_id": self.task_id,
            "final_output": self.final_output[:500] + "..." if len(self.final_output) > 500 else self.final_output,
            "total_tokens_used": self.total_tokens_used,
            "token_budget": self.token_budget,
            "execution_time_ms": self.execution_time_ms,
            "status": self.status.value,
            "rounds_executed": self.rounds_executed,
            "early_exit": self.early_exit,
            "within_budget": self.within_budget,
            "budget_efficiency": self.budget_efficiency,
        }


class LLMBackend:
    """
    Abstract LLM backend with token budget enforcement.
    
    Subclass this for different LLM providers (OpenAI, Anthropic, local).
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._encoder: Optional[tiktoken.Encoding] = None
        
    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback for unknown models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])
    
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System prompt for the agent
            user_message: User/task message
            max_tokens: Maximum tokens for response (HARD limit)
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        raise NotImplementedError("Subclass must implement generate()")


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        super().__init__(model)
        self.timeout = timeout
        
        # Import here to avoid requiring openai if not using this backend
        import openai
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key,  # Uses OPENAI_API_KEY env var if None
            timeout=timeout,
        )
    
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Generate response from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.7,  # Some creativity but not too random
            )
            
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.completion_tokens if response.usage else self.count_tokens(content)
            
            return content, tokens_used
            
        except Exception as e:
            # Return error message as response
            error_msg = f"[LLM Error: {str(e)[:100]}]"
            return error_msg, self.count_tokens(error_msg)


class MockBackend(LLMBackend):
    """Mock backend for testing without API calls."""
    
    def __init__(self, response_template: str = "Mock response for: {task}"):
        super().__init__("mock-model")
        self.response_template = response_template
        self.call_count = 0
        
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Generate mock response."""
        self.call_count += 1
        
        # Generate a plausible mock response
        response = self.response_template.format(
            task=user_message[:100],
            role=system_prompt[:50],
            call=self.call_count,
        )
        
        # Respect token limit
        response = self.truncate_to_tokens(response, max_tokens)
        tokens = self.count_tokens(response)
        
        return response, tokens


class MultiAgentExecutor:
    """
    Executes multi-agent architectures on tasks with token budget enforcement.
    
    This is where evolution meets execution. The executor:
    1. Takes a genome (architecture) and a task
    2. Runs agents according to the topology
    3. Enforces hard token budgets at every step
    4. Returns detailed execution trace for fitness calculation
    """
    
    def __init__(
        self,
        backend: LLMBackend,
        default_budget: int = 2000,
    ):
        self.backend = backend
        self.default_budget = default_budget
    
    async def execute(
        self,
        genome: MultiAgentGenome,
        task: str,
        token_budget: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute genome on a task with token budget.
        
        Args:
            genome: Multi-agent architecture to execute
            task: Task description/prompt
            token_budget: Total token budget for execution
            
        Returns:
            ExecutionResult with output and detailed trace
        """
        start_time = time.time()
        budget = token_budget or self.default_budget
        
        # Initialize agent states
        agent_states: dict[str, AgentState] = {
            agent.id: AgentState(agent=agent)
            for agent in genome.agents
        }
        
        message_trace: list[Message] = []
        total_tokens = 0
        rounds_executed = 0
        final_output = ""
        status = ExecutionStatus.RUNNING
        
        try:
            # Execute rounds
            for round_num in range(genome.max_rounds):
                rounds_executed = round_num + 1
                
                # Check budget before round
                if total_tokens >= budget:
                    status = ExecutionStatus.BUDGET_EXCEEDED
                    break
                
                # Determine which agents to run this round
                agents_to_run = self._get_agents_for_round(
                    genome, agent_states, round_num
                )
                
                # Run agents (potentially in parallel based on topology)
                round_results = await self._execute_round(
                    genome=genome,
                    agents=agents_to_run,
                    agent_states=agent_states,
                    task=task,
                    round_num=round_num,
                    remaining_budget=budget - total_tokens,
                )
                
                # Update state
                for msg in round_results:
                    message_trace.append(msg)
                    total_tokens += msg.tokens
                    
                    # Update sender state
                    if msg.sender_id in agent_states:
                        agent_states[msg.sender_id].messages_sent.append(msg)
                        agent_states[msg.sender_id].tokens_used += msg.tokens
                        agent_states[msg.sender_id].last_response = msg.content
                    
                    # Update receiver state
                    if msg.receiver_id in agent_states:
                        agent_states[msg.receiver_id].messages_received.append(msg)
                
                # Check for early exit
                if self._should_exit_early(genome, agent_states, round_num):
                    break
            
            # Get final output from output agent
            output_agent_id = genome.output_agent_id
            if output_agent_id and output_agent_id in agent_states:
                final_output = agent_states[output_agent_id].last_response or ""
            elif message_trace:
                final_output = message_trace[-1].content
            
            if status == ExecutionStatus.RUNNING:
                status = ExecutionStatus.COMPLETED
                
        except Exception as e:
            status = ExecutionStatus.FAILED
            final_output = f"Execution error: {str(e)}"
        
        execution_time = (time.time() - start_time) * 1000
        
        return ExecutionResult(
            genome_id=genome.id,
            task_id=str(hash(task))[:8],
            final_output=final_output,
            total_tokens_used=total_tokens,
            token_budget=budget,
            execution_time_ms=execution_time,
            status=status,
            agent_states=agent_states,
            message_trace=message_trace,
            rounds_executed=rounds_executed,
            early_exit=rounds_executed < genome.max_rounds,
            error_message=final_output if status == ExecutionStatus.FAILED else None,
        )
    
    def _get_agents_for_round(
        self,
        genome: MultiAgentGenome,
        agent_states: dict[str, AgentState],
        round_num: int,
    ) -> list[AgentGene]:
        """Determine which agents should run in this round."""
        if round_num == 0:
            # First round: start with entry agent
            entry_id = genome.entry_agent_id
            if entry_id:
                return [agent_states[entry_id].agent]
            return [genome.agents[0]]
        
        # Subsequent rounds: agents that received messages
        agents = []
        for agent_id, state in agent_states.items():
            # Agent runs if it received a message in the previous round
            recent_msgs = [
                m for m in state.messages_received 
                if m.round_number == round_num - 1
            ]
            if recent_msgs:
                agents.append(state.agent)
        
        # If no agents scheduled, run output agent
        if not agents and genome.output_agent_id:
            agents = [agent_states[genome.output_agent_id].agent]
        
        return agents
    
    async def _execute_round(
        self,
        genome: MultiAgentGenome,
        agents: list[AgentGene],
        agent_states: dict[str, AgentState],
        task: str,
        round_num: int,
        remaining_budget: int,
    ) -> list[Message]:
        """Execute a round of agent interactions."""
        if not agents:
            return []
        
        # Calculate per-agent budget for this round
        per_agent_budget = max(50, remaining_budget // (len(agents) + 1))
        
        messages = []
        
        # Execute agents (could parallelize for independent agents)
        for agent in agents:
            # Skip if no budget
            if remaining_budget <= 0:
                break
            
            # Build context from received messages
            context = self._build_context(
                agent, agent_states[agent.id], task, round_num
            )
            
            # Calculate this agent's token limit
            agent_max_tokens = min(
                agent.max_response_tokens,
                per_agent_budget,
                remaining_budget,
            )
            
            # Generate response
            response, tokens_used = await self.backend.generate(
                system_prompt=agent.system_prompt,
                user_message=context,
                max_tokens=agent_max_tokens,
            )
            
            remaining_budget -= tokens_used
            
            # Route message to connected agents
            targets = genome.topology.get(agent.id, [])
            if not targets:
                # No targets: output message goes to "output"
                targets = ["output"]
            
            for target_id in targets:
                msg = Message(
                    sender_id=agent.id,
                    receiver_id=target_id,
                    content=response,
                    tokens=tokens_used // len(targets),  # Split tokens
                    round_number=round_num,
                )
                messages.append(msg)
        
        return messages
    
    def _build_context(
        self,
        agent: AgentGene,
        state: AgentState,
        task: str,
        round_num: int,
    ) -> str:
        """Build context/prompt for an agent."""
        parts = []
        
        if round_num == 0:
            # First round: give the task
            parts.append(f"Task: {task}")
        else:
            # Include messages from other agents
            parts.append(f"Task: {task}\n\n")
            parts.append("Messages from other agents:\n")
            
            for msg in state.messages_received[-5:]:  # Last 5 messages
                parts.append(f"[From {msg.sender_id}]: {msg.content[:500]}")
            
            parts.append("\nProvide your response:")
        
        return "\n".join(parts)
    
    def _should_exit_early(
        self,
        genome: MultiAgentGenome,
        agent_states: dict[str, AgentState],
        round_num: int,
    ) -> bool:
        """Check if we should exit before max_rounds."""
        # For now, simple heuristic: exit if output agent has responded
        if genome.output_agent_id:
            output_state = agent_states.get(genome.output_agent_id)
            if output_state and output_state.last_response:
                # Check if response seems complete (heuristic)
                response = output_state.last_response
                if len(response) > 50 and not response.endswith("..."):
                    return True
        
        return False


async def execute_population(
    genomes: list[MultiAgentGenome],
    task: str,
    backend: LLMBackend,
    token_budget: int = 2000,
    max_concurrent: int = 5,
) -> list[ExecutionResult]:
    """
    Execute multiple genomes on a task with concurrency control.
    
    Args:
        genomes: Population of architectures to evaluate
        task: Task to solve
        backend: LLM backend to use
        token_budget: Budget per genome
        max_concurrent: Max concurrent executions
        
    Returns:
        List of execution results
    """
    executor = MultiAgentExecutor(backend, default_budget=token_budget)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_one(genome: MultiAgentGenome) -> ExecutionResult:
        async with semaphore:
            return await executor.execute(genome, task, token_budget)
    
    results = await asyncio.gather(
        *[execute_one(g) for g in genomes],
        return_exceptions=True,
    )
    
    # Handle any exceptions
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append(ExecutionResult(
                genome_id=genomes[i].id,
                task_id=str(hash(task))[:8],
                final_output=f"Error: {str(result)}",
                total_tokens_used=0,
                token_budget=token_budget,
                execution_time_ms=0,
                status=ExecutionStatus.FAILED,
                error_message=str(result),
            ))
        else:
            processed.append(result)
    
    return processed
