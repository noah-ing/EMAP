# EMAP: Evolution under Multi-Agent Pressure

**Resource-Constrained Evolution of Multi-Agent Programming Architectures**

Noah Ingwers, 2025

## Overview

EMAP is a framework for studying how resource constraints during evolutionary optimization shape multi-agent LLM architectures. Unlike existing approaches that treat computational cost as a secondary objective, EMAP makes budget constraints the primary evolutionary pressure - forcing architectures to adapt to resource scarcity rather than simply trading off against it.

## Key Findings

We conducted 12 experiments across 4 budget regimes (TIGHT 2K, MEDIUM 5K, LOOSE 10K, UNCONSTRAINED 50K tokens) with 3 random seeds each, evolving multi-agent architectures on HumanEval (164 programming tasks).

### Main Results

| Regime | Budget | Pass@1 | Avg Agents | Avg Edges |
|--------|--------|--------|------------|-----------|
| TIGHT | 2,000 tokens | 98.4% ± 2.3% | 3.0 | 2.6 |
| MEDIUM | 5,000 tokens | 100.0% ± 0.0% | 3.0 | 2.4 |
| LOOSE | 10,000 tokens | 97.6% ± 2.4% | 3.0 | 1.8 |
| UNCONSTRAINED | 50,000 tokens | 100.0% ± 0.0% | 2.7 | 2.0 |

### Key Discoveries

1. **Evolution consistently produces 3-agent architectures** regardless of constraint severity, contradicting the hypothesis that tight constraints would favor minimal single-agent systems.

2. **Multiple viable topologies emerge** - evolution discovered diverse architectures achieving equivalent performance:
   - Traditional pipeline (planner → coder → reviewer)
   - Test-first pipeline (tester → reviewer → planner)
   - Hybrid architecture (generalist → coder → architect)
   - Hierarchical 4-agent systems under loose constraints

3. **Budget constraints preserve diversity** - under constrained regimes, different seeds produce different topologies (linear vs cyclic). Without constraints, evolution converges toward simpler linear solutions.

4. **The "Goldilocks Zone"** - MEDIUM budget (5K tokens) achieves perfect performance with lowest token usage, representing optimal constraint severity for HumanEval tasks.

## Computational Cost

Total experimental cost: **3.09M tokens** (~27K API calls) across all 12 experiments.
Estimated API cost: **<$1.00** using GPT-4o-mini.
Runtime: 1.5-5.2 hours per experiment depending on budget regime.

## Installation

```bash
git clone https://github.com/noah-ing/emap.git
cd emap

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

# Set up API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

## Usage

### Run Evolution Experiment

```bash
python experiments/run_evolution_experiments.py \
    --budget 5000 \
    --seed 42 \
    --generations 12 \
    --population 10
```

### Run Tests

```bash
pytest tests/ -v
```

### Basic API Usage

```python
import asyncio
from emap.genome.representation import create_pipeline, AgentRole
from emap.agents.executor import OpenAIBackend, MultiAgentExecutor

backend = OpenAIBackend(model="gpt-4o-mini")
executor = MultiAgentExecutor(backend, default_budget=5000)

genome = create_pipeline([AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER])

result = asyncio.run(executor.execute(
    genome=genome,
    task="Write a function that returns the factorial of n",
    token_budget=5000
))

print(f"Output: {result.final_output}")
print(f"Tokens: {result.total_tokens_used}")
```

## Project Structure

```
E_M_A_P/
├── src/emap/
│   ├── genome/
│   │   ├── representation.py    # MultiAgentGenome, AgentGene
│   │   └── operators.py         # Mutation, crossover operators
│   ├── evolution/
│   │   ├── fitness.py           # Hard budget constraint evaluation
│   │   ├── selection.py         # Tournament, roulette, elitist selection
│   │   └── integrated_eval.py   # Full evolution pipeline
│   ├── agents/
│   │   └── executor.py          # LLM backends and message routing
│   └── benchmarks/
│       ├── humaneval.py         # HumanEval loader
│       └── sandbox.py           # Safe code execution
├── experiments/
│   ├── run_evolution_experiments.py
│   └── results/                 # Experiment JSON outputs
├── paper/
│   ├── main.tex                 # Research paper
│   └── references.bib           # Bibliography
└── tests/
```

## Method

### Hard Budget Constraints

Unlike Pareto-based approaches that trade off accuracy against cost, EMAP enforces hard constraints:

```python
def fitness(architecture, benchmark, budget):
    for task in benchmark:
        output, tokens = execute(architecture, task)
        if tokens > budget:
            return 0.0  # Zero fitness for budget violation
    return accuracy(outputs, benchmark)
```

This creates genuine evolutionary pressure - architectures must adapt to constraints, not simply accept lower performance.

### Evolvable Genome

```python
@dataclass
class MultiAgentGenome:
    agents: List[AgentGene]           # Agents with roles and prompts
    topology: Dict[str, List[str]]    # Communication graph (adjacency list)
    message_format: MessageFormat     # STRUCTURED, FREEFORM, or MINIMAL
    aggregation_strategy: AggregationStrategy
    max_rounds: int
    early_exit_confidence: float
```

### Mutation Operators

- Add/remove agents
- Add/remove edges
- Swap agent roles
- Adjust hyperparameters (temperature, max tokens, message length)
- Change message format and aggregation strategy

## Research Questions

**RQ1 (Addressed):** Do architectures evolved under different budget regimes exhibit different structures?
- Finding: All regimes converge to 3-agent architectures, but topology type (linear vs cyclic) is seed-dependent.

**RQ2 (Addressed):** Do constraint-evolved architectures exhibit different coordination strategies?
- Finding: Evolution discovers multiple viable topologies achieving equivalent performance.

**RQ3 & RQ4 (Future Work):** Transfer to abundance and cross-benchmark generalization remain to be tested.

## Broader Implications

Our findings suggest something counterintuitive: constraints may be features, not bugs. The dominant paradigm in AI development is to maximize resources during training and optimization, then compress or distill afterward. But biological evolution tells a different story - organisms that evolved under scarcity develop fundamentally different adaptations than those subjected to scarcity after the fact. A desert plant has different anatomy than a rainforest plant experiencing drought.

We observe an analogous phenomenon in multi-agent architectures. Under budget pressure, evolution discovers cyclic feedback topologies, compressed message formats, and aggressive early-exit strategies. Remove that pressure, and evolution converges toward simpler linear pipelines. The constraint doesn't just filter solutions - it shapes the search process itself, maintaining diversity and preventing premature convergence to local optima.

This has implications beyond code generation. As LLM costs and environmental impact come under scrutiny, the ability to evolve efficient architectures under realistic constraints becomes increasingly valuable. Rather than designing for abundance and hoping efficiency follows, we can design for scarcity and let evolution discover what's truly necessary.

## Future Directions

- **Progressive constraints**: Curriculum learning for resource limits - start loose, tighten gradually
- **Multi-constraint evolution**: Simultaneously optimize for tokens, latency, and memory
- **Cross-benchmark transfer**: Do HumanEval-evolved architectures generalize to MBPP or SWE-bench?
- **Prompt co-evolution**: Allow agent prompts to mutate alongside topology
- **Meta-evolution**: Evolve the evolutionary process itself - mutation rates, selection pressure, constraint schedules
- **Cross-model transfer**: Do architectures evolved on GPT-4o-mini transfer to Claude, Gemini, or open-source models?

## Citation

```bibtex
@article{ingwers2025emap,
  title={Scarcity Breeds Efficiency: Resource-Constrained Evolution of
         Multi-Agent Programming Architectures},
  author={Ingwers, Noah},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License
