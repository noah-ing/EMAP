# EMAP: Evolution under Multi-Agent Pressure

**Resource-Constrained Evolution of Multi-Agent Programming Architectures**

Noah Ingwers, 2025

> **Research status:** Exploratory, non-peer-reviewed work. The findings below summarize
> 12 recorded GPT-4o-mini runs completed in December 2025; they are not results from a
> peer-reviewed paper or a full-benchmark evaluation.

## Overview

EMAP is a framework for studying how resource constraints during evolutionary optimization shape multi-agent LLM architectures. Unlike existing approaches that treat computational cost as a secondary objective, EMAP makes budget constraints the primary evolutionary pressure - forcing architectures to adapt to resource scarcity rather than simply trading off against it.

## Recorded Findings

The repository contains 12 completed experiment artifacts: 4 token-budget regimes
(TIGHT 2K, MEDIUM 5K, LOOSE 10K, and UNCONSTRAINED 50K) with seeds 42, 43, and 44.
Each recorded configuration used 12 generations and a population of 10. The runner
is configured to sample 12% of the intended 164-task HumanEval corpus per evaluation.
Each artifact reports a 41-task final evaluation (25%), consistent with that runner
configuration but not with a full-benchmark evaluation.

### Recorded Results

| Regime | Budget | Mean final score | Final-pop. agents | Final-pop. edges |
|--------|--------|------------------|-------------------|------------------|
| TIGHT | 2,000 tokens | 98.4% ± 2.3% | 3.2 | 2.6 |
| MEDIUM | 5,000 tokens | 100.0% ± 0.0% | 3.1 | 2.4 |
| LOOSE | 10,000 tokens | 98.4% ± 2.3% | 3.5 | 2.8 |
| UNCONSTRAINED | 50,000 tokens | 99.2% ± 1.1% | 2.9 | 2.1 |

Values are means and population standard deviations across three recorded runs.
Agent and edge counts are final-generation population averages, not the sizes of a
single selected architecture. These numbers are reproduced from the checked-in
[summary](experiments/results/focused/summary.json) and
[per-run JSON artifacts](experiments/results/focused/).

### Descriptive Observations

1. **The recorded final populations averaged roughly three agents.** Mean agent
   counts ranged from 2.9 to 3.5 across the four regimes.
2. **The selected genomes used several topologies.** Checked-in artifacts include a
   planner -> coder -> reviewer pipeline, a tester -> reviewer -> planner topology,
   a generalist -> coder -> architect topology, and two 4-agent LOOSE-budget systems.
3. **The MEDIUM runs were the strongest in this small sample.** All three recorded
   MEDIUM final evaluations passed 41/41 sampled tasks and together used 414,390
   tokens, the lowest observed total among the four regimes. This does not establish
   5K tokens as a general optimum.
4. **Topology diversity is a hypothesis, not a causal result.** The 12 artifacts show
   different structures across seeds, but three runs per regime on varying task
   samples cannot establish that budget pressure caused those differences.

## Checked-in Evidence and Cost

- [Aggregate summary](experiments/results/focused/summary.json)
- [Twelve per-run JSON records](experiments/results/focused/)
- [Recorded run visualizations](experiments/viz/)
- [Artifact-grounded research note source](paper/main.tex) and
  [rendered research note](paper/Scarcity_Breeds_Efficiency__Resource_Constrained_Evolution_of_Multi_Agent_Programming_Architectures.pdf)

The 12 JSON files record **3,093,390 tokens**, **27,360 API calls**, and runtimes of
approximately **1.7-5.2 hours per run**. They do not separate input from output tokens,
so a historical dollar cost cannot be independently recalculated from the artifacts.

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
python experiments/run_single_experiment.py \
    --budget 5000 \
    --seed 42 \
    --generations 12 \
    --population 10
```

The runner writes to `experiments/results/focused/` and skips a run when a completed
artifact with the same budget and seed already exists. Preserve the checked-in JSON
files before attempting a fresh rerun.

### Run Offline Tests

```bash
python -m pytest tests/ -v
```

The test suite uses mock backends and does not require an OpenAI API key.

The code-evaluation harness is **not a security sandbox**. It executes generated
Python with `exec` in a disposable subprocess and applies static checks,
restricted builtins/imports, a timeout, and best-effort Unix resource limits.
Run it only in an unprivileged, isolated environment with no secrets or sensitive
network/filesystem access.

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
EMAP/
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
│       └── sandbox.py           # Best-effort subprocess evaluator
├── experiments/
│   ├── run_single_experiment.py # One budget/seed run
│   ├── run_all_experiments.py   # Recorded 4 x 3 run matrix
│   └── results/focused/         # Checked-in JSON artifacts
├── paper/
│   ├── main.tex                 # Artifact-grounded research note
│   └── Scarcity_...pdf          # Rendered research note
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

## Reproducibility and Limitations

To run against HumanEval, place `humaneval.jsonl` in `data/`, in
`src/emap/benchmarks/data/`, or at `~/.cache/emap/humaneval.jsonl`. The loader falls
back to five simple placeholder tasks when it cannot find that file. The recorded
artifacts have `final_total: 41`, which is consistent with the configured 25% sample
of the 164-task HumanEval corpus rather than the placeholder set.

Important limits of the recorded study:

- The `--seed` value controls NumPy-based architecture evolution, but task subsets are
  selected with Python's unseeded `random` module. The OpenAI calls are also
  nondeterministic. Exact reruns are therefore not guaranteed, and different runs may
  have used different 41-task final subsets.
- The JSON files contain aggregate scores, selected genomes, timing, token totals, and
  configuration. They do not contain final task IDs, per-task outputs, a dataset hash,
  a Git commit, or dependency/model version metadata. The GPT-4o-mini attribution is
  supported by the checked-in runner configuration, not by a model field in each JSON.
- Project dependencies use lower bounds and no lockfile, so the exact December 2025
  software environment is not preserved.
- With three runs per regime and sampled final evaluations, the table supports
  descriptive comparisons only. It does not establish statistical significance,
  full-HumanEval performance, causal effects of budget pressure, or generalization to
  other models and benchmarks.
- This repository and its research note have not been peer reviewed.

## Research Questions

**RQ1 (Explored descriptively):** Do architectures evolved under different budget regimes exhibit different structures?

- Recorded observation: final populations averaged 2.9-3.5 agents, while selected
  genomes ranged from 2 to 4 agents and included both linear and cyclic topologies.
  The current sample does not isolate budget from seed, task sampling, or model noise.

**RQ2 (Explored descriptively):** Do constraint-evolved architectures exhibit different coordination strategies?

- Recorded observation: selected genomes used sequential, hierarchical, and voting
  aggregation with structured, freeform, and minimal message formats. Whether the
  constraints caused those choices remains untested.

**RQ3 & RQ4 (Not evaluated):** Transfer to abundance and cross-benchmark generalization remain future work.

## Broader Implications

EMAP investigates a counterintuitive hypothesis: constraints may be useful design
pressures rather than only deployment limits. Biological adaptation under scarcity is
the motivating analogy, not evidence for an equivalent mechanism in LLM systems.

The checked-in runs contain cyclic feedback topologies, compressed message formats,
and simpler linear pipelines. Because there are only three runs per regime and final
task subsets were not controlled, these observations motivate a larger preregistered
comparison; they do not demonstrate that budget pressure caused the structures or
preserved diversity.

If the effect holds under controlled, full-benchmark evaluation, evolving systems
under realistic resource constraints could be useful beyond code generation. Testing
that proposition across models, benchmarks, and cost measures remains future work.

## Future Directions

- **Progressive constraints**: Curriculum learning for resource limits - start loose, tighten gradually
- **Multi-constraint evolution**: Simultaneously optimize for tokens, latency, and memory
- **Cross-benchmark transfer**: Do HumanEval-evolved architectures generalize to MBPP or SWE-bench?
- **Prompt co-evolution**: Allow agent prompts to mutate alongside topology
- **Meta-evolution**: Evolve the evolutionary process itself - mutation rates, selection pressure, constraint schedules
- **Cross-model transfer**: Do architectures evolved on GPT-4o-mini transfer to Claude, Gemini, or open-source models?

## Citation

```bibtex
@misc{ingwers2025emap,
  title={EMAP: An Exploratory Artifact Report on Resource-Constrained
         Evolution of Multi-Agent Programming Architectures},
  author={Ingwers, Noah},
  year={2025},
  note={Unpublished, non-peer-reviewed research note and software repository}
}
```

## Security

Report suspected vulnerabilities privately as described in
[SECURITY.md](SECURITY.md). The generated-code evaluator is a best-effort
experiment harness, not a security sandbox; its documented boundary and
reporting scope are stated there.

## License

[MIT License](LICENSE)
