# Research Project Specification: Scarcity Breeds Efficiency
## Resource-Constrained Evolution of Multi-Agent Programming Architectures

**Project:** EMAP (Evolution under Multi-Agent Pressure)  
**Target Venues:** ICML, NeurIPS, ICLR (Top-Tier ML Conferences)  
**Status:** Literature Review Complete ✓ | Research Direction Validated ✓

---

## I. EXECUTIVE SUMMARY

### The Research Question

**"How do resource constraints during the evolutionary process shape the resulting multi-agent programming architectures, and do scarcity-evolved systems exhibit fundamentally different—and potentially superior—properties?"**

### What Makes This Novel

After systematic literature review (see `/paper/LITERATURE_REVIEW.md`), we identified that:

| Existing Work | What They Do | Gap |
|---------------|--------------|-----|
| **EvoAgentX** (Jul 2025) | Evolve multi-agent workflows | Cost is secondary objective, not evolutionary pressure |
| **ARTEMIS** (Dec 2025) | Semantic genetic operators for prompts | Single-agent focus; no topology evolution |
| **SWE-Effi** (Sep 2025) | Evaluate agents under resource constraints | Evaluation only—no evolution |
| **CoRL** (Nov 2025) | RL for budget-aware coordination | Learns policy, not architecture |
| **MALBO** (Nov 2025) | Bayesian optimization for team composition | Pareto tradeoff, not constraint as pressure |

**Our contribution:** First study of how resource constraints *during evolution* shape multi-agent architectures. We treat budget not as a metric to optimize, but as the *environment* that shapes adaptation.

### Core Hypothesis

> **"Poverty breeds innovation."** Multi-agent architectures evolved under strict resource constraints will develop qualitatively different strategies than unconstrained evolution—and these strategies may transfer advantageously to resource-abundant settings.

---

## II. THEORETICAL FOUNDATION

### Biological Analogy

In evolutionary biology, organisms adapted to resource-scarce environments exhibit distinct characteristics:
- **Smaller body size** (island dwarfism)
- **Higher metabolic efficiency** (desert adaptation)
- **Specialized niches** (extreme specialization under constraint)
- **Communication compression** (reduced signaling in harsh environments)

We hypothesize analogous phenomena in multi-agent LLM evolution:
- **Fewer agents** (architectural simplicity)
- **Token efficiency** (compressed prompts and messages)
- **Role specialization** (emergent division of labor)
- **Fail-fast strategies** (avoiding "expensive failures")

### Connection to Prior Findings

**SWE-Effi (Fan et al., 2025)** documented two failure patterns in existing agents:
1. **Token snowball:** Harder tasks consume disproportionately more tokens
2. **Expensive failures:** Agents waste resources on unsolvable problems

We hypothesize that evolution under budget constraints will naturally select against these patterns.

---

## III. RESEARCH DESIGN

### A. Primary Hypotheses

**H1 (Structural Differentiation):** Architectures evolved under tight constraints (2K tokens) differ structurally from unconstrained evolution in: agent count, topology, role distribution, and prompt length.

**H2 (Failure Avoidance):** Constraint-evolved architectures exhibit significantly fewer "expensive failures" and reduced "token snowball" effects compared to unconstrained-evolved architectures.

**H3 (Transfer to Abundance):** When given unlimited resources, constraint-evolved architectures match or exceed the performance of unconstrained-evolved architectures.

**H4 (Emergent Efficiency):** Constraint evolution produces emergent communication compression and role specialization without explicit pressure for these properties.

### B. Experimental Design

#### Independent Variable: Resource Budget During Evolution

| Regime | Token Budget | Rationale |
|--------|--------------|-----------|
| **Tight** | 2,000 tokens | Severe constraint; forces minimal communication |
| **Medium** | 5,000 tokens | Moderate; allows structured collaboration |
| **Loose** | 10,000 tokens | Mild; most reasonable architectures fit |
| **Unconstrained** | ∞ | Baseline; no limit |

#### Dependent Variables

1. **Architecture structure:** # agents, # edges, topology type, role distribution
2. **Task performance:** pass@1, pass@5 on code generation benchmarks
3. **Cost efficiency:** tokens per solved problem
4. **Failure patterns:** tokens spent on failed problems, correlation with difficulty
5. **Generalization:** transfer to unseen benchmarks

#### Control Variables

- Base LLM: GPT-4o-mini (consistent across all conditions)
- Benchmarks: HumanEval, MBPP (primary); SWE-bench-lite (transfer)
- Evolution parameters: Fixed across regimes (population=20, generations=50)
- Random seeds: 5 independent runs per condition

### C. Fitness Function

The fitness function instantiates a **hard constraint** formulation:

```python
def fitness(architecture: Architecture, benchmark: List[Task], budget: int) -> float:
    """
    Hard constraint fitness: architectures exceeding budget get zero fitness.
    This differs from soft Pareto approaches where cost is a secondary objective.
    """
    results = []
    for task in benchmark:
        output, tokens_used = execute_with_tracking(architecture, task)
        if tokens_used > budget:
            return 0.0  # Hard constraint violation
        results.append((output, tokens_used))
    
    # Accuracy on tasks that fit within budget
    accuracy = mean([passes_tests(r[0], task) for r, task in zip(results, benchmark)])
    return accuracy
```

**Rationale:** The hard constraint forces evolution to *adapt* to resource limits, rather than merely trading off against them. Architectures that cannot survive within budget are eliminated, creating genuine selective pressure.

### D. Genome Representation

```python
@dataclass
class MultiAgentGenome:
    """Evolvable multi-agent architecture specification."""
    
    # Structural genes
    num_agents: int                          # 1-5 agents
    agent_roles: List[AgentRole]             # e.g., [planner, coder, reviewer]
    topology: AdjacencyMatrix                # Who communicates with whom
    aggregation_strategy: str                # "voting", "sequential", "hierarchical"
    
    # Behavioral genes (per agent)
    system_prompts: Dict[AgentRole, str]     # Role-specific prompts
    max_response_tokens: Dict[AgentRole, int] # Per-agent token limits
    tool_access: Dict[AgentRole, List[Tool]] # Which tools each role can use
    
    # Communication genes
    message_format: str                      # "freeform", "structured", "minimal"
    max_message_length: int                  # Inter-agent message limit
    
    # Meta genes
    early_exit_threshold: float              # Confidence for early termination
    max_rounds: int                          # Maximum communication rounds
```

### E. Genetic Operators

**Mutation Operators:**
- **Add agent:** Insert new agent with random role and connections
- **Remove agent:** Delete agent and rewire topology
- **Rewire topology:** Add/remove edges between agents
- **Prompt mutation:** LLM-guided rewriting ("make this more concise")
- **Token budget mutation:** Adjust per-agent response limits
- **Role change:** Swap agent role type

**Crossover Operators:**
- **Subgraph exchange:** Swap connected components between parents
- **Prompt recombination:** Blend prompts from same-role agents
- **Parameter averaging:** Average numeric parameters

**Selection:** Tournament selection (k=3) with elitism (top 2)

---

## IV. EXPERIMENTAL PROTOCOL

### Phase 1: Infrastructure (Week 1)

**Objectives:**
1. Implement genome representation with serialization
2. Build token-counted evaluation harness
3. Implement sandbox code execution
4. Verify budget enforcement

**Validation:** 
- Unit tests for all components
- Manual verification on 5 HumanEval problems

### Phase 2: Evolution Engine (Week 2)

**Objectives:**
1. Implement genetic operators
2. Build selection and population management
3. Create evolution loop with logging
4. Parallelize fitness evaluation

**Validation:**
- Evolution produces valid offspring
- Fitness tracking works correctly
- Logging captures all required data

### Phase 3: Main Experiments (Week 3-4)

**Experiment 1: Constraint Regime Comparison**
- 4 budget regimes × 5 seeds = 20 independent evolution runs
- 50 generations per run
- Track: structure, fitness, diversity per generation

**Experiment 2: Expensive Failure Analysis**
- Take best architecture from each regime
- Run on full benchmark with per-problem token tracking
- Classify problems by difficulty (based on baseline performance)
- Measure: tokens on failures, token snowball coefficient

**Experiment 3: Transfer to Abundance**
- Take all evolved architectures
- Evaluate with unlimited budget
- Compare: constraint-evolved vs. unconstrained-evolved

**Experiment 4: Cross-Benchmark Transfer**
- Evolve on HumanEval → test on MBPP
- Evolve on MBPP → test on HumanEval
- Measure zero-shot transfer performance

### Phase 4: Analysis (Week 5)

**Quantitative Analysis:**
- Statistical tests (t-tests, ANOVA) with effect sizes
- Bootstrap confidence intervals
- Multiple comparison corrections (Bonferroni)

**Qualitative Analysis:**
- Emergent communication patterns
- Role differentiation analysis
- Architecture visualization and clustering

### Phase 5: Paper Writing (Week 6-7)

**Sections:**
1. Introduction: Frame the "poverty breeds innovation" hypothesis
2. Related Work: Position against SWE-Effi, CoRL, EvoAgentX, etc.
3. Method: Genome representation, fitness function, evolution
4. Experiments: Four main experiments with analysis
5. Discussion: Implications, limitations, future work
6. Conclusion: Summarize contributions

### Phase 6: Revision & Submission (Week 8)

---

## V. BASELINES & COMPARISONS

### Baseline Systems

1. **Single-Agent Baseline:** GPT-4o-mini with standard code generation prompt
2. **Fixed Multi-Agent:** 3-agent system (planner + coder + reviewer) with hand-tuned prompts
3. **Unconstrained Evolution:** Architecture evolved without budget constraint (our control)
4. **Random Architecture:** Randomly sampled architectures (ablation)

### Key Comparisons

| Comparison | Tests |
|------------|-------|
| Constrained-evolved vs. Unconstrained-evolved | H1, H2, H3 |
| Constrained-evolved vs. Fixed Multi-Agent | Practical improvement |
| Constrained-evolved vs. Single-Agent | Value of multi-agent |
| Cross-regime comparison | Effect of constraint severity |

---

## VI. STATISTICAL RIGOR

### Requirements

- **Effect sizes:** Report Cohen's d for all comparisons
- **Confidence intervals:** 95% bootstrap CIs
- **Multiple comparisons:** Bonferroni correction
- **Full distributions:** Box plots, not just means
- **Reproducibility:** 5 seeds per condition, report variance

### Pre-Registration

Before running experiments:
1. Register hypotheses on OSF
2. Specify primary outcomes
3. Define analysis plan
4. Lock experimental protocol

---

## VII. PAPER STRUCTURE

### Title
> "Scarcity Breeds Efficiency: Resource-Constrained Evolution of Multi-Agent Programming Architectures"

### Abstract (250 words)
- Context: Evolutionary optimization of multi-agent LLM systems
- Gap: Existing work treats cost as secondary objective
- Method: Hard constraint evolution with varying budgets
- Results: Constraint-evolved architectures are structurally different, avoid expensive failures, transfer to abundance
- Contribution: First study of constraint as evolutionary pressure

### Section Outline

1. **Introduction** (2 pages)
   - Hook: Counter-intuitive finding (constraint improves transfer)
   - Research questions
   - Contributions

2. **Related Work** (2 pages)
   - Evolutionary LLM optimization: EvoAgentX, ARTEMIS, AFlow
   - Resource efficiency: SWE-Effi, CoRL, Curriculum Learning
   - Multi-agent foundations: AutoGen, MetaGPT

3. **Method** (3 pages)
   - Problem formulation
   - Genome representation
   - Fitness function (hard constraint)
   - Evolutionary operators
   - Benchmarks

4. **Experiments** (4 pages)
   - RQ1: Structural differences
   - RQ2: Failure avoidance
   - RQ3: Transfer to abundance
   - RQ4: Cross-benchmark transfer

5. **Analysis** (2 pages)
   - Emergent communication strategies
   - Role differentiation patterns
   - What makes constraint-evolved architectures work

6. **Discussion** (1.5 pages)
   - Implications for practice
   - Limitations
   - Future work

7. **Conclusion** (0.5 pages)

### Key Figures

- **Figure 1:** Method overview diagram
- **Figure 2:** Structural comparison across regimes (violin plots)
- **Figure 3:** Expensive failure analysis (tokens vs. difficulty)
- **Figure 4:** Transfer to abundance results (bar chart)
- **Figure 5:** Evolved architecture examples (topology visualizations)
- **Figure 6:** Generalization to SWE-bench

---

## VIII. IMPLEMENTATION ARCHITECTURE

### Directory Structure

```
E_M_A_P/
├── paper/
│   ├── LITERATURE_REVIEW.md      # ✓ Complete
│   ├── RESEARCH_PLAN.md          # ✓ Complete
│   ├── main.tex                  # ✓ Skeleton
│   ├── references.bib            # ✓ 23 references
│   └── figures/                  # To be created
│
├── src/
│   ├── genome/
│   │   ├── representation.py     # MultiAgentGenome dataclass
│   │   ├── operators.py          # Mutation, crossover
│   │   └── serialization.py      # Save/load genomes
│   │
│   ├── evolution/
│   │   ├── fitness.py            # Token-counted evaluation
│   │   ├── selection.py          # Tournament, elitism
│   │   └── loop.py               # Main evolution loop
│   │
│   ├── agents/
│   │   ├── executor.py           # Run multi-agent architecture
│   │   ├── sandbox.py            # Code execution sandbox
│   │   └── token_counter.py      # Track API usage
│   │
│   ├── benchmarks/
│   │   ├── humaneval.py          # HumanEval loader
│   │   ├── mbpp.py               # MBPP loader
│   │   └── evaluator.py          # pass@k evaluation
│   │
│   └── analysis/
│       ├── structural.py         # Architecture metrics
│       ├── failure.py            # Expensive failure analysis
│       └── visualization.py      # Plotting
│
├── experiments/
│   ├── configs/                  # YAML experiment configs
│   ├── logs/                     # Run logs
│   └── results/                  # CSV/JSON outputs
│
├── notebooks/
│   ├── 01_exploration.ipynb      # Initial data exploration
│   ├── 02_analysis.ipynb         # Statistical analysis
│   └── 03_figures.ipynb          # Publication figures
│
├── tests/
│   ├── test_genome.py
│   ├── test_evolution.py
│   └── test_evaluation.py
│
├── pyproject.toml                # Project configuration
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
└── FULL_PLAN.md                  # This document
```

### Technology Stack

**Core:**
- Python 3.11+
- DEAP (evolutionary algorithms)
- OpenAI API (GPT-4o-mini)
- tiktoken (token counting)

**Execution:**
- Docker (code sandboxing)
- asyncio (parallel evaluation)

**Analysis:**
- pandas, numpy (data processing)
- scipy.stats (statistical tests)
- matplotlib, seaborn (visualization)

**Experiment Tracking:**
- MLflow or W&B (optional)

---

## IX. RISK MITIGATION

### High Risk

**Risk:** Evolution is computationally expensive  
**Impact:** Budget overrun, incomplete experiments  
**Mitigation:** 
- Use subset of benchmark for fitness (20%)
- Full benchmark only for final evaluation
- Early stopping if fitness plateaus
- Estimate: 20 pop × 50 gen × 4 regimes × 5 seeds × 0.2 × 164 problems ≈ 66,000 evaluations

**Risk:** No interesting structural differences emerge  
**Impact:** Weak contribution  
**Mitigation:**
- This is still a finding ("constraints don't shape architecture")
- Pivot to analyzing why evolution is robust to constraints

### Medium Risk

**Risk:** Evolved architectures don't outperform baselines  
**Impact:** Negative result  
**Mitigation:**
- Honest reporting
- Focus on qualitative differences
- "Poverty breeds efficiency, not superiority"

---

## X. SUCCESS CRITERIA

### Minimum Viable Paper

1. Clear structural differences between constraint regimes (H1)
2. Statistical significance on at least 2 of 4 hypotheses
3. Reproducible results (5 seeds, CIs reported)
4. Proper positioning against related work

### Strong Paper

All of above, plus:
5. Transfer to abundance effect (H3 confirmed)
6. Novel insights about emergent patterns
7. Generalization to SWE-bench

### Exceptional Paper

All of above, plus:
8. Discovered patterns transfer across benchmarks
9. Practical guidelines for practitioners
10. Interesting negative results that inform future work

---

## XI. CITATIONS (Core 18)

### Evolutionary LLM Optimization
1. EvoAgentX (Wang et al., 2025) - arXiv:2507.03616
2. ARTEMIS (Brookes et al., 2025) - arXiv:2512.09108
3. AFlow (Zhang et al., 2024) - arXiv:2410.10762
4. AutoMaAS (Ma et al., 2025) - arXiv:2510.02669
5. AgentNet (Yang et al., 2025) - arXiv:2504.00587
6. MALBO (Sabbatella, 2025) - arXiv:2511.11788

### Resource Efficiency
7. SWE-Effi (Fan et al., 2025) - arXiv:2509.09853
8. CoRL (Jin et al., 2025) - arXiv:2511.02755
9. Curriculum for Constraints (Tzannetos et al., 2025) - NeurIPS'25

### Benchmarks
10. HumanEval (Chen et al., 2021)
11. MBPP (Austin et al., 2021)
12. SWE-bench (Jimenez et al., 2024)

### Multi-Agent Foundations
13. AutoGen (Wu et al., 2023)
14. MetaGPT (Hong et al., 2023)
15. CAMEL (Li et al., 2023)

### Evolutionary Computation
16. DEAP (Fortin et al., 2012)
17. MAP-Elites (Mouret & Clune, 2015)
18. Regularized Evolution (Real et al., 2019)

---

## XII. IMMEDIATE NEXT STEPS

### Completed ✅

1. ✅ Literature review complete (18 papers)
2. ✅ Research direction validated (resource-constrained evolution is novel)
3. ✅ Paper skeleton created (main.tex with all sections)
4. ✅ Bibliography complete (23 references)
5. ✅ Python project structure set up
6. ✅ Genome representation implemented (`MultiAgentGenome`, `AgentGene`)
7. ✅ Genetic operators implemented (7 mutation types + crossover)
8. ✅ Selection operators implemented (tournament, roulette, elitist)
9. ✅ Token-counted fitness evaluation implemented
10. ✅ Agent executor implemented (LLM backends, message routing)
11. ✅ Code sandbox implemented (safe execution, security checks)
12. ✅ Benchmark loaders implemented (HumanEval with placeholders)
13. ✅ Integrated evaluator connecting all components
14. ✅ Demo script working (mock backend)
15. ✅ 44 unit tests passing

### Next Steps ⬜

1. ⬜ Set OPENAI_API_KEY and test with real LLM
2. ⬜ Download full HumanEval dataset
3. ⬜ Run pilot evolution (1 regime, 1 seed) with real LLM
4. ⬜ Full experimental runs (4 regimes × 5 seeds)
5. ⬜ Statistical analysis of results
6. ⬜ Complete paper writing

### Current Demo Status

The demo (`examples/demo_evolution.py`) runs successfully:
```
$ python examples/demo_evolution.py --generations 5 --population 10 --seed 42

Generation 4: Best fitness: 0.000, Diversity: 0.600
```

**Why 0% fitness?** Mock backend generates placeholder code. Real experiments need OpenAI API.

---

*This document serves as the authoritative research specification. It reflects validated novelty claims and proper academic positioning based on systematic literature review.*

**Last Updated:** December 15, 2025
**Infrastructure Status:** Complete and tested (44/44 tests passing)
