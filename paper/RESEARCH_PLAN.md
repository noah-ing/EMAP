# EMAP Research Plan

**Project:** Scarcity Breeds Efficiency: Resource-Constrained Evolution of Multi-Agent Programming Architectures  
**Target Venue:** ICML/NeurIPS/ICLR (Fall 2025 deadline cycle)  
**Status:** Literature Review Complete → Experiment Design Phase

---

## Current Phase: Experiment Design

### Validated Research Gaps (see LITERATURE_REVIEW.md)
1. **Nobody treats resource constraints as evolutionary pressure** - existing work (SWE-Effi, CoRL, MALBO) either evaluates under constraints or optimizes for cost as secondary objective
2. **Nobody studies what architectures EMERGE from constraint evolution** - only what happens when you constrain existing architectures
3. **The "poverty breeds innovation" hypothesis is testable** - do constraint-evolved systems develop qualitatively different strategies?

---

## Experimental Design

### Primary Experiments

#### Experiment 1: Constraint Regime Comparison
**Goal:** Show that evolution under different budgets produces structurally different architectures

**Setup:**
- Evolve architectures under 4 budget regimes: 2K, 5K, 10K, unlimited tokens
- 5 independent runs per regime (different seeds)
- Population size: 20, Generations: 50
- Base LLM: GPT-4o-mini (cost control)

**Metrics:**
- Structural: # agents, # edges, topology type (star, chain, mesh), role distribution
- Performance: pass@1 on held-out eval split
- Cost: avg tokens per problem

**Hypothesis:** Tight-budget evolution produces simpler, more specialized architectures

---

#### Experiment 2: Expensive Failure Analysis
**Goal:** Test whether constraint-evolved architectures avoid "expensive failures" (SWE-Effi phenomenon)

**Setup:**
- Take best architecture from each regime
- Run on full benchmark with per-problem token tracking
- Classify problems as easy/medium/hard based on baseline single-agent performance

**Metrics:**
- Tokens spent on eventually-failed problems
- Tokens spent on hard problems vs. easy problems
- "Token snowball" coefficient (correlation between difficulty and tokens)

**Hypothesis:** Tight-budget-evolved architectures "fail fast" rather than persisting on unsolvable problems

---

#### Experiment 3: Transfer to Abundance
**Goal:** Test whether constraint-evolved architectures remain competitive when given unlimited resources

**Setup:**
- Take evolved architectures from each regime
- Evaluate ALL architectures under unlimited budget

**Metrics:**
- Pass@1 on HumanEval, MBPP
- Comparison: constraint-evolved vs unconstrained-evolved when both have unlimited resources

**Hypothesis:** Constraint-evolved architectures will match or exceed unconstrained performance (the "frugal generalization" effect)

---

#### Experiment 4: Cross-Benchmark Transfer
**Goal:** Test generalization of evolved architectures

**Setup:**
- Evolve on HumanEval, test on MBPP (and vice versa)
- Evolve on HumanEval+MBPP combined, test on SWE-bench-lite subset

**Metrics:**
- Zero-shot transfer accuracy (no re-evolution)
- Performance drop vs. within-benchmark evolution

**Hypothesis:** Constraint-evolved architectures may transfer better (learned general efficiency strategies)

---

### Analysis Components

#### A1: Emergent Communication Strategies
- Analyze inter-agent message content in tight vs. loose regimes
- Measure: message length, information density, redundancy
- Look for: compression, abbreviation, structured formats

#### A2: Role Differentiation
- Cluster evolved agents by behavior (prompt similarity, tool usage)
- Compare role diversity across regimes
- Question: Do tight budgets force more OR less specialization?

#### A3: Architecture Visualization
- Generate DAG visualizations of evolved topologies
- Identify common motifs (star, chain, hierarchical)
- Track evolution of topology over generations

---

## Implementation Requirements

### Core Components Needed:
1. **Genome representation** - encode architecture as mutable data structure
2. **Fitness evaluation** - run architecture on benchmark, measure accuracy + cost
3. **Genetic operators** - mutation (add/remove agent, rewire, modify prompt), crossover
4. **Evolution loop** - selection, variation, evaluation
5. **Budget enforcement** - hard cutoff when token limit reached
6. **Logging** - per-generation metrics, architecture snapshots

### Technical Decisions:
- **Framework:** Build on AutoGen or custom (AutoGen may be too heavy)
- **LLM API:** OpenAI GPT-4o-mini (balance of cost/capability)
- **Token counting:** tiktoken for accurate budget tracking
- **Evolution library:** DEAP or custom (DEAP is well-tested)
- **Parallelization:** Ray for distributed fitness evaluation

---

## Timeline

| Week | Tasks |
|------|-------|
| 1 | Implement genome representation + basic fitness evaluation |
| 2 | Implement genetic operators + evolution loop |
| 3 | Run Experiment 1 (constraint regime comparison) |
| 4 | Run Experiment 2 (expensive failure analysis) |
| 5 | Run Experiments 3-4 (transfer studies) |
| 6 | Analysis components (A1-A3) |
| 7 | Write paper draft |
| 8 | Polish, ablations, supplementary |

---

## Risk Analysis

### High Risk:
- **Evolution is expensive** - 20 individuals × 50 generations × 4 regimes × 5 runs = 20,000 evaluations
- **Mitigation:** Use subset of benchmark for fitness, full benchmark only for final eval

### Medium Risk:
- **No interesting structure emerges** - all regimes produce similar architectures
- **Mitigation:** This is also a finding! "Budget doesn't significantly constrain evolution"

### Low Risk:
- **Constraint-evolved architectures underperform** - poverty doesn't breed innovation, just poverty
- **Mitigation:** Honest reporting of negative result; still contributes to understanding

---

## Next Steps (Immediate)

1. [ ] Set up project structure with proper Python packaging
2. [ ] Implement genome representation for multi-agent architectures
3. [ ] Implement token-counted evaluation harness
4. [ ] Test on 5 HumanEval problems before scaling
5. [ ] Verify budget enforcement is working correctly

---

*Last updated: December 15, 2025*
