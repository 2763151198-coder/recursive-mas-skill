---
name: recursive-mas
description: |
  Recursive Multi-Agent Systems skill based on arXiv:2604.25917.
  Extends recursive computation from single model to multi-agent collaboration.
  Four collaboration patterns: Sequential, Mixture, Distillation, Deliberation.
read_when:
  - User mentions RecursiveMAS, recursive multi-agent, latent space reasoning
  - Task involves multi-agent coordination in WorkBuddy
  - Need to design agent collaboration workflows
  - Comparing text-mediated MAS vs latent-space MAS
  - Building multi-agent systems with RecursiveLink modules
---

# RecursiveMAS Skill — WorkBuddy Edition

Based on arXiv:2604.25917 — UIUC, Stanford, NVIDIA, MIT.

## Core Idea

Treat the entire MAS as a unified latent-space recursive computation.
Instead of Agent A → decode to text → Agent B → encode from text,
RecursiveMAS uses lightweight **RecursiveLink** modules to pass latent states
directly between agents, saving 34.6%–75.6% tokens and achieving 8.3% accuracy gain.

```
A1 →[R_out]→ A2 →[R_out]→ ... →[R_out]→ An
     (latent vector)           (latent vector)
```

## WorkBuddy Commands

The following commands are available when this skill is loaded:

| Command | Effect |
|---------|--------|
| `/mas:sequential` | Chain agents: Planner → Critic → Solver (step-by-step reasoning) |
| `/mas:mixture` | Parallel specialists + Summarizer (multi-domain synthesis) |
| `/mas:distillation` | Expert ↔ Learner pair (knowledge distillation) |
| `/mas:deliberation` | Reflector ↔ Tool-Caller (deliberative reasoning with tools) |
| `/mas:demo` | Run an interactive RecursiveLink simulation (no GPU needed) |
| `/mas:scaffold` | Generate a RecursiveMAS PyTorch project |

## Four Collaboration Patterns

### 1. Sequential (Sequential)
Chain: Agent1 → Agent2 → Agent3, each building on previous output.

**Use when**: step-by-step reasoning, math problem solving, code review pipeline.

**WorkBuddy usage**:
```
User asks a multi-step question
→ Agent A (Planner) decomposes the task
→ Passes result to Agent B (Critic) for verification
→ Passes to Agent C (Solver) for final solution
```

### 2. Mixture (Mixture)
Parallel: Multiple Domain Specialists work simultaneously, then a Summarizer aggregates.

**Use when**: multi-domain problems (code + math + science), large-scale parallel tasks.

**WorkBuddy usage**:
```
User asks a complex multi-domain task
→ Spawn N Domain Specialist agents in parallel (each handles one domain)
→ Each specialist processes independently and writes results to files
→ Summarizer agent reads all results and produces final output
```

### 3. Distillation (Distillation)
Teacher-Student: Expert agent and Learner agent work in pair, learner improves from expert.

**Use when**: code optimization, solution improvement, knowledge transfer.

**WorkBuddy usage**:
```
User provides a solution
→ Expert Agent (large model) produces an optimized version
→ Learner Agent (smaller model) analyzes the differences
→ Learner refines its approach and outputs improved result
```

### 4. Deliberation (Deliberation)
Reflection + Tools: Agent reflects on its own output, uses tools to verify/correct, then produces final.

**Use when**: tasks requiring tool use (web search, code execution, file operations).

**WorkBuddy usage**:
```
Agent produces initial output
→ Reflector checks for errors/gaps
→ Tool-Caller invokes external tools (Search, Bash, etc.) to fill gaps
→ Combines reflection and tool results into final answer
```

## Quick Start Demo

Load this skill and run:
```
/mas:demo
```

This runs a Python simulation of RecursiveLink (no GPU, no PyTorch):

```bash
python3 scripts/demo_recursivelink.py
```

It demonstrates:
- Inner RecursiveLink (agent internal latent thought loop)
- Outer RecursiveLink (agent-to-agent latent state transfer)
- Comparison with text-mediated communication (token savings)
- Visual ASCII output of the process

## Generating a PyTorch Project

For actual training (with GPU), run:
```
/mas:scaffold
```

Or manually:
```bash
python3 scripts/scaffold.py my_project --pattern mixture
```

## Paper Reference

- **Title**: Recursive Multi-Agent Systems
- **arXiv**: 2604.25917
- **Project**: https://recursivemas.github.io