# RecursiveMAS WorkBuddy Skill

[![arXiv](https://img.shields.io/badge/arXiv-2604.25917-b31b1b)](https://arxiv.org/abs/2604.25917)

RecursiveMAS is a WorkBuddy skill that implements the **Recursive Multi-Agent Systems** pattern from arXiv:2604.25917 (UIUC, Stanford, NVIDIA, MIT).

## What is RecursiveMAS?

Traditional multi-agent systems communicate through text — Agent A generates text, Agent B reads it, encodes it back, processes it, and generates more text. This is slow and token-inefficient.

RecursiveMAS extends **Recursive Language Models (RLM)** to multi-agent systems: agents communicate through direct **latent space** transfers using lightweight **RecursiveLink** modules, skipping the decode-re-encode cycle.

**Measured benefits** (9 benchmarks):
- +8.3% accuracy
- 1.2x–2.4x faster inference
- 34.6%–75.6% fewer tokens

## Installation

### Prerequisites
- [WorkBuddy](https://www.codebuddy.cn) installed
- GitHub CLI (`gh`) or manual download

### Install via GitHub
```bash
cd ~/.workbuddy/skills/
git clone https://github.com/2763151198-coder/recursive-mas-skill.git recursive-mas
```

### Install from .skill file
Place `recursive-mas.skill` in any directory, then:
```bash
mkdir -p ~/.workbuddy/skills/recursive-mas
unzip path/to/recursive-mas.skill -d ~/.workbuddy/skills/recursive-mas
```

## Usage

Load the skill in WorkBuddy:
```
Skill(skill="recursive-mas")
```

Once loaded, use these commands:

| Command | Action |
|---------|--------|
| `/mas:sequential` | Chain agents: Planner → Critic → Solver |
| `/mas:mixture` | Parallel specialists → Summarizer |
| `/mas:distillation` | Expert ↔ Learner teacher-student |
| `/mas:deliberation` | Reflector ↔ Tool-Caller |
| `/mas:demo` | Run RecursiveLink simulation (no GPU) |
| `/mas:scaffold` | Generate PyTorch project |

### Quick Demo (no GPU required)
```bash
python3 ~/.workbuddy/skills/recursive-mas/scripts/demo_recursivelink.py
```

### Generate a PyTorch Project
```bash
python3 ~/.workbuddy/skills/recursive-mas/scripts/scaffold.py my_project --pattern mixture
```

## Collaboration Patterns

### 1. Sequential
Chain of agents: `Planner → Critic → Solver`. Each agent builds on the previous output.
Best for: step-by-step reasoning, math problem solving, code review.

### 2. Mixture
Parallel specialists: Multiple domain specialists work simultaneously, then a Summarizer aggregates.
Best for: multi-domain problems, large-scale parallel tasks.

### 3. Distillation
Teacher-student: Expert ↔ Learner pair. Learner improves by analyzing Expert's approach.
Best for: code optimization, solution improvement, knowledge transfer.

### 4. Deliberation
Reflection + tools: Reflector checks output for errors, Tool-Caller invokes external tools.
Best for: tasks requiring web search, code execution, or file operations.

## Project Structure

```
~/.workbuddy/skills/recursive-mas/
├── SKILL.md                       # Main skill definition
├── scripts/
│   ├── demo_recursivelink.py      # Quick numpy demo (no GPU)
│   └── scaffold.py                # PyTorch project generator
└── references/
    ├── implementation.md          # PyTorch implementation guide
    ├── theory.md                  # Mathematical foundations
    └── comparison.md              # Comparison with other MAS approaches
```

## Paper Reference

- **Title**: Recursive Multi-Agent Systems
- **Authors**: Xiyuan Yang, Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, Jindong Jiang, Hanghang Tong, Tong Zhang, Markus J. Buehler, Jingrui He, James Zou
- **Affiliations**: UIUC, Stanford, NVIDIA, MIT
- **arXiv**: [2604.25917](https://arxiv.org/abs/2604.25917)
- **Project Page**: https://recursivemas.github.io

## License

MIT