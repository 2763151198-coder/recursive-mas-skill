# RecursiveMAS WorkBuddy Skill — `.workbuddy/skills/` directory

This directory contains the skill files for WorkBuddy.

## Structure

```
.workbuddy/skills/recursive-mas/
├── SKILL.md                       # Main skill definition
├── scripts/
│   ├── demo_recursivelink.py      # Quick numpy demo (no GPU)
│   └── scaffold.py                # PyTorch project generator
└── references/
    ├── implementation.md          # PyTorch implementation guide
    ├── theory.md                  # Mathematical foundations
    └── comparison.md              # Comparison with other MAS approaches
```

## Installation

Place this directory under `~/.workbuddy/skills/`:

```bash
mkdir -p ~/.workbuddy/skills/
cp -r .workbuddy/skills/recursive-mas ~/.workbuddy/skills/
```

Or install as a .skill package:
```bash
cd ~/.workbuddy/skills/
git clone https://github.com/2763151198-coder/recursive-mas-skill.git recursive-mas
```

## Usage

In WorkBuddy, load the skill:
```
Skill(skill="recursive-mas")
```