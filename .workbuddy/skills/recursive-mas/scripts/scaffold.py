#!/usr/bin/env python3
"""RecursiveMAS 项目脚手架 — 一键生成项目结构和配置文件"""

import argparse
import json
import os
import sys
from pathlib import Path

TEMPLATE_DIR = "recursivemas_project"

AGENT_CONFIGS = {
    "sequential-light": {
        "pattern": "sequential",
        "agents": [
            {"role": "planner", "model": "Qwen/Qwen3-1.7B"},
            {"role": "critic", "model": "meta-llama/Llama-3.2-1B-Instruct"},
            {"role": "solver", "model": "Qwen/Qwen2.5-Math-1.5B-Instruct"},
        ],
    },
    "sequential-scaled": {
        "pattern": "sequential",
        "agents": [
            {"role": "planner", "model": "google/gemma-3-4b-it"},
            {"role": "critic", "model": "meta-llama/Llama-3.2-3B-Instruct"},
            {"role": "solver", "model": "Qwen/Qwen3.5-4B"},
        ],
    },
    "mixture": {
        "pattern": "mixture",
        "agents": [
            {"role": "code_specialist", "model": "Qwen/Qwen2.5-Coder-3B-Instruct"},
            {"role": "science_specialist", "model": "BioMistral/BioMistral-7B"},
            {"role": "math_specialist", "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
            {"role": "summarizer", "model": "Qwen/Qwen3.5-2B"},
        ],
    },
    "distillation": {
        "pattern": "distillation",
        "agents": [
            {"role": "learner", "model": "Qwen/Qwen3.5-4B"},
            {"role": "expert", "model": "Qwen/Qwen3.5-9B"},
        ],
    },
    "deliberation": {
        "pattern": "deliberation",
        "agents": [
            {"role": "reflector", "model": "Qwen/Qwen3.5-4B"},
            {"role": "tool_caller", "model": "Qwen/Qwen3.5-4B"},
        ],
    },
}

TRAINING_CONFIG = {
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "scheduler": "cosine",
    "batch_size": 4,
    "inner_loop_epochs": 3,
    "outer_loop_epochs": 5,
    "recursion_rounds": 2,
    "latent_steps": 4,
    "inference_temperature": 0.6,
    "inference_top_p": 0.95,
    "max_new_tokens": 512,
}

REQUIREMENTS = [
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "wandb>=0.16.0",
]


def generate_agent_config(pattern_name: str) -> dict:
    return AGENT_CONFIGS.get(pattern_name, AGENT_CONFIGS["sequential-light"])


def generate_train_script(project_dir: Path, config: dict):
    """生成训练启动脚本"""
    script = f'''#!/usr/bin/env python3
"""RecursiveMAS 训练脚本 — 自动生成"""

import torch
from pathlib import Path

# 项目配置
PROJECT_DIR = Path("{project_dir}")
AGENT_CONFIG = {json.dumps(config, indent=4, ensure_ascii=False)}
TRAINING_CONFIG = {json.dumps(TRAINING_CONFIG, indent=4)}

def main():
    print("=" * 60)
    print("RecursiveMAS Training Pipeline")
    print("=" * 60)
    print(f"Pattern: {{AGENT_CONFIG['pattern']}}")
    print(f"Agents: {{len(AGENT_CONFIG['agents'])}}")
    print(f"Recursion rounds: {{TRAINING_CONFIG['recursion_rounds']}}")
    print()
    
    # Step 1: Inner-Loop Training
    print("[Stage 1] Inner-Loop Training (per-agent warm-start)")
    for agent_info in AGENT_CONFIG["agents"]:
        print(f"  Training ℛ_in for {{agent_info['role']}} ({{agent_info['model']}})")
        # TODO: Implement inner-loop training
    print()
    
    # Step 2: Outer-Loop Training
    print("[Stage 2] Outer-Loop Training (system-level co-optimization)")
    print(f"  Training {{len(AGENT_CONFIG['agents'])}} outer RecursiveLinks")
    print(f"  Recursion rounds: {{TRAINING_CONFIG['recursion_rounds']}}")
    # TODO: Implement outer-loop training
    print()
    
    print("Training complete! Check outputs in", PROJECT_DIR / "checkpoints")

if __name__ == "__main__":
    main()
'''
    return script


def main():
    parser = argparse.ArgumentParser(description="RecursiveMAS project scaffold")
    parser.add_argument("name", nargs="?", default="my_recursivemas", help="Project name")
    parser.add_argument(
        "--pattern",
        choices=list(AGENT_CONFIGS.keys()),
        default="sequential-light",
        help="Collaboration pattern",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    project_dir = Path(args.output_dir) / args.name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Agent 配置
    agent_config = generate_agent_config(args.pattern)

    # 写入配置文件
    config_path = project_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"pattern": agent_config["pattern"], "agents": agent_config["agents"], "training": TRAINING_CONFIG},
            f, indent=2, ensure_ascii=False,
        )
    print(f"✅ 配置文件: {config_path}")

    # 写入训练脚本
    train_script = project_dir / "train.py"
    with open(train_script, "w", encoding="utf-8") as f:
        f.write(generate_train_script(str(project_dir), agent_config))
    print(f"✅ 训练脚本: {train_script}")

    # 写入 requirements
    req_path = project_dir / "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REQUIREMENTS) + "\n")
    print(f"✅ 依赖文件: {req_path}")

    # 创建目录结构
    for d in ["checkpoints", "data", "logs"]:
        (project_dir / d).mkdir(exist_ok=True)
    print(f"✅ 目录结构: checkpoints/, data/, logs/")

    # 写入 README
    readme = project_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"""# {args.name}

RecursiveMAS 项目 — {agent_config['pattern']} 协作模式

## 协作模式
{agent_config['pattern']}

## Agent 配置
""")
        for a in agent_config["agents"]:
            f.write(f"- **{a['role']}**: {a['model']}\n")
        f.write(f"""
## 训练

```bash
pip install -r requirements.txt
python train.py
```

## 论文
- arXiv:2604.25917
- 项目页: https://recursivemas.github.io
""")
    print(f"✅ 说明文件: {readme}")

    print(f"\n🎉 项目创建完成: {project_dir}")
    print(f"   协作模式: {args.pattern}")
    print(f"   Agent 数量: {len(agent_config['agents'])}")


if __name__ == "__main__":
    main()
