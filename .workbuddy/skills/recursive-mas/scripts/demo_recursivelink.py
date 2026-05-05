#!/usr/bin/env python3
"""RecursiveMAS Demo — Quick simulation of RecursiveLink (no GPU needed)"""

import numpy as np
import time

# ---------- Config ----------
HIDDEN_DIM = 64
LATENT_STEPS = 4
NUM_AGENTS = 3
NUM_TOKENS = 32000  # typical vocabulary size

np.random.seed(42)

# ---------- RecursiveLink Module Simulation ----------
class InnerRecursiveLink:
    """Simulates R_in(h) = h + W2 * GELU(W1 * h)"""
    def __init__(self, dim):
        self.W1 = np.random.randn(dim * 2, dim) * 0.01
        self.W2 = np.random.randn(dim, dim * 2) * 0.01
    
    def gelu(self, x):
        return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def forward(self, h):
        return h + self.W2 @ self.gelu(self.W1 @ h)


class OuterRecursiveLink:
    """Simulates R_out(h) = W3*h + W2*GELU(W1*h)"""
    def __init__(self, src_dim, tgt_dim):
        self.W3 = np.random.randn(tgt_dim, src_dim) * 0.01
        self.W1 = np.random.randn(src_dim * 2, src_dim) * 0.01
        self.W2 = np.random.randn(tgt_dim, src_dim * 2) * 0.01
    
    def gelu(self, x):
        return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def forward(self, h):
        return self.W3 @ h + self.W2 @ self.gelu(self.W1 @ h)


# ---------- Text-Mediated MAS simulation ----------
class TextMediatedAgent:
    """Simulates an agent that decodes to text and encodes back"""
    def __init__(self, dim):
        self.dim = dim
        self.W = np.random.randn(dim, dim) * 0.01
    
    def process(self, h):
        # Decode: hidden -> token logits -> softmax -> embedding lookup
        W_decode = np.random.randn(NUM_TOKENS, self.dim) * 0.01
        logits = W_decode @ h
        # Softmax approximation: one-hot of max
        token_idx = int(np.argmax(logits))
        # Re-embed
        W_embed = np.random.randn(self.dim, NUM_TOKENS) * 0.01
        h_new = W_embed[:, token_idx]
        # Residual processing
        return h_new + self.W @ h_new


# ---------- Main Demo ----------
def main():
    sep = "=" * 65
    print(f"\n  {sep}")
    print(f"   RecursiveMAS — Quick Simulation")
    print(f"  {sep}\n")
    print(f"   Hidden dim: {HIDDEN_DIM}, Latent steps: {LATENT_STEPS}")
    print(f"   Agents: {NUM_AGENTS}, Vocabulary: {NUM_TOKENS:,}")
    print()
    
    # Step 1: Inner-Loop (latent thoughts generation)
    print(f"  ── Stage 1: Inner-Loop (Latent Thoughts) ──")
    inner = InnerRecursiveLink(HIDDEN_DIM)
    # Start with a random thought vector
    h = np.random.randn(HIDDEN_DIM) * 0.1
    
    print(f"     Initial state norm: {np.linalg.norm(h):.4f}")
    for step in range(LATENT_STEPS):
        h_new = inner.forward(h)
        delta = np.linalg.norm(h_new - h)
        h = h_new
        print(f"     Step {step+1}: norm={np.linalg.norm(h):.4f}, delta={delta:.4f}")
    
    print(f"\n     Final latent thought ready for next agent...\n")
    
    # Step 2: Outer-Loop (agent-to-agent transfer)
    print(f"  ── Stage 2: Outer-Loop (Agent Communication) ──")
    
    # Compare RecursiveMAS vs Text-Mediated
    outer_links = []
    for i in range(NUM_AGENTS - 1):
        outer_links.append(OuterRecursiveLink(HIDDEN_DIM, HIDDEN_DIM))
    
    text_agents = [TextMediatedAgent(HIDDEN_DIM) for _ in range(NUM_AGENTS)]
    
    h_rmas = h.copy()
    h_text = h.copy()
    
    times_rmas = []
    times_text = []
    
    for agent_idx in range(NUM_AGENTS):
        # Inner-loop within each agent
        for _ in range(2):
            h_rmas = inner.forward(h_rmas)
        
        # Measure RecursiveMAS (latent -> latent)
        t0 = time.time()
        h_text_agent = text_agents[agent_idx].process(h_text)
        t_text = time.time() - t0
        times_text.append(t_text)
        
        # Measure Text-Mediated (decode -> text -> encode)
        if agent_idx < NUM_AGENTS - 1:
            t0 = time.time()
            h_rmas = outer_links[agent_idx].forward(h_rmas)
            t_rmas = time.time() - t0
            times_rmas.append(t_rmas)
    
    text_time = sum(times_text) * 1e6
    rmas_time = sum(times_rmas) * 1e6
    
    print(f"     Text-Mediated MAS total:   {text_time:.1f} us  (decode->embed for each agent)")
    print(f"     RecursiveMAS total:        {rmas_time:.1f} us  (latent->latent transfer)")
    print(f"     Speedup:                   {text_time / max(rmas_time, 0.001):.1f}x")
    
    # Token savings estimation
    text_tokens = NUM_AGENTS * 100  # each agent generates ~100 tokens
    rmas_tokens = 100  # only last agent decodes
    token_saving = (text_tokens - rmas_tokens) / text_tokens * 100
    
    print(f"\n     Text tokens (per round):   ~{text_tokens}")
    print(f"     RMAS tokens (per round):   ~{rmas_tokens}")
    print(f"     Token savings:             ~{token_saving:.0f}%")
    print()
    
    # Step 3: Demo four collaboration patterns
    print(f"  ── Stage 3: Collaboration Patterns ──")
    patterns = [
        ("Sequential",    "Planner -> Critic -> Solver (chain)"),
        ("Mixture",       "Domain Specialists -> Summarizer (parallel)"),
        ("Distillation",  "Expert -> Learner (teacher-student)"),
        ("Deliberation",  "Reflector -> Tool-Caller (reflection+tools)"),
    ]
    for name, desc in patterns:
        print(f"     {name:15s}  {desc}")
    
    print(f"\n  {sep}")
    print(f"   Demo complete! Use /mas:scaffold to generate a PyTorch project.")
    print(f"  {sep}\n")


if __name__ == "__main__":
    main()