# Integrator Language Model (INL-LLM)

> **Deterministic equilibrium-based neural architecture with adaptive dynamic diversity**
>
> A new paradigm for language modeling built on self-balancing integrator dynamics,
> learnable equilibrium attractors, and controlled exploration through harmonic excitation.

---

## Overview

**INL-LLM** is a research architecture that replaces traditional stochastic neural networks with a **deterministic equilibrium-based system**. Instead of relying on random initialization and noise-driven optimization, it learns through:

- **Learnable equilibrium attractors** that adapt to data
- **Deterministic harmonic excitation** for controlled exploration
- **Variance-weighted self-regulation** for hierarchical balance
- **Dynamic integration control** for energy-aware convergence
- **Rhythmic training cycles** alternating stability and discovery

The core principle: **learning emerges from motion toward adaptive equilibrium, not from random drift**.

---

## Core Dynamics

### Integrator Neural Layer (INL)

Each neuron evolves as a **second-order dynamical system** with state $x$ and velocity $v$:

```
Velocity update:
  v_{t+1} = α·v_t + (1-α)·v_cand - β·(x_t - μ) + β_exc·sin(γ·t + φ)

State update:
  x_{t+1} = x_t + Δt·g·v_{t+1}
```

**Components:**
- $\mu$: **learnable equilibrium attractor** (adapts during training)
- $\alpha$: **dynamic integration gain** (responds to system imbalance)
- $\beta$: correction strength (learned)
- $g$: gating coefficient (learned)
- $v_\text{cand}$: candidate velocity from context (learned MLP)
- $\beta_\text{exc} \sin(\gamma t + \phi)$: **deterministic harmonic excitation**

### Key Innovation: Everything is Adaptive

Unlike fixed-attractor systems, INL-LLM **learns its own equilibrium**:

$$\mu_{t+1} = (1 - \eta_\mu) \mu_t + \eta_\mu \bar{h}_t$$

where $\eta_\mu \approx 10^{-4}$ ensures slow, stable adaptation.

---

## Five Pillars of the Architecture

### 1. Learnable Equilibrium Attractor

**Traditional approach:** Fixed target (e.g., mean ≈ 5)
**INL-LLM approach:** Each dimension learns its own equilibrium

**Equation:**
$$\mu_k(t+1) = 0.9999 \cdot \mu_k(t) + 0.0001 \cdot \bar{h}_k(t)$$

**Benefits:**
- Adapts to data distribution automatically
- Stable across domain shifts
- No manual tuning of equilibrium point

**Implementation:**
```python
IntegratorNeuronLayer(
    learnable_mu=True,           # Enabled by default
    mu_adaptation_rate=1e-4      # Slow EMA update
)
```

---

### 2. Deterministic Harmonic Excitation

**Problem:** Pure equilibrium systems can stagnate in local minima
**Solution:** Add deterministic periodic perturbations

**Equation:**
$$v_{t+1} = \alpha v_t + (1-\alpha)v_\text{cand} - \beta(x - \mu) + \beta_\text{exc} \sin(\gamma t + \phi)$$

where $\gamma$ (frequency) and $\phi$ (phase) are **learnable parameters**.

**Benefits:**
- Prevents stagnation without randomness
- Fully reproducible (deterministic)
- Structured micro-chaos enriches learning
- Frequency and phase adapt to layer needs

**Implementation:**
```python
IntegratorNeuronLayer(
    excitation_amplitude=0.03,   # Default amplitude
    # excitation_gamma: learned per dimension
    # excitation_phi: learned per dimension
)
```

**Visualization concept:**
```
Without excitation:    With excitation:
    ↓                      ↓ ~~~
    equilibrium            equilibrium (breathing)
```

---

### 3. Variance-Weighted Regularization

**Problem:** Uniform penalties treat all neurons equally
**Solution:** Weight penalties by neuron variance

**Equation:**
$$L_\text{mean} = \lambda \sum_i w_i \|\bar{h}_i - \mu_i\|^2$$

where
$$w_i = \frac{1}{1 + \text{Var}(h_i)}$$

**Interpretation:**
- **Low variance neurons** (stable) → low weight → **less penalized**
- **High variance neurons** (active) → high weight → **gently regularized**

**Benefits:**
- Self-regulating hierarchy
- Stable neurons remain stable
- Active neurons maintain expressiveness
- Automatic load balancing

**Implementation:**
```python
IntegratorLoss(
    variance_weighted=True  # Enabled by default
)
```

---

### 4. Dynamic Integration Gain (α-Control)

**Problem:** Fixed integration gain ignores system state
**Solution:** Modulate α based on imbalance

**Equation:**
$$\alpha_t = \alpha_\text{base} \cdot (1 - e^{-\kappa \|x_t - \bar{x}_t\|})$$

**Behavior:**
- **Near equilibrium** ($\|x - \bar{x}\| \approx 0$) → $\alpha \approx 0$ → **slow integration**
- **Far from equilibrium** ($\|x - \bar{x}\|$ large) → $\alpha \approx \alpha_\text{base}$ → **fast integration**

**Benefits:**
- Energy-aware convergence
- Smooth transitions
- Automatically adjusts integration speed
- Prevents overshooting near equilibrium

**Implementation:**
```python
IntegratorNeuronLayer(
    dynamic_alpha=True,   # Enabled by default
    alpha_kappa=1.0       # Sensitivity parameter
)
```

---

### 5. Equilibrium-Exploration Cycles

**Problem:** Constant training regime lacks rhythm
**Solution:** Alternate between two complementary phases

| Phase | Duration | λ_mean | β_exc | Goal |
|-------|----------|---------|-------|------|
| **Equilibrium** | 10 epochs | 0.5 | 0.0 | Stabilization, consolidation |
| **Exploration** | 20 epochs | 0.05 | 0.05 | Discovery, diversity |

**Training rhythm:**
```
Warmup (10 epochs)
  ↓
Cycle 1: Equilibrium (10) → Exploration (20)
  ↓
Cycle 2: Equilibrium (10) → Exploration (20)
  ↓
...
  ↓
Cycle N: Equilibrium (10) → Exploration (20)
```

**Benefits:**
- Natural learning rhythm (like sleep-wake cycles)
- Consolidation alternates with exploration
- No randomness needed
- Prevents catastrophic forgetting

**Implementation:**
```python
from integrator_scheduler_v2 import create_cycle_scheduler

# Three presets available
scheduler = create_cycle_scheduler(preset='balanced')
# Options: 'conservative', 'balanced', 'aggressive'

# Or custom:
scheduler = create_cycle_scheduler(
    equilibrium_config={
        'lambda_mean': 0.5,
        'excitation_amplitude': 0.0,
        'duration_epochs': 10
    },
    exploration_config={
        'lambda_mean': 0.05,
        'excitation_amplitude': 0.05,
        'duration_epochs': 20
    },
    num_cycles=5,
    warmup_epochs=10
)
```

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                  Integrator Language Model                 │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Token Embeddings + Positional Encoding                   │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │           N × Integrator Blocks                      │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │  Integrator Neuron Layer                        │ │ │
│  │  │                                                  │ │ │
│  │  │  • Learnable μ (equilibrium attractor)          │ │ │
│  │  │  • Dynamic α (integration gain control)         │ │ │
│  │  │  • Harmonic excitation (sin(γt + φ))            │ │ │
│  │  │  • Velocity integration (second-order dynamics) │ │ │
│  │  │  • Context-dependent control (α, β, g MLPs)     │ │ │
│  │  │                                                  │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  │                                                       │ │
│  │  Feedforward + Residual Connection                   │ │
│  │                                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  Output Projection (tied embeddings)                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Information flow:**
1. Tokens → Embeddings
2. For each block:
   - Context embedding via backbone MLP
   - Multiple integration steps (continuous-time approximation)
   - Learned μ adapts based on batch statistics
   - Harmonic excitation adds controlled perturbation
   - Dynamic α adjusts based on imbalance
3. Final state → Token predictions

---

## Model Sizes

| Config | d_model | Layers | Steps/layer | FF dim | Context | Params |
|--------|---------|--------|-------------|---------|---------|--------|
| **small** | 512 | 6 | 5 | 2048 | 512 | 43M |
| **medium** | 768 | 12 | 7 | 3072 | 1024 | 112M |
| **large** | 1024 | 24 | 10 | 4096 | 2048 | 303M |
| **xlarge** | 1536 | 32 | 12 | 6144 | 4096 | 808M |
| **3B** | 2048 | 40 | 15 | 8192 | 4096 | ~3B |

---

## Installation

```bash
git clone <your-repo-url>
cd architecture
pip install torch transformers tqdm pandas pyarrow matplotlib
```

**Dependencies:**
- PyTorch ≥ 2.0
- transformers
- tqdm, pandas, pyarrow (for data)
- matplotlib (for visualization)

---

## Quick Start

### Simple Training

```bash
python train_inl_llm_large.py
```

Interactive prompts will ask for:
- Model size (`small`, `medium`, `large`, etc.)
- Dataset path (`.txt` or `.parquet`)
- Training epochs and batch size

### Advanced Training with Cycles

```python
from integrator_neuron_layer import IntegratorModel
from integrator_losses import IntegratorLoss
from integrator_scheduler_v2 import create_cycle_scheduler
import torch.optim as optim

# 1. Create model (all features enabled by default)
model = IntegratorModel(
    input_dim=512,
    hidden_dim=768,
    num_iterations=10,
    # Defaults: learnable_mu=True, dynamic_alpha=True,
    #           excitation_amplitude=0.03
)

# 2. Create loss (variance weighting enabled)
loss_fn = IntegratorLoss(
    target_value=5.0,
    variance_weighted=True,  # default
    lambda_mean_init=1.0,
    lambda_energy=0.01
)

# 3. Setup cycle scheduler
scheduler = create_cycle_scheduler(preset='balanced')

# 4. Optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# 5. Training loop
for epoch in range(scheduler.get_total_epochs()):

    # Update training phase
    phase_info = scheduler.step(epoch)
    loss_fn.set_exploration_phase(phase_info['is_exploration'])

    # Update model excitation amplitude
    if hasattr(model, 'inl'):
        model.inl.excitation_amplitude = phase_info['excitation_amplitude']

    # Train epoch
    for batch in dataloader:
        inputs, targets = batch

        # Forward with trajectory tracking
        outputs, trajectory = model(inputs, return_trajectory=True)

        # Get learned equilibrium for loss
        learned_mu = model.get_learned_mu()

        # Compute losses
        losses = loss_fn(
            predictions=outputs,
            targets=targets,
            trajectory=trajectory,
            epoch=epoch,
            learned_mu=learned_mu
        )

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Log
        print(f"Phase: {phase_info['phase_name']}, Loss: {losses['total'].item():.4f}")
```

---

## Loss Function

Total loss combines four components:

$$L = L_\text{task} + L_\text{mean} + L_\text{speed} + L_\text{energy}$$

### 1. Task Loss
Standard cross-entropy or MSE for prediction

### 2. Mean Loss (Variance-Weighted)
$$L_\text{mean} = \lambda_\text{mean} \sum_i \frac{1}{1 + \text{Var}(h_i)} \|\bar{h}_i - \mu_i\|^2$$

**Annealing:** $\lambda_\text{mean}$ decays from 1.0 → 0.1 over 100 epochs

**Phase modulation:**
- Equilibrium: $\lambda_\text{mean} = 0.5$
- Exploration: $\lambda_\text{mean} = 0.05$

### 3. Speed Loss
Penalizes slow convergence in early iterations:
$$L_\text{speed} = \lambda_\text{speed} \sum_t w_t |x_t - \mu|$$

where $w_t = e^{-t/\tau}$ prioritizes early steps

### 4. Energy Loss
Regularizes velocity magnitude:
$$L_\text{energy} = \lambda_\text{energy} \mathbb{E}[|v|^2]$$

**Phase modulation:**
- Equilibrium: $\lambda_\text{energy} = 0.01$
- Exploration: $\lambda_\text{energy} = 0.001$ (reduced)

---

## Monitoring & Metrics

### Convergence Metrics

```python
from integrator_losses import compute_convergence_metrics

metrics = compute_convergence_metrics(
    x_trajectory=trajectory['x'],
    target_value=model.inl.mu.mean().item()
)

# Returns:
# {
#   'time_to_converge': 7.3,      # steps to equilibrium
#   'final_rmse': 0.12,            # deviation at end
#   'final_mean': 5.01,            # actual mean
#   'final_std': 0.34,             # variance
#   'fraction_converged': 0.92     # % within tolerance
# }
```

### Learned Parameters to Track

1. **Equilibrium evolution:** `model.inl.mu` over epochs
2. **Harmonic frequencies:** `model.inl.excitation_gamma`
3. **Harmonic phases:** `model.inl.excitation_phi`
4. **Dynamic α patterns:** Distribution of α values per batch
5. **Phase transitions:** Loss smoothness at cycle boundaries

---

## Key Parameters Reference

### IntegratorNeuronLayer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | — | Context embedding dimension |
| `output_dim` | 1 | State dimension |
| `target_value` | 5.0 | Initial equilibrium (if not learnable) |
| `dt` | 0.1 | Integration timestep |
| **Learnable Equilibrium** | | |
| `learnable_mu` | `True` | Enable adaptive equilibrium |
| `mu_adaptation_rate` | `1e-4` | EMA rate for μ update |
| **Harmonic Excitation** | | |
| `excitation_amplitude` | `0.03` | Noise amplitude (0=off) |
| **Dynamic Control** | | |
| `dynamic_alpha` | `True` | Enable α-control |
| `alpha_kappa` | `1.0` | Sensitivity to imbalance |
| **Controller Init** | | |
| `init_alpha` | `0.8` | Initial inertia |
| `init_beta` | `0.5` | Initial correction |
| `init_gate` | `0.5` | Initial gating |
| `velocity_scale` | `1.0` | Global velocity scaling |

### IntegratorLoss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_value` | 5.0 | Reference equilibrium |
| `lambda_mean_init` | 1.0 | Initial equilibrium penalty |
| `lambda_speed` | 0.1 | Speed penalty weight |
| `lambda_energy` | 0.01 | Energy penalty weight |
| `annealing_epochs` | 100 | Epochs to anneal λ_mean |
| **Variance Weighting** | | |
| `variance_weighted` | `True` | Enable adaptive weighting |
| **Phase Control** | | |
| `exploration_lambda_mean` | 0.05 | λ during exploration |
| `exploration_lambda_energy` | 0.001 | Energy λ during exploration |

---

## Data Formats

### Plain Text
```
The quick brown fox jumps over the lazy dog.
Integrator Neural Layers learn through equilibrium dynamics.
```

### Parquet (Structured)
```python
import pandas as pd

df = pd.DataFrame({
    'prompt': ['What is AI?', 'Explain quantum physics'],
    'completion': ['Artificial Intelligence...', 'Quantum mechanics...']
})
df.to_parquet('data.parquet')
```

---

## Design Philosophy

### 1. Determinism Without Rigidity

**Deterministic:**
- No random initialization
- Reproducible training (same seed → same result)
- Harmonic excitation uses deterministic sine waves

**Adaptive:**
- Equilibrium learns from data
- Integration gain responds to state
- Variance-based self-regulation

### 2. Energy-Based Learning

The model doesn't learn by random drift but by:
1. **Attraction to equilibrium** (stability)
2. **Harmonic perturbation** (exploration)
3. **Equilibrium adaptation** (plasticity)
4. **Variance regulation** (balance)

### 3. Continuous-Time Reasoning

Multiple integration steps approximate **continuous dynamics**:
- Not layer-by-layer discrete jumps
- Smooth state evolution
- Energy-aware convergence

### 4. Self-Organization

The system self-regulates through:
- Learned equilibrium points
- Variance-weighted penalties
- Dynamic integration control
- Natural training rhythms

---

## Mathematical Summary

### Complete Dynamics

**Velocity update:**
$$v_{t+1} = \alpha_t \cdot v_t + (1-\alpha_t) \cdot v_\text{cand} - \beta(x_t - \mu) + \beta_\text{exc} \sin(\gamma t + \phi)$$

**State update:**
$$x_{t+1} = x_t + \Delta t \cdot g \cdot v_{t+1}$$

**Dynamic alpha:**
$$\alpha_t = \alpha_\text{base} \cdot (1 - e^{-\kappa \|x_t - \bar{x}_t\|})$$

**Learnable mu:**
$$\mu_{t+1} = (1 - \eta_\mu) \mu_t + \eta_\mu \bar{h}_t$$

**Variance weights:**
$$w_i = \frac{1}{1 + \text{Var}(h_i)}$$

### Loss Function

$$L = L_\text{task} + \lambda_\text{mean}(t) \sum_i w_i \|\bar{h}_i - \mu_i\|^2 + \lambda_\text{speed} \sum_t e^{-t/\tau}|x_t - \mu| + \lambda_\text{energy} \mathbb{E}[|v|^2]$$

---

## Future Research Directions

1. **Theoretical Analysis**
   - Convergence guarantees for learned equilibrium
   - Stability analysis of harmonic excitation
   - Energy landscape characterization

2. **Scaling Studies**
   - 10B+ parameter models
   - Multi-GPU training optimization
   - Memory-efficient integration

3. **Multimodal Extensions**
   - INL-Vision (image dynamics)
   - INL-Audio (speech equilibrium)
   - Cross-modal equilibrium alignment

4. **Interpretability**
   - Equilibrium trajectory visualization
   - Harmonic frequency analysis
   - Energy flow through layers

5. **Applications**
   - Transfer learning with frozen equilibria
   - Continual learning with adaptive μ
   - Few-shot via equilibrium adaptation

---

## Citation

```bibtex
@software{inl_llm_2025,
  title={Integrator Language Model: Adaptive Equilibrium Learning with Deterministic Diversity},
  author={Peyriguère, Boris},
  year={2025},
  url={https://github.com/<your-repo>/inl-llm}
}
```

---

## License

MIT License © 2025 Boris Peyriguère

---

## Status Overview

| Feature | Status | Default Setting |
|---------|--------|-----------------|
| Learnable μ | **Active** | `learnable_mu=True` |
| Harmonic excitation | **Active** | `excitation_amplitude=0.03` |
| Variance weighting | **Active** | `variance_weighted=True` |
| Dynamic α | **Active** | `dynamic_alpha=True` |
| Cycle scheduler | **Optional** | Use `create_cycle_scheduler()` |

**All core innovations are enabled by default. No flags needed.**

---

## Contact

- **Author:** Boris Peyriguère
- **Email:** peyriguere.boris@gmail.com
- **Issues:** GitHub Issues
- **Contributions:** PRs welcome

---

*"Learning through adaptive equilibrium, exploring through deterministic diversity."*