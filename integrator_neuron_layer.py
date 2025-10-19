"""
IntegratorNeuronLayer (INL) - Learnable Dynamics Architecture

This module implements a neural network layer with learnable integrator/velocity dynamics.
Key features:
- Initial convergence towards 5 (configurable target)
- Learnable controller parameters (alpha, beta, gating)
- Soft constraints allowing deviation when data requires it
- Deterministic and fully differentiable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class IntegratorNeuronLayer(nn.Module):
    """
    Implements learnable integrator dynamics with velocity control.

    Equations:
        v_{t+1} = alpha * v_t + (1 - alpha) * v_cand - beta * (x_t - target)
        x_{t+1} = x_t + dt * g * scale(v_{t+1})

    where alpha, beta, g are context-dependent learnable parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        target_value: float = 5.0,
        dt: float = 0.1,
        hidden_controller: int = 64,
        init_alpha: float = 0.8,
        init_beta: float = 0.5,
        init_gate: float = 0.5,
        velocity_scale: float = 1.0,
        excitation_amplitude: float = 0.03,
        learnable_mu: bool = True,
        mu_adaptation_rate: float = 1e-4,
        dynamic_alpha: bool = True,
        alpha_kappa: float = 1.0
    ):
        """
        Args:
            hidden_dim: Dimension of context embedding h_t
            output_dim: Dimension of state x (typically 1 for scalar prediction)
            target_value: Initial target value (default 5.0)
            dt: Time step for integration
            hidden_controller: Hidden size for controller MLPs
            init_alpha: Initial inertia coefficient
            init_beta: Initial correction coefficient
            init_gate: Initial gating value
            velocity_scale: Scale factor for velocity
            excitation_amplitude: Amplitude of deterministic harmonic noise
            learnable_mu: Use learnable equilibrium attractor
            mu_adaptation_rate: Learning rate for mu adaptation
            dynamic_alpha: Use dynamic integration gain (α-control)
            alpha_kappa: Sensitivity parameter for dynamic alpha
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.velocity_scale = velocity_scale
        self.dynamic_alpha = dynamic_alpha
        self.alpha_kappa = alpha_kappa

        # Learnable equilibrium attractor
        if learnable_mu:
            self.mu = nn.Parameter(torch.full((output_dim,), target_value))
            self.mu_adaptation_rate = mu_adaptation_rate
            self.learnable_mu = True
        else:
            self.register_buffer('mu', torch.full((output_dim,), target_value))
            self.learnable_mu = False

        # For backward compatibility
        self.target_value = target_value

        # Deterministic harmonic excitation
        self.excitation_amplitude = excitation_amplitude
        # Learnable frequency and phase per dimension
        self.excitation_gamma = nn.Parameter(torch.randn(output_dim) * 0.1 + 1.0)
        self.excitation_phi = nn.Parameter(torch.randn(output_dim) * 2 * 3.14159)
        self.register_buffer('global_step', torch.tensor(0))

        # Input dimension for controller: [h, x, v]
        controller_input_dim = hidden_dim + 2 * output_dim

        # MLP for alpha (inertia coefficient) - output in (0, 1)
        self.mlp_alpha = nn.Sequential(
            nn.Linear(controller_input_dim, hidden_controller),
            nn.ReLU(),
            nn.Linear(hidden_controller, output_dim),
        )
        # Initialize to produce init_alpha
        with torch.no_grad():
            self.mlp_alpha[-1].bias.fill_(self._inverse_sigmoid(init_alpha))
            self.mlp_alpha[-1].weight.fill_(0.0)

        # MLP for beta (correction coefficient) - output >= 0
        self.mlp_beta = nn.Sequential(
            nn.Linear(controller_input_dim, hidden_controller),
            nn.ReLU(),
            nn.Linear(hidden_controller, output_dim),
        )
        # Initialize to produce init_beta
        with torch.no_grad():
            self.mlp_beta[-1].bias.fill_(self._inverse_softplus(init_beta))
            self.mlp_beta[-1].weight.fill_(0.0)

        # MLP for gating g - output in (0, 1)
        self.mlp_gate = nn.Sequential(
            nn.Linear(controller_input_dim, hidden_controller),
            nn.ReLU(),
            nn.Linear(hidden_controller, output_dim),
        )
        # Initialize to produce init_gate
        with torch.no_grad():
            self.mlp_gate[-1].bias.fill_(self._inverse_sigmoid(init_gate))
            self.mlp_gate[-1].weight.fill_(0.0)

        # MLP for candidate velocity v_cand
        self.mlp_v_cand = nn.Sequential(
            nn.Linear(controller_input_dim, hidden_controller),
            nn.ReLU(),
            nn.Linear(hidden_controller, output_dim),
        )
        # Initialize to produce small velocities
        with torch.no_grad():
            self.mlp_v_cand[-1].bias.fill_(0.0)
            self.mlp_v_cand[-1].weight.fill_(0.0)

    @staticmethod
    def _inverse_sigmoid(y: float) -> float:
        """Inverse of sigmoid function for initialization."""
        y = max(min(y, 0.999), 0.001)  # Clamp to avoid inf
        return torch.tensor(y / (1 - y)).log().item()

    @staticmethod
    def _inverse_softplus(y: float) -> float:
        """Inverse of softplus function for initialization."""
        y = max(y, 0.001)
        return torch.tensor(y).expm1().log().item()

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass computing one integration step.

        Args:
            h: Context embedding [batch_size, hidden_dim]
            x: Current state [batch_size, output_dim]
            v: Current velocity [batch_size, output_dim]

        Returns:
            x_next: Next state [batch_size, output_dim]
            v_next: Next velocity [batch_size, output_dim]
            aux: Dictionary with controller parameters for monitoring
        """
        # Concatenate inputs for controller
        controller_input = torch.cat([h, x, v], dim=-1)

        # Compute controller parameters
        alpha_base = torch.sigmoid(self.mlp_alpha(controller_input))
        beta = F.softplus(self.mlp_beta(controller_input))
        gate = torch.sigmoid(self.mlp_gate(controller_input))
        v_cand = self.mlp_v_cand(controller_input)

        # Dynamic integration gain (α-control)
        if self.dynamic_alpha:
            # Compute imbalance as deviation from equilibrium attractor
            imbalance = torch.norm(x - self.mu, dim=-1, keepdim=True)
            # Reduce alpha (increase correction) when far from equilibrium
            alpha = alpha_base * torch.exp(-self.alpha_kappa * imbalance)
        else:
            alpha = alpha_base

        # Update velocity with error correction term
        error = x - self.mu
        v_next = alpha * v + (1 - alpha) * v_cand - beta * error

        # Add deterministic harmonic excitation
        if self.excitation_amplitude > 0:
            # Deterministic noise based on global step counter
            t = self.global_step.float()
            harmonic_noise = self.excitation_amplitude * torch.sin(
                self.excitation_gamma * t + self.excitation_phi
            )
            v_next = v_next + harmonic_noise.unsqueeze(0)

            # Increment global step counter
            if self.training:
                self.global_step += 1

        # Update state with gated velocity
        x_next = x + self.dt * gate * self.velocity_scale * v_next

        # Return auxiliary info for monitoring/loss
        aux = {
            'alpha': alpha,
            'alpha_base': alpha_base if self.dynamic_alpha else alpha,
            'beta': beta,
            'gate': gate,
            'v_cand': v_cand,
            'error': error,
            'mu': self.mu
        }

        return x_next, v_next, aux

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize state x and velocity v.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            x0: Initial state [batch_size, output_dim] initialized to learned mu
            v0: Initial velocity [batch_size, output_dim] initialized to 0
        """
        # Initialize to current learned equilibrium
        x0 = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)
        v0 = torch.zeros((batch_size, self.output_dim), device=device)
        return x0, v0


class IntegratorModel(nn.Module):
    """
    Complete model: Backbone + IntegratorNeuronLayer + Readout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_iterations: int = 10,
        output_dim: int = 1,
        target_value: float = 5.0,
        **inl_kwargs
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for backbone and INL
            num_layers: Number of layers in backbone MLP
            num_iterations: Number of integration steps T
            output_dim: Output dimension (1 for scalar regression)
            target_value: Target value for convergence (default 5.0)
            **inl_kwargs: Additional arguments for IntegratorNeuronLayer
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.output_dim = output_dim
        self.target_value = target_value

        # Backbone: simple MLP (can be replaced with Transformer)
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Integrator Neuron Layer
        self.inl = IntegratorNeuronLayer(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            target_value=target_value,
            **inl_kwargs
        )

        # Readout layer
        self.readout = nn.Linear(output_dim, output_dim)
        # Initialize readout to identity transformation (no bias shift)
        # Since x is already initialized to target_value, we just pass it through
        with torch.no_grad():
            self.readout.weight.fill_(0.0)
            self.readout.weight.diagonal().fill_(1.0)
            self.readout.bias.fill_(0.0)  # No bias - x already at target_value

    def forward(
        self,
        inputs: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through complete model.

        Args:
            inputs: Input features [batch_size, input_dim]
            return_trajectory: If True, return full trajectory and aux info

        Returns:
            output: Final prediction [batch_size, output_dim]
            trajectory: Optional dict with trajectory info if return_trajectory=True
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Compute context from backbone
        h = self.backbone(inputs)  # [B, hidden_dim]

        # Initialize state and velocity
        x, v = self.inl.init_state(batch_size, device)

        # Store trajectory if requested
        if return_trajectory:
            x_traj = [x.clone()]
            v_traj = [v.clone()]
            aux_traj = []

        # Run integration steps
        for t in range(self.num_iterations):
            x, v, aux = self.inl(h, x, v)

            if return_trajectory:
                x_traj.append(x.clone())
                v_traj.append(v.clone())
                aux_traj.append({k: v.clone() for k, v in aux.items()})

        # Final readout
        output = self.readout(x)

        if return_trajectory:
            trajectory = {
                'x': torch.stack(x_traj, dim=1),  # [B, T+1, output_dim]
                'v': torch.stack(v_traj, dim=1),  # [B, T+1, output_dim]
                'aux': aux_traj
            }
            return output, trajectory

        return output, None

    def get_final_state(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get final state x_T before readout."""
        batch_size = inputs.shape[0]
        device = inputs.device

        h = self.backbone(inputs)
        x, v = self.inl.init_state(batch_size, device)

        for _ in range(self.num_iterations):
            x, v, _ = self.inl(h, x, v)

        return x

    def get_learned_mu(self) -> Optional[torch.Tensor]:
        """
        Get the learned equilibrium attractor.

        Returns:
            Learned mu tensor if learnable_mu enabled, else None
        """
        if hasattr(self.inl, 'learnable_mu') and self.inl.learnable_mu:
            return self.inl.mu
        return None
