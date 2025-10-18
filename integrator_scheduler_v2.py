"""
INL-LLM: Equilibrium-Exploration Cycle Scheduler

Implements rhythmic training phases that alternate between:
- Equilibrium Phase: Strong stability constraint, low excitation (stabilization)
- Exploration Phase: Weak stability constraint, high excitation (discovery)

This deterministic cycling encourages structured exploration without randomness.
"""

from typing import Dict, NamedTuple
import torch


class PhaseConfig(NamedTuple):
    """Configuration for a training phase."""
    name: str
    lambda_mean: float
    excitation_amplitude: float
    duration_epochs: int


class EquilibriumExplorationScheduler:
    """
    Manages equilibrium-exploration cycles for v2 training.

    Example cycle:
    - Equilibrium (10 epochs): lambda_mean=0.5, excitation=0.0
    - Exploration (20 epochs): lambda_mean=0.05, excitation=0.05
    - Repeat...
    """

    def __init__(
        self,
        equilibrium_config: Dict = None,
        exploration_config: Dict = None,
        num_cycles: int = 5,
        warmup_epochs: int = 10
    ):
        """
        Args:
            equilibrium_config: Config for equilibrium phase
            exploration_config: Config for exploration phase
            num_cycles: Number of complete cycles to perform
            warmup_epochs: Initial warmup before cycling starts
        """
        # Default equilibrium phase: stabilization
        if equilibrium_config is None:
            equilibrium_config = {
                'lambda_mean': 0.5,
                'excitation_amplitude': 0.0,
                'duration_epochs': 10
            }

        # Default exploration phase: discovery
        if exploration_config is None:
            exploration_config = {
                'lambda_mean': 0.05,
                'excitation_amplitude': 0.05,
                'duration_epochs': 20
            }

        self.equilibrium_phase = PhaseConfig(
            name='equilibrium',
            **equilibrium_config
        )
        self.exploration_phase = PhaseConfig(
            name='exploration',
            **exploration_config
        )

        self.num_cycles = num_cycles
        self.warmup_epochs = warmup_epochs

        # Build phase schedule
        self.schedule = self._build_schedule()

        # Current state
        self.current_epoch = 0
        self.current_phase = None

    def _build_schedule(self):
        """Build the complete phase schedule."""
        schedule = []

        # Warmup: equilibrium phase
        if self.warmup_epochs > 0:
            schedule.append({
                'name': 'warmup',
                'phase': self.equilibrium_phase,
                'start_epoch': 0,
                'end_epoch': self.warmup_epochs
            })

        # Cycles
        epoch = self.warmup_epochs
        for cycle in range(self.num_cycles):
            # Equilibrium phase
            schedule.append({
                'name': f'cycle_{cycle}_equilibrium',
                'phase': self.equilibrium_phase,
                'start_epoch': epoch,
                'end_epoch': epoch + self.equilibrium_phase.duration_epochs
            })
            epoch += self.equilibrium_phase.duration_epochs

            # Exploration phase
            schedule.append({
                'name': f'cycle_{cycle}_exploration',
                'phase': self.exploration_phase,
                'start_epoch': epoch,
                'end_epoch': epoch + self.exploration_phase.duration_epochs
            })
            epoch += self.exploration_phase.duration_epochs

        return schedule

    def get_phase_config(self, epoch: int) -> PhaseConfig:
        """
        Get the phase configuration for a given epoch.

        Args:
            epoch: Current training epoch

        Returns:
            PhaseConfig for the current epoch
        """
        for entry in self.schedule:
            if entry['start_epoch'] <= epoch < entry['end_epoch']:
                return entry['phase']

        # Default to exploration phase after all cycles
        return self.exploration_phase

    def is_exploration_phase(self, epoch: int) -> bool:
        """Check if current epoch is in exploration phase."""
        phase = self.get_phase_config(epoch)
        return phase.name == 'exploration'

    def step(self, epoch: int) -> Dict[str, any]:
        """
        Update scheduler state and return current phase info.

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary with phase information
        """
        self.current_epoch = epoch
        self.current_phase = self.get_phase_config(epoch)

        return {
            'phase_name': self.current_phase.name,
            'lambda_mean': self.current_phase.lambda_mean,
            'excitation_amplitude': self.current_phase.excitation_amplitude,
            'is_exploration': self.current_phase.name == 'exploration'
        }

    def get_total_epochs(self) -> int:
        """Get total number of epochs in the schedule."""
        if not self.schedule:
            return 0
        return self.schedule[-1]['end_epoch']

    def print_schedule(self):
        """Print the complete phase schedule."""
        print("=" * 70)
        print("EQUILIBRIUM-EXPLORATION CYCLE SCHEDULE")
        print("=" * 70)

        for entry in self.schedule:
            phase = entry['phase']
            print(f"\n{entry['name'].upper()}")
            print(f"  Epochs: {entry['start_epoch']}-{entry['end_epoch']} "
                  f"({entry['end_epoch'] - entry['start_epoch']} epochs)")
            print(f"  Lambda Mean: {phase.lambda_mean:.3f}")
            print(f"  Excitation Amplitude: {phase.excitation_amplitude:.3f}")
            print(f"  Phase Type: {phase.name}")

        print(f"\nTotal Training Epochs: {self.get_total_epochs()}")
        print("=" * 70)


class CycleTrainingMixin:
    """
    Mixin class to add cycle scheduling to existing trainers.

    Usage:
        class MyTrainer(CycleTrainingMixin, BaseTrainer):
            ...
    """

    def setup_cycle_scheduler(
        self,
        equilibrium_config: Dict = None,
        exploration_config: Dict = None,
        num_cycles: int = 5,
        warmup_epochs: int = 10
    ):
        """
        Initialize the phase scheduler.

        Call this in your trainer's __init__ method.
        """
        self.cycle_scheduler = EquilibriumExplorationScheduler(
            equilibrium_config=equilibrium_config,
            exploration_config=exploration_config,
            num_cycles=num_cycles,
            warmup_epochs=warmup_epochs
        )

        self.cycle_enabled = True
        self.cycle_scheduler.print_schedule()

    def update_phase(self, epoch: int, model, loss_fn):
        """
        Update model and loss function for current phase.

        Args:
            epoch: Current training epoch
            model: IntegratorModel
            loss_fn: IntegratorLoss
        """
        if not hasattr(self, 'cycle_enabled') or not self.cycle_enabled:
            return

        # Get current phase config
        phase_info = self.cycle_scheduler.step(epoch)

        # Update loss function phase
        if hasattr(loss_fn, 'set_exploration_phase'):
            loss_fn.set_exploration_phase(phase_info['is_exploration'])

        # Update model excitation amplitude
        if hasattr(model, 'inl') and hasattr(model.inl, 'excitation_amplitude'):
            model.inl.excitation_amplitude = phase_info['excitation_amplitude']

        # Update all INL blocks in language model if applicable
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                if hasattr(block, 'inl') and hasattr(block.inl, 'excitation_amplitude'):
                    block.inl.excitation_amplitude = phase_info['excitation_amplitude']

        return phase_info


# Example configuration presets
CYCLE_PRESETS = {
    'conservative': {
        'equilibrium_config': {
            'lambda_mean': 0.8,
            'excitation_amplitude': 0.0,
            'duration_epochs': 15
        },
        'exploration_config': {
            'lambda_mean': 0.1,
            'excitation_amplitude': 0.02,
            'duration_epochs': 15
        },
        'num_cycles': 4,
        'warmup_epochs': 20
    },

    'balanced': {
        'equilibrium_config': {
            'lambda_mean': 0.5,
            'excitation_amplitude': 0.0,
            'duration_epochs': 10
        },
        'exploration_config': {
            'lambda_mean': 0.05,
            'excitation_amplitude': 0.05,
            'duration_epochs': 20
        },
        'num_cycles': 5,
        'warmup_epochs': 10
    },

    'aggressive': {
        'equilibrium_config': {
            'lambda_mean': 0.3,
            'excitation_amplitude': 0.0,
            'duration_epochs': 5
        },
        'exploration_config': {
            'lambda_mean': 0.01,
            'excitation_amplitude': 0.08,
            'duration_epochs': 25
        },
        'num_cycles': 6,
        'warmup_epochs': 5
    }
}


def create_cycle_scheduler(preset: str = 'balanced', **overrides) -> EquilibriumExplorationScheduler:
    """
    Create a cycle scheduler from a preset configuration.

    Args:
        preset: One of 'conservative', 'balanced', 'aggressive'
        **overrides: Override any preset parameters

    Returns:
        Configured EquilibriumExplorationScheduler
    """
    if preset not in CYCLE_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(CYCLE_PRESETS.keys())}")

    config = CYCLE_PRESETS[preset].copy()
    config.update(overrides)

    return EquilibriumExplorationScheduler(**config)


if __name__ == '__main__':
    # Demo: print different scheduler configurations
    print("\n" + "=" * 70)
    print("CYCLE SCHEDULER DEMONSTRATION")
    print("=" * 70)

    for preset_name in ['conservative', 'balanced', 'aggressive']:
        print(f"\n\nPRESET: {preset_name.upper()}")
        scheduler = create_cycle_scheduler(preset_name)
        scheduler.print_schedule()

        # Show epoch-by-epoch evolution for first 50 epochs
        print(f"\nFirst 50 epochs evolution:")
        for epoch in range(min(50, scheduler.get_total_epochs())):
            if epoch % 10 == 0:
                phase_info = scheduler.step(epoch)
                print(f"  Epoch {epoch:3d}: {phase_info['phase_name']:20s} "
                      f"λ={phase_info['lambda_mean']:.3f} β={phase_info['excitation_amplitude']:.3f}")
