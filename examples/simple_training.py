"""
Simple training example for INL-LLM

This script demonstrates basic training with:
- Optimized model (Level 1)
- Synthetic data
- Equilibrium-exploration cycles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from inl_llm.models import UltraOptimizedIntegratorLanguageModel
from inl_llm.core import IntegratorLoss, create_cycle_scheduler

# Import tokenizer
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("⚠️ transformers not installed. Install with: pip install transformers")


class SimpleTextDataset(Dataset):
    """Simple synthetic dataset with learnable patterns (not pure random)."""

    def __init__(self, vocab_size=5000, num_samples=1000, seq_len=128):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate structured sequences with patterns (easier to learn than pure random)
        # Use mix of: counting sequences, repeated patterns, and some randomness
        base_pattern = torch.arange(0, self.seq_len) % min(100, self.vocab_size)
        noise = torch.randint(-5, 6, (self.seq_len,))  # Small perturbations
        seq = (base_pattern + noise + idx * 7) % self.vocab_size  # Add variety per sample
        # Target is shifted by 1 (next token prediction)
        return seq[:-1], seq[1:]


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device='cpu', epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        logits, trajectory = model(inputs, return_aux=True)

        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # ✅ FIX #1: Use IntegratorLoss with trajectory (was only using CrossEntropyLoss)
        if loss_fn is not None and trajectory is not None:
            # Use full IntegratorLoss with all components
            loss_components = loss_fn(
                predictions=logits_flat,
                targets=targets_flat,
                trajectory=trajectory,
                epoch=epoch
            )
            loss = loss_components['total']

            # Log detailed loss components occasionally
            if batch_idx % 10 == 0:
                L_task = loss_components.get('L_task', 0)
                L_mean = loss_components.get('L_mean', 0)
                L_speed = loss_components.get('L_speed', 0)
                L_energy = loss_components.get('L_energy', 0)
                print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f} '
                      f'[Task: {L_task:.4f}, Mean: {L_mean:.4f}, Speed: {L_speed:.4f}, Energy: {L_energy:.4f}]')
        else:
            # Fallback to simple CrossEntropy
            loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ✅ FIX #3: Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    print("="*70)
    print("SIMPLE TRAINING EXAMPLE - INL-LLM")
    print("="*70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    if TOKENIZER_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        print(f"✅ Tokenizer loaded: GPT-2 (vocab_size={vocab_size})")
    else:
        tokenizer = None
        vocab_size = 50000
        print(f"⚠️ Using synthetic data (vocab_size={vocab_size})")

    # Configuration
    batch_size = 2
    num_epochs = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Model size: 1B parameters")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")

    # Create custom 1B parameter model
    print("\nCreating custom 1B parameter model (all optimizations enabled)...")
    model = UltraOptimizedIntegratorLanguageModel(
        vocab_size=vocab_size,
        d_model=1600,           # Dimension du modèle (augmenté de 1536 à 1600)
        num_layers=28,          # Nombre de couches
        num_heads=25,           # Nombre de têtes d'attention (1600/25 = 64 dim par tête)
        num_iterations_per_layer=10,  # Itérations par couche
        feedforward_dim=6400,   # Dimension FFN (4x d_model)
        max_seq_len=2048,
        # Toutes les optimizations activées
        use_lowrank_embeddings=True,
        lowrank_ratio=0.125,
        use_gradient_checkpointing=True,
        use_shared_controllers=True,
        hierarchical_group_size=64,
        excitation_sparsity=0.1
    )
    model = model.to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = SimpleTextDataset(vocab_size=vocab_size, num_samples=100, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ✅ FIX #3: Lower learning rate for large model (was 3e-4, too high)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Add learning rate scheduler with warmup
    from torch.optim.lr_scheduler import OneCycleLR
    total_steps = num_epochs * len(dataloader)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    print(f"✅ Optimizer: AdamW with lr=5e-5, warmup={int(0.1*total_steps)} steps")

    # ✅ FIX #1: Create IntegratorLoss (was not being used at all)
    integrator_loss_fn = IntegratorLoss(
        target_value=5.0,
        lambda_mean_init=1.0,
        lambda_speed=0.1,
        lambda_energy=0.01,
        annealing_epochs=num_epochs,
        variance_weighted=True,
        task_loss_type='ce'  # ✅ Use CrossEntropy for language modeling (not MSE)
    )
    print(f"✅ Loss function: IntegratorLoss with CrossEntropy + trajectory regularization")

    # Create scheduler (optional)
    cycle_scheduler = create_cycle_scheduler(preset='balanced')

    # Training loop
    print("\nStarting training...")
    print("="*70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Update phase
        phase_info = cycle_scheduler.step(epoch)
        print(f"  Phase: {phase_info['phase_name']}")

        # Update IntegratorLoss phase
        integrator_loss_fn.set_exploration_phase(phase_info['phase_name'] == 'exploration')

        # Train with IntegratorLoss
        avg_loss = train_epoch(model, dataloader, integrator_loss_fn, optimizer, lr_scheduler, device, epoch)
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Get inference stats (adaptive early stopping)
        if epoch % 2 == 0:
            stats = model.get_inference_stats()
            if stats:
                print(f"  Inference stats: {stats}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    # Save model
    save_path = 'checkpoints/simple_model.pt'
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        if tokenizer:
            # Test with real text
            prompt_text = "Once upon a time"
            print(f"\n📝 Prompt: '{prompt_text}'")

            # Tokenize
            prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
            print(f"   Tokenized: {prompt_ids.shape}")

            # Generate
            output = model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )

            # Decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n🎯 Generated text:")
            print(f"   {generated_text}")
        else:
            # Test with synthetic data
            prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
            output = model.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            print(f"Generated {output.size(1)} tokens")
            print(f"Output shape: {output.shape}")

    print("\n✅ Example completed successfully!")


if __name__ == '__main__':
    main()
