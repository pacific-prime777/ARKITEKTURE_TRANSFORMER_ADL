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
from inl_llm import create_optimized_model
from inl_llm.core import IntegratorLoss, create_cycle_scheduler


class SimpleTextDataset(Dataset):
    """Simple synthetic dataset for demo purposes."""

    def __init__(self, vocab_size=5000, num_samples=1000, seq_len=128):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequences
        seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Target is shifted by 1 (next token prediction)
        return seq[:-1], seq[1:]


def train_epoch(model, dataloader, loss_fn, optimizer, device='cpu'):
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

        # Compute losses
        task_loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)

        # For integrator loss, we'd need trajectory - simplified here
        loss = task_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    return total_loss / num_batches


def main():
    print("="*70)
    print("SIMPLE TRAINING EXAMPLE - INL-LLM")
    print("="*70)

    # Configuration
    vocab_size = 5000
    batch_size = 4
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")

    # Create model
    print("\nCreating optimized model...")
    model = create_optimized_model(
        size='small',
        vocab_size=vocab_size,
        enable_all_optimizations=True
    )
    model = model.to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = SimpleTextDataset(vocab_size=vocab_size, num_samples=100, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

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

        # Train
        avg_loss = train_epoch(model, dataloader, None, optimizer, device)
        print(f"  Average Loss: {avg_loss:.4f}")

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
