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
