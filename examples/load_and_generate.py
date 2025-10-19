"""
Load trained INL-LLM model and tokenizer for text generation.

Usage:
    python load_and_generate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from inl_llm.models import UltraOptimizedIntegratorLanguageModel


def load_model_and_tokenizer(checkpoint_dir='checkpoints/inl_1b_model'):
    """Load model and tokenizer from checkpoint directory."""

    print(f"Loading from: {checkpoint_dir}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    print(f"   ✅ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Create model architecture (same as training)
    print("\n2. Creating model architecture...")
    model = UltraOptimizedIntegratorLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=1600,
        num_layers=28,
        num_heads=25,
        num_iterations_per_layer=10,
        feedforward_dim=6400,
        max_seq_len=2048,
        use_lowrank_embeddings=True,
        lowrank_ratio=0.125,
        use_gradient_checkpointing=False,  # Disable for inference
        use_shared_controllers=True,
        hierarchical_group_size=64,
        excitation_sparsity=0.1
    )
    print(f"   ✅ Model created ({model.get_num_params():,} parameters)")

    # Load weights
    print("\n3. Loading model weights...")
    model_path = os.path.join(checkpoint_dir, 'pytorch_model.pt')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"   ✅ Weights loaded from {model_path}")

    # Set to eval mode
    model.eval()

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9, device='cpu'):
    """Generate text from a prompt."""

    # Move model to device
    model = model.to(device)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    print("="*70)
    print("INL-LLM TEXT GENERATION")
    print("="*70)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load model and tokenizer
    checkpoint_dir = 'checkpoints/inl_1b_model'

    if not os.path.exists(checkpoint_dir):
        print(f"\n❌ Checkpoint not found: {checkpoint_dir}")
        print("   Please train the model first: python simple_training.py")
        return

    model, tokenizer = load_model_and_tokenizer(checkpoint_dir)

    # Test prompts
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far, far away",
    ]

    print("\n" + "="*70)
    print("GENERATING TEXT")
    print("="*70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
        print("-" * 50)

        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            device=device
        )

        print(f"Generated:\n{generated}\n")

    print("="*70)
    print("✅ Generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
