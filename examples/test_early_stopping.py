"""
Test Adaptive Early Stopping Performance

Compares inference speed with and without early stopping.
Shows the speedup gained from adaptive iteration control.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from inl_llm.models import UltraOptimizedIntegratorLanguageModel


def benchmark_inference(
    model,
    input_ids,
    num_runs=10,
    warmup=3
):
    """Benchmark inference speed."""
    device = next(model.parameters()).device

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)

    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize() if device.type == 'cuda' else None

    end = time.time()

    avg_time = (end - start) / num_runs
    return avg_time


def get_avg_iterations(model):
    """Get average iterations used from adaptive layers."""
    total_avg = 0.0
    count = 0

    for layer in model.layers:
        if hasattr(layer.inl, 'avg_iterations'):
            total_avg += layer.inl.avg_iterations.item()
            count += 1

    return total_avg / count if count > 0 else None


def main():
    print("="*70)
    print("ADAPTIVE EARLY STOPPING - PERFORMANCE TEST")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test configuration
    vocab_size = 50257
    batch_size = 4
    seq_len = 64
    num_runs = 20

    # Create test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Benchmark runs: {num_runs}")

    # Model 1: WITH adaptive early stopping
    print(f"\n{'='*70}")
    print("Creating model WITH adaptive early stopping...")

    model_adaptive = UltraOptimizedIntegratorLanguageModel(
        vocab_size=vocab_size,
        d_model=1600,
        num_layers=28,
        num_heads=25,
        num_iterations_per_layer=10,
        feedforward_dim=6400,
        use_adaptive_stopping=True,  # ✅ Early stopping ON
        adaptive_convergence_threshold=0.001
    ).to(device)

    model_adaptive.eval()

    print(f"✅ Model created with adaptive early stopping")
    print(f"   Max iterations: 10")
    print(f"   Min iterations: 3")
    print(f"   Convergence threshold: 0.001")

    # Model 2: WITHOUT adaptive early stopping
    print(f"\n{'='*70}")
    print("Creating model WITHOUT adaptive early stopping...")

    model_fixed = UltraOptimizedIntegratorLanguageModel(
        vocab_size=vocab_size,
        d_model=1600,
        num_layers=28,
        num_heads=25,
        num_iterations_per_layer=10,
        feedforward_dim=6400,
        use_adaptive_stopping=False  # ❌ Early stopping OFF
    ).to(device)

    model_fixed.eval()

    print(f"✅ Model created with fixed iterations")
    print(f"   Fixed iterations: 10 (always)")

    # Benchmark
    print(f"\n{'='*70}")
    print("Running benchmarks...")
    print(f"{'='*70}\n")

    # Fixed iterations
    print("⏱️  Benchmarking FIXED iterations model...")
    time_fixed = benchmark_inference(model_fixed, input_ids, num_runs)
    print(f"   Average time: {time_fixed*1000:.2f} ms/batch")

    # Adaptive iterations
    print("\n⏱️  Benchmarking ADAPTIVE iterations model...")
    time_adaptive = benchmark_inference(model_adaptive, input_ids, num_runs)
    avg_iters = get_avg_iterations(model_adaptive)

    print(f"   Average time: {time_adaptive*1000:.2f} ms/batch")
    if avg_iters:
        print(f"   Average iterations used: {avg_iters:.2f} / 10")

    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    speedup = time_fixed / time_adaptive
    time_saved = (time_fixed - time_adaptive) * 1000

    print(f"📊 Performance Comparison:")
    print(f"  Fixed iterations:     {time_fixed*1000:.2f} ms  (10 iterations always)")
    print(f"  Adaptive iterations:  {time_adaptive*1000:.2f} ms  ({avg_iters:.2f} iterations avg)" if avg_iters else f"  Adaptive iterations:  {time_adaptive*1000:.2f} ms")
    print(f"\n🚀 Speedup: {speedup:.2f}× faster")
    print(f"⏱️  Time saved: {time_saved:.2f} ms per batch")

    if avg_iters:
        theoretical_speedup = 10 / avg_iters
        print(f"📈 Theoretical speedup: {theoretical_speedup:.2f}× (based on iteration count)")

    # Memory test
    print(f"\n{'='*70}")
    print("Memory Usage:")
    print(f"{'='*70}\n")

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model_adaptive(input_ids)

        mem_adaptive = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model_fixed(input_ids)

        mem_fixed = torch.cuda.max_memory_allocated() / 1024**2

        print(f"  Fixed iterations:     {mem_fixed:.2f} MB")
        print(f"  Adaptive iterations:  {mem_adaptive:.2f} MB")
        print(f"  Memory saved:         {mem_fixed - mem_adaptive:.2f} MB")
    else:
        print("  (GPU memory stats not available on CPU)")

    print(f"\n{'='*70}")
    print("✅ EARLY STOPPING BENCHMARK COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
