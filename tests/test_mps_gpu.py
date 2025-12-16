#!/usr/bin/env python
"""
Test script to verify Apple M2 Pro GPU (MPS) is working.
"""
import torch
from sentence_transformers import SentenceTransformer
import time
import sys

def test_pytorch_mps():
    """Test PyTorch MPS availability."""
    print("="*60)
    print("TEST 1: PyTorch MPS Availability")
    print("="*60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        print(f"Test tensor on MPS: {x}")
        print("✅ PyTorch MPS is working!\n")
        return True
    else:
        print("❌ MPS not available\n")
        return False


def test_sentence_transformers():
    """Test sentence transformers with MPS."""
    print("="*60)
    print("TEST 2: Sentence Transformers on MPS")
    print("="*60)

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple M2 Pro GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ Using NVIDIA GPU")
    else:
        device = "cpu"
        print("⚠️  Using CPU (no GPU detected)")

    # Load small model for testing
    print("\nLoading model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    # Test encoding
    texts = [
        "This is a test sentence about artificial intelligence",
        "Machine learning models can process natural language",
        "Neural networks are used in computer vision tasks"
    ]

    print("Encoding 3 test sentences...")
    embeddings = model.encode(texts, show_progress_bar=False)

    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"   Embedding shape: {embeddings[0].shape}")
    print(f"   Device used: {device}\n")

    return device


def benchmark_performance(device):
    """Benchmark encoding speed."""
    print("="*60)
    print("TEST 3: Performance Benchmark")
    print("="*60)

    print(f"Testing on: {device.upper()}")
    print("Loading model: anferico/bert-for-patents ...")

    model = SentenceTransformer('anferico/bert-for-patents', device=device)

    # Create test data
    test_texts = [
        "A method for neural network training using backpropagation"
    ] * 100

    # Warmup
    print("Warming up...")
    _ = model.encode(test_texts[:10], show_progress_bar=False)

    # Benchmark
    print("Benchmarking 100 texts...")
    start = time.time()
    embeddings = model.encode(test_texts, batch_size=32, show_progress_bar=False)
    elapsed = time.time() - start

    speed = len(test_texts) / elapsed

    print(f"\n✅ Performance Results:")
    print(f"   Processed: {len(test_texts)} texts")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {speed:.1f} texts/sec")

    if device == "mps":
        memory_gb = torch.mps.current_allocated_memory() / 1024**3
        print(f"   GPU Memory: {memory_gb:.2f} GB")

    print(f"\n   Estimated time for 900,000 patents: {900000/speed/60:.1f} minutes")

    return speed


def main():
    """Run all tests."""
    print("\n" + "🍎" * 30)
    print("Apple M2 Pro GPU Test Suite")
    print("🍎" * 30 + "\n")

    # Test 1: PyTorch MPS
    mps_available = test_pytorch_mps()

    if not mps_available:
        print("⚠️  MPS not available. Make sure you have:")
        print("   - macOS 12.3 or later")
        print("   - PyTorch 2.0 or later")
        print("   - Apple Silicon Mac (M1/M2/M3)")
        sys.exit(1)

    # Test 2: Sentence Transformers
    device = test_sentence_transformers()

    # Test 3: Benchmark
    speed = benchmark_performance(device)

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    if device == "mps" and speed > 30:
        print("✅ All tests passed!")
        print("✅ GPU acceleration is working properly")
        print(f"✅ Performance: {speed:.1f} texts/sec")
        print("\nYou're ready to process patents on your M2 Pro! 🚀")
    elif device == "mps":
        print("⚠️  MPS is available but performance seems low")
        print("   Consider:")
        print("   - Closing other applications")
        print("   - Increasing batch size")
        print("   - Checking Activity Monitor")
    else:
        print("⚠️  Not using MPS acceleration")
        print("   Check the troubleshooting section in docs/apple_silicon_setup.md")

    print("\n")


if __name__ == "__main__":
    main()
