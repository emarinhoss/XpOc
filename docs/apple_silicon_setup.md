# Apple Silicon (M2 Pro) Setup Guide

Complete guide to set up the patent categorization pipeline on Apple M2 Pro with GPU acceleration.

## Overview

Apple M2 Pro has:
- **Unified memory architecture**: GPU and CPU share memory
- **Metal Performance Shaders (MPS)**: Apple's GPU acceleration framework
- **PyTorch MPS backend**: Native GPU support for ML workloads
- **Excellent performance**: Often comparable to mid-range NVIDIA GPUs

## Prerequisites

### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Conda (Miniforge for Apple Silicon)

**Recommended: Miniforge (optimized for ARM64)**

```bash
# Download Miniforge for Apple Silicon
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

# Install
bash Miniforge3-MacOSX-arm64.sh

# Follow prompts, then restart terminal
```

**Alternative: Anaconda**
```bash
# Download from https://www.anaconda.com/download
# Choose "Apple Silicon" version
```

### 3. Verify Installation

```bash
conda --version
# Should show: conda 23.x.x or newer

python --version
# Should show: Python 3.x.x
```

## Environment Setup

### Option 1: Using environment.yml (Recommended)

**Create `environment_apple_silicon.yml`:**

```yaml
name: xpoc-m2
channels:
  - conda-forge
  - pytorch
  - defaults

dependencies:
  - python=3.11
  - pip

  # Core scientific computing (ARM64 optimized)
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scipy

  # PyTorch with MPS support (Apple Silicon)
  - pytorch::pytorch>=2.0.0
  - pytorch::torchvision

  # Visualization
  - matplotlib
  - seaborn
  - plotly>=5.0.0

  # Utilities
  - pyyaml
  - tqdm
  - openpyxl

  # Development tools
  - jupyter
  - pytest

  # Install via pip (better compatibility)
  - pip:
      - sentence-transformers>=2.2.0
      - transformers>=4.30.0
      - accelerate>=0.20.0
      - openai>=1.0.0
      - kaleido
      - scikit-learn
```

**Create the environment:**

```bash
# Navigate to repo root
cd /path/to/XpOc

# Create environment from file
conda env create -f environment_apple_silicon.yml

# Activate environment
conda activate xpoc-m2
```

### Option 2: Manual Setup

```bash
# Create environment
conda create -n xpoc-m2 python=3.11 -y

# Activate
conda activate xpoc-m2

# Install PyTorch with MPS support
conda install pytorch::pytorch torchvision -c pytorch -y

# Install scientific computing packages
conda install numpy pandas scipy matplotlib seaborn pyyaml tqdm openpyxl jupyter pytest -c conda-forge -y

# Install plotly
conda install plotly>=5.0.0 -c plotly -y

# Install ML packages via pip (better compatibility)
pip install sentence-transformers>=2.2.0
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install openai>=1.0.0
pip install scikit-learn
pip install kaleido
```

## Verify GPU Acceleration

### Test 1: Check PyTorch MPS

Create `test_mps.py`:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    print(f"\nTest tensor on MPS: {x}")
    print("✅ MPS is working!")
else:
    print("❌ MPS not available")
```

Run:
```bash
python test_mps.py
```

**Expected output:**
```
PyTorch version: 2.1.0
MPS available: True
MPS built: True

Test tensor on MPS: tensor([1., 1., 1., 1., 1.], device='mps:0')
✅ MPS is working!
```

### Test 2: Test Sentence Transformers

Create `test_embeddings.py`:

```python
from sentence_transformers import SentenceTransformer
import torch

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple M2 Pro GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ Using NVIDIA GPU")
else:
    device = "cpu"
    print("⚠️  Using CPU (no GPU detected)")

# Load model
print("\nLoading model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Test encoding
texts = [
    "This is a test sentence",
    "Here is another sentence",
    "And one more for good measure"
]

print("Encoding texts...")
embeddings = model.encode(texts, show_progress_bar=True)

print(f"\n✅ Success! Generated {len(embeddings)} embeddings")
print(f"Embedding shape: {embeddings[0].shape}")
print(f"Device used: {device}")
```

Run:
```bash
python test_embeddings.py
```

**Expected output:**
```
✅ Using Apple M2 Pro GPU (MPS)

Loading model...
Encoding texts...
100%|██████████| 1/1 [00:00<00:00, 12.34it/s]

✅ Success! Generated 3 embeddings
Embedding shape: (384,)
Device used: mps
```

### Test 3: Performance Benchmark

Create `benchmark_gpu.py`:

```python
from sentence_transformers import SentenceTransformer
import torch
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Testing on: {device.upper()}")

# Load model
model = SentenceTransformer('anferico/bert-for-patents', device=device)

# Create test data (100 patents)
test_texts = ["AI patent about neural networks"] * 100

# Warmup
_ = model.encode(test_texts[:10], show_progress_bar=False)

# Benchmark
print("\nBenchmarking...")
start = time.time()
embeddings = model.encode(test_texts, batch_size=32, show_progress_bar=True)
elapsed = time.time() - start

print(f"\nProcessed {len(test_texts)} texts in {elapsed:.2f}s")
print(f"Speed: {len(test_texts)/elapsed:.1f} texts/sec")
print(f"Memory allocated: {torch.mps.current_allocated_memory()/1024**3:.2f} GB" if device == "mps" else "N/A")
```

Run:
```bash
python benchmark_gpu.py
```

**Expected performance (M2 Pro):**
```
Testing on: MPS

Benchmarking...
100%|██████████| 4/4 [00:02<00:00, 1.84it/s]

Processed 100 texts in 2.17s
Speed: 46.1 texts/sec
Memory allocated: 0.85 GB
```

## Running the Scripts

### 1. Patent Categorization (Zero-Shot)

**Basic usage:**
```bash
conda activate xpoc-m2

python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/patents.csv \
    --output-file data/patents_categorized.csv \
    --device mps \
    --batch-size 32
```

**Key parameters for M2 Pro:**
- `--device mps`: Use Apple GPU
- `--batch-size 32`: Good balance for M2 Pro (16-64 typical range)
- `--model-name anferico/bert-for-patents`: Default, optimized for patents

### 2. Ensemble Voting

```bash
python src/scripts/categorize_patents_ensemble.py \
    --input-file data/patents.csv \
    --output-file data/patents_ensemble.csv \
    --device mps \
    --batch-size 32
```

**Note:** Ensemble runs 5 models sequentially, so it takes ~5x longer than single model.

### 3. Category-Specific Matching

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/patents_categorized.csv \
    --onet-file data/onet/Task_Ratings.xlsx \
    --output-dir results/category_specific \
    --device mps \
    --batch-size 32
```

### 4. Visualizations

```bash
# Sankey diagram
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --output-format html

# Scatterplot
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --animation --static
```

## Performance Optimization

### Batch Size Guidelines

**For M2 Pro (16 GB unified memory):**

| Model Size | Batch Size | Memory Usage | Speed |
|------------|------------|--------------|-------|
| Small (110M params) | 64-128 | ~2 GB | Fastest |
| Medium (330M params) | 32-64 | ~4 GB | Fast |
| Large (700M params) | 16-32 | ~8 GB | Medium |
| Very Large (1B+ params) | 8-16 | ~12 GB | Slower |

**Finding optimal batch size:**

```bash
# Start with 32
--batch-size 32

# If memory error → reduce to 16
--batch-size 16

# If using <50% memory → increase to 64
--batch-size 64
```

### Memory Management

**Monitor memory usage:**

```python
# Add to scripts
import torch

if torch.backends.mps.is_available():
    allocated = torch.mps.current_allocated_memory() / 1024**3
    print(f"GPU memory allocated: {allocated:.2f} GB")
```

**Clear memory between runs:**

```python
import torch
torch.mps.empty_cache()
```

### Speed Expectations

**M2 Pro performance estimates:**

| Task | Model | Batch Size | Speed | Time (900k patents) |
|------|-------|------------|-------|---------------------|
| Zero-shot (single) | bert-for-patents | 32 | ~400-600/sec | ~25-40 min |
| Zero-shot (single) | MiniLM | 64 | ~800-1200/sec | ~12-20 min |
| Ensemble (5 models) | Various | 32 | ~80-120/sec | ~2-3 hours |
| Matching | bert-for-patents | 32 | Varies | ~1-2 hours |

**Factors affecting speed:**
- Model size (smaller = faster)
- Batch size (larger = faster, up to memory limit)
- Input text length (shorter = faster)
- Background processes (close other apps for max speed)

## Troubleshooting

### MPS Not Available

**Problem:** `MPS available: False`

**Solutions:**

1. **Check macOS version:**
   ```bash
   sw_vers
   ```
   Requires macOS 12.3+ for MPS

2. **Update PyTorch:**
   ```bash
   pip install --upgrade torch torchvision
   ```

3. **Check system:**
   ```bash
   system_profiler SPDisplaysDataType | grep Metal
   ```
   Should show Metal support

### Out of Memory Errors

**Problem:** `RuntimeError: MPS out of memory`

**Solutions:**

1. **Reduce batch size:**
   ```bash
   --batch-size 16  # or even 8
   ```

2. **Use smaller model:**
   ```bash
   --model-name sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Close other apps:**
   - Close browsers, IDEs, etc.
   - Free up RAM

4. **Clear cache:**
   ```python
   import torch
   torch.mps.empty_cache()
   ```

### Slow Performance

**Problem:** Running slower than expected

**Solutions:**

1. **Verify MPS is being used:**
   ```python
   print(model.device)  # Should be 'mps:0'
   ```

2. **Increase batch size:**
   ```bash
   --batch-size 64  # if memory allows
   ```

3. **Check Activity Monitor:**
   - Look for "Python" process
   - Should show ~100% CPU + GPU activity

4. **Reduce concurrent processes:**
   - Close background apps
   - Disable Time Machine during processing

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
# Reinstall package
pip install --upgrade <package-name>

# Or reinstall environment
conda deactivate
conda env remove -n xpoc-m2
conda env create -f environment_apple_silicon.yml
```

## Best Practices

### 1. Always Activate Environment

```bash
# Before running any scripts
conda activate xpoc-m2

# Check active environment
conda info --envs
```

### 2. Monitor System

```bash
# Terminal 1: Run script
python src/scripts/categorize_patents_zeroshot.py ...

# Terminal 2: Monitor
watch -n 1 "ps aux | grep python"
```

### 3. Use Appropriate Models

**Fast (small models):**
- `sentence-transformers/all-MiniLM-L6-v2` (80M params)
- `sentence-transformers/all-MiniLM-L12-v2` (110M params)

**Balanced (medium models):**
- `anferico/bert-for-patents` (110M params) - **Recommended**
- `sentence-transformers/all-mpnet-base-v2` (330M params)

**High quality (large models):**
- `allenai/scibert_scivocab_uncased` (110M params)
- `google/embeddinggemma-300m` (300M params)

### 4. Save Checkpoints

For large datasets, process in chunks:

```bash
# Process first 100k
head -100001 data/patents.csv > data/patents_batch1.csv
python categorize_patents_zeroshot.py --input-file data/patents_batch1.csv ...

# Process next 100k
tail -n +100002 data/patents.csv | head -100000 > data/patents_batch2.csv
python categorize_patents_zeroshot.py --input-file data/patents_batch2.csv ...

# Combine results
tail -n +2 data/patents_categorized_batch2.csv >> data/patents_categorized_batch1.csv
```

## Updating the Environment

### Add New Package

```bash
conda activate xpoc-m2
conda install package-name
# or
pip install package-name
```

### Update All Packages

```bash
conda activate xpoc-m2
conda update --all
pip list --outdated | cut -d ' ' -f1 | xargs -n1 pip install -U
```

### Export Environment

```bash
# Save current environment
conda env export > environment_backup.yml

# Or just pip packages
pip freeze > requirements_backup.txt
```

## Comparison: M2 Pro vs Other Hardware

**900,000 patents with bert-for-patents:**

| Hardware | Batch Size | Speed | Total Time |
|----------|------------|-------|------------|
| **M2 Pro (16GB)** | 32 | 400-600/sec | ~25-40 min |
| M1 Max (32GB) | 64 | 600-900/sec | ~17-25 min |
| M2 Ultra (64GB) | 128 | 1000-1500/sec | ~10-15 min |
| RTX 3090 (24GB) | 128 | 1500-2000/sec | ~8-12 min |
| RTX 4090 (24GB) | 256 | 2500-3500/sec | ~4-6 min |
| CPU (16 cores) | 32 | 50-100/sec | ~2.5-5 hours |

**M2 Pro is excellent for this workload!** About 5-8x faster than CPU-only.

## Next Steps

1. ✅ Create conda environment
2. ✅ Run test scripts to verify GPU
3. ✅ Benchmark performance
4. 📊 Start processing your patents
5. 🎨 Create visualizations

Your M2 Pro is well-suited for this pipeline! Expect to process 900k patents in under an hour with proper optimization.
