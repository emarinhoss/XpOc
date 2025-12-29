# Ensemble Aggregation Workflow

This guide explains how to perform ensemble voting by running models separately and then aggregating their results. This approach is more flexible and efficient than running all models together.

## Why Aggregate Separately?

**Benefits:**
- **Parallel execution**: Run models on different machines simultaneously
- **Lower memory**: Only one model loaded at a time
- **Flexibility**: Mix and match model results without re-running
- **Fault tolerance**: If one model fails, others can continue
- **Experimentation**: Try different model combinations easily
- **Time management**: Run fast models first, slow ones later

**Comparison:**

| Approach | Memory | Parallelizable | Flexibility | Restart Cost |
|----------|--------|----------------|-------------|--------------|
| **Integrated** (`categorize_patents_ensemble.py`) | ~1.7 GB | No | Low | High (re-run all) |
| **Aggregation** (`aggregate_ensemble_voting.py`) | ~400 MB | Yes | High | Low (re-run one) |

## Complete Workflow

### Step 1: Run Individual Models

Run `categorize_patents_zeroshot.py` with different models to generate separate result files.

```bash
# Model 1: BERT for Patents
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file results/patents_bert.csv \
    --model-name anferico/bert-for-patents \
    --model-type sentence-transformers \
    --device cuda \
    --batch-size 64

# Model 2: EmbeddingGemma
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file results/patents_gemma.csv \
    --model-name google/embeddinggemma-300m \
    --model-type auto-model \
    --pooling mean \
    --device cuda \
    --batch-size 64

# Model 3: MPNet
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file results/patents_mpnet.csv \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --model-type sentence-transformers \
    --device cuda \
    --batch-size 64

# Model 4: MiniLM
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file results/patents_minilm.csv \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --model-type sentence-transformers \
    --device cuda \
    --batch-size 128

# Model 5: SciBERT
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file results/patents_scibert.csv \
    --model-name allenai/scibert_scivocab_uncased \
    --model-type auto-model \
    --pooling mean \
    --device cuda \
    --batch-size 64
```

**Pro tip:** Run these in parallel on different machines or GPUs!

```bash
# On Machine 1 (or GPU 0)
CUDA_VISIBLE_DEVICES=0 python ... --output-file results/patents_bert.csv &
CUDA_VISIBLE_DEVICES=0 python ... --output-file results/patents_gemma.csv &

# On Machine 2 (or GPU 1)
CUDA_VISIBLE_DEVICES=1 python ... --output-file results/patents_mpnet.csv &
CUDA_VISIBLE_DEVICES=1 python ... --output-file results/patents_minilm.csv &

# On Machine 3
python ... --output-file results/patents_scibert.csv &
```

### Step 2: Aggregate Results

Once all models have finished, aggregate their predictions:

```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_bert.csv \
                  results/patents_gemma.csv \
                  results/patents_mpnet.csv \
                  results/patents_minilm.csv \
                  results/patents_scibert.csv \
    --model-names bert gemma mpnet minilm scibert \
    --output-file results/patents_ensemble.csv
```

**Quality filtering** (optional - keep only high-confidence predictions):

```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_bert.csv \
                  results/patents_gemma.csv \
                  results/patents_mpnet.csv \
                  results/patents_minilm.csv \
                  results/patents_scibert.csv \
    --model-names bert gemma mpnet minilm scibert \
    --min-confidence 0.7 \
    --min-votes 4 \
    --output-file results/patents_ensemble_high_quality.csv
```

This will only include patents where:
- Ensemble confidence ≥ 0.7
- At least 4 out of 5 models agreed on the category

**Alternative: Using a file list**

Create a text file `model_results.txt`:
```
results/patents_bert.csv
results/patents_gemma.csv
results/patents_mpnet.csv
results/patents_minilm.csv
results/patents_scibert.csv
```

Then run:
```bash
python src/scripts/aggregate_ensemble_voting.py \
    --file-list model_results.txt \
    --model-names bert gemma mpnet minilm scibert \
    --output-file results/patents_ensemble.csv
```

### Step 3: Analyze Results

```python
import pandas as pd

df = pd.read_csv('results/patents_ensemble.csv')

print(f"Total patents: {len(df)}")
print(f"\nCategory distribution:")
print(df['ensemble_category'].value_counts())

print(f"\nAverage confidence: {df['ensemble_confidence'].mean():.3f}")

print(f"\nVote distribution:")
print(df['ensemble_votes'].value_counts().sort_index())

# High confidence cases (4-5 votes)
high_conf = df[df['ensemble_votes'] >= 4]
print(f"\nHigh agreement: {len(high_conf)} ({100*len(high_conf)/len(df):.1f}%)")
```

## Quality Control with Filtering

The aggregation script supports filtering to keep only high-quality predictions.

### Confidence Filtering

Keep only patents where the ensemble has high confidence:

```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_*.csv \
    --min-confidence 0.75 \
    --output-file results/ensemble_high_confidence.csv
```

This filters out patents where the average confidence of agreeing models is below 0.75.

### Vote Filtering

Keep only patents where models strongly agree:

```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_*.csv \
    --min-votes 4 \
    --output-file results/ensemble_strong_agreement.csv
```

With 5 models, `--min-votes 4` means at least 4 models voted for the same category.

### Combined Filtering

For maximum quality, combine both filters:

```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_*.csv \
    --min-confidence 0.7 \
    --min-votes 4 \
    --output-file results/ensemble_highest_quality.csv
```

**Quality levels**:
- `--min-votes 5`: Unanimous agreement (highest quality, fewer results)
- `--min-votes 4`: Strong agreement (high quality, balanced)
- `--min-votes 3`: Majority agreement (moderate quality, more results)
- No filter: All predictions (includes uncertain cases)

**Example output with filtering**:
```
Applying filters...
  Confidence filter (>= 0.7): removed 12,451 rows, 887,549 remaining
  Vote filter (>= 4): removed 156,332 rows, 731,217 remaining

Total filtered: 168,783 rows (18.8%)
Remaining: 731,217 rows (81.2%)
```

## Advanced Workflows

### Experiment with Different Model Combinations

You only need to run each model once, then you can try different combinations:

```bash
# Combination 1: All 5 models
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_*.csv \
    --output-file results/ensemble_all5.csv

# Combination 2: Only patent-specific models
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_bert.csv \
                  results/patents_scibert.csv \
                  results/patents_mpnet.csv \
    --output-file results/ensemble_domain.csv

# Combination 3: Fast models only
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_minilm.csv \
                  results/patents_gemma.csv \
                  results/patents_bert.csv \
    --output-file results/ensemble_fast.csv
```

### Add More Models Later

Already have 5 results? Easy to add a 6th:

```bash
# Run new model
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file results/patents_roberta.csv \
    --model-name sentence-transformers/all-roberta-large-v1

# Aggregate with 6 models
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_bert.csv \
                  results/patents_gemma.csv \
                  results/patents_mpnet.csv \
                  results/patents_minilm.csv \
                  results/patents_scibert.csv \
                  results/patents_roberta.csv \
    --model-names bert gemma mpnet minilm scibert roberta \
    --output-file results/patents_ensemble6.csv
```

### Process Subsets Differently

Run fast models on everything, slow models on subset:

```bash
# Fast models on all 900k patents
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/all_patents.csv \
    --output-file results/all_minilm.csv \
    --model-name sentence-transformers/all-MiniLM-L6-v2

# Slow model only on recent patents (2020+)
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/all_patents.csv \
    --output-file results/recent_mpnet.csv \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --start-year 2020

# Aggregate (will only use overlapping patents)
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/all_minilm.csv results/recent_mpnet.csv \
    --output-file results/ensemble_hybrid.csv
```

## Time and Cost Comparison

### For 900k Patents

**Integrated Approach** (`categorize_patents_ensemble.py`):
```
Single GPU: 38 hours straight
Cannot parallelize
Memory: 1.7 GB
```

**Aggregation Approach** (run separately + aggregate):
```
5 GPUs in parallel: ~8 hours (5 models * ~8 hours / 5 GPUs)
Or 1 GPU sequential: ~40 hours (5 models * ~8 hours)
Memory per model: ~400 MB
Aggregation: ~5 minutes
```

**Cost on Cloud GPU (V100 @ $2/hour):**
- Integrated: $76 (38 hours * $2)
- Aggregation (5 GPUs parallel): $80 (5 GPUs * 8 hours * $2)
- Aggregation (1 GPU sequential): $80 (40 hours * $2)

**Winner:** Aggregation with parallel execution (8 hours vs 38 hours!)

## Output Format

The aggregation script produces the same output format as the integrated ensemble:

### Individual Model Columns
```
bert_category, bert_confidence,
gemma_category, gemma_confidence,
mpnet_category, mpnet_confidence,
minilm_category, minilm_confidence,
scibert_category, scibert_confidence
```

### Ensemble Columns
```
ensemble_category         # Final voted category
ensemble_confidence       # Average confidence of agreeing models
ensemble_votes           # Number of votes winner received
```

## Troubleshooting

### Issue: Files have different patent IDs

**Problem:** Files were generated from different input sources or subsets.

**Solution:** The script uses `inner join`, keeping only patents present in ALL files.

```bash
# Check overlap before aggregating
python -c "
import pandas as pd
files = ['results/patents_bert.csv', 'results/patents_gemma.csv']
df1 = pd.read_csv(files[0])
df2 = pd.read_csv(files[1])
overlap = set(df1['application_id']) & set(df2['application_id'])
print(f'Overlap: {len(overlap)} patents')
print(f'File 1 only: {len(df1) - len(overlap)}')
print(f'File 2 only: {len(df2) - len(overlap)}')
"
```

### Issue: Column names don't match

**Problem:** Files have different column names (e.g., `category` vs `ai_category`).

**Solution:** Use `--category-column` and `--confidence-column` flags:

```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/*.csv \
    --category-column category \
    --confidence-column confidence_score \
    --output-file results/ensemble.csv
```

### Issue: One model file is corrupted

**Problem:** One of the 5 files has errors or missing data.

**Solution:** Just exclude it and run with 4 models:

```bash
# Skip the problematic file
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_bert.csv \
                  results/patents_gemma.csv \
                  results/patents_mpnet.csv \
                  results/patents_minilm.csv \
    --output-file results/ensemble_4models.csv
```

Or re-run just that one model.

### Issue: Want to weight models differently

**Problem:** Want to give some models more votes than others.

**Solution:** Duplicate the file in the input list:

```bash
# Give bert 2 votes, others 1 vote each
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_bert.csv \
                  results/patents_bert.csv \
                  results/patents_gemma.csv \
                  results/patents_mpnet.csv \
                  results/patents_minilm.csv \
    --output-file results/ensemble_weighted.csv
```

## Best Practices

### 1. Use Consistent Input Data

All models should process the **exact same patents** from the **same input file**:

```bash
INPUT_FILE="data/processed/patents.csv"

# All models use same input
python src/scripts/categorize_patents_zeroshot.py --input-file $INPUT_FILE --output-file results/bert.csv ...
python src/scripts/categorize_patents_zeroshot.py --input-file $INPUT_FILE --output-file results/gemma.csv ...
```

### 2. Name Files Systematically

Use clear, consistent naming:
```
results/patents_bert.csv
results/patents_gemma.csv
results/patents_mpnet.csv
```

Not:
```
results/output1.csv
results/final_v2.csv
results/test_results.csv
```

### 3. Keep Model Names Consistent

Use the same model names across runs:

```bash
# Define model names once
MODEL_NAMES="bert gemma mpnet minilm scibert"

# Use in aggregation
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/patents_*.csv \
    --model-names $MODEL_NAMES \
    --output-file results/ensemble.csv
```

### 4. Validate Before Aggregating

Check that all files are complete:

```bash
# Count rows in each file
for file in results/patents_*.csv; do
    lines=$(wc -l < "$file")
    echo "$file: $lines patents"
done

# Should all be the same (plus 1 for header)
```

### 5. Save Individual Results

Don't delete individual model results after aggregation - they're useful for:
- Debugging disagreements
- Trying different combinations
- Analyzing model-specific patterns
- Re-aggregating with new models

## Shell Script for Complete Workflow

```bash
#!/bin/bash
# run_ensemble_workflow.sh

set -e  # Exit on error

INPUT_FILE="data/processed/patents.csv"
RESULTS_DIR="results"
DEVICE="cuda"
BATCH_SIZE=64

mkdir -p "$RESULTS_DIR"

echo "Step 1: Running individual models..."

# Model 1: BERT for Patents
echo "Running BERT for Patents..."
python src/scripts/categorize_patents_zeroshot.py \
    --input-file "$INPUT_FILE" \
    --output-file "$RESULTS_DIR/patents_bert.csv" \
    --model-name anferico/bert-for-patents \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"

# Model 2: EmbeddingGemma
echo "Running EmbeddingGemma..."
python src/scripts/categorize_patents_zeroshot.py \
    --input-file "$INPUT_FILE" \
    --output-file "$RESULTS_DIR/patents_gemma.csv" \
    --model-name google/embeddinggemma-300m \
    --model-type auto-model \
    --pooling mean \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"

# Model 3: MPNet
echo "Running MPNet..."
python src/scripts/categorize_patents_zeroshot.py \
    --input-file "$INPUT_FILE" \
    --output-file "$RESULTS_DIR/patents_mpnet.csv" \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"

# Model 4: MiniLM
echo "Running MiniLM..."
python src/scripts/categorize_patents_zeroshot.py \
    --input-file "$INPUT_FILE" \
    --output-file "$RESULTS_DIR/patents_minilm.csv" \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --device "$DEVICE" \
    --batch-size 128

# Model 5: SciBERT
echo "Running SciBERT..."
python src/scripts/categorize_patents_zeroshot.py \
    --input-file "$INPUT_FILE" \
    --output-file "$RESULTS_DIR/patents_scibert.csv" \
    --model-name allenai/scibert_scivocab_uncased \
    --model-type auto-model \
    --pooling mean \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"

echo "Step 2: Aggregating ensemble results..."
python src/scripts/aggregate_ensemble_voting.py \
    --input-files "$RESULTS_DIR"/patents_bert.csv \
                  "$RESULTS_DIR"/patents_gemma.csv \
                  "$RESULTS_DIR"/patents_mpnet.csv \
                  "$RESULTS_DIR"/patents_minilm.csv \
                  "$RESULTS_DIR"/patents_scibert.csv \
    --model-names bert gemma mpnet minilm scibert \
    --output-file "$RESULTS_DIR/patents_ensemble.csv"

echo "Done! Results saved to $RESULTS_DIR/patents_ensemble.csv"
```

Run with:
```bash
chmod +x run_ensemble_workflow.sh
./run_ensemble_workflow.sh
```

## Summary

**When to use aggregation approach:**
- ✅ Running models on multiple machines/GPUs in parallel
- ✅ Want to experiment with different model combinations
- ✅ Have limited memory per machine
- ✅ Want fault tolerance (re-run individual models if needed)
- ✅ Processing very large datasets (900k+ patents)

**When to use integrated approach:**
- ✅ Running everything on one machine sequentially
- ✅ Want simplicity (single command)
- ✅ Have enough memory for all models (~1.7 GB)
- ✅ Processing smaller datasets (<100k patents)

For **900k patents**, the **aggregation approach with parallel execution** is recommended - it's 5x faster and more flexible!
