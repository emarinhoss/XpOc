# Ensemble Voting for Patent Categorization

This document explains the ensemble voting approach for patent categorization, which uses multiple models to improve classification accuracy through majority voting.

## Overview

Instead of relying on a single model, the ensemble approach:
1. **Runs 5 different embedding models** on each patent
2. **Each model votes** for a category (equal weight: 1 vote each)
3. **Majority wins**: The category with the most votes is selected
4. **Confidence**: Average confidence of models that agreed on the winning category

## Why Use Ensemble Voting?

**Benefits:**
- **Higher accuracy**: Reduces errors from any single model's weaknesses
- **More robust**: Less sensitive to edge cases or model-specific biases
- **Confidence calibration**: Multi-model agreement indicates reliability
- **No training required**: Leverages pre-trained models without fine-tuning

**Trade-offs:**
- **5x slower**: Must run 5 models instead of 1
- **Higher memory**: All 5 models loaded in memory
- **More complex**: More components that can fail

## The 5 Default Models

| Model | Type | Strength | Size |
|-------|------|----------|------|
| **anferico/bert-for-patents** | sentence-transformers | Patent-specific vocabulary | 420MB |
| **google/embeddinggemma-300m** | auto-model | Efficient, general-purpose | 300MB |
| **all-mpnet-base-v2** | sentence-transformers | High quality semantic matching | 420MB |
| **all-MiniLM-L6-v2** | sentence-transformers | Fast, captures key patterns | 80MB |
| **allenai/scibert_scivocab_uncased** | auto-model | Scientific/technical text | 440MB |

**Total memory**: ~1.7 GB

### Why These 5 Models?

- **Diverse architectures**: Mix of BERT, MPNet, MiniLM for different perspectives
- **Diverse training**: Patents, scientific text, general web text
- **Diverse sizes**: From 80MB (fast) to 440MB (comprehensive)
- **Proven performance**: Well-established models with good benchmarks

## How Voting Works

### Example 1: Clear Winner

Patent: "Convolutional neural network for image classification"

| Model | Predicted Category | Confidence |
|-------|-------------------|------------|
| bert-for-patents | computer vision | 0.89 |
| embeddinggemma | computer vision | 0.85 |
| mpnet | computer vision | 0.91 |
| minilm | machine learning | 0.72 |
| scibert | computer vision | 0.87 |

**Vote count:**
- computer vision: 4 votes
- machine learning: 1 vote

**Result:**
- Category: **computer vision**
- Confidence: (0.89 + 0.85 + 0.91 + 0.87) / 4 = **0.88**
- Votes: **4**

### Example 2: Tie-Breaking

Patent: "Deep learning model for natural language understanding"

| Model | Predicted Category | Confidence |
|-------|-------------------|------------|
| bert-for-patents | NLP | 0.82 |
| embeddinggemma | NLP | 0.79 |
| mpnet | machine learning | 0.85 |
| minilm | machine learning | 0.80 |
| scibert | NLP | 0.65 |

**Vote count:**
- NLP: 3 votes
- machine learning: 2 votes

**Result:**
- Category: **NLP** (has more votes)
- Confidence: (0.82 + 0.79 + 0.65) / 3 = **0.75**
- Votes: **3**

### Example 3: Perfect Tie (2-2-1)

Patent: "Genetic algorithm for parameter optimization"

| Model | Predicted Category | Confidence |
|-------|-------------------|------------|
| bert-for-patents | evolutionary computation | 0.78 |
| embeddinggemma | machine learning | 0.82 |
| mpnet | evolutionary computation | 0.75 |
| minilm | machine learning | 0.85 |
| scibert | optimization | 0.70 |

**Vote count:**
- evolutionary computation: 2 votes (avg confidence: 0.765)
- machine learning: 2 votes (avg confidence: 0.835)
- optimization: 1 vote

**Tie-breaking:** Compare average confidence of tied categories
- evolutionary computation: (0.78 + 0.75) / 2 = 0.765
- machine learning: (0.82 + 0.85) / 2 = **0.835** ← Higher

**Result:**
- Category: **machine learning** (higher confidence in tie)
- Confidence: **0.835**
- Votes: **2**

## Usage

### Basic Usage (Default Models)

```bash
python src/scripts/categorize_patents_ensemble.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_ensemble.csv
```

### With GPU Acceleration

```bash
python src/scripts/categorize_patents_ensemble.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_ensemble.csv \
    --device cuda \
    --batch-size 64
```

### Custom Models Configuration

Create a YAML file with your own 5 models:

```yaml
# my_models.yaml
ensemble_models:
  - name: model1
    model_name: some/model-1
    model_type: sentence-transformers
    pooling: mean

  - name: model2
    model_name: some/model-2
    model_type: auto-model
    pooling: cls
  # ... 3 more models
```

Then use it:

```bash
python src/scripts/categorize_patents_ensemble.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_ensemble.csv \
    --models-config my_models.yaml
```

## Output Format

The output CSV contains:

### Individual Model Predictions

For each model (5 columns per model):
- `{model_name}_category`: Predicted category
- `{model_name}_confidence`: Confidence score

Example columns:
```
bert-for-patents_category, bert-for-patents_confidence,
embeddinggemma_category, embeddinggemma_confidence,
mpnet_category, mpnet_confidence,
minilm_category, minilm_confidence,
scibert_category, scibert_confidence
```

### Ensemble Results

- `ai_category_ensemble`: Final category from voting
- `ai_category_ensemble_confidence`: Average confidence of agreeing models
- `ai_category_ensemble_votes`: Number of votes the winner received (1-5)

### Example Output Row

```csv
application_id,application_title,...,
bert-for-patents_category,bert-for-patents_confidence,
embeddinggemma_category,embeddinggemma_confidence,
mpnet_category,mpnet_confidence,
minilm_category,minilm_confidence,
scibert_category,scibert_confidence,
ai_category_ensemble,ai_category_ensemble_confidence,ai_category_ensemble_votes

US12345,"Neural network for...",
computer vision,0.89,
computer vision,0.85,
computer vision,0.91,
machine learning,0.72,
computer vision,0.87,
computer vision,0.88,4
```

## Analyzing Results

### Check Agreement Levels

```python
import pandas as pd

df = pd.read_csv('data/processed/patents_ensemble.csv')

# High agreement (4-5 votes)
high_agreement = df[df['ai_category_ensemble_votes'] >= 4]
print(f"High agreement: {len(high_agreement)} patents ({100*len(high_agreement)/len(df):.1f}%)")

# Low agreement (2-3 votes)
low_agreement = df[df['ai_category_ensemble_votes'] <= 3]
print(f"Low agreement: {len(low_agreement)} patents ({100*len(low_agreement)/len(df):.1f}%)")
```

### Compare Individual Models

```python
# See which models agree most often
from itertools import combinations

models = ['bert-for-patents', 'embeddinggemma', 'mpnet', 'minilm', 'scibert']

for m1, m2 in combinations(models, 2):
    agreement = (df[f'{m1}_category'] == df[f'{m2}_category']).mean()
    print(f"{m1} vs {m2}: {agreement:.1%} agreement")
```

### Find Disagreement Cases

```python
# Patents where all 5 models disagree (very rare)
def all_different(row):
    cats = [row[f'{m}_category'] for m in models]
    return len(set(cats)) == 5

fully_disagreed = df[df.apply(all_different, axis=1)]
print(f"Complete disagreement: {len(fully_disagreed)} patents")

# Review these manually - might be ambiguous or need new category
if len(fully_disagreed) > 0:
    fully_disagreed[['application_id', 'application_title'] +
                    [f'{m}_category' for m in models]].to_csv('disagreements.csv', index=False)
```

### Compare with Single Model

```python
# How often does ensemble differ from bert-for-patents?
different = df['ai_category_ensemble'] != df['bert-for-patents_category']
print(f"Ensemble differs from BERT: {different.sum()} patents ({100*different.mean():.1f}%)")

# Show examples where they differ
differences = df[different][['application_title', 'bert-for-patents_category',
                             'ai_category_ensemble', 'ai_category_ensemble_votes']]
print(differences.head(10))
```

## Performance

### Processing Time (10,000 patents, CPU)

| Configuration | Time | Notes |
|---------------|------|-------|
| **Single model** (BERT) | 22 min | Baseline |
| **Ensemble (5 models)** | 95 min | ~4.3x slower |
| **Ensemble + GPU** | 25 min | Only ~1.1x slower! |

### Processing Time (900k patents)

| Configuration | Time | Cost |
|---------------|------|------|
| Single model (CPU) | ~30 hours | $0 |
| Ensemble (CPU) | ~130 hours (~5.4 days) | $0 |
| Ensemble (GPU - V100) | ~38 hours | ~$76 (cloud GPU) |
| OpenAI API | ~10 days | $9,000-$18,000 |

**Recommendation:** Use GPU for ensemble to keep processing time reasonable.

## When to Use Ensemble vs Single Model

### Use Ensemble When:
- ✅ Accuracy is critical
- ✅ You have time/compute for 5x longer processing
- ✅ You want detailed analysis of model agreement
- ✅ You need confidence calibration (high votes = high confidence)
- ✅ Processing <100k patents with GPU

### Use Single Model When:
- ✅ Speed is critical
- ✅ Processing 500k+ patents quickly
- ✅ Doing exploratory analysis
- ✅ Limited compute resources
- ✅ The single model is domain-specific (e.g., bert-for-patents)

## Tips for Best Results

1. **Use GPU**: Makes ensemble practical for large datasets
2. **Review low-vote cases**: Patents with 2-3 votes may need manual review
3. **Analyze disagreements**: Learn about edge cases and ambiguities
4. **Customize models**: Replace models in config for your specific domain
5. **Save intermediate results**: Keep individual predictions for analysis

## Troubleshooting

### Issue: Out of memory

**Solution:** Process in smaller chunks or use fewer models
```bash
# Process 10k at a time
head -10001 patents.csv > patents_chunk1.csv
python categorize_patents_ensemble.py --input-file patents_chunk1.csv
```

### Issue: One model fails

**Solution:** The script will stop. Check the error and either:
- Fix the model configuration
- Replace the failing model in config file
- Use a different device (CPU vs CUDA)

### Issue: Very slow on CPU

**Solutions:**
1. Use GPU: `--device cuda`
2. Use smaller batch size: `--batch-size 16`
3. Process subset: `--start-year 2020`
4. Consider single model instead

### Issue: Models disagree often

**Analysis:** This might indicate:
- Patents are genuinely ambiguous
- Categories need refinement
- One model is miscalibrated

**Action:** Review disagreement cases manually and potentially:
- Adjust category definitions
- Add more categories
- Remove problematic model

## Comparing Ensemble vs Single Model

Run both and compare:

```bash
# Single model (fast)
python src/scripts/categorize_patents_zeroshot.py \
    --input-file test_patents.csv \
    --output-file results_single.csv

# Ensemble (accurate)
python src/scripts/categorize_patents_ensemble.py \
    --input-file test_patents.csv \
    --output-file results_ensemble.csv

# Compare
python -c "
import pandas as pd
single = pd.read_csv('results_single.csv')
ensemble = pd.read_csv('results_ensemble.csv')

# Merge on ID
merged = single.merge(ensemble, on='application_id', suffixes=('_single', '_ensemble'))

# How often do they agree?
agreement = (merged['ai_category_single'] == merged['ai_category_ensemble']).mean()
print(f'Agreement: {agreement:.1%}')

# Where ensemble has high confidence (4-5 votes), how often do they agree?
high_conf = merged[merged['ai_category_ensemble_votes'] >= 4]
high_conf_agreement = (high_conf['ai_category_single'] == high_conf['ai_category_ensemble']).mean()
print(f'Agreement on high-confidence cases: {high_conf_agreement:.1%}')
"
```

## Next Steps

1. Test on a small sample (1000 patents)
2. Review disagreement cases
3. Decide if improved accuracy is worth 5x processing time
4. Run on full dataset with GPU if accuracy gains are significant
5. Use ensemble results as ground truth for evaluating single models
