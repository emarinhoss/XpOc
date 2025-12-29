# Quick Start: Patent Categorization

This guide helps you get started with categorizing patents into AI categories.

## Quick Decision Guide

**Choose Zero-Shot BERT** if you:
- Have 10k+ patents to categorize
- Want to save money
- Have access to a computer (CPU or GPU)
- Don't need 99% accuracy

**Choose OpenAI API** if you:
- Have <5k patents
- Need maximum accuracy
- Have budget for API calls
- Don't want to set up local infrastructure

---

## Method 1: Zero-Shot BERT (Recommended for 900k patents)

### Step 1: Ensure Dependencies

```bash
pip install pandas numpy torch sentence-transformers tqdm
```

### Step 2: Prepare Your Data

Your input CSV/TSV should have:
- `application_title`: Patent title
- `application_abstract`: Patent abstract (optional but recommended)

Example:
```csv
application_id,application_title,application_abstract,year
US001,Neural network for image classification,A method for...,2023
US002,Natural language model training,Systems and methods...,2023
```

### Step 3: Run Classification

**Basic usage (CPU):**
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv
```

**With GPU (much faster):**
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --device cuda \
    --batch-size 128
```

**Filter by year:**
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --start-year 2020
```

### Step 4: Check Results

Output CSV will have two new columns:
- `ai_category`: The predicted category
- `ai_category_confidence`: Confidence score (0.0 to 1.0)

**Interpret confidence scores:**
- `> 0.7`: High confidence, likely correct
- `0.5 - 0.7`: Moderate confidence
- `< 0.5`: Low confidence, may need review

---

## Method 2: OpenAI API

### Step 1: Set Up API Key

```bash
export RAND_OPENAI_API_KEY="your-api-key-here"
```

Or for Azure OpenAI:
```bash
export RAND_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
```

### Step 2: Run Classification

**Standard OpenAI:**
```bash
python src/scripts/categorize_patents.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --api-key-env RAND_OPENAI_API_KEY
```

**Azure OpenAI:**
```bash
python src/scripts/categorize_patents.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --use-azure \
    --azure-deployment your-deployment-name \
    --api-key-env RAND_OPENAI_API_KEY
```

**Resume interrupted run:**
The script automatically resumes from where it left off if the output file exists.

---

## Advanced: Hybrid Approach

For best accuracy with minimal cost:

### Step 1: Zero-Shot Classification
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_zeroshot.csv
```

### Step 2: Extract Low-Confidence Patents
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/processed/patents_zeroshot.csv')
low_conf = df[df['ai_category_confidence'] < 0.5]
low_conf.to_csv('data/processed/patents_low_conf.csv', index=False)
print(f'Found {len(low_conf)} patents with low confidence ({100*len(low_conf)/len(df):.1f}%)')
"
```

### Step 3: Re-classify Only Low-Confidence Patents
```bash
python src/scripts/categorize_patents.py \
    --input-file data/processed/patents_low_conf.csv \
    --output-file data/processed/patents_low_conf_api.csv
```

### Step 4: Merge Results
```bash
python -c "
import pandas as pd

# Load both results
df_zero = pd.read_csv('data/processed/patents_zeroshot.csv')
df_api = pd.read_csv('data/processed/patents_low_conf_api.csv')

# Update low-confidence patents with API results
df_zero.set_index('application_id', inplace=True)
df_api.set_index('application_id', inplace=True)
df_zero.update(df_api)
df_zero.reset_index(inplace=True)

# Save final results
df_zero.to_csv('data/processed/patents_final.csv', index=False)
print('Merged results saved!')
"
```

---

## Troubleshooting

### Problem: "Out of memory" error

**Solution:** Reduce batch size
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --batch-size 8  # Lower value
```

### Problem: Very slow on CPU

**Options:**
1. Use GPU if available (`--device cuda`)
2. Process in chunks by year (`--start-year 2023`)
3. Reduce batch size for better progress visibility

### Problem: Low confidence scores

**Causes:**
- Patent text is ambiguous
- Patent doesn't fit any category well
- Title/abstract missing or too short

**Solutions:**
1. Check patents with confidence < 0.5 manually
2. Use API classification for low-confidence cases
3. Add more detailed category descriptions

### Problem: Wrong categories assigned

**Check:**
1. Are title/abstract fields correct?
2. Is the patent actually about AI?
3. Review confidence scores

**Fix:**
1. Manually review and correct
2. Use API classification for validation
3. Refine category descriptions in the script

---

## Performance Benchmarks

### Zero-Shot BERT Classification

| Dataset Size | Hardware | Batch Size | Time |
|--------------|----------|------------|------|
| 10,000 | CPU (16 cores) | 32 | ~20 min |
| 10,000 | GPU (V100) | 128 | ~2 min |
| 100,000 | CPU (16 cores) | 32 | ~3 hours |
| 100,000 | GPU (V100) | 128 | ~15 min |
| 900,000 | CPU (16 cores) | 32 | ~24 hours |
| 900,000 | GPU (V100) | 128 | ~2-4 hours |

### OpenAI API Classification

| Dataset Size | Rate Limit | Time |
|--------------|------------|------|
| 1,000 | 60/min | ~20 min |
| 10,000 | 60/min | ~3 hours |
| 100,000 | 60/min | ~28 hours |
| 900,000 | 60/min | ~10 days |

*Note: Times assume 1-second delay between requests for rate limiting*

---

## Next Steps

After categorization:
1. Analyze category distribution
2. Review low-confidence predictions
3. Use categorized patents in the main matching pipeline
4. Generate reports and visualizations

See [patent_categorization_comparison.md](patent_categorization_comparison.md) for more details.
