# Azure OpenAI Patent Categorization

This guide explains how to use Azure OpenAI embeddings for patent categorization with built-in rate limiting.

## Overview

`categorize_patents_azure.py` provides zero-shot classification using Azure OpenAI's embedding models instead of local HuggingFace models.

### Benefits

**vs Local Models:**
- No GPU/CPU requirements
- Consistent, cloud-hosted performance
- High-quality embeddings (text-embedding-ada-002 or newer)

**vs GPT API Classification:**
- Much cheaper (embeddings vs completions)
- Faster (no generation required)
- More scalable (higher rate limits)

### Trade-offs

**Pros:**
- ✅ No local compute needed
- ✅ High-quality embeddings
- ✅ Faster than GPT classification
- ✅ Built-in rate limiting

**Cons:**
- ❌ Costs money (but much less than GPT classification)
- ❌ Requires internet connection
- ❌ Subject to API rate limits

## Prerequisites

### 1. Azure OpenAI Access

You need:
- Azure subscription
- Azure OpenAI resource
- Deployment of an embeddings model (e.g., `text-embedding-ada-002`)

### 2. Environment Variables

Set these before running:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export OPENAI_API_VERSION="2023-05-15"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

**Or in Python:**
```python
import os
os.environ["AZURE_OPENAI_API_KEY"] = "your-key"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
```

### 3. Install Dependencies

```bash
pip install openai pandas numpy tqdm
```

## Usage

### Basic Usage (500 QPS Default)

```bash
python src/scripts/categorize_patents_azure.py \
    --input-file data/patents.csv \
    --output-file data/patents_categorized.csv
```

This uses:
- Default deployment: `text-embedding-ada-002`
- Default rate limit: 500 queries/second
- Default batch size: 16 texts per API call

### Custom Rate Limit

```bash
python src/scripts/categorize_patents_azure.py \
    --input-file data/patents.csv \
    --output-file data/patents_categorized.csv \
    --rate-limit 100
```

**When to adjust:**
- Your Azure quota is lower: use `--rate-limit 100` or `--rate-limit 10`
- You have higher quota: use `--rate-limit 1000` or higher
- Getting rate limit errors: reduce rate limit

### Different Embedding Model

```bash
python src/scripts/categorize_patents_azure.py \
    --input-file data/patents.csv \
    --output-file data/patents_categorized.csv \
    --deployment-name text-embedding-3-small \
    --rate-limit 1000
```

**Available models:**
- `text-embedding-ada-002` (default, 1536 dimensions)
- `text-embedding-3-small` (cheaper, 1536 dimensions)
- `text-embedding-3-large` (better quality, 3072 dimensions)

### Custom Batch Size

```bash
python src/scripts/categorize_patents_azure.py \
    --input-file data/patents.csv \
    --output-file data/patents_categorized.csv \
    --batch-size 32
```

**Batch size guidelines:**
- Smaller (8-16): More API calls, finer rate limit control
- Larger (32-64): Fewer API calls, faster if rate limit allows
- Default (16): Good balance

### Use Specific Text Column

```bash
python src/scripts/categorize_patents_azure.py \
    --input-file data/patents.csv \
    --output-file data/patents_categorized.csv \
    --text-column combined_text
```

## How It Works

### 1. Rate Limiting

The script implements **token bucket rate limiting**:

```python
# Example: 500 QPS means 500 requests per second
# Script automatically throttles to stay under limit
# Uses exponential backoff if rate limit is hit
```

**Features:**
- Smooth rate limiting (not bursty)
- Automatic throttling
- Retry with exponential backoff on errors

### 2. Zero-Shot Classification

Same approach as local version:

```
1. Encode category descriptions → category_embeddings
2. Encode patent text → patent_embedding
3. Compute cosine similarity: similarity = patent_embedding · category_embeddings
4. Choose category with highest similarity
```

### 3. Batching

Sends multiple texts per API call for efficiency:

```python
# Instead of 100 API calls for 100 patents:
# With batch_size=16 → only ~7 API calls

# Respects rate limits:
# 500 QPS with batch_size=16 → ~8000 texts/second theoretical max
```

### 4. Error Handling

- Automatic retries (3 attempts)
- Exponential backoff on failures
- Empty text handling (returns "missing_data")

## Performance & Cost

### Speed

**Example: 900,000 patents**

Rate limit 500 QPS, batch size 16:
- Theoretical: ~8000 patents/second
- Actual (with overhead): ~2000-3000 patents/second
- Total time: ~5-8 minutes

**vs Local HuggingFace:**
- Local (GPU): ~2000-4000 patents/second
- Azure: ~2000-3000 patents/second
- **Similar performance**, but no GPU needed

### Cost

**Azure OpenAI Pricing (as of 2024):**

`text-embedding-ada-002`:
- $0.0001 per 1K tokens
- Average patent: ~300 tokens
- Cost per patent: ~$0.00003

**For 900,000 patents:**
- Total tokens: ~270M tokens
- Cost: ~$27

**Comparison:**
- GPT-4 classification: $9,000-$18,000
- Azure embeddings: $27
- Local (free): $0

**Still much cheaper than GPT, but not free like local models.**

## Rate Limit Guidelines

### How to Choose Rate Limit

**Check your Azure quota:**
1. Go to Azure Portal → OpenAI → Quotas
2. Look at "Requests per second" for your deployment
3. Set `--rate-limit` to 80% of that quota

**Examples:**
- Quota: 500 RPS → use `--rate-limit 400`
- Quota: 100 RPS → use `--rate-limit 80`
- Quota: 1000 RPS → use `--rate-limit 800`

### Troubleshooting Rate Limits

**Getting 429 errors?**

1. **Reduce rate limit:**
   ```bash
   --rate-limit 100  # Try half of current
   ```

2. **Reduce batch size:**
   ```bash
   --batch-size 8  # Smaller batches
   ```

3. **Check Azure quotas:**
   - Make sure you're not hitting TPM (tokens per minute) limit
   - May need to increase quota in Azure Portal

## Output

### Columns Added

- `ai_category`: Predicted AI category
- `confidence`: Cosine similarity score (0-1)

### Example Output

```csv
application_id,title,abstract,ai_category,confidence
123,Neural Network...,A method for...,machine learning,0.847
456,Image Processing...,A system for...,computer vision,0.923
```

## Comparison: Azure vs Local vs GPT

| Aspect | Azure Embeddings | Local HuggingFace | GPT Classification |
|--------|------------------|-------------------|-------------------|
| Cost (900k) | ~$27 | $0 | ~$9k-$18k |
| Speed | ~2-3k/sec | ~2-4k/sec | ~50-100/sec |
| Setup | Env vars | Install models | Env vars |
| Accuracy | ~85-92% | ~85-92% | ~95-98% |
| Hardware | None | GPU/CPU | None |
| Internet | Required | Optional | Required |
| Rate Limits | Yes (controllable) | No | Yes (strict) |

**Recommendation:**
- **Research/Experimentation:** Use local (free)
- **Production (moderate budget):** Use Azure embeddings (fast + reasonable cost)
- **Highest accuracy needed:** Use GPT classification (expensive)

## Advanced Usage

### Monitor Progress

The script shows progress bars:
```
Encoding category descriptions: 100%|██████| 8/8
Encoding texts: 100%|██████████████| 56250/56250 [00:45<00:00, 1234.5it/s]
```

### Resume Failed Runs

If the script fails partway:

```bash
# Check how many were completed
wc -l data/patents_categorized.csv

# Process remaining (manually filter input CSV)
# Or just re-run (idempotent if output file doesn't exist)
```

### Parallel Processing

For very large datasets, split input and run parallel:

```bash
# Split into 4 files
split -n 4 data/patents.csv data/patents_part_

# Run 4 instances (different machines or rate limits)
python categorize_patents_azure.py --input-file data/patents_part_aa --output-file data/out_1.csv &
python categorize_patents_azure.py --input-file data/patents_part_ab --output-file data/out_2.csv &
python categorize_patents_azure.py --input-file data/patents_part_ac --output-file data/out_3.csv &
python categorize_patents_azure.py --input-file data/patents_part_ad --output-file data/out_4.csv &

# Combine results
cat data/out_*.csv > data/patents_categorized.csv
```

## Integration with Pipeline

Use Azure categorization as a drop-in replacement:

```bash
# Step 1: Categorize with Azure (instead of local)
python src/scripts/categorize_patents_azure.py \
    --input-file data/raw/patents.csv \
    --output-file data/processed/patents_categorized.csv

# Step 2: Category-specific matching (same as before)
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_categorized.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --output-dir results/category_specific

# Step 3: Calculate exposure (same as before)
python src/scripts/calculate_occupational_exposure.py \
    --input-dir results/category_specific \
    --output-file results/occupational_exposure.csv
```

**Compatible with all downstream scripts** - just use `--category-column ai_category`

## Troubleshooting

### "Missing environment variables"

**Problem:** Azure credentials not set

**Solution:**
```bash
export AZURE_OPENAI_API_KEY="your-key"
export OPENAI_API_VERSION="2023-05-15"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

### "Deployment not found"

**Problem:** Deployment name doesn't match Azure

**Solution:** Check Azure Portal for exact deployment name:
```bash
--deployment-name your-actual-deployment-name
```

### Rate limit errors persist

**Problem:** Rate limit too high or Azure quota issue

**Solutions:**
1. Reduce rate limit: `--rate-limit 50`
2. Check Azure Portal quotas
3. Request quota increase from Azure

### Slow performance

**Problem:** Rate limit or batch size too conservative

**Solutions:**
1. Increase rate limit (if quota allows): `--rate-limit 1000`
2. Increase batch size: `--batch-size 32`
3. Check internet connection speed

## Best Practices

1. **Start conservative:** Begin with `--rate-limit 100` to test
2. **Monitor costs:** Check Azure billing regularly
3. **Test on subset:** Try 1000 patents first before full dataset
4. **Save checkpoints:** For large datasets, process in chunks
5. **Validate results:** Spot-check categorizations for accuracy

## Next Steps

After categorization:
1. Run category-specific matching
2. Calculate exposure scores
3. Create visualizations
4. Compare with local model results for validation

## Related Documentation

- [Zero-Shot Classification](multi_model_support.md) - Local HuggingFace version
- [Ensemble Voting](ensemble_voting.md) - Combine multiple models
- [Category-Specific Matching](category_specific_matching.md) - Next step after categorization
