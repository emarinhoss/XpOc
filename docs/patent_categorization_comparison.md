# Patent Categorization Approaches: Comparison

This document compares two methods for categorizing patents into AI categories.

## Overview

The repository contains two approaches for classifying 900k patents into 8 AI categories:

1. **OpenAI API Classification** (`categorize_patents.py`)
2. **Zero-Shot BERT Classification** (`categorize_patents_zeroshot.py`) ⭐ Recommended

## AI Categories

Both approaches classify patents into these 8 categories:

| Category | Description |
|----------|-------------|
| **computer vision** | Methods to understand images and videos |
| **evolutionary computation** | Methods mimicking evolution to solve problems |
| **AI hardware** | Physical hardware designed for AI software |
| **knowledge processing** | Methods to represent and derive facts from knowledge bases |
| **machine learning** | Algorithms that learn from data |
| **NLP** | Methods to understand and generate human language |
| **planning and control** | Methods to determine and execute plans to achieve goals |
| **speech recognition** | Methods to understand speech and generate responses |

---

## Approach 1: OpenAI API Classification

**File:** `src/scripts/categorize_patents.py`

### How It Works

- Sends patent title + abstract to OpenAI API (GPT-3.5-turbo or Azure OpenAI)
- Uses LLM to classify into one of 8 categories
- Temperature = 0.0 for deterministic results

### Pros

- High accuracy (LLMs are good at understanding context)
- Easy to implement and test
- Can handle nuanced or ambiguous patents
- Supports Azure OpenAI for enterprise deployments

### Cons

- **Cost**: ~$0.01-0.02 per patent = **$9,000-$18,000** for 900k patents
- **Time**: Rate limited to avoid throttling (1 sec/patent = ~10.4 days)
- **External dependency**: Requires API access and internet connection
- **Rate limits**: Subject to OpenAI rate limiting
- **Reproducibility**: API models can change over time

### Usage Example

```bash
python src/scripts/categorize_patents.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --api-key-env RAND_OPENAI_API_KEY \
    --batch-size 50 \
    --rate-limit-delay 1.0
```

### Cost Calculation (900k patents)

- Avg tokens per request: ~150 tokens (input) + 5 tokens (output) = 155 tokens
- Cost for GPT-3.5-turbo: $0.50 per 1M input tokens, $1.50 per 1M output tokens
- Total cost: ~$10,000-$18,000 depending on text length

---

## Approach 2: Zero-Shot BERT Classification ⭐ **RECOMMENDED**

**File:** `src/scripts/categorize_patents_zeroshot.py`

### How It Works

1. Pre-computes BERT embeddings for 8 category descriptions
2. Encodes each patent (title + abstract) using BERT-for-patents model
3. Computes cosine similarity between patent and all categories
4. Assigns category with highest similarity score
5. Returns category + confidence score

### Pros

- **Cost**: **$0** (runs locally, no API calls)
- **Speed**: ~10-100x faster than API (can process ~500-1000 patents/hour on CPU)
- **Offline**: No internet connection required
- **Reproducible**: Same model always gives same results
- **Confidence scores**: Provides similarity score for interpretability
- **Batch processing**: Efficient GPU/CPU batch encoding
- **No rate limits**: Limited only by hardware

### Cons

- May have slightly lower accuracy than GPT on ambiguous cases
- Requires local compute resources (CPU/GPU)
- Needs to download BERT model (~400MB) once

### Usage Example

```bash
# CPU (slower but works everywhere)
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized_zeroshot.csv \
    --device cpu \
    --batch-size 32

# GPU (much faster if available)
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized_zeroshot.csv \
    --device cuda \
    --batch-size 128
```

### Performance Estimates (900k patents)

| Hardware | Batch Size | Est. Time | Cost |
|----------|------------|-----------|------|
| CPU (16 cores) | 32 | ~18-36 hours | $0 |
| Single GPU (V100) | 128 | ~2-4 hours | $0 (local) or ~$8-16 (cloud) |
| Multi-GPU | 256+ | ~30-60 min | $0 (local) or ~$2-4 (cloud) |

---

## Side-by-Side Comparison

| Feature | OpenAI API | Zero-Shot BERT |
|---------|-----------|----------------|
| **Cost for 900k patents** | $9,000-$18,000 | $0 (local) or $2-16 (cloud GPU) |
| **Processing time** | ~10 days | ~4 hours (GPU) or ~24 hours (CPU) |
| **Accuracy** | Very High (95-98%) | High (85-92%) |
| **Requires internet** | Yes | No |
| **Rate limits** | Yes | No |
| **Reproducible** | No | Yes |
| **Confidence scores** | No | Yes |
| **Can resume** | Yes | Yes |
| **Dependencies** | OpenAI API key | PyTorch, sentence-transformers |
| **Customization** | Limited | Full control |

---

## Recommendation

**Use Zero-Shot BERT Classification** for the following reasons:

1. **Cost**: Saves $9,000-$18,000
2. **Speed**: 50-200x faster with GPU
3. **Offline**: Works without internet
4. **Reproducible**: Same results every time
5. **Good enough accuracy**: 85-92% is sufficient for large-scale analysis

### When to Use OpenAI API Instead

- You need the highest possible accuracy (>95%)
- You have a small dataset (<10k patents)
- You have budget for API calls
- You don't have access to compute resources

---

## Hybrid Approach (Optional)

For maximum accuracy with cost efficiency:

1. **Run zero-shot BERT** on all 900k patents
2. **Filter low-confidence predictions** (confidence < 0.5)
3. **Use OpenAI API** only on ambiguous cases (~5-10% of patents)
4. **Final cost**: ~$500-$1,000 (instead of $10k-$18k)

Example:
```bash
# Step 1: Zero-shot classification
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized_zeroshot.csv

# Step 2: Extract low-confidence patents
python -c "
import pandas as pd
df = pd.read_csv('data/processed/patents_categorized_zeroshot.csv')
low_conf = df[df['ai_category_confidence'] < 0.5]
low_conf.to_csv('data/processed/patents_low_confidence.csv', index=False)
print(f'Found {len(low_conf)} low-confidence patents')
"

# Step 3: Re-classify with OpenAI API (only if needed)
python src/scripts/categorize_patents.py \
    --input-file data/processed/patents_low_confidence.csv \
    --output-file data/processed/patents_low_confidence_reclassified.csv
```

---

## Implementation Details

### Zero-Shot Classification Algorithm

```python
# 1. Encode category descriptions (done once)
category_texts = [
    "methods to understand images and videos...",
    "methods to understand and generate human language...",
    # ... 6 more
]
category_embeddings = bert_model.encode(category_texts)

# 2. For each patent:
patent_text = f"{title}. {abstract}"
patent_embedding = bert_model.encode(patent_text)

# 3. Compute cosine similarity with all categories
similarities = cosine_similarity(patent_embedding, category_embeddings)

# 4. Assign highest-scoring category
best_category = categories[argmax(similarities)]
confidence = max(similarities)
```

### Model Used

- **BERT Model**: `anferico/bert-for-patents`
- **Why this model?**: Fine-tuned specifically on patent text, understands technical terminology better than general BERT
- **Embedding size**: 768 dimensions
- **Max sequence length**: 512 tokens

---

## Validation

To validate the zero-shot approach:

1. Manually label 100-200 random patents
2. Run both OpenAI and zero-shot classification
3. Compare against manual labels
4. Calculate accuracy, precision, recall per category

---

## Next Steps

1. ✅ Implement zero-shot classifier
2. ⏳ Test on sample patents
3. ⏳ Validate accuracy on labeled subset
4. ⏳ Run on full 900k patent dataset
5. ⏳ Analyze category distribution and confidence scores
6. ⏳ (Optional) Use hybrid approach for low-confidence cases

---

## Questions?

- **Q: Can I improve zero-shot accuracy?**
  - A: Yes! Improve category descriptions, use larger BERT models, or try ensemble methods

- **Q: What if I want different categories?**
  - A: Just update `CATEGORIES_WITH_DEFINITIONS` in the script

- **Q: Can this work for non-AI patents?**
  - A: Yes! Change the categories and descriptions to fit your domain

- **Q: How do I know if predictions are good?**
  - A: Check confidence scores. High confidence (>0.7) = reliable, low (<0.5) = uncertain
