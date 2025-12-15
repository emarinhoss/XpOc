# Category-Specific Patent-Occupation Matching

This guide explains how to perform **category-specific analysis** where patent-occupation matching is run separately for each of the 8 AI categories.

## Why Category-Specific Analysis?

Instead of matching **all AI patents** to occupational tasks in one run, this approach matches patents **separately by AI category** (machine learning, NLP, computer vision, etc.).

### Benefits

**Reveals differential impact:**
- Which tasks are most affected by **computer vision** vs **NLP**?
- How does **machine learning** impact differ from **AI hardware** impact?
- Which AI categories have fastest growing impact on specific occupations?

**Example insights:**
- "Data scientist tasks show 115 ML patent matches in 2025 (up from 45 in 2021)"
- "Computer vision has 3x more impact on 'Image Analysis' tasks than NLP"
- "Planning & control patents increasingly match robotics tasks (trend analysis)"

## Overview

### Workflow

```
Input: Categorized patents (from ensemble voting)
         ↓
For each of 8 AI categories:
  ├─ Filter patents to this category
  ├─ Match filtered patents to O*NET tasks
  ├─ For each year:
  │   ├─ Find top-k most similar patents per task
  │   └─ Count matches above threshold
  └─ Save: category_specific_results.csv
         ↓
Output: 8 separate CSV files (one per category)
```

### Output Files

Creates files like:
```
results/category_specific/
├── AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv
├── AI_2021to2025_abstractTitleFor_nlp_0.75.csv
├── AI_2021to2025_abstractTitleFor_computer_vision_0.75.csv
├── AI_2021to2025_abstractTitleFor_evolutionary_computation_0.75.csv
├── AI_2021to2025_abstractTitleFor_ai_hardware_0.75.csv
├── AI_2021to2025_abstractTitleFor_knowledge_processing_0.75.csv
├── AI_2021to2025_abstractTitleFor_planning_and_control_0.75.csv
└── AI_2021to2025_abstractTitleFor_speech_recognition_0.75.csv
```

### Output Format

Each CSV contains:
```csv
O*NET-SOC Code,Title,Task,2021,2022,2023,2024,2025
15-2051.00,Data Scientist,"Analyze data using ML",45,67,89,102,115
15-2051.00,Data Scientist,"Build predictive models",38,52,71,85,98
15-1252.00,Software Engineer,"Implement ML algorithms",12,18,24,31,42
...
```

**Columns:**
- `O*NET-SOC Code`: Occupation code
- `Title`: Occupation title
- `Task`: Task description
- `2021`, `2022`, ...: Count of patents matching this task (similarity ≥ threshold)

## Usage

### Prerequisites

1. **Categorized patents** - Run ensemble voting first:
```bash
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/*.csv \
    --output-file data/processed/patents_ensemble.csv
```

2. **O*NET tasks data** - Download from O*NET (Task Ratings file)

### Basic Usage

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --output-dir results/category_specific
```

This will:
- Process all 8 AI categories
- Use default threshold (0.75)
- Use BERT-for-patents model
- Save results to `results/category_specific/`

### Process Specific Categories Only

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --categories "machine learning" "NLP" "computer vision" \
    --output-dir results/ml_nlp_cv
```

### Custom Threshold

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --threshold 0.8 \
    --output-dir results/strict_threshold
```

### Different Embedding Model

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --model-name google/embeddinggemma-300m \
    --model-type auto-model \
    --pooling mean \
    --output-dir results/gemma
```

### GPU Acceleration

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --device cuda \
    --batch-size 64 \
    --output-dir results/category_specific
```

## Complete Workflow Example

### Step 1: Categorize Patents (if not already done)

```bash
# Option A: Single model (fast)
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv

# Option B: Ensemble voting (more accurate)
# Run 5 models separately, then aggregate
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/bert.csv results/gemma.csv ... \
    --output-file data/processed/patents_ensemble.csv
```

### Step 2: Run Category-Specific Analysis

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --output-dir results/category_specific \
    --device cuda \
    --batch-size 64
```

### Step 3: Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results for machine learning
ml_df = pd.read_csv('results/category_specific/AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv')

# Find tasks with highest ML impact in 2025
top_tasks_2025 = ml_df.nlargest(10, '2025')[['Title', 'Task', '2025']]
print("Top 10 tasks most affected by ML patents in 2025:")
print(top_tasks_2025)

# Track trend for a specific occupation
data_scientist_tasks = ml_df[ml_df['Title'] == 'Data Scientist']
years = ['2021', '2022', '2023', '2024', '2025']
trend = data_scientist_tasks[years].sum()

plt.figure(figsize=(10, 6))
plt.plot(years, trend, marker='o')
plt.title('ML Patent Impact on Data Scientist Tasks Over Time')
plt.xlabel('Year')
plt.ylabel('Total Matches')
plt.show()

# Compare impact across categories for a specific task
task_id = 123  # Example task
categories = ['machine_learning', 'nlp', 'computer_vision']
impacts_2025 = []

for cat in categories:
    df = pd.read_csv(f'results/category_specific/AI_2021to2025_abstractTitleFor_{cat}_0.75.csv')
    impact = df.iloc[task_id]['2025']
    impacts_2025.append(impact)

plt.figure(figsize=(10, 6))
plt.bar(categories, impacts_2025)
plt.title('Impact of Different AI Categories on Task #123 (2025)')
plt.ylabel('Number of Matching Patents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Configuration Options

### All Command-Line Arguments

```bash
python src/scripts/categorize_by_ai_category.py --help

Required:
  --patents-file PATH         Categorized patents CSV
  --onet-file PATH           O*NET tasks file (Excel or CSV)
  --output-dir PATH          Output directory

Categories:
  --categories [CAT ...]     Specific categories to process
  --category-column NAME     Category column name (default: ensemble_category)

Years:
  --years [YEAR ...]         Specific years to process (default: all)

Matching:
  --threshold FLOAT          Similarity threshold (default: 0.75)
  --top-k INT               Top matches per task (default: 500)
  --text-column NAME         Patent text column (default: title_abstract)

Model:
  --config PATH              YAML config file (optional)
  --model-name NAME          Model name (default: anferico/bert-for-patents)
  --model-type TYPE          sentence-transformers | auto-model
  --pooling STRATEGY         mean | cls | max
  --device DEVICE            cpu | cuda
  --batch-size INT           Batch size (default: 32)
```

### Input File Requirements

**Patents file must have:**
- Category column (e.g., `ensemble_category`, `ai_category`)
- `year` column
- Text column (e.g., `title_abstract`, `application_title`)

**O*NET file must have:**
- `O*NET-SOC Code`
- `Title`
- `Task`

## Performance

### Processing Time (900k patents, 8 categories)

| Configuration | Time per Category | Total Time (8 categories) |
|---------------|-------------------|---------------------------|
| CPU, batch_size=32 | ~4 hours | ~32 hours |
| GPU (V100), batch_size=64 | ~45 minutes | ~6 hours |
| GPU (A100), batch_size=128 | ~25 minutes | ~3.5 hours |

**Note:** Categories can be run in parallel on different machines/GPUs to reduce total time.

### Memory Requirements

- **Model loading**: ~400 MB (sentence-transformers) to ~600 MB (larger models)
- **O*NET embeddings**: ~50 MB (cached across categories)
- **Patent embeddings per year**: ~100-300 MB depending on patent count
- **Total**: ~1-2 GB per category

## Parallel Execution

Run categories in parallel for faster processing:

### Approach 1: Screen/Tmux Sessions

```bash
# Terminal 1
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --categories "machine learning" "NLP" \
    --output-dir results/category_specific

# Terminal 2
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --categories "computer vision" "evolutionary computation" \
    --output-dir results/category_specific

# Terminal 3
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --categories "AI hardware" "knowledge processing" \
    --output-dir results/category_specific

# Terminal 4
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --categories "planning and control" "speech recognition" \
    --output-dir results/category_specific
```

### Approach 2: Shell Script

```bash
#!/bin/bash
# run_categories_parallel.sh

PATENTS_FILE="data/processed/patents_ensemble.csv"
ONET_FILE="data/raw/onet/Task_Ratings.xlsx"
OUTPUT_DIR="results/category_specific"

# Run 4 processes, each handling 2 categories
python src/scripts/categorize_by_ai_category.py \
    --patents-file "$PATENTS_FILE" \
    --onet-file "$ONET_FILE" \
    --categories "machine learning" "NLP" \
    --output-dir "$OUTPUT_DIR" &

python src/scripts/categorize_by_ai_category.py \
    --patents-file "$PATENTS_FILE" \
    --onet-file "$ONET_FILE" \
    --categories "computer vision" "evolutionary computation" \
    --output-dir "$OUTPUT_DIR" &

python src/scripts/categorize_by_ai_category.py \
    --patents-file "$PATENTS_FILE" \
    --onet-file "$ONET_FILE" \
    --categories "AI hardware" "knowledge processing" \
    --output-dir "$OUTPUT_DIR" &

python src/scripts/categorize_by_ai_category.py \
    --patents-file "$PATENTS_FILE" \
    --onet-file "$ONET_FILE" \
    --categories "planning and control" "speech recognition" \
    --output-dir "$OUTPUT_DIR" &

wait
echo "All categories processed!"
```

## Comparison: Combined vs Category-Specific

### Combined Analysis (main_pipeline.py)

**One run for all AI patents:**
```
All 900k AI patents → Match to tasks → Output: 1 file
Time: ~4 hours (GPU)
Output: patent_occupation_matches.csv
```

**Best for:**
- Understanding overall AI impact
- Faster processing (single run)
- When category distinction not needed

### Category-Specific Analysis (categorize_by_ai_category.py)

**8 runs (one per category):**
```
ML patents → Match → ml_results.csv
NLP patents → Match → nlp_results.csv
...
Time: ~6 hours (GPU, sequential) or ~45 min (4 GPUs parallel)
Output: 8 category-specific files
```

**Best for:**
- Understanding differential impact by technology type
- Tracking category-specific trends
- Research requiring fine-grained analysis

## Analysis Examples

### Example 1: Which Category Most Affects Data Scientists?

```python
import pandas as pd

categories = ['machine_learning', 'nlp', 'computer_vision', 'ai_hardware',
              'knowledge_processing', 'evolutionary_computation',
              'planning_and_control', 'speech_recognition']

impacts_2025 = {}

for cat in categories:
    df = pd.read_csv(f'results/category_specific/AI_2021to2025_abstractTitleFor_{cat}_0.75.csv')
    # Sum all data scientist task matches
    ds_impact = df[df['Title'] == 'Data Scientist']['2025'].sum()
    impacts_2025[cat] = ds_impact

# Find top category
top_cat = max(impacts_2025, key=impacts_2025.get)
print(f"Category with most impact on Data Scientists: {top_cat}")
print(f"Total matches: {impacts_2025[top_cat]}")
```

### Example 2: Trend Analysis Across Categories

```python
import pandas as pd
import matplotlib.pyplot as plt

years = ['2021', '2022', '2023', '2024', '2025']
categories = ['machine_learning', 'nlp', 'computer_vision']

plt.figure(figsize=(12, 6))

for cat in categories:
    df = pd.read_csv(f'results/category_specific/AI_2021to2025_abstractTitleFor_{cat}_0.75.csv')
    # Total impact per year
    trend = [df[str(year)].sum() for year in years]
    plt.plot(years, trend, marker='o', label=cat.replace('_', ' ').title())

plt.xlabel('Year')
plt.ylabel('Total Task Matches')
plt.title('AI Category Impact Trends Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('category_trends.png', dpi=300)
plt.show()
```

### Example 3: Cross-Category Comparison for Specific Task

```python
import pandas as pd

task_description = "Analyze large datasets using machine learning"

# Find this task in one file to get details
ml_df = pd.read_csv('results/category_specific/AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv')
task_row = ml_df[ml_df['Task'].str.contains(task_description, case=False, na=False)].iloc[0]

print(f"Task: {task_row['Task']}")
print(f"Occupation: {task_row['Title']}")
print("\nImpact by category (2025):")

categories = ['machine_learning', 'nlp', 'computer_vision', 'ai_hardware']

for cat in categories:
    df = pd.read_csv(f'results/category_specific/AI_2021to2025_abstractTitleFor_{cat}_0.75.csv')
    matching_row = df[df['Task'] == task_row['Task']].iloc[0]
    impact = matching_row['2025']
    print(f"  {cat.replace('_', ' ').title()}: {impact} matches")
```

## Troubleshooting

### Issue: "No patents found for category"

**Cause:** Category name mismatch between patents file and script.

**Solution:** Check category names in your patents file:
```python
import pandas as pd
df = pd.read_csv('patents_ensemble.csv')
print(df['ensemble_category'].unique())
```

Then use exact names with `--categories`.

### Issue: "Text column 'title_abstract' not found"

**Cause:** Patents file doesn't have combined title+abstract field.

**Solutions:**
1. Create it beforehand:
```python
df['title_abstract'] = df['application_title'] + '. ' + df['application_abstract']
df.to_csv('patents_with_text.csv', index=False)
```

2. Or specify existing column:
```bash
--text-column application_title
```

### Issue: Very slow processing

**Solutions:**
1. Use GPU: `--device cuda`
2. Increase batch size: `--batch-size 128`
3. Process categories in parallel (see above)
4. Use faster model: `--model-name sentence-transformers/all-MiniLM-L6-v2`

### Issue: Out of memory

**Solutions:**
1. Reduce batch size: `--batch-size 16`
2. Process fewer years at once
3. Use CPU instead of GPU
4. Process categories one at a time

## Next Steps

After generating category-specific results:

1. **Visualization**: Create charts showing category impacts over time
2. **Statistical analysis**: Test for significant differences between categories
3. **Occupational profiles**: Identify which occupations are most affected by each category
4. **Policy implications**: Understand differential retraining needs by AI type
5. **Trend forecasting**: Project future impacts based on historical trends

## Summary

This script enables **fine-grained analysis** of AI's impact on work by:
- Separating analysis by AI technology type
- Revealing differential impacts across categories
- Supporting longitudinal trend analysis
- Enabling category-specific policy recommendations

Perfect for research questions like:
- "How does NLP affect communication jobs differently than other AI?"
- "Which AI categories are growing fastest in manufacturing tasks?"
- "What retraining is needed for jobs most affected by computer vision?"
