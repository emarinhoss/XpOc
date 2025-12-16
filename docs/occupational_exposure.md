# Occupational Exposure to AI Technologies

This guide explains how to calculate **occupational exposure scores** that measure how much each occupation's important tasks are matched by AI patents.

## What is Occupational Exposure?

Occupational exposure quantifies the degree to which an occupation's critical tasks are affected by AI technologies, weighted by task importance.

### Formula

For each occupation, AI category, and year:

```
exposure = Σ (patent matches for task / total patent matches) × task importance
```

Where:
- **patent matches for task**: Number of AI patents matching this task (similarity ≥ threshold)
- **total patent matches**: Total patent matches across all tasks for this occupation
- **task importance**: O*NET importance rating (from Data Value column)
- **Σ**: Sum over all importance-rated tasks for the occupation

### Interpretation

**Exposure score represents a weighted average of task importance:**
- Higher exposure = occupation's more important tasks are heavily matched by AI patents
- Lower exposure = AI patents match less important tasks, or few matches overall
- Zero exposure = no AI patents matched any tasks for this occupation

**Example:**
If an occupation has exposure of 4.5 for machine learning in 2025:
- This means AI patents are matching tasks with average importance ~4.5
- On O*NET's 1-5 scale, this indicates high importance tasks are being matched

## Workflow

### Input
Category-specific matching results from `categorize_by_ai_category.py`:
```
results/category_specific/
├── AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv
├── AI_2021to2025_abstractTitleFor_nlp_0.75.csv
├── AI_2021to2025_abstractTitleFor_computer_vision_0.75.csv
└── ... (8 files total)
```

Each file contains:
```csv
O*NET-SOC Code,Title,Task,Scale Name,Category,Data Value,2021,2022,2023
15-2051.00,Data Scientist,"Analyze data",Importance,Core,4.5,100,120,150
15-2051.00,Data Scientist,"Build models",Importance,Core,4.2,80,95,110
```

### Processing
1. **Filter for importance**: Keep only rows where `Scale Name == "Importance"`
2. **Extract category**: Detect AI category from filename (e.g., "machine_learning")
3. **Calculate exposure**: For each occupation in each year, compute weighted exposure
4. **Combine results**: Merge all 8 categories into one file

### Output
Single CSV with exposure scores:
```csv
O*NET-SOC Code,Title,AI_Category,2021,2022,2023,2024,2025
15-2051.00,Data Scientist,machine_learning,4.52,4.58,4.61,4.65,4.68
15-2051.00,Data Scientist,nlp,4.31,4.35,4.39,4.42,4.45
15-2051.00,Data Scientist,computer_vision,3.89,3.95,4.01,4.05,4.09
15-1252.00,Software Engineer,machine_learning,3.78,3.82,3.85,3.88,3.91
...
```

## Usage

### Basic Usage

```bash
python src/scripts/calculate_occupational_exposure.py \
    --input-dir results/category_specific \
    --output-file results/occupational_exposure.csv
```

This will:
- Process all 8 category-specific CSV files
- Calculate exposure for each occupation × AI category × year
- Save combined results to one file

### Filter by Threshold

If you have multiple threshold levels (0.75, 0.80, etc.), process only one:

```bash
python src/scripts/calculate_occupational_exposure.py \
    --input-dir results/category_specific \
    --pattern "0.75" \
    --output-file results/occupational_exposure_075.csv
```

This processes only files containing "0.75" in the filename.

## Complete Workflow Example

### Step 1: Category-Specific Matching

First, generate category-specific matching results:

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --output-dir results/category_specific \
    --threshold 0.75
```

Output: 8 CSV files (one per AI category)

### Step 2: Calculate Exposure

Then calculate occupational exposure:

```bash
python src/scripts/calculate_occupational_exposure.py \
    --input-dir results/category_specific \
    --output-file results/occupational_exposure.csv
```

Output: Single CSV with exposure scores for all occupations × categories × years

## Example Calculation

### Input Data (simplified)

**Occupation: Data Scientist (15-2051.00)**
**Category: Machine Learning**
**Year: 2023**

| Task | Scale Name | Data Value (Importance) | 2023 (Matches) |
|------|------------|------------------------|----------------|
| Analyze data using ML | Importance | 4.5 | 100 |
| Build predictive models | Importance | 4.2 | 50 |
| Clean data | Importance | 3.0 | 50 |

### Calculation

```
Total matches = 100 + 50 + 50 = 200

Task 1 contribution: (100/200) × 4.5 = 0.50 × 4.5 = 2.25
Task 2 contribution: (50/200) × 4.2 = 0.25 × 4.2 = 1.05
Task 3 contribution: (50/200) × 3.0 = 0.25 × 3.0 = 0.75

Exposure = 2.25 + 1.05 + 0.75 = 4.05
```

**Result:** Data Scientist has exposure of **4.05** to machine learning in 2023.

This means ML patents are matching tasks with average importance ~4.05 (on 1-5 scale).

## Output Analysis

### Typical Analysis Questions

**1. Which occupations have highest exposure to AI?**
```python
import pandas as pd

df = pd.read_csv('results/occupational_exposure.csv')

# Average exposure across all categories and years
year_cols = ['2021', '2022', '2023', '2024', '2025']
avg_exposure = df.groupby(['O*NET-SOC Code', 'Title'])[year_cols].mean().mean(axis=1)
top_10 = avg_exposure.nlargest(10)
print(top_10)
```

**2. How is exposure changing over time?**
```python
# Calculate year-over-year growth
df['growth_2021_2025'] = (df['2025'] - df['2021']) / df['2021'] * 100

# Top growing exposures
fastest_growing = df.nlargest(10, 'growth_2021_2025')[
    ['Title', 'AI_Category', '2021', '2025', 'growth_2021_2025']
]
print(fastest_growing)
```

**3. Which AI categories have broadest impact?**
```python
# Count occupations with non-zero exposure per category
by_category = df.groupby('AI_Category').apply(
    lambda x: (x[year_cols] > 0).any(axis=1).sum()
)
print(by_category.sort_values(ascending=False))
```

**4. Differential exposure across categories**
```python
# Compare ML vs NLP exposure for same occupation
data_scientist = df[df['Title'] == 'Data Scientist']
pivot = data_scientist.pivot(
    index='AI_Category',
    columns=None,
    values='2025'
)
print(pivot.sort_values(ascending=False))
```

## Understanding Exposure Scores

### Score Ranges

Given O*NET importance ratings typically range 1-5:

- **4.0-5.0**: Very high exposure - AI heavily matches critical tasks
- **3.0-4.0**: High exposure - AI matches important tasks
- **2.0-3.0**: Moderate exposure - AI matches less critical tasks
- **1.0-2.0**: Low exposure - AI matches minor tasks
- **0.0-1.0**: Minimal exposure - Few or no matches

### Zero Exposure

An occupation can have zero exposure in a year if:
- No AI patents matched any tasks (total matches = 0)
- The occupation has no importance-rated tasks in O*NET data

### Comparing Across Categories

Exposure scores are comparable across AI categories because they're based on the same importance scale.

**Example interpretation:**
- Data Scientist: ML exposure 4.5, NLP exposure 3.8
- Interpretation: ML patents match more critical tasks than NLP patents for this occupation

## Key Features

### Handles Edge Cases

- **Zero matches**: If occupation has no patent matches in a year, exposure = 0
- **Missing years**: Only processes years present in input files
- **Multiple thresholds**: Use `--pattern` flag to filter files
- **Missing importance**: Only processes tasks with Scale Name = "Importance"

### Batch Processing

Automatically processes all category files in the input directory:
- Detects AI category from filename
- Combines results from all 8 categories
- Outputs single unified file

### Summary Statistics

The script outputs useful summaries:
- Average exposure by AI category
- Top 10 most exposed occupations
- Processing time and record counts

## Prerequisites

### Input Requirements

1. **Category-specific matching results** from `categorize_by_ai_category.py`
2. **Required columns in input files:**
   - `O*NET-SOC Code`
   - `Title`
   - `Task`
   - `Scale Name`
   - `Data Value`
   - Year columns (e.g., `2021`, `2022`, etc.)

3. **Scale Name filtering:** Only rows with `Scale Name == "Importance"` are used

### Dependencies

- pandas
- numpy
- tqdm (for progress bars)

## Troubleshooting

### "No importance tasks found"

**Problem:** Input file has no rows with `Scale Name == "Importance"`

**Solution:** Check your O*NET Task Ratings file includes importance ratings

### "No year columns found"

**Problem:** Input files don't have numeric year columns (2021, 2022, etc.)

**Solution:** Verify you're using output from `categorize_by_ai_category.py`

### "Missing required columns"

**Problem:** Input files missing required columns

**Solution:** Ensure input files have all required columns listed above

## Next Steps

After calculating exposure scores, you can:

1. **Visualize trends**: Plot exposure over time for specific occupations
2. **Statistical analysis**: Correlate exposure with employment/wage data
3. **Policy insights**: Identify occupations most affected by AI
4. **Workforce planning**: Predict skill demand based on exposure trends
