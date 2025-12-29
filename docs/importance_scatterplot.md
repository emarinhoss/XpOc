# Task Importance vs AI Match Count Scatterplot

This guide explains how to create scatterplot visualizations that reveal whether AI is automating critical tasks versus routine ones.

## What Does It Show?

The scatterplot reveals the **relationship between task importance and AI impact**:

- **X-axis:** Number of AI patent matches for the task
- **Y-axis:** Task importance (O*NET 1-5 scale)
- **Points:** Individual tasks
- **Color:** AI category (optional)

### Four Quadrants

```
High Importance (5)
    │
    │  Protected          │  HIGH CONCERN
    │  (few AI matches)   │  (many AI matches)
    │                     │
3.5 ├─────────────────────┼──────────────────
    │                     │
    │  Minimal Impact     │  Routine Automation
    │  (few matches)      │  (many matches)
    │                     │
Low (1)
    └─────────────────────┴──────────────────
              Median              High
           AI Matches         AI Matches
```

**Interpretations:**
- **Top-right (High Concern):** Important tasks being heavily automated → worker concern
- **Bottom-right (Routine Automation):** Low-importance tasks automated → less concern
- **Top-left (Protected):** Important tasks with little AI → currently safe
- **Bottom-left (Minimal Impact):** Low-importance, low-match → negligible impact

## Why This Visualization?

### Shows Automation Quality

Unlike simple counts, this reveals **what kind of work** is being automated:
- Are AI patents targeting high-skilled, important work?
- Or are they automating routine, low-importance tasks?

### Policy Insights

**For workforce planning:**
- Top-right quadrant → invest in retraining programs
- Top-left quadrant → tasks to emphasize in education
- Bottom-right → natural automation progression

### Research Value

- Validates or challenges assumptions about AI's impact
- Shows whether automation follows "routine task" hypothesis
- Identifies unexpected patterns

## Prerequisites

### Input Data

Category-specific matching results from `categorize_by_ai_category.py`:

```
results/category_specific/
├── AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv
├── AI_2021to2025_abstractTitleFor_nlp_0.75.csv
└── ... (8 files total)
```

Required columns:
- `Task`, `Data Value` (importance), year columns (2021, 2022, etc.)
- `Scale Name` = "Importance"

### Python Packages

```bash
pip install plotly pandas numpy tqdm
```

## Usage

### Basic Static Plot (Current Year)

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --static \
    --years 2025
```

Creates: `importance_scatter_2025.html`

### Animated Scatterplot (All Years)

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --animation
```

Creates: `importance_scatter_animated.html`
Shows evolution from 2021 → 2025 with play/pause controls.

### Both Animated + Static

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --animation --static
```

Creates:
- `importance_scatter_animated.html` (interactive timeline)
- `importance_scatter_2021.html` through `importance_scatter_2025.html` (individual years)

### Aggregate View (No Category Colors)

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --static \
    --no-show-categories
```

All points in single color, aggregated across AI categories.

### Customize Number of Labels

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --static \
    --top-n-labels 15
```

Labels top 15 outliers in high-concern quadrant (default: 10).

### With PNG Export (Optional)

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --static \
    --save-png
```

**Note:** Requires kaleido. May not work in some environments. HTML is recommended.

## Output Files

### Interactive HTML

**Features:**
- Hover over points to see task details
- Zoom and pan
- Toggle categories on/off (click legend)
- Animated version has play/pause and slider

**Use cases:**
- Exploratory data analysis
- Presentations with Q&A
- Embedding in research websites
- Sharing with stakeholders

### Static PNG (Optional)

**Features:**
- Publication-ready
- High resolution (1000×700 px)
- No interactivity

**Use cases:**
- Papers and reports
- Slides without interactivity needed
- Print materials

## Interpreting the Visualization

### What to Look For

**1. Quadrant Distribution**

```python
# Example interpretation:
- 60% of tasks in bottom-right → mostly routine automation
- 20% in top-right → some concern (important tasks automated)
- 15% in top-left → many important tasks still protected
- 5% in bottom-left → minimal AI activity
```

**2. Trend Over Time (Animation)**

Watch for:
- Movement from left to right → increasing AI activity
- Movement from top-left to top-right → important tasks becoming automated
- Clustering patterns → certain importance levels targeted

**3. Category Patterns (Colored View)**

- Which AI categories target high-importance tasks?
- Do some categories focus on routine vs critical work?
- Cross-category differences

**4. Labeled Outliers**

The script automatically labels tasks in the high-concern quadrant (top-right).
These are specific tasks policymakers should monitor.

### Example Insights

**Finding:** "Many points cluster in top-right quadrant"
**Interpretation:** AI is targeting important, high-skill tasks
**Implication:** Workers in these roles face displacement risk

**Finding:** "Most points in bottom-right, few in top-right"
**Interpretation:** AI primarily automates routine tasks
**Implication:** Supports "routine task automation" hypothesis

**Finding:** "Computer vision points higher than NLP"
**Interpretation:** Vision AI targets more important tasks
**Implication:** Different retraining needs by AI type

## Advanced Analysis

### Quadrant Statistics

After generating the visualization, analyze quadrant distributions:

```python
import pandas as pd

# Load data
df = pd.read_csv('results/category_specific/AI_..._machine_learning_0.75.csv')
df = df[df['Scale Name'] == 'Importance']

# Define quadrants
median_matches = df[df['2025'] > 0]['2025'].median()
high_importance = 4.0

# Count by quadrant
high_concern = len(df[(df['Data Value'] >= high_importance) & (df['2025'] >= median_matches)])
protected = len(df[(df['Data Value'] >= high_importance) & (df['2025'] < median_matches)])
routine = len(df[(df['Data Value'] < high_importance) & (df['2025'] >= median_matches)])
minimal = len(df[(df['Data Value'] < high_importance) & (df['2025'] < median_matches)])

print(f"High Concern: {high_concern} ({100*high_concern/len(df):.1f}%)")
print(f"Protected: {protected} ({100*protected/len(df):.1f}%)")
print(f"Routine Automation: {routine} ({100*routine/len(df):.1f}%)")
print(f"Minimal Impact: {minimal} ({100*minimal/len(df):.1f}%)")
```

### Correlation Analysis

```python
import numpy as np
from scipy.stats import pearsonr

# Calculate correlation
corr, p_value = pearsonr(df['Data Value'], df['2025'])

print(f"Correlation: {corr:.3f} (p={p_value:.4f})")

# Interpretation:
# corr > 0: AI targets MORE important tasks (concerning)
# corr ≈ 0: No relationship (AI targets all importance levels equally)
# corr < 0: AI targets LESS important tasks (routine automation)
```

### Temporal Change

```python
# Compare 2021 vs 2025
df['change'] = df['2025'] - df['2021']

high_importance_tasks = df[df['Data Value'] >= 4.0]
avg_change_important = high_importance_tasks['change'].mean()

low_importance_tasks = df[df['Data Value'] < 3.0]
avg_change_routine = low_importance_tasks['change'].mean()

print(f"Important tasks: +{avg_change_important:.1f} matches on average")
print(f"Routine tasks: +{avg_change_routine:.1f} matches on average")

# Which grew faster?
if avg_change_important > avg_change_routine:
    print("→ AI increasingly targeting important tasks (CONCERN)")
else:
    print("→ AI still focused on routine tasks (expected pattern)")
```

## Customization

### Adjust Quadrant Thresholds

Edit the script to change importance threshold:

```python
# Default: 3.5
# Change to 4.0 for stricter "high importance"
fig.add_hline(
    y=4.0,  # Changed from 3.5
    line_dash="dash",
    ...
)
```

### Change Match Threshold

```python
# Default: median
# Change to 75th percentile for stricter "high match"
percentile_75 = plot_df['matches'].quantile(0.75)
fig.add_vline(x=percentile_75, ...)
```

### Custom Annotations

```python
# Annotate specific tasks of interest
tasks_of_interest = [
    "Analyze medical images",
    "Draft legal documents"
]

for task in tasks_of_interest:
    task_row = plot_df[plot_df['Task'] == task].iloc[0]
    fig.add_annotation(
        x=task_row['matches'],
        y=task_row['importance'],
        text=task,
        ...
    )
```

## Comparison with Other Visualizations

**vs Sankey Diagram:**
- Sankey shows flow/pathways (categorical)
- Scatterplot shows correlation/relationship (quantitative)
- Use both: Sankey for storytelling, scatter for analysis

**vs Exposure Scores:**
- Exposure is aggregated (occupation-level)
- Scatterplot is granular (task-level)
- Use both: Exposure for policy, scatter for research

**vs Heat Map:**
- Heat map shows matrix of values
- Scatterplot shows correlation
- Use scatter when relationship is the focus

## Common Questions

### Q: Why are some high-importance tasks in bottom-left?

**A:** These are important tasks that AI hasn't matched much (yet).
- Could be tasks requiring human judgment
- Or areas where AI hasn't been applied
- These are "safe" tasks for now

### Q: What if most points are in bottom-right?

**A:** This is actually good news!
- Suggests AI is automating routine work (as expected)
- Less worker displacement concern
- But monitor movement over time

### Q: Should I worry about tasks in top-right?

**A:** Yes, these deserve attention:
- Identify which occupations perform these tasks
- Design retraining for workers
- Consider whether automation is inevitable or can be resisted

### Q: Can I combine categories?

**A:** Yes, use `--no-show-categories` flag:
- Aggregates all AI categories
- Shows total AI impact per task
- Simpler visualization for general audiences

## Integration with Policy Analysis

### Step 1: Identify High-Concern Tasks

```python
# Get tasks in top-right quadrant
high_concern = df[
    (df['Data Value'] >= 4.0) &
    (df['2025'] >= median_matches)
]

# Export for further analysis
high_concern[['Task', 'Title', 'Data Value', '2025']].to_csv(
    'high_concern_tasks.csv',
    index=False
)
```

### Step 2: Link to Occupations

```python
# Which occupations perform high-concern tasks?
high_concern_occs = high_concern.groupby('Title').agg({
    'Task': 'count',
    'Data Value': 'mean',
    '2025': 'sum'
}).sort_values('Task', ascending=False)

print("Occupations with most high-concern tasks:")
print(high_concern_occs.head(10))
```

### Step 3: Prioritize Interventions

```python
# Score occupations by (# high-concern tasks) × (avg importance) × (total matches)
high_concern_occs['priority_score'] = (
    high_concern_occs['Task'] *
    high_concern_occs['Data Value'] *
    high_concern_occs['2025']
)

print("\nTop priority occupations for intervention:")
print(high_concern_occs.nlargest(5, 'priority_score'))
```

## Next Steps

After creating the scatterplot:

1. **Analyze Patterns:** Use quadrant analysis and correlation
2. **Compare Years:** Look for temporal trends in animation
3. **Identify Outliers:** Investigate tasks in top-right
4. **Cross-reference:** Compare with occupational exposure scores
5. **Policy Reports:** Use findings to inform workforce strategy

## Related Documentation

- [Category-Specific Matching](category_specific_matching.md) - How to generate input data
- [Sankey Visualization](sankey_visualization.md) - Complementary pathway view
- [Occupational Exposure](occupational_exposure.md) - Aggregate scores for policy
