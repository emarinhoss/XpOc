# Sankey Visualization: AI Category → Task → Occupation

This guide explains how to create animated Sankey diagrams that visualize how AI technologies flow through specific tasks to impact occupations.

## What Does It Show?

The Sankey diagram reveals the **pathway of AI impact**:

```
AI Category ══════> Specific Task ══════> Occupation
     ↓                    ↓                    ↓
Machine Learning  "Analyze medical   →  Radiologist
                   images"

Computer Vision   "Inspect products" →  Quality Inspector

NLP              "Draft legal       →  Legal Assistant
                  documents"
```

### Three Levels

1. **Source (Left):** 8 AI technology categories
2. **Middle:** Top tasks per category (specific work activities)
3. **Target (Right):** Top affected occupations

### Flow Thickness

Thicker flows = higher exposure contribution (more impact)

### Animation

Shows how these flows change from 2021 → 2025

## Why This Visualization?

### Tells Concrete Stories

Instead of abstract numbers, you see:
- "Machine Learning patents → 'Analyze patient data' → Radiologist"
- "NLP patents → 'Draft contracts' → Legal Assistant"

### Shows Mechanisms

Reveals **how** AI affects jobs (through specific tasks), not just that it does.

### Breadth View

With 20 occupations × 8 categories × 3 tasks = shows diverse impacts across the economy.

### Time Evolution

Animation shows which pathways are growing vs shrinking.

## Prerequisites

### 1. Install Required Packages

```bash
pip install plotly kaleido
```

- **plotly**: Creates interactive Sankey diagrams
- **kaleido**: Exports static PNG images

### 2. Input Data

You need category-specific matching results from `categorize_by_ai_category.py`:

```
results/category_specific/
├── AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv
├── AI_2021to2025_abstractTitleFor_nlp_0.75.csv
├── AI_2021to2025_abstractTitleFor_computer_vision_0.75.csv
└── ... (8 files total)
```

Each file must have:
- `O*NET-SOC Code`, `Title`, `Task`
- `Scale Name`, `Data Value` (importance)
- Year columns (2021, 2022, etc.)

## Usage

### Basic Usage (Default Settings)

```bash
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations
```

**Outputs:**
- `sankey_ai_impact.html` - Interactive animation
- `sankey_ai_impact_2021.png` - Static image for 2021
- `sankey_ai_impact_2022.png` - Static image for 2022
- ... (one PNG per year)

**Defaults:**
- Top 20 occupations
- Top 3 tasks per AI category
- All years in data
- Both HTML and PNG outputs

### Customize Number of Items

```bash
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --top-occupations 15 \
    --top-tasks-per-category 5 \
    --output-dir results/visualizations
```

**Trade-offs:**
- More items = more comprehensive, but more cluttered
- Fewer items = cleaner, but might miss important stories

**Recommendations:**
- **General public:** 15 occupations, 3 tasks (cleaner)
- **Researchers:** 25 occupations, 5 tasks (comprehensive)
- **Presentations:** 10 occupations, 2 tasks (focus)

### Specific Years Only

```bash
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --years 2021 2023 2025 \
    --output-dir results/visualizations
```

Shows only selected years (faster processing, focused story).

### HTML Only (No PNGs)

```bash
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --output-format html \
    --output-dir results/visualizations
```

Skip PNG generation (faster, for web-only use).

### Custom Output Name

```bash
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --output-name my_sankey_diagram \
    --output-dir results/visualizations
```

Creates `my_sankey_diagram.html` and `my_sankey_diagram_2021.png`, etc.

## How It Works

### 1. Data Selection Strategy

The script selects items to show based on **exposure contribution**:

**For Occupations:**
1. Calculate total exposure for each occupation (across all categories)
2. Select top N (default 20) by total exposure
3. These occupations appear consistently across all years

**For Tasks:**
1. For each AI category, calculate each task's contribution to top occupations
2. Select top M tasks (default 3) per category
3. Contribution = (task matches / total matches) × task importance

**Why this matters:**
- Focuses on high-impact pathways
- Ensures visualization tells important stories
- Avoids cluttering with minor connections

### 2. Sankey Structure

**Nodes:**
- 8 AI category nodes (sources)
- Up to 24 task nodes (3 per category, middle layer)
- 20 occupation nodes (targets)

**Links:**
- Category → Task: Sum of task's contribution across all occupations
- Task → Occupation: Task's contribution to specific occupation

**Flow thickness:**
Proportional to exposure contribution (normalized)

### 3. Animation Implementation

- One Sankey diagram per year
- Plotly animation frames transition between years
- Slider control lets user scrub through timeline
- Play/Pause buttons for automatic animation

### 4. Color Scheme

**AI Categories:** Distinct, colorblind-friendly colors
- Machine Learning: Blue
- NLP: Orange
- Computer Vision: Green
- Speech Recognition: Red
- Knowledge Processing: Purple
- Planning & Control: Brown
- AI Hardware: Pink
- Evolutionary Computation: Gray

**Tasks:** Lighter shades of their source category color

**Occupations:** Neutral gray (targets)

## Output Files

### Interactive HTML

**Features:**
- Hover: See exact values and labels
- Animation: Play/Pause controls
- Slider: Scrub through years
- Responsive: Adjusts to window size
- Shareable: Single file, works in any browser

**Use cases:**
- Embed in website or blog
- Share via email or cloud storage
- Present with interactive exploration

**File size:** ~500KB - 2MB (depends on data size)

### Static PNG Images

**Features:**
- High resolution (1400×900 pixels)
- One image per year
- Publication ready
- No dependencies

**Use cases:**
- Reports and papers
- Presentations (PowerPoint, etc.)
- Social media posts
- Print materials

**File size:** ~200KB per image

## Example Insights

### What You Can Learn

**1. Technology-Specific Impacts**

"Computer Vision heavily impacts Radiologists through 'Analyze medical images' task, but also impacts Quality Inspectors through 'Inspect products' task."

**2. Multi-Technology Exposure**

"Data Scientists are exposed to both Machine Learning AND NLP, showing convergence of AI technologies."

**3. Task-Level Mechanisms**

"NLP's impact on Legal Assistants flows primarily through 'Draft documents' and 'Review contracts' - specific, automatable tasks."

**4. Temporal Trends**

"From 2021 to 2025, the flow from ML → 'Predict outcomes' → Financial Analyst doubled, while other pathways remained stable."

### Storytelling with the Visualization

**For general public:**
- "See how AI patents connect to real work tasks in your occupation"
- "Which AI technologies are affecting which jobs?"

**For policymakers:**
- "Identify occupations with concentrated exposure (single pathway) vs diversified (multiple pathways)"
- "Target retraining programs based on specific task-level impacts"

**For researchers:**
- "Validate or challenge assumptions about automation pathways"
- "Compare predicted impacts vs observed employment changes"

## Customization

### Adjust Visual Parameters

Edit the script to customize:

**Node positioning:**
```python
# In create_sankey_figure function
node=dict(
    pad=15,        # Space between nodes
    thickness=20,  # Node width
    ...
)
```

**Animation speed:**
```python
'frame': {'duration': 1500, 'redraw': True}  # 1.5 seconds per year
```

**Figure size:**
```python
height=900,   # Increase for more vertical space
width=1400    # Increase for more horizontal space
```

### Filter by Exposure Threshold

Add minimum exposure filter to script:

```python
# After calculating contributions
df_contrib = df_contrib[df_contrib['contribution'] > 0.5]  # Only show high-impact
```

## Troubleshooting

### "No module named 'plotly'"

**Solution:**
```bash
pip install plotly kaleido
```

### "No CSV files found"

**Problem:** Input directory doesn't contain category-specific files

**Solution:** Run `categorize_by_ai_category.py` first to generate input files

### Visualization is too cluttered

**Solutions:**
1. Reduce `--top-occupations` (try 15 or 10)
2. Reduce `--top-tasks-per-category` (try 2)
3. Filter to specific years only
4. Increase figure size in script

### Animation doesn't play in browser

**Problem:** Some browsers block autoplay

**Solution:** Click the Play button manually, or use Firefox/Chrome

### PNG generation fails

**Problem:** kaleido not installed or incompatible

**Solution:**
```bash
pip install -U kaleido
```

If still fails, use `--output-format html` to skip PNG generation.

### Flows look disconnected

**Problem:** Top tasks don't match top occupations (different filtering)

**Solution:** This is expected - the script filters tasks that contribute to top occupations. Some categories may have sparse connections.

## Advanced Usage

### Create Multiple Variants

Compare different configurations:

```bash
# Broad view (many items)
python create_sankey_visualization.py \
    --input-dir results/category_specific \
    --top-occupations 30 \
    --top-tasks-per-category 5 \
    --output-name sankey_broad

# Focused view (key items only)
python create_sankey_visualization.py \
    --input-dir results/category_specific \
    --top-occupations 10 \
    --top-tasks-per-category 2 \
    --output-name sankey_focused

# Latest year only
python create_sankey_visualization.py \
    --input-dir results/category_specific \
    --years 2025 \
    --output-format png \
    --output-name sankey_2025_static
```

### Embed in Website

```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Impact Visualization</title>
</head>
<body>
    <h1>How AI Technologies Impact Occupations</h1>

    <iframe src="sankey_ai_impact.html"
            width="100%"
            height="950"
            frameborder="0">
    </iframe>

    <p>
        This visualization shows how AI patents in different categories
        flow through specific work tasks to impact various occupations.
        Use the slider to see changes from 2021 to 2025.
    </p>
</body>
</html>
```

### Share on Social Media

**For Twitter/LinkedIn:**
1. Generate static PNG for latest year (2025)
2. Add descriptive caption highlighting key insight
3. Post image with link to interactive HTML

**Example caption:**
"New analysis: Machine Learning patents are transforming Data Science jobs primarily through 3 tasks: analyzing data, building models, and making predictions. See the full interactive visualization: [link]"

## Performance Notes

**Processing time:**
- HTML only: ~30 seconds
- HTML + PNGs: ~2-3 minutes (depends on number of years)

**Memory usage:**
- ~500MB - 1GB (loads all 8 category files)

**File sizes:**
- HTML: ~500KB - 2MB
- PNG: ~200KB per image

## Integration with Other Analyses

### Combine with Exposure Scores

1. Create Sankey showing pathways
2. Create separate chart showing overall exposure scores
3. Tell complete story: "Which occupations affected" (exposure) + "How they're affected" (Sankey)

### Occupation Deep Dives

1. Use Sankey to identify highly-exposed occupations
2. Create detailed reports for those occupations
3. Interview workers in those roles to validate findings

### Policy Recommendations

1. Identify occupations with single, critical pathway (high risk)
2. Recommend skill diversification or retraining
3. Use task-level insights to design specific interventions

## Next Steps

After creating the visualization:

1. **Interpret:** What stories does it tell?
2. **Validate:** Do the pathways make sense? Ask domain experts
3. **Share:** Publish interactive version, share static images
4. **Iterate:** Try different parameter combinations
5. **Extend:** Add filters, drill-downs, or comparison views

## Related Documentation

- [Category-Specific Matching](category_specific_matching.md) - How to generate input data
- [Occupational Exposure](occupational_exposure.md) - Alternative numerical view of impact
- [Ensemble Voting](ensemble_voting.md) - How patents are categorized
