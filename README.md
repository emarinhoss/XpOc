# Patent-Occupation Matcher

A system for matching AI patents with occupational tasks using BERT embeddings and ScaNN similarity search. This project helps to understand the impact of technological innovations, as represented by patents, on the labor market structure, as represented by O*NET occupational tasks.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Patent Categorization](#patent-categorization)
- [Patent-Occupation Matching](#patent-occupation-matching)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data](#data)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **End-to-End Pipeline**: A complete pipeline from data loading and preprocessing to embedding and matching.
- **BERT-based Embeddings**: Uses a fine-tuned BERT model (`anferico/bert-for-patents`) for generating high-quality embeddings of patent titles and O*NET task descriptions.
- **Efficient Similarity Search**: Implements ScaNN (Scalable Nearest Neighbors) for fast and accurate similarity search, with a fallback to a simpler matcher if ScaNN is not available.
- **Configurable**: The pipeline is highly configurable through a single `config.yaml` file.
- **Modular Design**: The codebase is organized into modular components for data loading, embedding, and matching, making it easy to extend and maintain.
- **Docker Support**: Includes a `Dockerfile` and `docker-compose.yml` for easy setup and deployment.

## System Architecture

The pipeline consists of the following main components:

1.  **Data Loaders**: `PatentLoader` and `ONetLoader` are responsible for loading patent and O*NET data from raw files.
2.  **Data Preprocessor**: Cleans and preprocesses the data before embedding.
3.  **Embedder**: `BertEmbedder` uses a pre-trained BERT model to generate embeddings for patent titles and O*NET tasks.
4.  **Matcher**: `ScannMatcher` or `SimpleMatcher` builds an index of patent embeddings and performs similarity searches to find the best matches for each O*NET task.
5.  **Main Pipeline**: The `PatentOccupationPipeline` class orchestrates the entire workflow, from loading data to saving the results.

The pipeline processes patents year by year, generates embeddings, and finds the top-k most similar patents for each O*NET task. The results are then saved to a CSV file.

## Patent Categorization

Before matching patents with occupations, patents must first be classified into AI categories. This repository provides two approaches:

### Approach 1: Zero-Shot Classification ⭐ **RECOMMENDED**

Uses embedding models and cosine similarity to classify patents into 8 AI categories **without API calls**.

**Supports any HuggingFace model:**
- `anferico/bert-for-patents` (default, specialized for patents)
- `google/embeddinggemma-300m` (efficient, general-purpose)
- `sentence-transformers/all-MiniLM-L6-v2` (very fast)
- Any other sentence-transformers or AutoModel

**Benefits:**
- **Free**: No API costs ($0 vs $9,000-$18,000 for 900k patents)
- **Fast**: ~4 hours on GPU or ~24 hours on CPU for 900k patents
- **Flexible**: Choose from hundreds of HuggingFace models
- **Offline**: Works without internet connection
- **Reproducible**: Same results every time

**Usage (Default BERT-for-patents):**
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv
```

**Usage (Google EmbeddingGemma):**
```bash
python src/scripts/categorize_patents_zeroshot.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --model-name google/embeddinggemma-300m \
    --model-type auto-model \
    --pooling mean
```

See [docs/multi_model_support.md](docs/multi_model_support.md) for more model options.

### Approach 1a: Azure OpenAI Embeddings ⭐ **CLOUD-BASED**

Uses Azure OpenAI embedding models for zero-shot classification with built-in rate limiting.

**Benefits:**
- **No GPU needed**: Runs on any machine
- **Fast**: ~2-3k patents/second with proper rate limits
- **High quality**: Azure OpenAI embeddings (ada-002, ada-003, etc.)
- **Cheaper than GPT**: ~$27 for 900k patents vs $9k-$18k

**Trade-offs:**
- **Not free**: Costs money (but ~300x cheaper than GPT classification)
- **Requires Azure**: Need Azure OpenAI subscription
- **Internet required**: API calls

**Usage:**
```bash
# Set environment variables
export AZURE_OPENAI_API_KEY="your-key"
export OPENAI_API_VERSION="2023-05-15"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Run with default settings (500 QPS)
python src/scripts/categorize_patents_azure.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv

# Custom rate limit (adjust based on your Azure quota)
python src/scripts/categorize_patents_azure.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --rate-limit 100 \
    --deployment-name text-embedding-ada-002
```

**When to use:**
- Production environment without GPU infrastructure
- Moderate budget (~$30 for 900k patents)
- Need consistent cloud-based performance
- Already using Azure services

See [docs/azure_categorization.md](docs/azure_categorization.md) for complete guide and rate limiting configuration.

### Approach 1b: Ensemble Voting ⭐⭐ **HIGHEST ACCURACY**

Uses **5 different models with majority voting** for maximum accuracy. Each model votes on a category, and the winner is selected by majority (with tie-breaking by confidence).

**Default models:**
- `anferico/bert-for-patents` (patents domain)
- `google/embeddinggemma-300m` (efficient)
- `all-mpnet-base-v2` (high quality)
- `all-MiniLM-L6-v2` (fast)
- `allenai/scibert_scivocab_uncased` (scientific)

**Benefits:**
- **Highest accuracy**: Reduces errors through multi-model agreement
- **Confidence calibration**: High vote count = high reliability
- **No API costs**: Still $0 (runs locally)
- **Detailed analysis**: See individual model predictions

**Trade-offs:**
- **5x slower**: Must run 5 models (~38 hours on GPU for 900k patents)
- **More memory**: ~1.7 GB total

**Usage:**
```bash
python src/scripts/categorize_patents_ensemble.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_ensemble.csv \
    --device cuda
```

See [docs/ensemble_voting.md](docs/ensemble_voting.md) for detailed explanation.

**Alternative: Ensemble Aggregation** (Recommended for large datasets)

For more flexibility, run models separately and aggregate afterwards:

```bash
# Step 1: Run each model separately (can be parallelized)
python src/scripts/categorize_patents_zeroshot.py \
    --model-name anferico/bert-for-patents \
    --output-file results/bert.csv ...

# Step 2: Aggregate results from all models
python src/scripts/aggregate_ensemble_voting.py \
    --input-files results/bert.csv results/gemma.csv ... \
    --output-file results/ensemble.csv
```

**Benefits:** 5x faster with parallel execution, lower memory per model, more flexible.
See [docs/ensemble_aggregation_workflow.md](docs/ensemble_aggregation_workflow.md) for complete workflow.

### Approach 2: OpenAI API Classification

Uses GPT-3.5-turbo or Azure OpenAI to classify patents via API calls.

**Benefits:**
- Higher accuracy on ambiguous cases (~95-98%)
- Easy to use

**Drawbacks:**
- Expensive (~$9,000-$18,000 for 900k patents)
- Slow (rate-limited, ~10 days for 900k patents)
- Requires API key and internet

**Usage:**
```bash
python src/scripts/categorize_patents.py \
    --input-file data/processed/patents.csv \
    --output-file data/processed/patents_categorized.csv \
    --api-key-env RAND_OPENAI_API_KEY
```

### AI Categories

Both approaches classify patents into these 8 categories:
- **computer vision**: Image and video understanding
- **evolutionary computation**: Evolution-inspired optimization
- **AI hardware**: Specialized AI processors
- **knowledge processing**: Knowledge bases and reasoning
- **machine learning**: Learning algorithms
- **NLP**: Natural language processing
- **planning and control**: Planning and robotics
- **speech recognition**: Speech understanding

See [docs/patent_categorization_comparison.md](docs/patent_categorization_comparison.md) for a detailed comparison.

## Patent-Occupation Matching

After categorizing patents, match them with O*NET occupational tasks to understand AI's impact on work.

### Standard Matching (All AI Patents Combined)

Match all AI patents to occupational tasks in one run:

```bash
python src/scripts/run_pipeline.py --config config/config.yaml
```

**Output:** Single file with all AI patent matches across years.

**Best for:** Understanding overall AI impact on occupations.

### Category-Specific Matching ⭐ **For Detailed Analysis**

Run matching separately for each of the 8 AI categories to reveal differential impacts:

```bash
python src/scripts/categorize_by_ai_category.py \
    --patents-file data/processed/patents_ensemble.csv \
    --onet-file data/raw/onet/Task_Ratings.xlsx \
    --output-dir results/category_specific \
    --device cuda
```

**Output:** 8 separate files showing how each AI category affects tasks:
- `AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv`
- `AI_2021to2025_abstractTitleFor_nlp_0.75.csv`
- `AI_2021to2025_abstractTitleFor_computer_vision_0.75.csv`
- ... (5 more categories)

**Benefits:**
- **Differential impact analysis**: See which tasks are affected by computer vision vs NLP
- **Category-specific trends**: Track how ML impact grows differently than other categories
- **Fine-grained insights**: Understand which AI technologies affect which occupations

**Best for:** Research requiring detailed understanding of different AI technology impacts.

See [docs/category_specific_matching.md](docs/category_specific_matching.md) for complete guide.

### Occupational Exposure Calculation ⭐ **Final Analysis Step**

Calculate **occupational exposure scores** that measure how much each occupation's important tasks are matched by AI patents, weighted by task importance.

```bash
python src/scripts/calculate_occupational_exposure.py \
    --input-dir results/category_specific \
    --output-file results/occupational_exposure.csv
```

**What is exposure?**

Exposure quantifies the degree to which an occupation's critical tasks are affected by AI technologies:

```
exposure = Σ (patent matches for task / total matches) × task importance
```

**Output:** Single file with exposure scores for all occupations × AI categories × years:

```csv
O*NET-SOC Code,Title,AI_Category,2021,2022,2023,2024,2025
15-2051.00,Data Scientist,machine_learning,4.52,4.58,4.61,4.65,4.68
15-2051.00,Data Scientist,nlp,4.31,4.35,4.39,4.42,4.45
15-1252.00,Software Engineer,machine_learning,3.78,3.82,3.85,3.88,3.91
...
```

**Interpretation:**
- Higher score = occupation's more important tasks are heavily matched by AI
- Scores range ~1-5 (based on O*NET importance scale)
- Compare across AI categories to see differential impacts

**Benefits:**
- **Weighted by importance**: Focuses on critical tasks, not just any matches
- **Cross-category comparison**: See which AI technologies affect which occupations most
- **Trend analysis**: Track how exposure changes over time
- **Policy insights**: Identify occupations most affected by AI for workforce planning

**Best for:** Final analysis, policy research, identifying high-impact occupations.

See [docs/occupational_exposure.md](docs/occupational_exposure.md) for detailed explanation and analysis examples.

## Visualization

### Sankey Diagram: AI Category → Task → Occupation ⭐ **Interactive Storytelling**

Create animated flow diagrams showing how AI technologies impact occupations through specific tasks.

```bash
python src/scripts/create_sankey_visualization.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --top-occupations 20 \
    --top-tasks-per-category 3
```

**What it shows:**

```
Machine Learning ══════> "Analyze medical images" ══════> Radiologist
                ════════> "Build predictive models" ═════> Data Scientist

Computer Vision  ══════> "Inspect products" ════════════> Quality Inspector
                ════════> "Analyze medical images" ══════> Radiologist

NLP             ═══════> "Draft legal documents" ═══════> Legal Assistant
```

**Features:**
- **Animated:** Shows change from 2021 → 2025
- **Interactive HTML:** Hover for details, play/pause controls
- **Static PNGs:** One per year for presentations
- **Concrete stories:** See specific task-level mechanisms of AI impact
- **Breadth view:** 20 occupations × 8 AI categories × 3 tasks = comprehensive overview

**Outputs:**
- `sankey_ai_impact.html` - Interactive, shareable visualization
- `sankey_ai_impact_2021.png` through `sankey_ai_impact_2025.png` - Static images

**Best for:** Public communication, presentations, understanding HOW AI affects jobs (not just that it does).

See [docs/sankey_visualization.md](docs/sankey_visualization.md) for complete guide and customization options.

### Scatterplot: Task Importance vs AI Match Count ⭐ **Research & Analysis**

Create scatterplots revealing whether AI is automating critical tasks versus routine ones.

```bash
python src/scripts/create_importance_scatter.py \
    --input-dir results/category_specific \
    --output-dir results/visualizations \
    --animation --static
```

**What it shows:**

Four quadrants revealing automation patterns:

```
High        │ Protected      │ HIGH CONCERN
Importance  │ (few matches)  │ (many matches)
            ├────────────────┼────────────────
Low         │ Minimal Impact │ Routine Auto
Importance  │ (few matches)  │ (many matches)
            └────────────────┴────────────────
                 Low              High
              AI Matches        AI Matches
```

**Key insights:**
- **Top-right:** Important tasks being automated → worker concern
- **Bottom-right:** Routine task automation → expected pattern
- **Top-left:** Important tasks still safe from AI
- **Animation:** Shows how tasks move between quadrants over time

**Features:**
- **Interactive plots:** Hover for task details, zoom, filter
- **Animated timeline:** Watch automation progression 2021→2025
- **Quadrant analysis:** Automatic labeling of high-concern outliers
- **Category colors:** See which AI technologies target which tasks
- **Statistical insights:** Correlation between importance and automation

**Outputs:**
- `importance_scatter_animated.html` - Interactive timeline
- `importance_scatter_2021.html` through `2025.html` - Individual years

**Best for:** Research papers, policy analysis, understanding whether AI targets skilled vs routine work.

See [docs/importance_scatterplot.md](docs/importance_scatterplot.md) for analysis guide and interpretation.

## Installation

### Prerequisites

- Python 3.8 or higher
- A C++ compiler for ScaNN (e.g., `g++` on Linux, Build Tools for Visual Studio on Windows)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/patent-occupation-matcher.git
    cd patent-occupation-matcher
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Using Docker:**

    Alternatively, you can use Docker to set up the environment:

    ```bash
    docker-compose build
    docker-compose up -d
    ```

## Usage

To run the main pipeline, use the `run_pipeline.py` script with the path to your configuration file.

```bash
python src/scripts/run_pipeline.py --config config/config.yaml
```

### Command-line Arguments

-   `--config`: Path to the configuration YAML file. Default: `config/config.yaml`.
-   `--help`: Show the help message and exit.

## Configuration

The pipeline is configured using the `config/config.yaml` file. Here's an overview of the main configuration options:

```yaml
# Data paths
data:
  patents:
    raw_path: "data/raw/patents/"
    processed_path: "data/processed/patents.csv"
    ai_categories_path: "data/raw/categories/patent_categories.xlsx"
  onet:
    raw_path: "data/raw/onet/"
    task_ratings_path: "data/raw/onet/Task Ratings-3.xlsx"
    processed_path: "data/processed/onet_tasks.csv"

# Model configuration
embedding:
  model_name: "anferico/bert-for-patents"
  device: "cpu"  # "cuda" or "cpu"
  batch_size: 32
  max_length: 512
  normalize: true

# Matching configuration (for ScaNN)
matching:
  num_leaves: 200
  num_leaves_to_search: 100
  training_sample_size: 250000
  reorder_num_neighbors: 100

# Processing
processing:
  years_to_process: [2021, 2022, 2023, 2024, 2025]
  ai_topic_filter: 0
  similarity_threshold: 0.75
  top_k_matches: 200

# Output
output:
  results_path: "data/processed/matches/"
  format: "csv"
  save_embeddings: true
```

## Data

The project requires two main data sources:

-   **Patents Data**: From the USPTO Patent Grant Full Text Data. The expected format is a TSV or CSV file with fields like `application_id`, `application_title`, and `filing_date`.
-   **O\*NET Data**: From the O\*NET Task Ratings database. The expected format is an Excel file with fields like `Task ID`, `Task`, and `Title`.

You can use the provided script to download sample data:

```bash
python src/scripts/download_data.py
```

This will download sample data to the `data/raw` directory.

## Testing

To run the tests, use the following command:

```bash
python -m unittest discover src/tests
```

You can also run the main pipeline test directly:

```bash
python src/tests/test_pipeline.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature`).
6.  Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
