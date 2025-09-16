# Patent-Occupation Matcher

A system for matching AI patents with occupational tasks using BERT embeddings and ScaNN similarity search. This project helps to understand the impact of technological innovations, as represented by patents, on the labor market structure, as represented by O*NET occupational tasks.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
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
