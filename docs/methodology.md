# Methodology Documentation

## Overview
This document describes the methodology used for matching AI patents with occupational tasks from O*NET.

## Data Sources

### Patents Data
- Source: USPTO Patent Grant Full Text Data
- Format: TSV/CSV files
- Fields Used:
  - `application_id`: Unique patent identifier
  - `application_title`: Patent title text
  - `topic_label`: Patent classification
  - `filing_date`: Date of filing
  - `year`: Year extracted from filing date

### O*NET Data
- Source: O*NET Task Ratings database
- Version: 28.3
- Fields Used:
  - `Task`: Task description text
  - `Title`: Occupation title
  - `Task ID`: Unique task identifier

## Processing Pipeline

### 1. Data Preprocessing
- Remove duplicate patents
- Filter for AI-related patents using topic categories
- Clean text fields (lowercase, remove special characters)
- Handle missing values

### 2. Embedding Generation
- Model: BERT fine-tuned on patents (`anferico/bert-for-patents`)
- Embedding dimension: 768
- Normalization: L2 normalization for cosine similarity

### 3. Similarity Matching
- Algorithm: ScaNN (Scalable Nearest Neighbors)
- Metric: Cosine similarity (via dot product on normalized vectors)
- Parameters:
  - Number of leaves: 200
  - Training sample size: 250,000
  - Reordering neighbors: 100

### 4. Results Storage
- Top-k matches per task: 200
- Stored fields:
  - Neighbor indices
  - Similarity scores
  - Year-wise results

## Evaluation Metrics
- Mean similarity score
- Distribution of similarity scores
- Number of high-confidence matches (>0.8 similarity)
- Coverage of occupations

## Limitations
- Patent titles may not fully represent patent content
- Temporal aspects not considered in embeddings
- Limited to English language patents