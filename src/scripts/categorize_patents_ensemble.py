#!/usr/bin/env python
"""
Ensemble voting patent categorization using multiple embedding models.
This approach runs 5 different models and uses majority voting to classify patents.

Each model gets one vote, and the final classification is determined by the category
with the most votes. The confidence is the average of confidences from models that
agreed on the winning category.
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embedding.generic_embedder import GenericEmbedder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- AI Category Definitions ---
CATEGORIES_WITH_DEFINITIONS = {
    "computer vision": "methods to understand images and videos, including object detection, image recognition, facial recognition, and visual processing",
    "evolutionary computation": "methods mimicking evolution to solve problems using genetic algorithms, evolutionary strategies, and nature-inspired optimization",
    "AI hardware": "physical hardware designed specifically to implement AI software, including neural network processors, AI accelerators, and specialized chips",
    "knowledge processing": "methods to represent and derive new facts from knowledge bases, including ontologies, reasoning systems, and expert systems",
    "machine learning": "algorithms that learn from data, including supervised learning, unsupervised learning, reinforcement learning, and statistical models",
    "NLP": "methods to understand and generate human language, including text processing, sentiment analysis, machine translation, and language models",
    "planning and control": "methods to determine and execute plans to achieve goals, including robotics control, automated planning, and decision systems",
    "speech recognition": "methods to understand speech and generate responses, including voice recognition, speech-to-text, and audio processing"
}

# --- Default Model Configurations ---
DEFAULT_MODELS = [
    {
        'name': 'bert-for-patents',
        'model_name': 'anferico/bert-for-patents',
        'model_type': 'sentence-transformers',
        'pooling': 'mean'
    },
    {
        'name': 'embeddinggemma',
        'model_name': 'google/embeddinggemma-300m',
        'model_type': 'auto-model',
        'pooling': 'mean'
    },
    {
        'name': 'mpnet',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'model_type': 'sentence-transformers',
        'pooling': 'mean'
    },
    {
        'name': 'minilm',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'model_type': 'sentence-transformers',
        'pooling': 'mean'
    },
    {
        'name': 'scibert',
        'model_name': 'allenai/scibert_scivocab_uncased',
        'model_type': 'auto-model',
        'pooling': 'mean'
    }
]


class EnsemblePatentClassifier:
    """Ensemble classifier using multiple models with majority voting."""

    def __init__(self, models_config: List[Dict], device: str = 'cpu', batch_size: int = 32, max_length: int = 512):
        """
        Initialize ensemble classifier with multiple models.

        Args:
            models_config: List of model configurations
            device: Device to run models on
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
        """
        self.models_config = models_config
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.categories = list(CATEGORIES_WITH_DEFINITIONS.keys())

        # Initialize all models
        self.models = []
        self.model_names = []

        logger.info(f"Initializing {len(models_config)} models for ensemble voting...")
        for i, model_cfg in enumerate(models_config, 1):
            logger.info(f"Loading model {i}/{len(models_config)}: {model_cfg['name']} ({model_cfg['model_name']})")

            config = {
                'embedding': {
                    'model_name': model_cfg['model_name'],
                    'model_type': model_cfg['model_type'],
                    'device': self.device,
                    'batch_size': self.batch_size,
                    'max_length': self.max_length,
                    'normalize': True,
                    'pooling': model_cfg.get('pooling', 'mean')
                }
            }

            embedder = GenericEmbedder(config)

            # Pre-compute category embeddings for this model
            category_texts = list(CATEGORIES_WITH_DEFINITIONS.values())
            category_embeddings = embedder.encode_texts(category_texts, show_progress=False)

            self.models.append({
                'name': model_cfg['name'],
                'embedder': embedder,
                'category_embeddings': category_embeddings
            })
            self.model_names.append(model_cfg['name'])

        logger.info(f"All {len(self.models)} models loaded successfully")

    def classify_single_with_model(
        self,
        text: str,
        model_data: Dict
    ) -> Tuple[str, float]:
        """
        Classify a single text using one model.

        Args:
            text: Patent text
            model_data: Model configuration and embeddings

        Returns:
            Tuple of (category, confidence)
        """
        if not text or text.strip() == "":
            return "missing_data", 0.0

        # Encode the patent text
        patent_embedding = model_data['embedder'].encode_texts([text], show_progress=False)[0]

        # Compute cosine similarity with all categories
        similarities = np.dot(model_data['category_embeddings'], patent_embedding)

        # Get the category with highest similarity
        best_idx = np.argmax(similarities)
        best_category = self.categories[best_idx]
        confidence = float(similarities[best_idx])

        return best_category, confidence

    def ensemble_vote(
        self,
        predictions: List[Tuple[str, float]]
    ) -> Tuple[str, float, int]:
        """
        Perform ensemble voting on predictions from all models.

        Args:
            predictions: List of (category, confidence) tuples from each model

        Returns:
            Tuple of (final_category, ensemble_confidence, vote_count)
        """
        # Extract categories and confidences
        categories = [pred[0] for pred in predictions]

        # Count votes
        vote_counts = Counter(categories)

        # Find maximum vote count
        max_votes = max(vote_counts.values())

        # Get all categories with maximum votes (for tie-breaking)
        tied_categories = [cat for cat, count in vote_counts.items() if count == max_votes]

        if len(tied_categories) == 1:
            # Clear winner
            winning_category = tied_categories[0]
        else:
            # Tie-breaking: choose category with highest average confidence
            logger.debug(f"Tie detected among {tied_categories}. Breaking tie by average confidence.")

            tie_avg_confidences = {}
            for cat in tied_categories:
                # Get confidences of models that voted for this category
                cat_confidences = [conf for pred_cat, conf in predictions if pred_cat == cat]
                tie_avg_confidences[cat] = np.mean(cat_confidences)

            winning_category = max(tie_avg_confidences, key=tie_avg_confidences.get)

        # Calculate ensemble confidence: average of confidences from agreeing models
        agreeing_confidences = [conf for cat, conf in predictions if cat == winning_category]
        ensemble_confidence = float(np.mean(agreeing_confidences))

        return winning_category, ensemble_confidence, max_votes

    def classify_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[Dict], List[Tuple[str, float, int]]]:
        """
        Classify multiple patents using ensemble voting.

        Args:
            texts: List of patent texts

        Returns:
            Tuple of (individual_predictions, ensemble_results)
            - individual_predictions: List of dicts with each model's prediction
            - ensemble_results: List of (category, confidence, vote_count) tuples
        """
        all_individual_predictions = []
        ensemble_results = []

        logger.info(f"Processing {len(texts)} patents with {len(self.models)} models...")

        # Get predictions from each model
        for model_data in self.models:
            model_name = model_data['name']
            logger.info(f"Running model: {model_name}")

            model_predictions = []

            for i in tqdm(range(0, len(texts), self.batch_size),
                         desc=f"Classifying with {model_name}"):
                batch_texts = texts[i:i+self.batch_size]

                for text in batch_texts:
                    category, confidence = self.classify_single_with_model(text, model_data)
                    model_predictions.append((category, confidence))

            all_individual_predictions.append(model_predictions)

        # Perform ensemble voting for each patent
        logger.info("Performing ensemble voting...")
        for patent_idx in tqdm(range(len(texts)), desc="Ensemble voting"):
            # Gather predictions from all models for this patent
            patent_predictions = []
            for model_idx in range(len(self.models)):
                patent_predictions.append(all_individual_predictions[model_idx][patent_idx])

            # Vote
            ensemble_result = self.ensemble_vote(patent_predictions)
            ensemble_results.append(ensemble_result)

        # Reorganize individual predictions into per-patent format
        individual_results = []
        for patent_idx in range(len(texts)):
            patent_result = {}
            for model_idx, model_name in enumerate(self.model_names):
                category, confidence = all_individual_predictions[model_idx][patent_idx]
                patent_result[f'{model_name}_category'] = category
                patent_result[f'{model_name}_confidence'] = confidence
            individual_results.append(patent_result)

        return individual_results, ensemble_results


def combine_text_fields(row: pd.Series, title_col: str, abstract_col: str) -> str:
    """Combine title and abstract into single text for classification."""
    title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else ""
    abstract = str(row.get(abstract_col, "")) if pd.notna(row.get(abstract_col)) else ""

    # Combine with title given more weight
    combined = f"{title}. {abstract}".strip()
    return combined if combined != "." else ""


def load_models_config(config_file: Optional[str]) -> List[Dict]:
    """Load models configuration from file or use defaults."""
    if config_file and Path(config_file).exists():
        logger.info(f"Loading models configuration from {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('ensemble_models', DEFAULT_MODELS)
    else:
        logger.info("Using default model configuration")
        return DEFAULT_MODELS


def main(args):
    """Main function to run ensemble patent categorization."""

    # Setup paths
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input data
    try:
        logger.info(f"Loading patents from {input_path}...")
        if input_path.suffix == '.tsv':
            df = pd.read_csv(input_path, sep='\t')
        else:
            df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} patents")
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        return

    # Filter by year if specified
    if args.start_year:
        if 'year' not in df.columns:
            if 'published_date' in df.columns:
                df['published_date'] = pd.to_datetime(df['published_date'])
                df['year'] = df['published_date'].dt.year
            else:
                logger.warning("Cannot filter by year: 'year' or 'published_date' column not found.")

        if 'year' in df.columns:
            original_count = len(df)
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df = df[df['year'] >= args.start_year]
            logger.info(f"Filtered from {original_count} to {len(df)} patents (year >= {args.start_year})")

    # Load models configuration
    models_config = load_models_config(args.models_config)
    logger.info(f"Using {len(models_config)} models for ensemble voting:")
    for i, model_cfg in enumerate(models_config, 1):
        logger.info(f"  {i}. {model_cfg['name']}: {model_cfg['model_name']}")

    # Initialize ensemble classifier
    classifier = EnsemblePatentClassifier(
        models_config=models_config,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Check if output exists for resumption
    if output_path.exists() and not args.overwrite:
        logger.info(f"Output file found at {output_path}. Loading existing results...")
        df_existing = pd.read_csv(output_path)

        # Check if we can resume
        if args.category_column in df_existing.columns and len(df_existing) == len(df):
            logger.info("Resuming from existing results...")
            df = df_existing
        else:
            logger.warning("Existing output file incompatible. Starting fresh...")

    # Check what needs processing
    ensemble_col = f'{args.category_column}_ensemble'
    if ensemble_col not in df.columns:
        df[ensemble_col] = None

    to_process = df[ensemble_col].isnull()
    num_to_process = to_process.sum()

    if num_to_process == 0:
        logger.info("All patents already categorized. Exiting.")
        return

    logger.info(f"Categorizing {num_to_process} patents with ensemble voting...")

    # Combine title and abstract
    df_to_process = df[to_process].copy()
    texts = df_to_process.apply(
        lambda row: combine_text_fields(row, args.title_column, args.abstract_column),
        axis=1
    ).tolist()

    # Classify with ensemble
    individual_results, ensemble_results = classifier.classify_batch(texts)

    # Update dataframe with individual model results
    for i, individual in enumerate(individual_results):
        idx = df_to_process.index[i]
        for key, value in individual.items():
            df.loc[idx, key] = value

    # Update dataframe with ensemble results
    for i, (category, confidence, vote_count) in enumerate(ensemble_results):
        idx = df_to_process.index[i]
        df.loc[idx, ensemble_col] = category
        df.loc[idx, f'{args.category_column}_ensemble_confidence'] = confidence
        df.loc[idx, f'{args.category_column}_ensemble_votes'] = vote_count

    # Save results
    logger.info(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE CATEGORIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total patents: {len(df)}")
    logger.info(f"Newly categorized: {num_to_process}")

    logger.info("\nEnsemble category distribution:")
    category_counts = df[ensemble_col].value_counts()
    for cat, count in category_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    logger.info(f"\nAverage ensemble confidence: {df[f'{args.category_column}_ensemble_confidence'].mean():.3f}")

    # Vote distribution
    logger.info("\nVote count distribution:")
    vote_counts = df[f'{args.category_column}_ensemble_votes'].value_counts().sort_index()
    for votes, count in vote_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {int(votes)} votes: {count} patents ({pct:.1f}%)")

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble voting patent categorization using 5 different models. "
                    "Uses majority voting with tie-breaking by confidence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default 5 models
  python categorize_patents_ensemble.py --input-file patents.csv

  # Custom models via config file
  python categorize_patents_ensemble.py \\
      --input-file patents.csv \\
      --models-config config/ensemble_models.yaml

  # With GPU acceleration
  python categorize_patents_ensemble.py \\
      --input-file patents.csv \\
      --device cuda \\
      --batch-size 64
        """
    )

    # Input/Output
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/patents.csv",
        help="Path to input patent CSV/TSV file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/patents_categorized_ensemble.csv",
        help="Path to save output CSV file with categories"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming"
    )

    # Column names
    parser.add_argument(
        "--category-column",
        type=str,
        default="ai_category",
        help="Base name for category columns (will add _ensemble suffix)"
    )
    parser.add_argument(
        "--title-column",
        type=str,
        default="application_title",
        help="Name of column containing patent title"
    )
    parser.add_argument(
        "--abstract-column",
        type=str,
        default="application_abstract",
        help="Name of column containing patent abstract"
    )

    # Model configuration
    parser.add_argument(
        "--models-config",
        type=str,
        default=None,
        help="Path to YAML file with custom model configurations (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run models on (cpu or cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding patents"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )

    # Filtering
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Only process patents from this year onwards"
    )

    args = parser.parse_args()
    main(args)
