#!/usr/bin/env python
"""
Zero-shot patent categorization using BERT embeddings.
This approach uses cosine similarity between patent text and category descriptions
to classify patents without requiring API calls or training data.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embedding.bert_embedder import BertEmbedder

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


class ZeroShotPatentClassifier:
    """Zero-shot classifier for patents using BERT embeddings."""

    def __init__(self, model_config: Dict):
        """Initialize the classifier with BERT model configuration."""
        logger.info("Initializing BERT embedder for zero-shot classification...")
        self.embedder = BertEmbedder(model_config)
        self.categories = list(CATEGORIES_WITH_DEFINITIONS.keys())
        self.category_embeddings = None
        self._encode_categories()

    def _encode_categories(self):
        """Pre-compute embeddings for all category descriptions."""
        logger.info(f"Encoding {len(self.categories)} category descriptions...")
        category_texts = list(CATEGORIES_WITH_DEFINITIONS.values())
        self.category_embeddings = self.embedder.encode_texts(
            category_texts,
            show_progress=False
        )
        logger.info("Category embeddings computed successfully")

    def classify_single(self, text: str) -> Tuple[str, float]:
        """
        Classify a single patent text using zero-shot classification.

        Args:
            text: Patent title and/or abstract

        Returns:
            Tuple of (category_name, confidence_score)
        """
        if not text or text.strip() == "":
            return "missing_data", 0.0

        # Encode the patent text
        patent_embedding = self.embedder.encode_texts([text], show_progress=False)[0]

        # Compute cosine similarity with all categories
        similarities = np.dot(self.category_embeddings, patent_embedding)

        # Get the category with highest similarity
        best_idx = np.argmax(similarities)
        best_category = self.categories[best_idx]
        confidence = float(similarities[best_idx])

        return best_category, confidence

    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Tuple[str, float]]:
        """
        Classify multiple patents in batch for efficiency.

        Args:
            texts: List of patent texts (title + abstract)
            batch_size: Number of patents to encode at once

        Returns:
            List of tuples (category_name, confidence_score)
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying patents"):
            batch_texts = texts[i:i+batch_size]

            # Filter out empty texts
            valid_indices = [j for j, t in enumerate(batch_texts) if t and t.strip() != ""]
            valid_texts = [batch_texts[j] for j in valid_indices]

            if not valid_texts:
                # All texts in batch are empty
                results.extend([("missing_data", 0.0)] * len(batch_texts))
                continue

            # Encode batch
            patent_embeddings = self.embedder.encode_texts(valid_texts, show_progress=False)

            # Compute similarities for all patents in batch
            # Shape: (num_patents, num_categories)
            similarities = np.dot(patent_embeddings, self.category_embeddings.T)

            # Get best category for each patent
            best_indices = np.argmax(similarities, axis=1)
            confidences = similarities[np.arange(len(similarities)), best_indices]

            # Map back to original batch with empty text handling
            batch_results = []
            valid_idx = 0
            for j in range(len(batch_texts)):
                if j in valid_indices:
                    category = self.categories[best_indices[valid_idx]]
                    confidence = float(confidences[valid_idx])
                    batch_results.append((category, confidence))
                    valid_idx += 1
                else:
                    batch_results.append(("missing_data", 0.0))

            results.extend(batch_results)

        return results


def combine_text_fields(row: pd.Series, title_col: str, abstract_col: str) -> str:
    """Combine title and abstract into single text for classification."""
    title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else ""
    abstract = str(row.get(abstract_col, "")) if pd.notna(row.get(abstract_col)) else ""

    # Combine with title given more weight
    combined = f"{title}. {abstract}".strip()
    return combined if combined != "." else ""


def main(args):
    """Main function to run zero-shot patent categorization."""

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

    # Initialize classifier
    model_config = {
        'embedding': {
            'model_name': args.model_name,
            'device': args.device,
            'batch_size': args.batch_size,
            'max_length': 512,
            'normalize': True
        }
    }

    classifier = ZeroShotPatentClassifier(model_config)

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

    # Create category column if needed
    if args.category_column not in df.columns:
        df[args.category_column] = None
        df[f'{args.category_column}_confidence'] = None

    # Find patents that need categorization
    to_process = df[args.category_column].isnull()
    num_to_process = to_process.sum()

    if num_to_process == 0:
        logger.info("All patents already categorized. Exiting.")
        return

    logger.info(f"Categorizing {num_to_process} patents...")

    # Combine title and abstract
    df_to_process = df[to_process].copy()
    texts = df_to_process.apply(
        lambda row: combine_text_fields(row, args.title_column, args.abstract_column),
        axis=1
    ).tolist()

    # Classify all patents
    results = classifier.classify_batch(texts, batch_size=args.batch_size)

    # Update dataframe
    categories = [r[0] for r in results]
    confidences = [r[1] for r in results]

    df.loc[to_process, args.category_column] = categories
    df.loc[to_process, f'{args.category_column}_confidence'] = confidences

    # Save results
    logger.info(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)

    # Print statistics
    logger.info("\n" + "="*50)
    logger.info("CATEGORIZATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Total patents: {len(df)}")
    logger.info(f"Newly categorized: {num_to_process}")
    logger.info("\nCategory distribution:")
    category_counts = df[args.category_column].value_counts()
    for cat, count in category_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    logger.info(f"\nAverage confidence: {df[f'{args.category_column}_confidence'].mean():.3f}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot patent categorization using BERT embeddings."
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
        default="data/processed/patents_categorized_zeroshot.csv",
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
        help="Name of column to store AI category"
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
        "--model-name",
        type=str,
        default="anferico/bert-for-patents",
        help="BERT model to use for embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on (cpu or cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding patents"
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
