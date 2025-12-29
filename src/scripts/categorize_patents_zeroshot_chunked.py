#!/usr/bin/env python
"""
Memory-efficient zero-shot patent categorization for large datasets.

This version processes patents in chunks to handle datasets with millions of records
without running out of memory.

Key features:
- Streams data in chunks (default 10,000 patents at a time)
- Appends results incrementally
- Supports resumption from partially completed runs
- Shows progress across chunks
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


class ZeroShotPatentClassifier:
    """Zero-shot classifier for patents using embedding models."""

    def __init__(self, model_config: Dict):
        """Initialize the classifier with embedding model configuration."""
        logger.info("Initializing embedder for zero-shot classification...")
        self.embedder = GenericEmbedder(model_config)
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

        for i in range(0, len(texts), batch_size):
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


def count_total_rows(file_path: Path, start_year: int = None) -> int:
    """Count total rows that will be processed (for progress bar)."""
    logger.info("Counting total patents to process...")

    total = 0
    for chunk in pd.read_csv(file_path, chunksize=10000):
        if start_year is not None:
            # Try to find year column
            year_col = None
            for col_name in ['year', 'Year', 'filing_year', 'application_year', 'grant_year']:
                if col_name in chunk.columns:
                    year_col = col_name
                    break

            if year_col:
                chunk = chunk[chunk[year_col] >= start_year]

        total += len(chunk)

    logger.info(f"Total patents to process: {total:,}")
    return total


def main(args):
    """Main function to run chunked zero-shot patent categorization."""

    # Setup paths
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        return

    # Initialize classifier
    logger.info("Initializing classifier...")
    model_config = {
        'embedding': {
            'model_name': args.model_name,
            'model_type': args.model_type,
            'device': args.device,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'normalize': True,
            'pooling': args.pooling
        }
    }

    classifier = ZeroShotPatentClassifier(model_config)

    # Count total rows for progress tracking
    total_patents = count_total_rows(input_path, args.start_year)

    # Check if output exists for resumption
    processed_count = 0
    if output_path.exists() and not args.overwrite:
        logger.info(f"Output file found. Checking existing progress...")
        existing_df = pd.read_csv(output_path)
        processed_count = len(existing_df)
        logger.info(f"Already processed: {processed_count:,} patents")

        if processed_count >= total_patents:
            logger.info("All patents already categorized. Exiting.")
            return

        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
        processed_count = 0

    # Process in chunks
    logger.info(f"Processing patents in chunks of {args.chunk_size:,}...")
    logger.info(f"Output file: {output_path}")

    chunk_num = 0
    patents_processed = processed_count

    with tqdm(total=total_patents, initial=processed_count, desc="Overall progress") as pbar:
        for chunk in pd.read_csv(input_path, chunksize=args.chunk_size):
            chunk_num += 1

            # Apply year filtering if requested
            if args.start_year is not None:
                # Try to find a year column
                year_col = None
                for col_name in ['year', 'Year', 'filing_year', 'application_year', 'grant_year']:
                    if col_name in chunk.columns:
                        year_col = col_name
                        break

                if year_col:
                    chunk = chunk[chunk[year_col] >= args.start_year].copy()

            if len(chunk) == 0:
                continue

            # Skip if we've already processed this chunk
            if patents_processed > 0:
                if len(chunk) <= patents_processed:
                    patents_processed -= len(chunk)
                    pbar.update(len(chunk))
                    continue
                else:
                    # Partial chunk processed, skip those rows
                    chunk = chunk.iloc[patents_processed:].copy()
                    patents_processed = 0

            # Combine title and abstract
            texts = chunk.apply(
                lambda row: combine_text_fields(row, args.title_column, args.abstract_column),
                axis=1
            ).tolist()

            # Classify patents in this chunk
            results = classifier.classify_batch(texts, batch_size=args.batch_size)

            # Add results to chunk
            chunk[args.category_column] = [r[0] for r in results]
            chunk[f'{args.category_column}_confidence'] = [r[1] for r in results]

            # Append to output file
            chunk.to_csv(output_path, mode=mode, header=header, index=False)

            # After first write, switch to append mode
            if mode == 'w':
                mode = 'a'
                header = False

            # Update progress
            pbar.update(len(chunk))

            # Log memory-friendly progress
            if chunk_num % 10 == 0:
                logger.info(f"Processed {chunk_num} chunks ({pbar.n:,} / {total_patents:,} patents)")

    logger.info("\n" + "="*60)
    logger.info("CATEGORIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total patents processed: {total_patents:,}")
    logger.info(f"Output saved to: {output_path}")

    # Print category distribution summary (read last chunk to get idea)
    logger.info("\nSample category distribution (last chunk):")
    category_counts = chunk[args.category_column].value_counts()
    for category, count in category_counts.items():
        pct = 100 * count / len(chunk)
        logger.info(f"  {category}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Categorize large patent datasets using zero-shot classification (memory-efficient chunked processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input patent CSV/TSV file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save output CSV file with categories"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of patents to process per chunk (default: 10000)"
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
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "auto-model"],
        help="Type of model"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "max"],
        help="Pooling strategy for auto-model type"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run model on (cpu, cuda, or mps for Apple Silicon)"
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

    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Progress has been saved and can be resumed.")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise
