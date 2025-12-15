#!/usr/bin/env python
"""
Category-specific patent-occupation matching analysis.

This script performs patent-occupation matching separately for each of the 8 AI
categories, allowing analysis of how different AI technologies impact different
types of work.

Similar to the approach where matching is run 8 times (once per category) rather
than once for all AI patents combined.
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embedding.generic_embedder import GenericEmbedder
from src.data.onet_loader import ONetLoader
from src.data.data_preprocessor import DataPreprocessor

# Try to import ScaNN, fall back to simple matcher
try:
    from src.matching.scann_matcher import ScannMatcher
    MATCHER_CLASS = ScannMatcher
    logger = logging.getLogger(__name__)
    logger.info("Using ScaNN matcher")
except (ImportError, Exception) as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"ScaNN not available ({e}), using simple matcher")
    from src.matching.simple_matcher import SimpleMatcher
    MATCHER_CLASS = SimpleMatcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI Categories
AI_CATEGORIES = [
    "computer vision",
    "evolutionary computation",
    "AI hardware",
    "knowledge processing",
    "machine learning",
    "NLP",
    "planning and control",
    "speech recognition"
]


class CategorySpecificMatcher:
    """Performs patent-occupation matching for a specific AI category."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize matcher with configuration."""
        self.config = config
        self.embedder = GenericEmbedder(config)
        self.matcher = MATCHER_CLASS(config)
        self.preprocessor = DataPreprocessor()

    def match_category_by_year(
        self,
        patents_df: pd.DataFrame,
        onet_df: pd.DataFrame,
        category: str,
        years: List[int],
        text_column: str = 'title_abstract',
        threshold: float = 0.75
    ) -> pd.DataFrame:
        """
        Match patents to O*NET tasks for a specific category, year by year.

        Args:
            patents_df: DataFrame with categorized patents
            onet_df: DataFrame with O*NET tasks
            category: AI category to filter
            years: List of years to process
            text_column: Column containing patent text
            threshold: Cosine similarity threshold for counting matches

        Returns:
            DataFrame with O*NET tasks and match counts per year
        """
        logger.info(f"Processing category: {category}")

        # Filter patents to this category
        category_patents = patents_df[patents_df[self.config['category_column']] == category].copy()

        if len(category_patents) == 0:
            logger.warning(f"No patents found for category '{category}'. Skipping.")
            return None

        logger.info(f"Found {len(category_patents)} patents for '{category}'")

        # Initialize results dataframe with required columns
        columns_to_keep = ['O*NET-SOC Code', 'Title', 'Task', 'Scale Name', 'Category', 'Data Value']
        result_df = onet_df[columns_to_keep].copy()

        # Get unique tasks and encode once
        tasks = onet_df['Task'].unique().tolist()
        logger.info(f"Encoding {len(tasks)} O*NET tasks...")
        task_embeddings = self.embedder.encode_texts(tasks, show_progress=True)

        # Create task to index mapping
        task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        task_indices = onet_df['Task'].map(task_to_idx).values

        # Process each year
        for year in years:
            logger.info(f"Processing year {year} for category '{category}'...")

            # Filter patents by year
            year_patents = category_patents[category_patents['year'] == year].copy()

            if len(year_patents) == 0:
                logger.warning(f"No patents found for year {year} in category '{category}'")
                result_df[str(year)] = 0
                continue

            logger.info(f"  Found {len(year_patents)} patents for year {year}")

            # Get patent text
            if text_column not in year_patents.columns:
                # Try to create title_abstract if not exists
                if 'application_title' in year_patents.columns and 'application_abstract' in year_patents.columns:
                    year_patents[text_column] = (
                        year_patents['application_title'].fillna('') + '. ' +
                        year_patents['application_abstract'].fillna('')
                    ).str.strip()
                else:
                    logger.error(f"Text column '{text_column}' not found and cannot be created")
                    result_df[str(year)] = 0
                    continue

            patent_texts = year_patents[text_column].fillna('').tolist()

            # Encode patents
            logger.info(f"  Encoding {len(patent_texts)} patents...")
            patent_embeddings = self.embedder.encode_texts(patent_texts, show_progress=True)

            # Build index and search
            logger.info(f"  Building search index...")
            self.matcher.build_index(patent_embeddings)

            # Search: for each task, find top-k most similar patents
            logger.info(f"  Searching for matches...")
            neighbors, distances = self.matcher.search_batch(task_embeddings)

            # Count matches above threshold for each task
            counts = []
            for task_idx, task_row_idx in enumerate(task_indices):
                # Get distances for this task (using the task's embedding index)
                task_distances = distances[task_row_idx]

                # Count how many are above threshold
                count = np.sum(task_distances >= threshold)
                counts.append(count)

            result_df[str(year)] = counts

        return result_df


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main function to run category-specific analysis."""

    start_time = time.time()

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'embedding': {
                'model_name': args.model_name,
                'model_type': args.model_type,
                'device': args.device,
                'batch_size': args.batch_size,
                'max_length': 512,
                'normalize': True,
                'pooling': args.pooling
            },
            'matching': {
                'num_leaves': 200,
                'num_leaves_to_search': 100,
                'training_sample_size': 250000,
                'reorder_num_neighbors': args.top_k,
                'pre_reorder_num_neighbors': 250,
                'anisotropic_quantization_threshold': 0.2
            },
            'category_column': args.category_column
        }

    # Load categorized patents
    logger.info(f"Loading categorized patents from {args.patents_file}...")
    patents_df = pd.read_csv(args.patents_file)
    logger.info(f"Loaded {len(patents_df)} patents")

    # Check for required columns
    if args.category_column not in patents_df.columns:
        logger.error(f"Category column '{args.category_column}' not found in patents file")
        logger.error(f"Available columns: {list(patents_df.columns)}")
        return

    if 'year' not in patents_df.columns:
        logger.error("'year' column not found in patents file")
        return

    # Determine which categories to process
    categories_to_process = args.categories if args.categories else AI_CATEGORIES

    logger.info(f"Categories to process: {categories_to_process}")

    # Check which categories have patents
    category_counts = patents_df[args.category_column].value_counts()
    logger.info("\nPatent counts by category:")
    for cat in categories_to_process:
        count = category_counts.get(cat, 0)
        logger.info(f"  {cat}: {count} patents")

    # Load O*NET tasks
    logger.info(f"\nLoading O*NET tasks from {args.onet_file}...")

    # Try different file formats
    onet_path = Path(args.onet_file)
    if onet_path.suffix == '.xlsx':
        onet_df = pd.read_excel(args.onet_file)
    elif onet_path.suffix == '.csv':
        onet_df = pd.read_csv(args.onet_file)
    else:
        logger.error(f"Unsupported O*NET file format: {onet_path.suffix}")
        return

    logger.info(f"Loaded {len(onet_df)} O*NET task records")

    # Check for required columns
    required_cols = ['O*NET-SOC Code', 'Title', 'Task', 'Scale Name', 'Category', 'Data Value']
    missing_cols = [col for col in required_cols if col not in onet_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in O*NET file: {missing_cols}")
        logger.error(f"Available columns: {list(onet_df.columns)}")
        logger.error("The O*NET Task Ratings file must include 'Scale Name', 'Category', and 'Data Value' columns.")
        return

    # Get years to process
    years = sorted(patents_df['year'].dropna().unique())
    if args.years:
        years = [y for y in args.years if y in years]

    logger.info(f"Years to process: {years}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize matcher
    logger.info("\nInitializing matcher...")
    matcher = CategorySpecificMatcher(config)

    # Process each category
    for i, category in enumerate(categories_to_process, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category {i}/{len(categories_to_process)}: {category}")
        logger.info(f"{'='*60}")

        # Run matching for this category
        result_df = matcher.match_category_by_year(
            patents_df=patents_df,
            onet_df=onet_df,
            category=category,
            years=years,
            text_column=args.text_column,
            threshold=args.threshold
        )

        if result_df is None:
            continue

        # Save results
        category_slug = category.replace(' ', '_').replace('*', '').lower()
        output_file = output_dir / f"AI_{min(years)}to{max(years)}_abstractTitleFor_{category_slug}_{args.threshold}.csv"

        logger.info(f"Saving results to {output_file}...")
        result_df.to_csv(output_file, index=False)

        # Print summary statistics
        year_cols = [str(y) for y in years]
        logger.info(f"\nSummary for '{category}':")
        logger.info(f"  Total tasks: {len(result_df)}")
        for year in years:
            total_matches = result_df[str(year)].sum()
            tasks_with_matches = (result_df[str(year)] > 0).sum()
            avg_matches = result_df[str(year)].mean()
            logger.info(f"  Year {year}: {total_matches} total matches, "
                       f"{tasks_with_matches} tasks with matches, "
                       f"avg {avg_matches:.1f} matches/task")

    # Final summary
    elapsed_time = (time.time() - start_time) / 60
    logger.info("\n" + "="*60)
    logger.info("CATEGORY-SPECIFIC ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Processed {len(categories_to_process)} categories")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total time: {elapsed_time:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Category-specific patent-occupation matching analysis. "
                    "Runs matching separately for each AI category to understand "
                    "differential impact of different AI technologies on tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 8 AI categories
  python categorize_by_ai_category.py \\
      --patents-file data/processed/patents_ensemble.csv \\
      --onet-file data/raw/onet/Task_Ratings.xlsx \\
      --output-dir results/category_specific

  # Process specific categories only
  python categorize_by_ai_category.py \\
      --patents-file data/processed/patents_ensemble.csv \\
      --onet-file data/raw/onet/Task_Ratings.xlsx \\
      --categories "machine learning" "NLP" "computer vision" \\
      --output-dir results/ml_nlp_cv

  # Custom model and threshold
  python categorize_by_ai_category.py \\
      --patents-file data/processed/patents_ensemble.csv \\
      --onet-file data/raw/onet/Task_Ratings.xlsx \\
      --model-name google/embeddinggemma-300m \\
      --model-type auto-model \\
      --threshold 0.8 \\
      --output-dir results/gemma_0.8
        """
    )

    # Input files
    parser.add_argument(
        "--patents-file",
        type=str,
        required=True,
        help="Path to categorized patents CSV file"
    )
    parser.add_argument(
        "--onet-file",
        type=str,
        required=True,
        help="Path to O*NET tasks file (Excel or CSV)"
    )

    # Category configuration
    parser.add_argument(
        "--categories",
        type=str,
        nargs='+',
        help="Specific categories to process (default: all 8 categories)"
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default="ensemble_category",
        help="Name of category column in patents file (default: ensemble_category)"
    )

    # Year configuration
    parser.add_argument(
        "--years",
        type=int,
        nargs='+',
        help="Specific years to process (default: all years in patents file)"
    )

    # Text column
    parser.add_argument(
        "--text-column",
        type=str,
        default="title_abstract",
        help="Column containing patent text (default: title_abstract)"
    )

    # Matching parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for counting matches (default: 0.75)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Number of top matches to retrieve per task (default: 500)"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (optional)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="anferico/bert-for-patents",
        help="Embedding model name (default: anferico/bert-for-patents)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "auto-model"],
        help="Model type (default: sentence-transformers)"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "max"],
        help="Pooling strategy for auto-model (default: mean)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save category-specific results"
    )

    args = parser.parse_args()
    main(args)
