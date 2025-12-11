#!/usr/bin/env python
"""
Aggregate ensemble voting from pre-computed patent categorization results.

This script takes multiple CSV files (each containing results from running
categorize_patents_zeroshot.py with different models) and performs ensemble
voting to produce a final classification.

This is useful for:
- Running models on different machines/times
- Experimenting with different model combinations
- Avoiding loading all models at once
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensemble_vote(
    predictions: List[Tuple[str, float]]
) -> Tuple[str, float, int]:
    """
    Perform ensemble voting on predictions from multiple models.

    Args:
        predictions: List of (category, confidence) tuples from each model

    Returns:
        Tuple of (final_category, ensemble_confidence, vote_count)
    """
    # Extract categories
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


def load_and_validate_files(
    input_files: List[str],
    model_names: List[str],
    id_column: str,
    category_column: str,
    confidence_column: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load all input files and validate they have consistent IDs.

    Args:
        input_files: List of CSV file paths
        model_names: List of model names (for column prefixes)
        id_column: Name of the ID column
        category_column: Name of the category column in each file
        confidence_column: Name of the confidence column in each file

    Returns:
        Tuple of (merged_dataframe, actual_model_names)
    """
    if len(input_files) < 2:
        raise ValueError("Need at least 2 input files for ensemble voting")

    logger.info(f"Loading {len(input_files)} input files...")

    # Load first file as base
    logger.info(f"Loading base file: {input_files[0]}")
    base_df = pd.read_csv(input_files[0])

    if id_column not in base_df.columns:
        raise ValueError(f"ID column '{id_column}' not found in {input_files[0]}. "
                        f"Available columns: {list(base_df.columns)}")

    # Keep all original columns from base file
    result_df = base_df.copy()

    # Rename category and confidence columns for first model
    actual_model_names = []
    if model_names and len(model_names) > 0:
        model_name = model_names[0]
    else:
        model_name = f"model_{1}"

    actual_model_names.append(model_name)

    if category_column in result_df.columns:
        result_df.rename(columns={
            category_column: f"{model_name}_category",
            confidence_column: f"{model_name}_confidence"
        }, inplace=True)
    else:
        # Try common variations
        for cat_col in [category_column, 'ai_category', 'category', 'predicted_category']:
            if cat_col in result_df.columns:
                result_df.rename(columns={
                    cat_col: f"{model_name}_category"
                }, inplace=True)
                category_column = cat_col
                break

        for conf_col in [confidence_column, 'ai_category_confidence', 'confidence']:
            if conf_col in result_df.columns:
                result_df.rename(columns={
                    conf_col: f"{model_name}_confidence"
                }, inplace=True)
                break

    # Load and merge remaining files
    for i, file_path in enumerate(input_files[1:], start=2):
        logger.info(f"Loading file {i}/{len(input_files)}: {file_path}")

        df = pd.read_csv(file_path)

        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in {file_path}")

        # Determine model name
        if model_names and len(model_names) >= i:
            model_name = model_names[i-1]
        else:
            model_name = f"model_{i}"

        actual_model_names.append(model_name)

        # Select only ID, category, and confidence columns
        cols_to_keep = [id_column]

        # Find category column
        cat_col_found = False
        for cat_col in [category_column, 'ai_category', 'category', 'predicted_category']:
            if cat_col in df.columns:
                cols_to_keep.append(cat_col)
                cat_col_found = True
                break

        if not cat_col_found:
            raise ValueError(f"Category column not found in {file_path}. "
                           f"Available columns: {list(df.columns)}")

        # Find confidence column
        for conf_col in [confidence_column, 'ai_category_confidence', 'confidence']:
            if conf_col in df.columns:
                cols_to_keep.append(conf_col)
                break

        df_subset = df[cols_to_keep].copy()

        # Rename columns
        rename_map = {id_column: id_column}
        if len(cols_to_keep) >= 2:
            rename_map[cols_to_keep[1]] = f"{model_name}_category"
        if len(cols_to_keep) >= 3:
            rename_map[cols_to_keep[2]] = f"{model_name}_confidence"

        df_subset.rename(columns=rename_map, inplace=True)

        # Merge with result
        result_df = result_df.merge(df_subset, on=id_column, how='inner')

        logger.info(f"  Merged: {len(result_df)} patents remaining")

    logger.info(f"Successfully loaded and merged {len(input_files)} files")
    logger.info(f"Final dataset: {len(result_df)} patents")

    return result_df, actual_model_names


def aggregate_ensemble(
    df: pd.DataFrame,
    model_names: List[str],
    output_prefix: str = 'ensemble'
) -> pd.DataFrame:
    """
    Perform ensemble voting aggregation on the dataframe.

    Args:
        df: DataFrame with model predictions
        model_names: List of model names
        output_prefix: Prefix for output columns

    Returns:
        DataFrame with ensemble results added
    """
    logger.info(f"Performing ensemble voting on {len(df)} patents...")

    ensemble_categories = []
    ensemble_confidences = []
    ensemble_votes = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ensemble voting"):
        # Gather predictions from all models
        predictions = []
        for model_name in model_names:
            cat_col = f"{model_name}_category"
            conf_col = f"{model_name}_confidence"

            if cat_col in df.columns and conf_col in df.columns:
                category = row[cat_col]
                confidence = row[conf_col]

                # Skip if missing
                if pd.notna(category) and pd.notna(confidence):
                    predictions.append((str(category), float(confidence)))

        if len(predictions) == 0:
            # No valid predictions
            ensemble_categories.append(None)
            ensemble_confidences.append(None)
            ensemble_votes.append(0)
        else:
            # Perform voting
            ens_cat, ens_conf, ens_votes = ensemble_vote(predictions)
            ensemble_categories.append(ens_cat)
            ensemble_confidences.append(ens_conf)
            ensemble_votes.append(ens_votes)

    # Add ensemble results to dataframe
    df[f'{output_prefix}_category'] = ensemble_categories
    df[f'{output_prefix}_confidence'] = ensemble_confidences
    df[f'{output_prefix}_votes'] = ensemble_votes

    return df


def main(args):
    """Main function to aggregate ensemble voting results."""

    # Get input files
    input_files = []

    if args.input_files:
        input_files.extend(args.input_files)

    if args.file_list:
        # Read file paths from a text file
        with open(args.file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    input_files.append(line)

    if len(input_files) < 2:
        logger.error("Need at least 2 input files for ensemble voting")
        logger.error("Provide files via --input-files or --file-list")
        return

    logger.info(f"Aggregating results from {len(input_files)} files")

    # Get model names
    model_names = args.model_names if args.model_names else None

    # Load and merge all files
    try:
        merged_df, actual_model_names = load_and_validate_files(
            input_files=input_files,
            model_names=model_names,
            id_column=args.id_column,
            category_column=args.category_column,
            confidence_column=args.confidence_column
        )
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return

    logger.info(f"Using models: {actual_model_names}")

    # Perform ensemble aggregation
    result_df = aggregate_ensemble(
        df=merged_df,
        model_names=actual_model_names,
        output_prefix=args.output_prefix
    )

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_path}...")
    result_df.to_csv(output_path, index=False)

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE AGGREGATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total patents: {len(result_df)}")
    logger.info(f"Number of models: {len(actual_model_names)}")

    logger.info(f"\n{args.output_prefix.title()} category distribution:")
    category_counts = result_df[f'{args.output_prefix}_category'].value_counts()
    for cat, count in category_counts.items():
        pct = (count / len(result_df)) * 100
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    logger.info(f"\nAverage {args.output_prefix} confidence: "
               f"{result_df[f'{args.output_prefix}_confidence'].mean():.3f}")

    # Vote distribution
    logger.info(f"\nVote count distribution:")
    vote_counts = result_df[f'{args.output_prefix}_votes'].value_counts().sort_index()
    for votes, count in vote_counts.items():
        pct = (count / len(result_df)) * 100
        logger.info(f"  {int(votes)} votes: {count} patents ({pct:.1f}%)")

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate ensemble voting from pre-computed patent categorization results. "
                    "Takes multiple CSV files with model predictions and performs majority voting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 5 files
  python aggregate_ensemble_voting.py \\
      --input-files results_bert.csv results_gemma.csv results_mpnet.csv \\
                    results_minilm.csv results_scibert.csv \\
      --output-file results_ensemble.csv

  # With custom model names
  python aggregate_ensemble_voting.py \\
      --input-files results_*.csv \\
      --model-names bert gemma mpnet minilm scibert \\
      --output-file results_ensemble.csv

  # Using file list
  python aggregate_ensemble_voting.py \\
      --file-list model_results.txt \\
      --output-file results_ensemble.csv

  # Where model_results.txt contains:
  #   results_bert.csv
  #   results_gemma.csv
  #   results_mpnet.csv
  #   results_minilm.csv
  #   results_scibert.csv
        """
    )

    # Input files
    parser.add_argument(
        "--input-files",
        type=str,
        nargs='+',
        help="List of input CSV files with model predictions (space-separated)"
    )
    parser.add_argument(
        "--file-list",
        type=str,
        help="Path to text file containing list of input CSV files (one per line)"
    )

    # Model configuration
    parser.add_argument(
        "--model-names",
        type=str,
        nargs='+',
        help="Names for each model (space-separated, same order as input files)"
    )

    # Column configuration
    parser.add_argument(
        "--id-column",
        type=str,
        default="application_id",
        help="Name of the ID column for merging files (default: application_id)"
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default="ai_category",
        help="Name of the category column in input files (default: ai_category)"
    )
    parser.add_argument(
        "--confidence-column",
        type=str,
        default="ai_category_confidence",
        help="Name of the confidence column in input files (default: ai_category_confidence)"
    )

    # Output configuration
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save output CSV file with ensemble results"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ensemble",
        help="Prefix for ensemble output columns (default: ensemble)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input_files and not args.file_list:
        parser.error("Must provide either --input-files or --file-list")

    main(args)
