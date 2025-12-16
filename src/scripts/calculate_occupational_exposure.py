#!/usr/bin/env python
"""
Calculate occupational exposure to AI technologies.

This script processes category-specific patent-occupation matching results and
calculates exposure scores for each occupation. Exposure measures how much an
occupation's important tasks are matched by AI patents, weighted by task importance.

Formula:
    exposure = Σ (patent matches for task / total patent matches) × task importance

Only tasks with Scale Name = "Importance" are included in the calculation.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI Categories (for reference)
AI_CATEGORIES = [
    "computer_vision",
    "evolutionary_computation",
    "ai_hardware",
    "knowledge_processing",
    "machine_learning",
    "nlp",
    "planning_and_control",
    "speech_recognition"
]


def extract_category_from_filename(filepath: Path) -> str:
    """
    Extract AI category from filename.

    Example: AI_2021to2025_abstractTitleFor_machine_learning_0.75.csv -> machine_learning
    """
    filename = filepath.stem  # Remove .csv extension
    parts = filename.split('_')

    # Find the threshold part (e.g., "0.75")
    for i, part in enumerate(parts):
        if part.replace('.', '').isdigit():
            # Everything before this is the category
            category_parts = parts[3:i]  # Skip "AI", year range, "abstractTitleFor"
            category = '_'.join(category_parts)
            return category

    # Fallback: assume everything after "abstractTitleFor" is the category
    if 'abstractTitleFor' in filename:
        idx = filename.index('abstractTitleFor')
        category = filename[idx + len('abstractTitleFor') + 1:]
        # Remove trailing threshold if present
        category = category.rsplit('_', 1)[0]
        return category

    return "unknown"


def calculate_exposure_for_category(
    df: pd.DataFrame,
    category: str,
    year_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate occupational exposure for a specific AI category.

    Args:
        df: DataFrame with task-level matching results (already filtered for Importance)
        category: AI category name
        year_columns: List of year column names (e.g., ['2021', '2022', '2023'])

    Returns:
        DataFrame with columns: O*NET-SOC Code, Title, AI_Category, year columns
    """
    logger.info(f"Calculating exposure for category: {category}")

    # Group by occupation
    occupations = df.groupby(['O*NET-SOC Code', 'Title'])

    results = []

    for (soc_code, title), occ_df in occupations:
        exposure_row = {
            'O*NET-SOC Code': soc_code,
            'Title': title,
            'AI_Category': category
        }

        # Calculate exposure for each year
        for year in year_columns:
            # Get task matches and importance scores
            task_matches = occ_df[year].values
            task_importance = occ_df['Data Value'].values

            # Total matches for this occupation in this year
            total_matches = task_matches.sum()

            if total_matches == 0:
                # No matches -> exposure = 0
                exposure_row[year] = 0.0
            else:
                # Calculate weighted exposure
                # exposure = Σ (task_matches / total_matches) × task_importance
                weights = task_matches / total_matches
                contributions = weights * task_importance
                exposure = contributions.sum()
                exposure_row[year] = round(exposure, 4)

        results.append(exposure_row)

    result_df = pd.DataFrame(results)
    logger.info(f"  Calculated exposure for {len(result_df)} occupations")

    return result_df


def process_category_file(filepath: Path, year_columns: List[str]) -> pd.DataFrame:
    """
    Process a single category-specific CSV file.

    Args:
        filepath: Path to category-specific CSV file
        year_columns: List of year column names

    Returns:
        DataFrame with exposure scores for this category
    """
    logger.info(f"\nProcessing file: {filepath.name}")

    # Extract category from filename
    category = extract_category_from_filename(filepath)
    logger.info(f"  Detected category: {category}")

    # Read CSV
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded {len(df)} task records")

    # Check for required columns
    required_cols = ['O*NET-SOC Code', 'Title', 'Task', 'Scale Name', 'Data Value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"  Missing required columns: {missing_cols}")
        return None

    # Filter for Importance only
    importance_df = df[df['Scale Name'] == 'Importance'].copy()
    logger.info(f"  Filtered to {len(importance_df)} importance tasks "
                f"({len(importance_df)/len(df)*100:.1f}% of total)")

    if len(importance_df) == 0:
        logger.warning(f"  No importance tasks found in {filepath.name}, skipping...")
        return None

    # Verify year columns exist
    missing_years = [year for year in year_columns if year not in importance_df.columns]
    if missing_years:
        logger.warning(f"  Missing year columns: {missing_years}")
        # Use only available years
        year_columns = [year for year in year_columns if year in importance_df.columns]

    # Calculate exposure
    exposure_df = calculate_exposure_for_category(importance_df, category, year_columns)

    return exposure_df


def main(args):
    """Main function to calculate occupational exposure."""

    start_time = time.time()

    # Find all category-specific CSV files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Get all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))

    if args.pattern:
        # Filter by pattern
        csv_files = [f for f in csv_files if args.pattern in f.name]

    if len(csv_files) == 0:
        logger.error(f"No CSV files found in {input_dir}")
        if args.pattern:
            logger.error(f"  Pattern filter: {args.pattern}")
        return

    logger.info(f"Found {len(csv_files)} category-specific files to process")
    for f in csv_files:
        logger.info(f"  - {f.name}")

    # Determine year columns from first file
    sample_df = pd.read_csv(csv_files[0])
    year_columns = [col for col in sample_df.columns if col.isdigit()]
    year_columns = sorted(year_columns)

    if len(year_columns) == 0:
        logger.error("No year columns found in input files")
        return

    logger.info(f"\nYear columns detected: {year_columns}")

    # Process each category file
    all_exposure_dfs = []

    for filepath in tqdm(csv_files, desc="Processing categories"):
        exposure_df = process_category_file(filepath, year_columns)
        if exposure_df is not None:
            all_exposure_dfs.append(exposure_df)

    if len(all_exposure_dfs) == 0:
        logger.error("No exposure data calculated from any files")
        return

    # Combine all category results
    logger.info("\nCombining results from all categories...")
    combined_df = pd.concat(all_exposure_dfs, ignore_index=True)

    # Sort by occupation and category
    combined_df = combined_df.sort_values(['O*NET-SOC Code', 'AI_Category'])

    logger.info(f"Total records: {len(combined_df)}")
    logger.info(f"Unique occupations: {combined_df['O*NET-SOC Code'].nunique()}")
    logger.info(f"AI categories: {combined_df['AI_Category'].nunique()}")

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved exposure scores to: {output_path}")

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("EXPOSURE CALCULATION SUMMARY")
    logger.info("="*60)

    logger.info(f"\nExposure by AI Category (average across all occupations and years):")
    for category in combined_df['AI_Category'].unique():
        category_df = combined_df[combined_df['AI_Category'] == category]
        avg_exposure = category_df[year_columns].mean().mean()
        logger.info(f"  {category}: {avg_exposure:.4f}")

    logger.info(f"\nTop 10 most exposed occupations (average across all categories and years):")
    # Calculate average exposure per occupation
    occ_exposure = combined_df.groupby(['O*NET-SOC Code', 'Title'])[year_columns].mean().mean(axis=1)
    top_occupations = occ_exposure.nlargest(10)
    for (soc_code, title), exposure in top_occupations.items():
        logger.info(f"  {title} ({soc_code}): {exposure:.4f}")

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"\nTotal processing time: {elapsed_time:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate occupational exposure to AI technologies. "
                    "Processes category-specific patent-occupation matching results and "
                    "calculates weighted exposure scores based on task importance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all category files in a directory
  python calculate_occupational_exposure.py \\
      --input-dir results/category_specific \\
      --output-file results/occupational_exposure.csv

  # Process only specific pattern of files
  python calculate_occupational_exposure.py \\
      --input-dir results/category_specific \\
      --pattern "0.75" \\
      --output-file results/occupational_exposure_075.csv

Output format:
  O*NET-SOC Code,Title,AI_Category,2021,2022,2023,2024,2025
  15-2051.00,Data Scientist,machine_learning,5.0,5.2,5.4,5.5,5.6
  15-2051.00,Data Scientist,nlp,4.8,4.9,5.0,5.1,5.2
  ...

The exposure score is calculated as:
  exposure = Σ (patent matches for task / total patent matches) × task importance

where the sum is over all importance-rated tasks for each occupation.
        """
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing category-specific CSV files from categorize_by_ai_category.py"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save combined occupational exposure scores"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help="Filter input files by pattern (e.g., '0.75' to only process files with threshold 0.75)"
    )

    args = parser.parse_args()
    main(args)
