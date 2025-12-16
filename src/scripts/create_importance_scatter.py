#!/usr/bin/env python
"""
Create scatterplot visualization: Task Importance vs AI Match Count

This script creates scatterplots showing the relationship between task importance
(O*NET ratings) and AI patent matches, revealing whether AI is targeting critical
vs routine tasks.

Key insights:
- Top-right quadrant: High importance + many matches (CONCERN - automation of critical tasks)
- Bottom-right: Low importance + many matches (routine task automation)
- Top-left: High importance + few matches (currently safe tasks)
- Bottom-left: Low importance + few matches (minimal AI impact)
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI category colors (consistent with Sankey)
CATEGORY_COLORS = {
    'machine_learning': '#1f77b4',
    'nlp': '#ff7f0e',
    'computer_vision': '#2ca02c',
    'speech_recognition': '#d62728',
    'knowledge_processing': '#9467bd',
    'planning_and_control': '#8c564b',
    'ai_hardware': '#e377c2',
    'evolutionary_computation': '#7f7f7f'
}


def extract_category_from_filename(filepath: Path) -> str:
    """Extract AI category from filename."""
    filename = filepath.stem
    parts = filename.split('_')

    for i, part in enumerate(parts):
        if part.replace('.', '').replace('-', '').isdigit():
            category_parts = parts[3:i]
            category = '_'.join(category_parts)
            return category

    if 'abstractTitleFor' in filename:
        idx = filename.index('abstractTitleFor')
        category = filename[idx + len('abstractTitleFor') + 1:]
        category = category.rsplit('_', 1)[0]
        return category

    return "unknown"


def load_all_category_data(input_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and combine all category-specific CSV files.

    Returns:
        (combined_df, year_columns)
    """
    csv_files = sorted(input_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    logger.info(f"Found {len(csv_files)} category files")

    all_data = []
    year_columns = None

    for filepath in csv_files:
        category = extract_category_from_filename(filepath)
        df = pd.read_csv(filepath)

        # Filter for importance tasks only
        df = df[df['Scale Name'] == 'Importance'].copy()

        # Detect year columns
        if year_columns is None:
            year_columns = []
            for col in df.columns:
                try:
                    year_val = float(col)
                    if 1900 <= year_val <= 2100:
                        year_columns.append(col)
                except (ValueError, TypeError):
                    continue
            year_columns = sorted(year_columns, key=lambda x: float(x))

        # Add category column
        df['AI_Category'] = category

        all_data.append(df)
        logger.info(f"  Loaded {category}: {len(df)} importance tasks")

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nCombined: {len(combined_df)} total task-category pairs")
    logger.info(f"Years detected: {year_columns}")

    return combined_df, year_columns


def create_scatterplot_for_year(
    df: pd.DataFrame,
    year: str,
    show_categories: bool = True,
    top_n_labels: int = 10
) -> go.Figure:
    """
    Create scatterplot for a specific year.

    Args:
        df: DataFrame with task data
        year: Year column to use
        show_categories: Color by AI category
        top_n_labels: Number of outliers to label

    Returns:
        Plotly figure
    """
    # Prepare data
    plot_df = df[['Task', 'Data Value', year, 'AI_Category', 'Title']].copy()
    plot_df = plot_df.rename(columns={
        'Data Value': 'importance',
        year: 'matches',
        'Title': 'occupation'
    })

    # Remove tasks with zero matches (would clutter bottom)
    if not show_categories:
        plot_df = plot_df[plot_df['matches'] > 0]

    # Aggregate by task (sum matches across all categories)
    if not show_categories:
        plot_df = plot_df.groupby(['Task', 'importance']).agg({
            'matches': 'sum',
            'occupation': 'first'
        }).reset_index()

    # Create figure
    if show_categories:
        fig = px.scatter(
            plot_df,
            x='matches',
            y='importance',
            color='AI_Category',
            color_discrete_map=CATEGORY_COLORS,
            hover_data={
                'Task': True,
                'occupation': True,
                'AI_Category': True,
                'matches': ':.0f',
                'importance': ':.2f'
            },
            labels={
                'matches': 'AI Patent Matches',
                'importance': 'Task Importance (1-5 scale)',
                'AI_Category': 'AI Category'
            },
            opacity=0.6
        )
    else:
        fig = px.scatter(
            plot_df,
            x='matches',
            y='importance',
            hover_data={
                'Task': True,
                'occupation': True,
                'matches': ':.0f',
                'importance': ':.2f'
            },
            labels={
                'matches': 'AI Patent Matches',
                'importance': 'Task Importance (1-5 scale)'
            },
            opacity=0.6,
            color_discrete_sequence=['#4472C4']
        )

    # Add quadrant lines
    max_matches = plot_df['matches'].max()

    # Vertical line at median matches (separates high/low AI impact)
    median_matches = plot_df[plot_df['matches'] > 0]['matches'].median()
    fig.add_vline(
        x=median_matches,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text=f"Median matches ({median_matches:.0f})",
        annotation_position="top"
    )

    # Horizontal line at importance 3.5 (separates high/low importance)
    fig.add_hline(
        y=3.5,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="High importance threshold (3.5)",
        annotation_position="right"
    )

    # Add quadrant labels
    fig.add_annotation(
        x=max_matches * 0.85,
        y=4.7,
        text="<b>High Concern</b><br>Important tasks<br>heavily automated",
        showarrow=False,
        font=dict(size=10, color='red'),
        bgcolor='rgba(255, 200, 200, 0.7)',
        bordercolor='red',
        borderwidth=1
    )

    fig.add_annotation(
        x=max_matches * 0.85,
        y=1.5,
        text="<b>Routine Automation</b><br>Low importance tasks<br>being automated",
        showarrow=False,
        font=dict(size=10, color='orange'),
        bgcolor='rgba(255, 230, 200, 0.7)',
        bordercolor='orange',
        borderwidth=1
    )

    fig.add_annotation(
        x=median_matches * 0.3,
        y=4.7,
        text="<b>Currently Protected</b><br>Important tasks<br>little AI impact",
        showarrow=False,
        font=dict(size=10, color='green'),
        bgcolor='rgba(200, 255, 200, 0.7)',
        bordercolor='green',
        borderwidth=1
    )

    # Label interesting outliers (top-right quadrant)
    high_importance_high_matches = plot_df[
        (plot_df['importance'] >= 4.0) &
        (plot_df['matches'] >= median_matches * 1.5)
    ].nlargest(top_n_labels, 'matches')

    for _, row in high_importance_high_matches.iterrows():
        task_label = row['Task'][:40] + "..." if len(row['Task']) > 40 else row['Task']
        fig.add_annotation(
            x=row['matches'],
            y=row['importance'],
            text=task_label,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='rgba(100,100,100,0.5)',
            ax=30,
            ay=-30,
            font=dict(size=8),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Task Importance vs AI Patent Matches<br><sub>Year: {year}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis_title="Number of AI Patent Matches",
        yaxis_title="Task Importance (O*NET 1-5 Scale)",
        height=700,
        width=1000,
        showlegend=show_categories,
        hovermode='closest'
    )

    # Set axis ranges
    fig.update_xaxes(range=[0, max_matches * 1.05])
    fig.update_yaxes(range=[0.8, 5.2])

    return fig


def create_animated_scatterplot(
    df: pd.DataFrame,
    years: List[str],
    show_categories: bool = True
) -> go.Figure:
    """
    Create animated scatterplot across years.

    Args:
        df: DataFrame with task data
        years: List of year columns
        show_categories: Color by AI category

    Returns:
        Animated plotly figure
    """
    # Prepare data for all years
    all_frames_data = []

    for year in years:
        year_df = df[['Task', 'Data Value', year, 'AI_Category']].copy()
        year_df = year_df.rename(columns={
            'Data Value': 'importance',
            year: 'matches'
        })
        year_df['year'] = year
        all_frames_data.append(year_df)

    combined = pd.concat(all_frames_data, ignore_index=True)

    # Remove zero matches for cleaner visualization
    combined = combined[combined['matches'] > 0]

    # Aggregate by task and year (sum across categories if not showing categories)
    if not show_categories:
        combined = combined.groupby(['Task', 'importance', 'year']).agg({
            'matches': 'sum'
        }).reset_index()

    # Create animated scatter
    if show_categories:
        fig = px.scatter(
            combined,
            x='matches',
            y='importance',
            animation_frame='year',
            color='AI_Category',
            color_discrete_map=CATEGORY_COLORS,
            hover_data=['Task'],
            labels={
                'matches': 'AI Patent Matches',
                'importance': 'Task Importance',
                'AI_Category': 'AI Category'
            },
            opacity=0.6,
            range_x=[0, combined['matches'].max() * 1.05],
            range_y=[0.8, 5.2]
        )
    else:
        fig = px.scatter(
            combined,
            x='matches',
            y='importance',
            animation_frame='year',
            hover_data=['Task'],
            labels={
                'matches': 'AI Patent Matches',
                'importance': 'Task Importance'
            },
            opacity=0.6,
            color_discrete_sequence=['#4472C4'],
            range_x=[0, combined['matches'].max() * 1.05],
            range_y=[0.8, 5.2]
        )

    # Add reference lines
    median_matches = combined['matches'].median()

    fig.add_hline(y=3.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_matches, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Task Importance vs AI Patent Matches (Over Time)",
        height=700,
        width=1000,
        showlegend=show_categories
    )

    return fig


def main(args):
    """Main function."""

    logger.info("="*60)
    logger.info("SCATTERPLOT: TASK IMPORTANCE vs AI MATCH COUNT")
    logger.info("="*60)

    # Load data
    input_dir = Path(args.input_dir)
    logger.info(f"\nLoading data from: {input_dir}")
    df, year_columns = load_all_category_data(input_dir)

    years = year_columns
    if args.years:
        years = [y for y in year_columns if float(y) in args.years]

    logger.info(f"Processing years: {years}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    if args.animation:
        logger.info("\nCreating animated scatterplot...")
        fig = create_animated_scatterplot(df, years, show_categories=args.show_categories)

        html_path = output_dir / f"{args.output_name}_animated.html"
        logger.info(f"Saving to: {html_path}")
        fig.write_html(str(html_path))

    # Static plots for each year
    if args.static:
        logger.info("\nCreating static scatterplots for each year...")
        for year in tqdm(years, desc="Generating plots"):
            fig = create_scatterplot_for_year(
                df,
                year,
                show_categories=args.show_categories,
                top_n_labels=args.top_n_labels
            )

            # Save HTML
            html_path = output_dir / f"{args.output_name}_{year}.html"
            fig.write_html(str(html_path))

            # Save PNG if requested
            if args.save_png:
                try:
                    png_path = output_dir / f"{args.output_name}_{year}.png"
                    fig.write_image(str(png_path), width=1000, height=700)
                except Exception as e:
                    logger.warning(f"Could not save PNG for {year}: {e}")

    # Generate summary statistics
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)

    for year in years:
        high_importance_tasks = df[df['Data Value'] >= 4.0]
        high_matches = high_importance_tasks[high_importance_tasks[year] >= 50]

        logger.info(f"\nYear {year}:")
        logger.info(f"  Tasks with importance ≥ 4.0: {len(high_importance_tasks)}")
        logger.info(f"  High-importance tasks with ≥50 matches: {len(high_matches)}")

        if len(high_matches) > 0:
            top_5 = high_matches.nlargest(5, year)
            logger.info(f"  Top 5 high-importance, high-match tasks:")
            for _, row in top_5.iterrows():
                task_short = row['Task'][:60] + "..." if len(row['Task']) > 60 else row['Task']
                logger.info(f"    - {task_short} ({row[year]:.0f} matches, {row['Data Value']:.2f} importance)")

    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create scatterplot showing Task Importance vs AI Patent Matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Animated scatterplot with categories
  python create_importance_scatter.py \\
      --input-dir results/category_specific \\
      --output-dir results/visualizations \\
      --animation

  # Static plots for each year
  python create_importance_scatter.py \\
      --input-dir results/category_specific \\
      --output-dir results/visualizations \\
      --static

  # Both animated and static
  python create_importance_scatter.py \\
      --input-dir results/category_specific \\
      --output-dir results/visualizations \\
      --animation --static --show-categories

  # Aggregate view (all categories combined)
  python create_importance_scatter.py \\
      --input-dir results/category_specific \\
      --output-dir results/visualizations \\
      --static --no-show-categories
        """
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing category-specific CSV files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Directory to save visualizations"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="importance_scatter",
        help="Base name for output files"
    )

    parser.add_argument(
        "--animation",
        action="store_true",
        help="Create animated scatterplot across years"
    )

    parser.add_argument(
        "--static",
        action="store_true",
        help="Create static scatterplots for each year"
    )

    parser.add_argument(
        "--show-categories",
        action="store_true",
        default=True,
        help="Color points by AI category (default: True)"
    )

    parser.add_argument(
        "--no-show-categories",
        action="store_false",
        dest="show_categories",
        help="Don't color by category (aggregate view)"
    )

    parser.add_argument(
        "--top-n-labels",
        type=int,
        default=10,
        help="Number of outliers to label (default: 10)"
    )

    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save PNG versions (requires kaleido)"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs='+',
        help="Specific years to process (default: all)"
    )

    args = parser.parse_args()

    # Default to static if neither specified
    if not args.animation and not args.static:
        args.static = True

    main(args)
