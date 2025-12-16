#!/usr/bin/env python
"""
Create Sankey diagram visualization: AI Category → Task → Occupation

This script creates animated Sankey diagrams showing how AI technologies flow
through specific tasks to impact occupations over time.

The visualization shows:
- Source: 8 AI categories
- Middle: Top tasks per category (by exposure contribution)
- Target: Top occupations (by total exposure)
- Animation: Changes across years
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI category colors (distinct, colorblind-friendly palette)
CATEGORY_COLORS = {
    'machine_learning': '#1f77b4',      # Blue
    'nlp': '#ff7f0e',                   # Orange
    'computer_vision': '#2ca02c',       # Green
    'speech_recognition': '#d62728',    # Red
    'knowledge_processing': '#9467bd',  # Purple
    'planning_and_control': '#8c564b',  # Brown
    'ai_hardware': '#e377c2',           # Pink
    'evolutionary_computation': '#7f7f7f'  # Gray
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Convert hex color to rgba format.

    Args:
        hex_color: Hex color string (e.g., '#1f77b4')
        alpha: Alpha/opacity value (0.0 to 1.0)

    Returns:
        RGBA string (e.g., 'rgba(31, 119, 180, 0.5)')
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'



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


def load_category_data(input_dir: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load all category-specific CSV files.

    Returns:
        (category_data, year_columns)
    """
    csv_files = sorted(input_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    logger.info(f"Found {len(csv_files)} category files")

    category_data = {}
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

        category_data[category] = df
        logger.info(f"  Loaded {category}: {len(df)} importance tasks")

    logger.info(f"Years detected: {year_columns}")

    return category_data, year_columns


def calculate_task_contributions(
    df: pd.DataFrame,
    year_col: str
) -> pd.DataFrame:
    """
    Calculate each task's contribution to its occupation's exposure.

    Returns DataFrame with additional 'contribution' column.
    """
    result = df.copy()

    # Group by occupation
    result['total_matches_occ'] = result.groupby('O*NET-SOC Code')[year_col].transform('sum')

    # Calculate task weight within occupation
    result['task_weight'] = result[year_col] / result['total_matches_occ'].replace(0, np.nan)
    result['task_weight'] = result['task_weight'].fillna(0)

    # Contribution = weight × importance
    result['contribution'] = result['task_weight'] * result['Data Value']

    return result


def select_top_items(
    category_data: Dict[str, pd.DataFrame],
    year_col: str,
    top_n_occupations: int,
    top_n_tasks_per_category: int
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Select top occupations and top tasks per category.

    Returns:
        (top_occupations, top_tasks_by_category)
    """
    # Calculate total exposure per occupation across all categories
    all_exposure = []

    for category, df in category_data.items():
        df_with_contrib = calculate_task_contributions(df, year_col)

        # Sum contribution by occupation
        occ_exposure = df_with_contrib.groupby(['O*NET-SOC Code', 'Title'])['contribution'].sum()
        occ_exposure = occ_exposure.reset_index()
        occ_exposure['category'] = category
        all_exposure.append(occ_exposure)

    # Combine all categories
    combined = pd.concat(all_exposure, ignore_index=True)

    # Get total exposure per occupation
    total_by_occ = combined.groupby(['O*NET-SOC Code', 'Title'])['contribution'].sum()
    total_by_occ = total_by_occ.sort_values(ascending=False)

    # Top N occupations
    top_occupations = list(total_by_occ.head(top_n_occupations).index)
    top_occupation_codes = [code for code, title in top_occupations]

    logger.info(f"\nTop {top_n_occupations} occupations by exposure:")
    for (code, title), exposure in total_by_occ.head(top_n_occupations).items():
        logger.info(f"  {title} ({code}): {exposure:.3f}")

    # For each category, get top tasks that contribute to these top occupations
    top_tasks_by_category = {}

    for category, df in category_data.items():
        # Filter to top occupations
        df_top_occ = df[df['O*NET-SOC Code'].isin(top_occupation_codes)].copy()

        if len(df_top_occ) == 0:
            top_tasks_by_category[category] = []
            continue

        # Calculate contributions
        df_with_contrib = calculate_task_contributions(df_top_occ, year_col)

        # Sum contribution by task across all top occupations
        task_contrib = df_with_contrib.groupby('Task')['contribution'].sum()
        task_contrib = task_contrib.sort_values(ascending=False)

        # Top N tasks for this category
        top_tasks = list(task_contrib.head(top_n_tasks_per_category).index)
        top_tasks_by_category[category] = top_tasks

        logger.info(f"\n{category.upper()} - Top {len(top_tasks)} tasks:")
        for task in top_tasks[:3]:  # Show first 3
            logger.info(f"  - {task[:80]}...")

    return top_occupations, top_tasks_by_category


def build_sankey_data(
    category_data: Dict[str, pd.DataFrame],
    year_col: str,
    top_occupations: List[Tuple[str, str]],
    top_tasks_by_category: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Build Sankey diagram data structure.

    Returns dict with 'nodes', 'links', and metadata.
    """
    # Build node lists
    nodes = []
    node_colors = []
    node_labels = []

    # 1. AI Category nodes (sources)
    categories = list(category_data.keys())
    for cat in categories:
        nodes.append(f"cat_{cat}")
        node_labels.append(cat.replace('_', ' ').title())
        node_colors.append(CATEGORY_COLORS.get(cat, '#999999'))

    # 2. Task nodes (middle)
    task_to_category = {}
    for cat, tasks in top_tasks_by_category.items():
        for task in tasks:
            task_id = f"task_{len(nodes)}"
            nodes.append(task_id)
            # Truncate long task names
            label = task[:60] + "..." if len(task) > 60 else task
            node_labels.append(label)
            # Use lighter shade of category color with transparency
            node_colors.append(hex_to_rgba(CATEGORY_COLORS.get(cat, '#999999'), 0.5))
            task_to_category[task] = cat

    # 3. Occupation nodes (targets)
    occ_code_to_idx = {}
    for code, title in top_occupations:
        occ_id = f"occ_{code}"
        nodes.append(occ_id)
        node_labels.append(title)
        node_colors.append('#cccccc')  # Gray for occupations
        occ_code_to_idx[code] = len(nodes) - 1

    # Build links: Category → Task → Occupation
    links_source = []
    links_target = []
    links_value = []
    links_color = []

    for cat_idx, cat in enumerate(categories):
        df = category_data[cat]

        # Filter to top tasks and top occupations
        top_tasks = top_tasks_by_category[cat]
        top_occ_codes = [code for code, title in top_occupations]

        df_filtered = df[
            (df['Task'].isin(top_tasks)) &
            (df['O*NET-SOC Code'].isin(top_occ_codes))
        ].copy()

        if len(df_filtered) == 0:
            continue

        # Calculate contributions
        df_contrib = calculate_task_contributions(df_filtered, year_col)

        # Category → Task links
        for task in top_tasks:
            task_data = df_contrib[df_contrib['Task'] == task]

            if len(task_data) == 0:
                continue

            # Sum contribution of this task across all occupations
            total_task_contrib = task_data['contribution'].sum()

            if total_task_contrib > 0.01:  # Threshold to avoid tiny links
                # Find task node index
                task_node_idx = None
                for i, node in enumerate(nodes):
                    if node.startswith('task_') and node_labels[i].startswith(task[:60]):
                        task_node_idx = i
                        break

                if task_node_idx is not None:
                    # Link: Category → Task
                    links_source.append(cat_idx)
                    links_target.append(task_node_idx)
                    links_value.append(total_task_contrib)
                    links_color.append(hex_to_rgba(CATEGORY_COLORS.get(cat, '#999999'), 0.25))

                    # Task → Occupation links
                    for _, row in task_data.iterrows():
                        occ_code = row['O*NET-SOC Code']
                        contrib = row['contribution']

                        if contrib > 0.01 and occ_code in occ_code_to_idx:
                            links_source.append(task_node_idx)
                            links_target.append(occ_code_to_idx[occ_code])
                            links_value.append(contrib)
                            links_color.append(hex_to_rgba(CATEGORY_COLORS.get(cat, '#999999'), 0.19))

    return {
        'nodes': nodes,
        'node_labels': node_labels,
        'node_colors': node_colors,
        'links_source': links_source,
        'links_target': links_target,
        'links_value': links_value,
        'links_color': links_color
    }


def create_sankey_figure(
    sankey_data_by_year: Dict[str, Dict[str, Any]],
    years: List[str]
) -> go.Figure:
    """
    Create animated Sankey figure with year frames.
    """
    # Get first year data for initial display
    first_year = years[0]
    initial_data = sankey_data_by_year[first_year]

    # Create figure
    fig = go.Figure()

    # Add initial Sankey
    fig.add_trace(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=0.5),
            label=initial_data['node_labels'],
            color=initial_data['node_colors']
        ),
        link=dict(
            source=initial_data['links_source'],
            target=initial_data['links_target'],
            value=initial_data['links_value'],
            color=initial_data['links_color']
        )
    ))

    # Create frames for animation
    frames = []
    for year in years:
        data = sankey_data_by_year[year]

        frame = go.Frame(
            data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='white', width=0.5),
                    label=data['node_labels'],
                    color=data['node_colors']
                ),
                link=dict(
                    source=data['links_source'],
                    target=data['links_target'],
                    value=data['links_value'],
                    color=data['links_color']
                )
            )],
            name=str(year)
        )
        frames.append(frame)

    fig.frames = frames

    # Add animation controls
    fig.update_layout(
        title=dict(
            text=f"AI Technology Impact Flow: Categories → Tasks → Occupations<br><sub>Year: {first_year}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        font=dict(size=12),
        height=900,
        width=1400,
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1500, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 500}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 1.15,
            'xanchor': 'left',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'args': [[year], {
                        'frame': {'duration': 500, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 500}
                    }],
                    'label': str(year),
                    'method': 'animate'
                }
                for year in years
            ],
            'x': 0.1,
            'len': 0.85,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'pad': {'b': 10, 't': 50},
            'currentvalue': {
                'visible': True,
                'prefix': 'Year: ',
                'xanchor': 'right',
                'font': {'size': 16}
            }
        }]
    )

    return fig


def main(args):
    """Main function."""

    logger.info("="*60)
    logger.info("SANKEY DIAGRAM: AI CATEGORY → TASK → OCCUPATION")
    logger.info("="*60)

    # Load data
    input_dir = Path(args.input_dir)
    logger.info(f"\nLoading data from: {input_dir}")
    category_data, year_columns = load_category_data(input_dir)

    years = year_columns
    if args.years:
        # Filter to specific years
        years = [y for y in year_columns if float(y) in args.years]

    logger.info(f"Processing years: {years}")

    # Build Sankey for each year
    logger.info("\nBuilding Sankey diagrams for each year...")
    sankey_data_by_year = {}

    # Use first year to select top items (consistent across animation)
    logger.info(f"\nSelecting top items based on {years[0]}...")
    top_occupations, top_tasks_by_category = select_top_items(
        category_data,
        years[0],
        args.top_occupations,
        args.top_tasks_per_category
    )

    for year in tqdm(years, desc="Processing years"):
        sankey_data = build_sankey_data(
            category_data,
            year,
            top_occupations,
            top_tasks_by_category
        )
        sankey_data_by_year[year] = sankey_data

    # Create figure
    logger.info("\nCreating animated Sankey figure...")
    fig = create_sankey_figure(sankey_data_by_year, years)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interactive HTML
    if args.output_format in ['html', 'both']:
        html_path = output_dir / f"{args.output_name}.html"
        logger.info(f"\nSaving interactive HTML to: {html_path}")
        fig.write_html(str(html_path))

    # Static images for each year
    if args.output_format in ['png', 'both']:
        logger.info("\nGenerating static images for each year...")
        for i, year in enumerate(tqdm(years, desc="Saving PNGs")):
            # Update to specific year
            fig.update_layout(
                title=f"AI Technology Impact Flow: Categories → Tasks → Occupations<br><sub>Year: {year}</sub>"
            )

            # Update data to specific year
            data = sankey_data_by_year[year]
            fig.data = [go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='white', width=0.5),
                    label=data['node_labels'],
                    color=data['node_colors']
                ),
                link=dict(
                    source=data['links_source'],
                    target=data['links_target'],
                    value=data['links_value'],
                    color=data['links_color']
                )
            )]

            png_path = output_dir / f"{args.output_name}_{year}.png"
            fig.write_image(str(png_path), width=1400, height=900)

    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Top occupations shown: {args.top_occupations}")
    logger.info(f"Top tasks per category: {args.top_tasks_per_category}")
    logger.info(f"Years: {len(years)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create animated Sankey diagram showing AI Category → Task → Occupation flows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with animation
  python create_sankey_visualization.py \\
      --input-dir results/category_specific \\
      --output-dir results/visualizations \\
      --output-format both

  # Customize number of items shown
  python create_sankey_visualization.py \\
      --input-dir results/category_specific \\
      --top-occupations 15 \\
      --top-tasks-per-category 5 \\
      --output-dir results/visualizations

  # Specific years only
  python create_sankey_visualization.py \\
      --input-dir results/category_specific \\
      --years 2021 2023 2025 \\
      --output-format html
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
        help="Directory to save visualizations (default: results/visualizations)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="sankey_ai_impact",
        help="Base name for output files (default: sankey_ai_impact)"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=['html', 'png', 'both'],
        default='both',
        help="Output format (default: both)"
    )

    parser.add_argument(
        "--top-occupations",
        type=int,
        default=20,
        help="Number of top occupations to show (default: 20)"
    )

    parser.add_argument(
        "--top-tasks-per-category",
        type=int,
        default=3,
        help="Number of top tasks per AI category (default: 3)"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs='+',
        help="Specific years to include (default: all years in data)"
    )

    args = parser.parse_args()

    # Check for required packages
    try:
        import plotly
        if args.output_format in ['png', 'both']:
            import kaleido
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install plotly kaleido")
        sys.exit(1)

    main(args)
