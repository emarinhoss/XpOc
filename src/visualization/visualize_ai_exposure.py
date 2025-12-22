#!/usr/bin/env python3
"""
Visualize AI Exposure Distribution by Occupation.

Creates static (PNG) and interactive (HTML) visualizations showing the distribution
of AI exposure scores across occupations, with highlighted specific occupations
and statistical reference points.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize AI Exposure Distribution by Occupation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visualize_ai_exposure.py \\
      --exposure-file data/exposure_scores.csv \\
      --occupations-file data/occupations_of_interest.csv \\
      --output-dir output/visualizations \\
      --highlight-socs "15-1252.00,29-1171.00,53-3032.00"
        """
    )

    parser.add_argument(
        '--exposure-file', '-e',
        type=str,
        required=True,
        help='Path to CSV file with exposure scores (columns: O*NET-SOC Code, Title, AI_Category, 2021-2025)'
    )

    parser.add_argument(
        '--occupations-file', '-o',
        type=str,
        required=True,
        help='Path to CSV file with occupations of interest (must have OccCode_2019ONET column)'
    )

    parser.add_argument(
        '--output-dir', '-d',
        type=str,
        default='output/visualizations',
        help='Directory to save output visualizations (default: output/visualizations)'
    )

    parser.add_argument(
        '--highlight-socs', '-s',
        type=str,
        default='',
        help='Comma-separated list of SOC codes to highlight (e.g., "15-1252.00,29-1171.00")'
    )

    return parser.parse_args()


def load_exposure_data(filepath: str) -> pd.DataFrame:
    """Load exposure data from CSV file."""
    logger.info(f"Loading exposure data from {filepath}")
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = ['O*NET-SOC Code', 'Title', 'AI_Category']
    year_cols = ['2021', '2022', '2023', '2024', '2025']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Check for year columns (may be int or string)
    available_year_cols = []
    for year in year_cols:
        if year in df.columns:
            available_year_cols.append(year)
        elif int(year) in df.columns:
            df.rename(columns={int(year): year}, inplace=True)
            available_year_cols.append(year)

    if not available_year_cols:
        raise ValueError(f"No year columns found. Expected: {year_cols}")

    logger.info(f"Loaded {len(df)} rows with {df['AI_Category'].nunique()} AI categories")
    logger.info(f"Year columns found: {available_year_cols}")

    return df, available_year_cols


def load_occupations_of_interest(filepath: str) -> pd.DataFrame:
    """Load occupations of interest from CSV file."""
    logger.info(f"Loading occupations of interest from {filepath}")
    df = pd.read_csv(filepath)

    if 'OccCode_2019ONET' not in df.columns:
        raise ValueError("Missing required column: OccCode_2019ONET")

    logger.info(f"Loaded {len(df)} occupations of interest")
    return df


def calculate_cumulative_exposure(
    exposure_df: pd.DataFrame,
    occupations_df: pd.DataFrame,
    year_cols: list
) -> pd.DataFrame:
    """
    Calculate cumulative exposure for each occupation per AI category.

    Returns a DataFrame with columns:
    - soc_code: O*NET-SOC Code
    - title: Occupation title
    - ai_category: AI category name
    - cumulative_exposure: Sum of exposure across all years
    """
    # Filter exposure data to only include occupations of interest
    valid_socs = set(occupations_df['OccCode_2019ONET'].dropna().unique())
    filtered_df = exposure_df[exposure_df['O*NET-SOC Code'].isin(valid_socs)].copy()

    logger.info(f"Matched {filtered_df['O*NET-SOC Code'].nunique()} occupations from {len(valid_socs)} occupations of interest")

    # Calculate cumulative exposure (sum across years)
    filtered_df['cumulative_exposure'] = filtered_df[year_cols].sum(axis=1)

    # Select and rename columns for clarity
    result = filtered_df[['O*NET-SOC Code', 'Title', 'AI_Category', 'cumulative_exposure']].copy()
    result.columns = ['soc_code', 'title', 'ai_category', 'cumulative_exposure']

    return result


def get_statistics(data: pd.Series) -> dict:
    """Calculate statistics for exposure data."""
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
        'count': len(data)
    }


def create_static_visualization(
    df: pd.DataFrame,
    ai_category: str,
    highlight_socs: list,
    output_path: Path,
    stats: dict
):
    """
    Create static matplotlib/seaborn visualization.

    Shows a beeswarm/strip plot with mean, median, min/max labels,
    and highlighted occupations.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data
    plot_data = df[df['ai_category'] == ai_category].copy()
    plot_data = plot_data.sort_values('cumulative_exposure').reset_index(drop=True)

    # Add jitter for y-axis to create beeswarm effect
    np.random.seed(42)
    plot_data['jitter'] = np.random.uniform(-0.3, 0.3, len(plot_data))

    # Identify highlighted occupations
    plot_data['is_highlighted'] = plot_data['soc_code'].isin(highlight_socs)

    # Find min and max occupations
    min_idx = plot_data['cumulative_exposure'].idxmin()
    max_idx = plot_data['cumulative_exposure'].idxmax()
    min_occ = plot_data.loc[min_idx]
    max_occ = plot_data.loc[max_idx]

    # Plot all points (non-highlighted)
    non_highlighted = plot_data[~plot_data['is_highlighted']]
    ax.scatter(
        non_highlighted['cumulative_exposure'],
        non_highlighted['jitter'],
        c='#4a90a4',
        alpha=0.5,
        s=30,
        label='All Occupations',
        edgecolors='white',
        linewidths=0.5
    )

    # Plot highlighted points
    highlighted = plot_data[plot_data['is_highlighted']]
    ax.scatter(
        highlighted['cumulative_exposure'],
        highlighted['jitter'],
        c='#e74c3c',
        alpha=1.0,
        s=120,
        label='Highlighted Occupations',
        edgecolors='black',
        linewidths=1.5,
        zorder=5
    )

    # Add labels for highlighted occupations
    for _, row in highlighted.iterrows():
        ax.annotate(
            row['title'],
            xy=(row['cumulative_exposure'], row['jitter']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1),
            zorder=6
        )

    # Add mean line
    ax.axvline(
        stats['mean'],
        color='#2ecc71',
        linestyle='--',
        linewidth=2,
        label=f"Mean: {stats['mean']:.2f}",
        zorder=3
    )

    # Add median line
    ax.axvline(
        stats['median'],
        color='#9b59b6',
        linestyle='-.',
        linewidth=2,
        label=f"Median: {stats['median']:.2f}",
        zorder=3
    )

    # Annotate min occupation
    ax.annotate(
        f"MIN: {min_occ['title']}\n({min_occ['cumulative_exposure']:.2f})",
        xy=(min_occ['cumulative_exposure'], min_occ['jitter']),
        xytext=(-15, -50),
        textcoords='offset points',
        fontsize=9,
        color='#1a5276',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4e6f1', edgecolor='#1a5276', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.5),
        ha='center',
        zorder=6
    )

    # Annotate max occupation
    ax.annotate(
        f"MAX: {max_occ['title']}\n({max_occ['cumulative_exposure']:.2f})",
        xy=(max_occ['cumulative_exposure'], max_occ['jitter']),
        xytext=(15, 50),
        textcoords='offset points',
        fontsize=9,
        color='#7b241c',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#7b241c', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#7b241c', lw=1.5),
        ha='center',
        zorder=6
    )

    # Styling
    ax.set_xlabel('Cumulative AI Exposure (2021-2025)', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title(
        f'AI Exposure Distribution: {ai_category}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add statistics box
    stats_text = (
        f"N = {stats['count']:,}  |  "
        f"Mean = {stats['mean']:.2f}  |  "
        f"Median = {stats['median']:.2f}  |  "
        f"Std Dev = {stats['std']:.2f}  |  "
        f"Range = [{stats['min']:.2f}, {stats['max']:.2f}]"
    )
    fig.text(
        0.5, 0.02,
        stats_text,
        ha='center',
        fontsize=10,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6')
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved static visualization: {output_path}")


def create_interactive_visualization(
    df: pd.DataFrame,
    ai_category: str,
    highlight_socs: list,
    output_path: Path,
    stats: dict
):
    """
    Create interactive Plotly visualization.

    Shows a strip plot with hover information for all occupations,
    highlighted occupations, and statistical reference lines.
    """
    plot_data = df[df['ai_category'] == ai_category].copy()
    plot_data = plot_data.sort_values('cumulative_exposure').reset_index(drop=True)

    # Add jitter for y-axis
    np.random.seed(42)
    plot_data['jitter'] = np.random.uniform(-0.3, 0.3, len(plot_data))

    # Identify highlighted occupations
    plot_data['is_highlighted'] = plot_data['soc_code'].isin(highlight_socs)

    # Find min and max
    min_idx = plot_data['cumulative_exposure'].idxmin()
    max_idx = plot_data['cumulative_exposure'].idxmax()
    min_occ = plot_data.loc[min_idx]
    max_occ = plot_data.loc[max_idx]

    fig = go.Figure()

    # Add non-highlighted points
    non_highlighted = plot_data[~plot_data['is_highlighted']]
    fig.add_trace(go.Scatter(
        x=non_highlighted['cumulative_exposure'],
        y=non_highlighted['jitter'],
        mode='markers',
        name='All Occupations',
        marker=dict(
            size=8,
            color='#4a90a4',
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'SOC Code: %{customdata[1]}<br>'
            'Exposure: %{x:.2f}<extra></extra>'
        ),
        customdata=non_highlighted[['title', 'soc_code']].values
    ))

    # Add highlighted points
    highlighted = plot_data[plot_data['is_highlighted']]
    fig.add_trace(go.Scatter(
        x=highlighted['cumulative_exposure'],
        y=highlighted['jitter'],
        mode='markers+text',
        name='Highlighted Occupations',
        marker=dict(
            size=14,
            color='#e74c3c',
            opacity=1.0,
            line=dict(width=2, color='black')
        ),
        text=highlighted['title'],
        textposition='top center',
        textfont=dict(size=10, color='#c0392b'),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'SOC Code: %{customdata[1]}<br>'
            'Exposure: %{x:.2f}<extra></extra>'
        ),
        customdata=highlighted[['title', 'soc_code']].values
    ))

    # Add mean line
    fig.add_vline(
        x=stats['mean'],
        line=dict(color='#2ecc71', width=2, dash='dash'),
        annotation_text=f"Mean: {stats['mean']:.2f}",
        annotation_position='top'
    )

    # Add median line
    fig.add_vline(
        x=stats['median'],
        line=dict(color='#9b59b6', width=2, dash='dashdot'),
        annotation_text=f"Median: {stats['median']:.2f}",
        annotation_position='bottom'
    )

    # Add min annotation
    fig.add_annotation(
        x=min_occ['cumulative_exposure'],
        y=min_occ['jitter'],
        text=f"MIN: {min_occ['title']}<br>({min_occ['cumulative_exposure']:.2f})",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#1a5276',
        bgcolor='#d4e6f1',
        bordercolor='#1a5276',
        font=dict(size=10, color='#1a5276')
    )

    # Add max annotation
    fig.add_annotation(
        x=max_occ['cumulative_exposure'],
        y=max_occ['jitter'],
        text=f"MAX: {max_occ['title']}<br>({max_occ['cumulative_exposure']:.2f})",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#7b241c',
        bgcolor='#fadbd8',
        bordercolor='#7b241c',
        font=dict(size=10, color='#7b241c')
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'AI Exposure Distribution: {ai_category}',
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        xaxis_title='Cumulative AI Exposure (2021-2025)',
        yaxis=dict(showticklabels=False, title=''),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=1200,
        height=600,
        annotations=[
            dict(
                text=(
                    f"N = {stats['count']:,}  |  Mean = {stats['mean']:.2f}  |  "
                    f"Median = {stats['median']:.2f}  |  Std Dev = {stats['std']:.2f}  |  "
                    f"Range = [{stats['min']:.2f}, {stats['max']:.2f}]"
                ),
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.12,
                showarrow=False,
                font=dict(size=11, color='#7f8c8d'),
                bgcolor='#f8f9fa',
                bordercolor='#dee2e6',
                borderwidth=1
            )
        ]
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=False)

    # Save as HTML
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)

    logger.info(f"Saved interactive visualization: {output_path}")


def save_summary_statistics(
    df: pd.DataFrame,
    output_path: Path
):
    """Save summary statistics for all AI categories to CSV."""
    stats_list = []

    for category in df['ai_category'].unique():
        cat_data = df[df['ai_category'] == category]['cumulative_exposure']
        stats = get_statistics(cat_data)
        stats['ai_category'] = category
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df[['ai_category', 'count', 'mean', 'median', 'std', 'min', 'max', 'q1', 'q3']]
    stats_df.to_csv(output_path, index=False)

    logger.info(f"Saved summary statistics: {output_path}")
    return stats_df


def main():
    """Main entry point."""
    args = parse_arguments()

    # Parse highlight SOCs
    highlight_socs = []
    if args.highlight_socs:
        highlight_socs = [soc.strip() for soc in args.highlight_socs.split(',') if soc.strip()]

    logger.info(f"SOC codes to highlight: {highlight_socs}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    exposure_df, year_cols = load_exposure_data(args.exposure_file)
    occupations_df = load_occupations_of_interest(args.occupations_file)

    # Calculate cumulative exposure
    cumulative_df = calculate_cumulative_exposure(exposure_df, occupations_df, year_cols)

    # Get unique AI categories
    ai_categories = cumulative_df['ai_category'].unique()
    logger.info(f"Found {len(ai_categories)} AI categories: {list(ai_categories)}")

    # Validate highlight SOCs
    valid_socs = set(cumulative_df['soc_code'].unique())
    invalid_socs = [soc for soc in highlight_socs if soc not in valid_socs]
    if invalid_socs:
        logger.warning(f"The following SOC codes were not found in the data: {invalid_socs}")

    valid_highlight_socs = [soc for soc in highlight_socs if soc in valid_socs]
    logger.info(f"Valid highlighted SOC codes: {valid_highlight_socs}")

    # Generate visualizations for each AI category
    for category in ai_categories:
        # Sanitize category name for filename
        safe_category = category.replace(' ', '_').replace('/', '_').replace('\\', '_')

        # Get statistics for this category
        cat_data = cumulative_df[cumulative_df['ai_category'] == category]['cumulative_exposure']
        stats = get_statistics(cat_data)

        # Create static visualization (PNG)
        static_path = output_dir / f"{safe_category}_exposure.png"
        create_static_visualization(
            cumulative_df,
            category,
            valid_highlight_socs,
            static_path,
            stats
        )

        # Create interactive visualization (HTML)
        interactive_path = output_dir / f"{safe_category}_exposure.html"
        create_interactive_visualization(
            cumulative_df,
            category,
            valid_highlight_socs,
            interactive_path,
            stats
        )

    # Save summary statistics
    stats_path = output_dir / "summary_statistics.csv"
    save_summary_statistics(cumulative_df, stats_path)

    logger.info(f"Visualization complete! Output saved to: {output_dir}")
    print(f"\nGenerated {len(ai_categories) * 2} visualizations ({len(ai_categories)} static + {len(ai_categories)} interactive)")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
