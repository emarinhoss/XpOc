"""Module for loading and preprocessing patent data."""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List
import csv

logger = logging.getLogger(__name__)


class PatentLoader:
    """Handles loading and filtering of patent data."""
    
    def __init__(self, config: dict):
        """Initialize patent loader with configuration."""
        self.config = config
        self.patents_df = None
        self.ai_patents_df = None
        
    def load_patents(self, filepath: str) -> pd.DataFrame:
        """Load patent data from CSV file."""
        logger.info(f"Loading patents from {filepath}")
        try:
            self.patents_df = pd.read_csv(
                filepath, 
                quoting=csv.QUOTE_NONNUMERIC
            ).drop_duplicates()
            logger.info(f"Loaded {len(self.patents_df)} patents")
            return self.patents_df
        except Exception as e:
            logger.error(f"Error loading patents: {e}")
            raise
            
    def load_ai_categories(self, filepath: str) -> pd.DataFrame:
        """Load AI patent categories."""
        logger.info(f"Loading AI categories from {filepath}")
        categories_df = pd.read_excel(filepath)
        return categories_df
    
    def filter_ai_patents(self, categories_df: pd.DataFrame) -> pd.DataFrame:
        """Filter patents to only include AI-related ones."""
        ai_topic = self.config['processing']['ai_topic_filter']
        ai_categories = categories_df[
            categories_df["AI topic"] == ai_topic
        ]["Topics"].tolist()
        
        self.ai_patents_df = self.patents_df[
            self.patents_df['topic_label'].isin(ai_categories)
        ]
        
        logger.info(f"Filtered to {len(self.ai_patents_df)} AI patents")
        return self.ai_patents_df
    
    def get_patents_by_year(self, year: int) -> pd.DataFrame:
        """Get patents for a specific year."""
        if self.ai_patents_df is None:
            raise ValueError("AI patents not loaded. Run filter_ai_patents first.")
        
        year_patents = self.ai_patents_df[
            self.ai_patents_df['year'] == float(year)
        ]
        logger.info(f"Found {len(year_patents)} patents for year {year}")
        return year_patents