"""Module for loading O*NET occupation task data."""
import pandas as pd
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class ONetLoader:
    """Handles loading and processing of O*NET data."""
    
    def __init__(self, config: dict):
        """Initialize O*NET loader with configuration."""
        self.config = config
        self.tasks_df = None
        
    def load_task_ratings(self, filepath: str) -> pd.DataFrame:
        """Load O*NET task ratings data."""
        logger.info(f"Loading O*NET tasks from {filepath}")
        try:
            self.tasks_df = pd.read_excel(filepath)
            logger.info(f"Loaded {len(self.tasks_df)} task records")
            return self.tasks_df
        except Exception as e:
            logger.error(f"Error loading O*NET data: {e}")
            raise
    
    def get_unique_tasks(self) -> List[str]:
        """Get list of unique task descriptions."""
        if self.tasks_df is None:
            raise ValueError("Tasks not loaded. Run load_task_ratings first.")
        
        unique_tasks = self.tasks_df['Task'].unique().tolist()
        logger.info(f"Found {len(unique_tasks)} unique tasks")
        return unique_tasks
    
    def add_matching_results(self, year: int, neighbors: list, distances: list):
        """Add matching results for a specific year to the dataframe."""
        self.tasks_df[f'neighbors_{year}'] = neighbors
        self.tasks_df[f'distances_{year}'] = distances
        logger.info(f"Added matching results for year {year}")