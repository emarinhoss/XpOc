"""Data preprocessing utilities."""
import pandas as pd
import numpy as np
import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing tasks."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep the text as-is for patent titles (they may contain important technical terms)
        return text.strip()
    
    @staticmethod
    def preprocess_patents(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess patent dataframe."""
        logger.info("Preprocessing patent data")
        
        # Ensure required columns exist
        required_cols = ['application_title']
        
        # If application_title doesn't exist, try other column names
        if 'application_title' not in df.columns:
            if 'title' in df.columns:
                df['application_title'] = df['title']
            elif 'patent_title' in df.columns:
                df['application_title'] = df['patent_title']
            else:
                # Create from first available text column
                text_cols = df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    df['application_title'] = df[text_cols[0]]
                    logger.warning(f"Using column '{text_cols[0]}' as application_title")
        
        # Ensure application_id exists
        if 'application_id' not in df.columns:
            if 'id' in df.columns:
                df['application_id'] = df['id']
            else:
                df['application_id'] = range(len(df))
        
        # Ensure year column exists
        if 'year' not in df.columns:
            if 'filing_date' in df.columns:
                df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
                df['year'] = df['filing_date'].dt.year.astype(float)
            elif 'published_date' in df.columns:
                df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
                df['year'] = df['published_date'].dt.year.astype(float)
            else:
                # Default to sample years
                df['year'] = 2021.0
        
        # Ensure topic_label exists
        if 'topic_label' not in df.columns:
            df['topic_label'] = 'Machine Learning'  # Default topic
        
        # Drop duplicates based on application_id
        df = df.drop_duplicates(subset=['application_id'])
        
        # Clean text fields
        df['application_title'] = df['application_title'].apply(DataPreprocessor.clean_text)
        
        # Handle missing values
        df = df.dropna(subset=['application_title'])
        
        logger.info(f"Preprocessed {len(df)} patents")
        return df
    
    @staticmethod
    def preprocess_onet(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess O*NET dataframe."""
        logger.info("Preprocessing O*NET data")
        
        # Ensure Task column exists
        if 'Task' not in df.columns:
            # Look for alternative column names
            task_cols = [col for col in df.columns if 'task' in col.lower()]
            if task_cols:
                df['Task'] = df[task_cols[0]]
            else:
                # Use first text column
                text_cols = df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    df['Task'] = df[text_cols[0]]
        
        # Ensure Title column exists
        if 'Title' not in df.columns:
            df['Title'] = 'Occupation'
        
        # Clean task descriptions
        df['Task'] = df['Task'].apply(DataPreprocessor.clean_text)
        
        # Handle missing values
        df = df.dropna(subset=['Task'])
        
        logger.info(f"Preprocessed {len(df)} O*NET records")
        return df