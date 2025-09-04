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
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def preprocess_patents(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess patent dataframe."""
        logger.info("Preprocessing patent data")
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['application_id'])
        
        # Clean text fields
        if 'application_title' in df.columns:
            df['application_title_clean'] = df['application_title'].apply(
                DataPreprocessor.clean_text
            )
        
        # Convert date columns
        date_columns = ['filing_date', 'published_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle missing values
        df = df.dropna(subset=['application_title'])
        
        logger.info(f"Preprocessed {len(df)} patents")
        return df
    
    @staticmethod
    def preprocess_onet(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess O*NET dataframe."""
        logger.info("Preprocessing O*NET data")
        
        # Clean task descriptions
        if 'Task' in df.columns:
            df['Task_clean'] = df['Task'].apply(DataPreprocessor.clean_text)
        
        # Handle missing values
        df = df.dropna(subset=['Task'])
        
        logger.info(f"Preprocessed {len(df)} O*NET records")
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns."""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        return True