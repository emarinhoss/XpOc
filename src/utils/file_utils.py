"""File handling utilities."""
import json
import yaml
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Union
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]):
        """Ensure directory exists, create if not."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Any, filepath: Union[str, Path]):
        """Save data as JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON to {filepath}")
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Any:
        """Load JSON data."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    
    @staticmethod
    def save_yaml(data: Any, filepath: Union[str, Path]):
        """Save data as YAML."""
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Saved YAML to {filepath}")
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Any:
        """Load YAML data."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Loaded YAML from {filepath}")
        return data
    
    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path]):
        """Save data as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved pickle to {filepath}")
    
    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Any:
        """Load pickle data."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle from {filepath}")
        return data
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], **kwargs):
        """Save DataFrame based on file extension."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif filepath.suffix == '.parquet':
            df.to_parquet(filepath, index=False, **kwargs)
        elif filepath.suffix == '.xlsx':
            df.to_excel(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Saved DataFrame to {filepath}")