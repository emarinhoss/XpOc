#!/usr/bin/env python
"""Script to download required data files."""
import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.file_utils import FileHandler

logger = logging.getLogger(__name__)


class DataDownloader:
    """Handle data downloading and extraction."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data downloader."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.raw_dir / "patents",
            self.raw_dir / "onet",
            self.raw_dir / "categories",
            self.data_dir / "processed" / "embeddings",
            self.data_dir / "processed" / "matches"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192):
        """Download file from URL with progress bar."""
        logger.info(f"Downloading from {url}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {filepath}")
    
    def download_patents(self):
        """Download patent data from USPTO."""
        logger.info("Downloading patent data...")
        
        # USPTO bulk data URL (example - adjust as needed)
        url = "https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_published_application.tsv.zip"
        filepath = self.raw_dir / "patents" / "pg_published_application.tsv.zip"
        
        if not filepath.exists():
            self.download_file(url, filepath)
            
            # Extract zip file
            logger.info("Extracting patent data...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir / "patents")
        else:
            logger.info("Patent data already exists")
    
    def download_onet(self):
        """Download O*NET data."""
        logger.info("Downloading O*NET data...")
        
        # O*NET task statements URL
        url = "https://www.onetcenter.org/dl_files/database/db_28_3_excel/Task%20Ratings.xlsx"
        filepath = self.raw_dir / "onet" / "Task Ratings-3.xlsx"
        
        if not filepath.exists():
            self.download_file(url, filepath)
        else:
            logger.info("O*NET data already exists")
    
    def create_sample_data(self):
        """Create sample data for testing."""
        logger.info("Creating sample data...")
        
        # Sample patent categories (AI-related)
        categories_data = {
            'AI topic': [1, 1, 1, 0, 0],
            'Topics': [
                'Machine Learning Training and Model Development',
                'Neural Network Architectures',
                'Natural Language Processing',
                'Traditional Software Development',
                'Hardware Manufacturing'
            ]
        }
        
        categories_df = pd.DataFrame(categories_data)
        categories_df.to_excel(
            self.raw_dir / "categories" / "patent_categories.xlsx",
            index=False
        )
        logger.info("Created sample patent categories")
    
    def run(self):
        """Run all download tasks."""
        try:
            self.download_patents()
            self.download_onet()
            self.create_sample_data()
            logger.info("Data download completed successfully")
        except Exception as e:
            logger.error(f"Data download failed: {e}")
            sys.exit(1)


def main():
    """Main execution function."""
    setup_logging(level="INFO")
    
    downloader = DataDownloader()
    downloader.run()


if __name__ == "__main__":
    main()