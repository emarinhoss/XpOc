"""Main pipeline for patent-occupation matching."""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
import time
import pandas as pd

from src.data.patent_loader import PatentLoader
from src.data.onet_loader import ONetLoader
from src.data.data_preprocessor import DataPreprocessor
from src.embedding.bert_embedder import BertEmbedder

# Try to import ScaNN, fall back to simple matcher if it fails
try:
    from src.matching.scann_matcher import ScannMatcher
    MATCHER_CLASS = ScannMatcher
    logger = logging.getLogger(__name__)
    logger.info("Using ScaNN matcher")
except (ImportError, Exception) as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"ScaNN not available ({e}), using simple matcher")
    from src.matching.simple_matcher import SimpleMatcher
    MATCHER_CLASS = SimpleMatcher

logger = logging.getLogger(__name__)


class PatentOccupationPipeline:
    """Main pipeline for matching patents with occupational tasks."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.patent_loader = PatentLoader(self.config)
        self.onet_loader = ONetLoader(self.config)
        self.embedder = BertEmbedder(self.config)
        self.matcher = MATCHER_CLASS(self.config)
        self.preprocessor = DataPreprocessor()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _ensure_processed_data(self):
        """Ensure processed data exists, create if not."""
        processed_patents_path = Path(self.config['data']['patents']['processed_path'])
        processed_onet_path = Path(self.config['data']['onet']['processed_path'])
        
        # Ensure directories exist
        processed_patents_path.parent.mkdir(parents=True, exist_ok=True)
        processed_onet_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if processed patents exist
        if not processed_patents_path.exists():
            logger.info("Processed patents not found. Creating sample data...")
            self._create_sample_patents(processed_patents_path)
        
        # Check if processed O*NET exists
        if not processed_onet_path.exists():
            logger.info("Processed O*NET not found. Creating sample data...")
            # Create as CSV for better compatibility
            if processed_onet_path.suffix == '.xlsx':
                processed_onet_path = processed_onet_path.with_suffix('.csv')
            self._create_sample_onet(processed_onet_path)
    
    def _create_sample_patents(self, output_path: Path):
        """Create sample patent data for demonstration."""
        sample_data = {
            'application_id': range(1, 101),
            'application_title': [f'AI Patent Title {i}' for i in range(1, 101)],
            'application_abstract': [f'Abstract for patent {i}' for i in range(1, 101)],
            'filing_date': ['2021-01-01'] * 30 + ['2022-01-01'] * 35 + ['2023-01-01'] * 35,
            'published_date': ['2021-06-01'] * 30 + ['2022-06-01'] * 35 + ['2023-06-01'] * 35,
            'year': [2021.0] * 30 + [2022.0] * 35 + [2023.0] * 35,
            'topic_label': ['Machine Learning'] * 50 + ['Neural Networks'] * 50
        }
        
        df = pd.DataFrame(sample_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Created sample patents at {output_path}")
    
    def _create_sample_onet(self, output_path: Path):
        """Create sample O*NET data for demonstration."""
        sample_data = {
            'Task ID': range(1, 51),
            'Task': [f'Perform task related to {area} using {tool}' 
                    for area in ['analysis', 'design', 'management', 'development', 'testing']
                    for tool in ['AI tools', 'ML algorithms', 'data systems', 'automation', 
                                 'neural networks', 'NLP', 'computer vision', 'robotics', 
                                 'expert systems', 'decision trees']],
            'Title': ['Data Scientist'] * 10 + ['Software Engineer'] * 10 + 
                    ['AI Specialist'] * 10 + ['ML Engineer'] * 10 + ['Research Scientist'] * 10,
            'O*NET-SOC Code': ['15-2051.00'] * 10 + ['15-1252.00'] * 10 + 
                             ['15-1299.00'] * 10 + ['15-1299.01'] * 10 + ['19-1021.00'] * 10
        }
        
        df = pd.DataFrame(sample_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Created sample O*NET data at {output_path}")
    
    def run(self):
        """Execute the main pipeline."""
        logger.info("Starting patent-occupation matching pipeline")
        start_time = time.time()
        
        # Ensure processed data exists
        self._ensure_processed_data()
        
        # Load processed data
        logger.info("Loading processed data...")
        patents_df = self.patent_loader.load_patents(
            self.config['data']['patents']['processed_path']
        )
        
        # Load or create AI categories
        categories_path = Path(self.config['data']['patents']['ai_categories_path'])
        if categories_path.exists():
            categories_df = self.patent_loader.load_ai_categories(str(categories_path))
        else:
            # Create default categories
            logger.warning("AI categories file not found. Using all patents as AI patents...")
            categories_df = pd.DataFrame({
                'AI topic': [1] * len(patents_df['topic_label'].unique()),
                'Topics': patents_df['topic_label'].unique()
            })
        
        ai_patents = self.patent_loader.filter_ai_patents(categories_df)
        
        # Load O*NET data
        onet_df = self.onet_loader.load_task_ratings(
            self.config['data']['onet']['processed_path']
        )
        
        # Generate O*NET embeddings
        logger.info("Generating O*NET task embeddings...")
        tasks = self.onet_loader.get_unique_tasks()
        task_embeddings = self.embedder.encode_texts(tasks)
        
        # Process by year
        years = self.config['processing']['years_to_process']
        
        for year in years:
            logger.info(f"Processing year {year}")
            
            # Get patents for year
            year_patents = self.patent_loader.get_patents_by_year(year)
            
            if len(year_patents) == 0:
                logger.warning(f"No patents found for year {year}, skipping...")
                continue
            
            patent_titles = year_patents['application_title'].tolist()
            
            # Generate patent embeddings
            logger.info(f"Generating embeddings for {len(patent_titles)} patents")
            patent_embeddings = self.embedder.encode_texts(patent_titles)
            
            # Build index and search
            self.matcher.build_index(patent_embeddings)
            neighbors, distances = self.matcher.search_batch(task_embeddings)
            
            # Store results
            self.onet_loader.add_matching_results(
                year, 
                neighbors.tolist(), 
                distances.tolist()
            )
        
        # Save results
        output_path = Path(self.config['output']['results_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "patent_occupation_matches.csv"
        self.onet_loader.tasks_df.to_csv(output_file, index=False)
        
        elapsed_time = (time.time() - start_time) / 3600
        logger.info(f"Pipeline completed in {elapsed_time:.2f} hours")
        logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    pipeline = PatentOccupationPipeline("config/config.yaml")
    pipeline.run()        
    # ... rest of the class remains the same ...