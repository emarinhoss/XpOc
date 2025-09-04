"""Main pipeline for patent-occupation matching."""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
import time

from src.data.patent_loader import PatentLoader
from src.data.onet_loader import ONetLoader
from src.embedding.bert_embedder import BertEmbedder
from src.matching.scann_matcher import ScannMatcher

logger = logging.getLogger(__name__)


class PatentOccupationPipeline:
    """Main pipeline for matching patents with occupational tasks."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.patent_loader = PatentLoader(self.config)
        self.onet_loader = ONetLoader(self.config)
        self.embedder = BertEmbedder(self.config)
        self.matcher = ScannMatcher(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run(self):
        """Execute the main pipeline."""
        logger.info("Starting patent-occupation matching pipeline")
        start_time = time.time()
        
        # Load data
        logger.info("Loading data...")
        patents_df = self.patent_loader.load_patents(
            self.config['data']['patents']['processed_path']
        )
        
        categories_df = self.patent_loader.load_ai_categories(
            self.config['data']['patents']['ai_categories_path']
        )
        
        ai_patents = self.patent_loader.filter_ai_patents(categories_df)
        
        onet_df = self.onet_loader.load_task_ratings(
            self.config['data']['onet']['task_ratings_path']
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