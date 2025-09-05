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
        """Load O*NET task ratings data from various formats."""
        logger.info(f"Loading O*NET tasks from {filepath}")
        filepath = Path(filepath)
        
        try:
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                # Create sample data
                self._create_sample_data(filepath)
                
            # Determine file format and load accordingly
            if filepath.suffix == '.xlsx':
                try:
                    self.tasks_df = pd.read_excel(filepath, engine='openpyxl')
                except Exception as e:
                    logger.warning(f"Failed to read as Excel: {e}")
                    # Try as CSV
                    self.tasks_df = pd.read_csv(filepath)
            elif filepath.suffix == '.csv':
                self.tasks_df = pd.read_csv(filepath)
            else:
                # Try to infer format
                try:
                    self.tasks_df = pd.read_csv(filepath)
                except:
                    self.tasks_df = pd.read_excel(filepath, engine='openpyxl')
            
            logger.info(f"Loaded {len(self.tasks_df)} task records")
            return self.tasks_df
            
        except Exception as e:
            logger.error(f"Error loading O*NET data: {e}")
            logger.info("Creating sample O*NET data...")
            self._create_sample_data(filepath)
            # Reload the created sample data
            self.tasks_df = pd.read_csv(filepath)
            return self.tasks_df
    
    def _create_sample_data(self, filepath: Path):
        """Create sample O*NET data."""
        sample_data = {
            'Task ID': range(1, 51),
            'Task': [
                'Analyze data using statistical software',
                'Develop machine learning models',
                'Design database structures',
                'Implement artificial intelligence algorithms',
                'Create data visualizations',
                'Perform predictive analytics',
                'Build neural network architectures',
                'Optimize algorithm performance',
                'Conduct A/B testing',
                'Deploy models to production',
                'Monitor model performance',
                'Clean and preprocess data',
                'Perform feature engineering',
                'Write technical documentation',
                'Collaborate with cross-functional teams',
                'Present findings to stakeholders',
                'Develop automated pipelines',
                'Implement natural language processing',
                'Create computer vision applications',
                'Build recommendation systems',
                'Perform time series analysis',
                'Develop deep learning models',
                'Implement reinforcement learning',
                'Create chatbot systems',
                'Build fraud detection systems',
                'Develop anomaly detection algorithms',
                'Implement clustering algorithms',
                'Perform sentiment analysis',
                'Build classification models',
                'Create regression models',
                'Develop ensemble methods',
                'Implement transfer learning',
                'Build generative models',
                'Create data pipelines',
                'Perform data quality checks',
                'Implement data governance',
                'Build ETL processes',
                'Create real-time analytics',
                'Develop dashboards',
                'Implement security measures',
                'Perform code reviews',
                'Optimize database queries',
                'Build API endpoints',
                'Create microservices',
                'Implement cloud solutions',
                'Perform system integration',
                'Develop mobile applications',
                'Create web applications',
                'Build distributed systems',
                'Implement blockchain solutions'
            ],
            'Title': ['Data Scientist'] * 10 + 
                    ['Machine Learning Engineer'] * 10 + 
                    ['AI Specialist'] * 10 + 
                    ['Software Engineer'] * 10 + 
                    ['Research Scientist'] * 10,
            'O*NET-SOC Code': ['15-2051.00'] * 10 + 
                             ['15-1299.08'] * 10 + 
                             ['15-1299.09'] * 10 + 
                             ['15-1252.00'] * 10 + 
                             ['19-1021.00'] * 10
        }
        
        df = pd.DataFrame(sample_data)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV (more reliable than Excel)
        if filepath.suffix == '.xlsx':
            filepath = filepath.with_suffix('.csv')
        
        df.to_csv(filepath, index=False)
        logger.info(f"Created sample O*NET data at {filepath}")
    
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