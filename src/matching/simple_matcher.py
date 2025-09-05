"""Simple fallback matcher using sklearn."""
import numpy as np
import logging
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimpleMatcher:
    """Simple similarity matcher using sklearn."""
    
    def __init__(self, config: dict):
        """Initialize matcher."""
        self.config = config.get('matching', {})
        self.dataset = None
        logger.info("Initialized SimpleMatcher (ScaNN fallback)")
        
    def build_index(self, dataset: np.ndarray):
        """Store dataset for matching."""
        logger.info(f"Building simple index for {len(dataset)} items")
        self.dataset = dataset
        logger.info("Simple index built successfully")
        
    def search_batch(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batch similarity search using cosine similarity."""
        if self.dataset is None:
            raise ValueError("Index not built. Run build_index first.")
        
        logger.info(f"Searching for {len(queries)} queries using cosine similarity")
        
        # Normalize if not already normalized
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
        dataset_norm = self.dataset / (np.linalg.norm(self.dataset, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarity
        similarities = np.dot(queries_norm, dataset_norm.T)
        
        # Get top-k for each query
        k = min(self.config.get('reorder_num_neighbors', 100), len(self.dataset))
        if k > len(self.dataset):
            k = len(self.dataset)
        
        neighbors = np.zeros((len(queries), k), dtype=np.int32)
        distances = np.zeros((len(queries), k))
        
        for i, sim_row in enumerate(similarities):
            # Get top-k indices (highest similarities first)
            top_k_idx = np.argsort(sim_row)[-k:][::-1]
            neighbors[i] = top_k_idx
            distances[i] = sim_row[top_k_idx]
        
        logger.info("Batch search completed")
        return neighbors, distances