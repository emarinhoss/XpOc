"""Module for ScaNN-based similarity matching."""
import scann
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ScannMatcher:
    """Handles similarity matching using ScaNN."""
    
    def __init__(self, config: dict):
        """Initialize ScaNN matcher with configuration."""
        self.config = config['matching']
        self.searcher = None
        
    def build_index(self, dataset: np.ndarray):
        """Build ScaNN search index from dataset."""
        logger.info(f"Building ScaNN index for {len(dataset)} items")
        
        self.searcher = scann.scann_ops_pybind.builder(
            dataset, 
            self.config['reorder_num_neighbors'], 
            "dot_product"
        ).tree(
            num_leaves=self.config['num_leaves'],
            num_leaves_to_search=self.config['num_leaves_to_search'],
            training_sample_size=self.config['training_sample_size']
        ).score_ah(
            2, 
            anisotropic_quantization_threshold=self.config['anisotropic_quantization_threshold']
        ).reorder(
            self.config['reorder_num_neighbors']
        ).build()
        
        logger.info("ScaNN index built successfully")
        
    def search_batch(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batch similarity search."""
        if self.searcher is None:
            raise ValueError("Index not built. Run build_index first.")
        
        logger.info(f"Searching for {len(queries)} queries")
        
        neighbors, distances = self.searcher.search_batched(
            queries,
            leaves_to_search=self.config['num_leaves_to_search'],
            pre_reorder_num_neighbors=self.config['pre_reorder_num_neighbors']
        )
        
        logger.info("Batch search completed")
        return neighbors, distances