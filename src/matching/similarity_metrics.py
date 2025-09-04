"""Similarity metrics for matching."""
import numpy as np
from typing import Callable, Optional
from scipy.spatial.distance import euclidean, manhattan
import logging

logger = logging.getLogger(__name__)


class SimilarityMetrics:
    """Collection of similarity metrics."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors."""
        return euclidean(vec1, vec2)
    
    @staticmethod
    def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Manhattan distance between two vectors."""
        return manhattan(vec1, vec2)
    
    @staticmethod
    def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate dot product between two vectors."""
        return np.dot(vec1, vec2)
    
    @staticmethod
    def get_metric(metric_name: str) -> Callable:
        """Get similarity metric function by name."""
        metrics = {
            'cosine': SimilarityMetrics.cosine_similarity,
            'euclidean': SimilarityMetrics.euclidean_distance,
            'manhattan': SimilarityMetrics.manhattan_distance,
            'dot': SimilarityMetrics.dot_product
        }
        
        if metric_name not in metrics:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics.keys())}")
        
        return metrics[metric_name]


class SimilarityAnalyzer:
    """Analyze similarity distributions."""
    
    @staticmethod
    def analyze_distribution(similarities: np.ndarray) -> dict:
        """Analyze similarity score distribution."""
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'q1': float(np.percentile(similarities, 25)),
            'q3': float(np.percentile(similarities, 75))
        }
    
    @staticmethod
    def filter_by_threshold(
        indices: np.ndarray,
        scores: np.ndarray,
        threshold: float
    ) -> tuple:
        """Filter matches by similarity threshold."""
        mask = scores >= threshold
        filtered_indices = indices[mask]
        filtered_scores = scores[mask]
        
        logger.info(f"Filtered {len(indices)} to {len(filtered_indices)} matches with threshold {threshold}")
        return filtered_indices, filtered_scores