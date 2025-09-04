"""Utilities for embedding operations."""
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingUtils:
    """Utility functions for embedding operations."""
    
    @staticmethod
    def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def batch_similarity(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute pairwise similarities between query and target embeddings."""
        # Normalize if not already normalized
        queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        targets_norm = targets / np.linalg.norm(targets, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(queries_norm, targets_norm.T)
        return similarity_matrix
    
    @staticmethod
    def get_top_k_similar(
        query_embedding: np.ndarray,
        target_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k most similar embeddings."""
        similarities = np.dot(target_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices, top_k_scores
    
    @staticmethod
    def reduce_dimensions(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {n_components} dimensions")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")
        
        return reduced_embeddings
    
    @staticmethod
    def save_embedding_metadata(
        embeddings: np.ndarray,
        texts: List[str],
        filepath: str
    ):
        """Save embeddings with metadata."""
        import json
        
        metadata = {
            'shape': embeddings.shape,
            'num_texts': len(texts),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 1,
            'sample_texts': texts[:5] if len(texts) > 5 else texts
        }
        
        base_path = Path(filepath).parent
        base_name = Path(filepath).stem
        
        # Save embeddings
        np.save(f"{base_path}/{base_name}_embeddings.npy", embeddings)
        
        # Save metadata
        with open(f"{base_path}/{base_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved embeddings and metadata to {base_path}")