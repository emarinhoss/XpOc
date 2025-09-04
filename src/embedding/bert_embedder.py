"""Module for generating BERT embeddings."""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BertEmbedder:
    """Handles BERT embedding generation for text."""
    
    def __init__(self, config: dict):
        """Initialize BERT embedder with configuration."""
        self.config = config
        self.model_name = config['embedding']['model_name']
        self.device = config['embedding']['device']
        self.batch_size = config['embedding']['batch_size']
        self.normalize = config['embedding']['normalize']
        
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        logger.info(f"Encoding {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)
            
        return embeddings
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        logger.info("Embeddings normalized")
        return normalized
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings