"""Generic embedder supporting multiple HuggingFace models."""
import numpy as np
import torch
from typing import List, Optional, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GenericEmbedder:
    """
    Generic embedder that supports multiple HuggingFace model types.

    Supports:
    - sentence-transformers models (e.g., anferico/bert-for-patents)
    - AutoModel + AutoTokenizer models (e.g., google/embeddinggemma-300m)
    - Custom pooling strategies (mean, cls, max)
    """

    SUPPORTED_MODEL_TYPES = ['sentence-transformers', 'auto-model']
    POOLING_STRATEGIES = ['mean', 'cls', 'max']

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedder with configuration.

        Args:
            config: Configuration dictionary with keys:
                - model_name: HuggingFace model name
                - model_type: 'sentence-transformers' or 'auto-model'
                - device: 'cpu' or 'cuda'
                - batch_size: Batch size for encoding
                - normalize: Whether to normalize embeddings
                - pooling: Pooling strategy ('mean', 'cls', 'max') for auto-model
                - max_length: Maximum sequence length
        """
        embedding_config = config.get('embedding', config)

        self.model_name = embedding_config['model_name']
        self.model_type = embedding_config.get('model_type', 'sentence-transformers')
        self.device = embedding_config.get('device', 'cpu')
        self.batch_size = embedding_config.get('batch_size', 32)
        self.normalize = embedding_config.get('normalize', True)
        self.pooling = embedding_config.get('pooling', 'mean')
        self.max_length = embedding_config.get('max_length', 512)

        # Validate configuration
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {self.model_type}. "
                           f"Supported types: {self.SUPPORTED_MODEL_TYPES}")

        if self.pooling not in self.POOLING_STRATEGIES:
            raise ValueError(f"Unsupported pooling: {self.pooling}. "
                           f"Supported strategies: {self.POOLING_STRATEGIES}")

        logger.info(f"Initializing {self.model_type} model: {self.model_name}")
        logger.info(f"Device: {self.device}, Batch size: {self.batch_size}, "
                   f"Normalize: {self.normalize}, Pooling: {self.pooling}")

        self._load_model()

    def _load_model(self):
        """Load the model based on model_type."""
        if self.model_type == 'sentence-transformers':
            self._load_sentence_transformer()
        elif self.model_type == 'auto-model':
            self._load_auto_model()

    def _load_sentence_transformer(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = None  # sentence-transformers handles tokenization internally
            logger.info("Sentence-transformers model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers")

    def _load_auto_model(self):
        """Load model using AutoModel and AutoTokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()

            logger.info("AutoModel loaded successfully")
        except ImportError:
            raise ImportError("transformers not installed. "
                            "Install with: pip install transformers")

    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply pooling strategy to token embeddings.

        Args:
            token_embeddings: Token-level embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling == 'cls':
            # Use [CLS] token (first token)
            return token_embeddings[:, 0, :]

        elif self.pooling == 'mean':
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == 'max':
            # Max pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Mask out padding
            return torch.max(token_embeddings, dim=1)[0]

    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings [num_texts, embedding_dim]
        """
        if self.model_type == 'sentence-transformers':
            return self._encode_sentence_transformers(texts, show_progress)
        elif self.model_type == 'auto-model':
            return self._encode_auto_model(texts, show_progress)

    def _encode_sentence_transformers(
        self,
        texts: List[str],
        show_progress: bool
    ) -> np.ndarray:
        """Encode using sentence-transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )

        if not self.normalize:
            # Apply normalization if needed and not done by model
            embeddings = self._normalize_embeddings(embeddings)

        return embeddings

    def _encode_auto_model(
        self,
        texts: List[str],
        show_progress: bool
    ) -> np.ndarray:
        """Encode using AutoModel with custom pooling."""
        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                # Forward pass
                outputs = self.model(**encoded)

                # Get token embeddings (usually from last_hidden_state)
                if hasattr(outputs, 'last_hidden_state'):
                    token_embeddings = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states'):
                    token_embeddings = outputs.hidden_states[-1]
                else:
                    # Fallback: try to get the first output tensor
                    token_embeddings = outputs[0]

                # Pool embeddings
                pooled = self._pool_embeddings(token_embeddings, encoded['attention_mask'])

                all_embeddings.append(pooled.cpu().numpy())

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        # Normalize if requested
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        return embeddings

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)  # Avoid division by zero
        return embeddings / norms

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        if self.model_type == 'sentence-transformers':
            return self.model.get_sentence_embedding_dimension()
        else:
            # Encode a dummy text to get dimension
            dummy_embedding = self.encode_texts(["test"], show_progress=False)
            return dummy_embedding.shape[1]
