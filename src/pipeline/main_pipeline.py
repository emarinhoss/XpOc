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
        
    # ... rest of the class remains the same ...