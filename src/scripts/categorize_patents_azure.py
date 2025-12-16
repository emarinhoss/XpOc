#!/usr/bin/env python
"""
Zero-shot patent categorization using Azure OpenAI embeddings.

This version uses Azure OpenAI embeddings API instead of local models,
with built-in rate limiting to handle API constraints.

Rate limiting:
- Default: 500 queries per second
- Configurable via --rate-limit parameter
- Automatic batching and throttling
"""
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque

import pandas as pd
import numpy as np
from tqdm import tqdm
import openai
from openai import AzureOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- AI Category Definitions ---
CATEGORIES_WITH_DEFINITIONS = {
    "computer vision": "methods to understand images and videos, including object detection, image recognition, facial recognition, and visual processing",
    "evolutionary computation": "methods mimicking evolution to solve problems using genetic algorithms, evolutionary strategies, and nature-inspired optimization",
    "AI hardware": "physical hardware designed specifically to implement AI software, including neural network processors, AI accelerators, and specialized chips",
    "knowledge processing": "methods to represent and derive new facts from knowledge bases, including ontologies, reasoning systems, and expert systems",
    "machine learning": "algorithms that learn from data, including supervised learning, unsupervised learning, reinforcement learning, and statistical models",
    "NLP": "methods to understand and generate human language, including text processing, sentiment analysis, machine translation, and language models",
    "planning and control": "methods to determine and execute plans to achieve goals, including robotics control, automated planning, and decision systems",
    "speech recognition": "methods to understand speech and generate responses, including voice recognition, speech-to-text, and audio processing"
}


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate_limit: int):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum queries per second
        """
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = time.time()
        self.min_interval = 1.0 / rate_limit if rate_limit > 0 else 0

    def acquire(self, num_tokens: int = 1):
        """
        Acquire tokens, blocking if necessary.

        Args:
            num_tokens: Number of tokens to acquire (default: 1)
        """
        while True:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(self.rate_limit, self.tokens + elapsed * self.rate_limit)
            self.last_update = now

            if self.tokens >= num_tokens:
                self.tokens -= num_tokens
                return

            # Need to wait
            sleep_time = (num_tokens - self.tokens) / self.rate_limit
            time.sleep(sleep_time)


class AzureEmbedder:
    """Azure OpenAI embedder with rate limiting."""

    def __init__(
        self,
        deployment_name: str = "text-embedding-ada-002",
        rate_limit: int = 500,
        max_retries: int = 3,
        batch_size: int = 16
    ):
        """
        Initialize Azure OpenAI embedder.

        Args:
            deployment_name: Azure deployment name for embeddings
            rate_limit: Maximum queries per second
            max_retries: Number of retry attempts
            batch_size: Number of texts to embed per API call
        """
        self.deployment_name = deployment_name
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.rate_limiter = RateLimiter(rate_limit)

        # Initialize Azure OpenAI client
        logger.info("Initializing Azure OpenAI client...")
        self.client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        )

        logger.info(f"Azure embedder initialized (rate limit: {rate_limit} QPS, batch size: {batch_size})")

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Call Azure OpenAI embeddings API with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        for attempt in range(self.max_retries):
            try:
                # Rate limiting: acquire tokens based on batch size
                self.rate_limiter.acquire(len(texts))

                # Call Azure OpenAI
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.deployment_name
                )

                # Extract embeddings
                embeddings = [item.embedding for item in response.data]
                return embeddings

            except openai.RateLimitError as e:
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error in embedding API call (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(f"Failed to get embeddings after {self.max_retries} retries")

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode multiple texts into embeddings.

        Args:
            texts: List of texts to encode
            show_progress: Show progress bar

        Returns:
            NumPy array of embeddings (n_texts, embedding_dim)
        """
        if len(texts) == 0:
            return np.array([])

        all_embeddings = []

        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", total=len(texts) // self.batch_size + 1)

        for i in iterator:
            batch = texts[i:i + self.batch_size]

            # Filter out empty texts
            valid_batch = []
            valid_indices = []
            for j, text in enumerate(batch):
                if text and text.strip():
                    valid_batch.append(text)
                    valid_indices.append(j)

            # Get embeddings for valid texts
            if valid_batch:
                batch_embeddings = self._embed_with_retry(valid_batch)

                # Fill in embeddings (use zero vector for empty texts)
                full_batch_embeddings = []
                valid_idx = 0
                for j in range(len(batch)):
                    if j in valid_indices:
                        full_batch_embeddings.append(batch_embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        # Use zero vector for empty text
                        full_batch_embeddings.append([0.0] * len(batch_embeddings[0]))

                all_embeddings.extend(full_batch_embeddings)
            else:
                # All texts in batch were empty
                # Use zero vectors (get dimension from first valid embedding)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                    dim = 1536  # Default for ada-002
                all_embeddings.extend([[0.0] * dim] * len(batch))

        embeddings_array = np.array(all_embeddings)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings_array = embeddings_array / norms

        return embeddings_array


class AzureZeroShotPatentClassifier:
    """Zero-shot classifier for patents using Azure OpenAI embeddings."""

    def __init__(
        self,
        deployment_name: str = "text-embedding-ada-002",
        rate_limit: int = 500,
        batch_size: int = 16
    ):
        """
        Initialize the classifier with Azure OpenAI configuration.

        Args:
            deployment_name: Azure deployment name for embeddings
            rate_limit: Maximum queries per second
            batch_size: Number of texts per API call
        """
        logger.info("Initializing Azure embedder for zero-shot classification...")
        self.embedder = AzureEmbedder(
            deployment_name=deployment_name,
            rate_limit=rate_limit,
            batch_size=batch_size
        )
        self.categories = list(CATEGORIES_WITH_DEFINITIONS.keys())
        self.category_embeddings = None
        self._encode_categories()

    def _encode_categories(self):
        """Pre-compute embeddings for all category descriptions."""
        logger.info(f"Encoding {len(self.categories)} category descriptions...")
        category_texts = list(CATEGORIES_WITH_DEFINITIONS.values())
        self.category_embeddings = self.embedder.encode_texts(
            category_texts,
            show_progress=False
        )
        logger.info("Category embeddings computed successfully")

    def classify_single(self, text: str) -> Tuple[str, float]:
        """
        Classify a single patent text using zero-shot classification.

        Args:
            text: Patent title and/or abstract

        Returns:
            Tuple of (category_name, confidence_score)
        """
        if not text or text.strip() == "":
            return "missing_data", 0.0

        # Encode the patent text
        patent_embedding = self.embedder.encode_texts([text], show_progress=False)[0]

        # Compute cosine similarity with all categories
        similarities = np.dot(self.category_embeddings, patent_embedding)

        # Get the category with highest similarity
        best_idx = np.argmax(similarities)
        best_category = self.categories[best_idx]
        confidence = float(similarities[best_idx])

        return best_category, confidence

    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Classify multiple patents in batch for efficiency.

        Args:
            texts: List of patent texts (title + abstract)

        Returns:
            List of tuples (category_name, confidence_score)
        """
        if len(texts) == 0:
            return []

        # Encode all patents
        logger.info(f"Encoding {len(texts)} patent texts...")
        patent_embeddings = self.embedder.encode_texts(texts, show_progress=True)

        # Compute similarities for all patents at once
        logger.info("Computing similarities with categories...")
        similarities = np.dot(patent_embeddings, self.category_embeddings.T)

        # Get best category for each patent
        results = []
        for i in range(len(texts)):
            if not texts[i] or texts[i].strip() == "":
                results.append(("missing_data", 0.0))
            else:
                best_idx = np.argmax(similarities[i])
                best_category = self.categories[best_idx]
                confidence = float(similarities[i, best_idx])
                results.append((best_category, confidence))

        return results


def main(args):
    """Main function to run Azure zero-shot classification."""

    logger.info("="*60)
    logger.info("AZURE ZERO-SHOT PATENT CATEGORIZATION")
    logger.info("="*60)
    logger.info(f"Deployment: {args.deployment_name}")
    logger.info(f"Rate limit: {args.rate_limit} queries/second")
    logger.info(f"Batch size: {args.batch_size}")

    # Load input data
    logger.info(f"\nLoading patents from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} patents")

    # Prepare text for classification
    if args.text_column:
        # Use specified column
        patent_texts = df[args.text_column].fillna("").astype(str).tolist()
    else:
        # Combine title and abstract
        if 'application_title' in df.columns and 'application_abstract' in df.columns:
            patent_texts = (
                df['application_title'].fillna("").astype(str) + ". " +
                df['application_abstract'].fillna("").astype(str)
            ).tolist()
        elif 'title' in df.columns and 'abstract' in df.columns:
            patent_texts = (
                df['title'].fillna("").astype(str) + ". " +
                df['abstract'].fillna("").astype(str)
            ).tolist()
        else:
            logger.error("Could not find title/abstract columns. Use --text-column to specify.")
            return

    # Initialize classifier
    logger.info("\nInitializing Azure classifier...")
    classifier = AzureZeroShotPatentClassifier(
        deployment_name=args.deployment_name,
        rate_limit=args.rate_limit,
        batch_size=args.batch_size
    )

    # Classify patents
    logger.info("\nClassifying patents...")
    start_time = time.time()

    results = classifier.classify_batch(patent_texts)

    elapsed_time = time.time() - start_time
    rate = len(patent_texts) / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Classified {len(patent_texts)} patents in {elapsed_time:.1f}s ({rate:.1f} patents/sec)")

    # Add results to dataframe
    df[args.output_category_column] = [cat for cat, conf in results]
    df[args.output_confidence_column] = [conf for cat, conf in results]

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving results to: {output_path}")
    df.to_csv(output_path, index=False)

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("CATEGORIZATION SUMMARY")
    logger.info("="*60)

    category_counts = df[args.output_category_column].value_counts()
    logger.info("\nCategory distribution:")
    for category, count in category_counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  {category}: {count} ({pct:.1f}%)")

    avg_confidence = df[args.output_confidence_column].mean()
    logger.info(f"\nAverage confidence: {avg_confidence:.3f}")

    # Confidence distribution
    high_conf = (df[args.output_confidence_column] >= 0.7).sum()
    med_conf = ((df[args.output_confidence_column] >= 0.5) & (df[args.output_confidence_column] < 0.7)).sum()
    low_conf = (df[args.output_confidence_column] < 0.5).sum()

    logger.info(f"\nConfidence distribution:")
    logger.info(f"  High (≥0.7): {high_conf} ({100*high_conf/len(df):.1f}%)")
    logger.info(f"  Medium (0.5-0.7): {med_conf} ({100*med_conf/len(df):.1f}%)")
    logger.info(f"  Low (<0.5): {low_conf} ({100*low_conf/len(df):.1f}%)")

    logger.info("\n" + "="*60)
    logger.info("CATEGORIZATION COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Categorize patents using Azure OpenAI embeddings (zero-shot classification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables required:
  AZURE_OPENAI_API_KEY      - Your Azure OpenAI API key
  OPENAI_API_VERSION        - API version (e.g., "2023-05-15")
  AZURE_OPENAI_ENDPOINT     - Azure endpoint URL

Examples:
  # Basic usage (500 QPS default)
  python categorize_patents_azure.py \\
      --input-file data/patents.csv \\
      --output-file data/patents_categorized.csv

  # Custom rate limit (100 QPS)
  python categorize_patents_azure.py \\
      --input-file data/patents.csv \\
      --output-file data/patents_categorized.csv \\
      --rate-limit 100

  # Different deployment and batch size
  python categorize_patents_azure.py \\
      --input-file data/patents.csv \\
      --output-file data/patents_categorized.csv \\
      --deployment-name text-embedding-3-small \\
      --batch-size 32 \\
      --rate-limit 1000

  # Use specific text column
  python categorize_patents_azure.py \\
      --input-file data/patents.csv \\
      --output-file data/patents_categorized.csv \\
      --text-column combined_text
        """
    )

    # Input/Output
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file with patents"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save categorized patents CSV"
    )

    # Text configuration
    parser.add_argument(
        "--text-column",
        type=str,
        help="Column containing patent text (default: combine title + abstract)"
    )

    # Azure configuration
    parser.add_argument(
        "--deployment-name",
        type=str,
        default="text-embedding-ada-002",
        help="Azure OpenAI deployment name for embeddings (default: text-embedding-ada-002)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=500,
        help="Maximum queries per second (default: 500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of texts to embed per API call (default: 16)"
    )

    # Output configuration
    parser.add_argument(
        "--output-category-column",
        type=str,
        default="ai_category",
        help="Name for category output column (default: ai_category)"
    )
    parser.add_argument(
        "--output-confidence-column",
        type=str,
        default="confidence",
        help="Name for confidence output column (default: confidence)"
    )

    args = parser.parse_args()

    # Validate environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these environment variables before running the script.")
        sys.exit(1)

    main(args)
