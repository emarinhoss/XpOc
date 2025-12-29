#!/usr/bin/env python
"""
Memory-efficient zero-shot patent categorization using Azure OpenAI embeddings.

This version processes patents in chunks to handle datasets with millions of records
without running out of memory.

Key features:
- Streams data in chunks (default 10,000 patents at a time)
- Appends results incrementally
- Supports resumption from partially completed runs
- Azure OpenAI embeddings with rate limiting
- Shows progress across chunks
"""
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple

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
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key,
            api_version=os.environ["OPENAI_API_VERSION"],
            default_headers={"Ocp-Apim-Subscription-Key": api_key}
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

    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
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

        # Encode all patent texts
        patent_embeddings = self.embedder.encode_texts(texts, show_progress=False)

        # Compute similarities for all patents
        # Shape: (num_patents, num_categories)
        similarities = np.dot(patent_embeddings, self.category_embeddings.T)

        # Get best category for each patent
        best_indices = np.argmax(similarities, axis=1)
        confidences = similarities[np.arange(len(similarities)), best_indices]

        # Build results
        results = []
        for i in range(len(texts)):
            if not texts[i] or not texts[i].strip():
                results.append(("missing_data", 0.0))
            else:
                category = self.categories[best_indices[i]]
                confidence = float(confidences[i])
                results.append((category, confidence))

        return results


def combine_text_fields(row: pd.Series, title_col: str, abstract_col: str) -> str:
    """Combine title and abstract into single text for classification."""
    title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else ""
    abstract = str(row.get(abstract_col, "")) if pd.notna(row.get(abstract_col)) else ""

    # Combine with title given more weight
    combined = f"{title}. {abstract}".strip()
    return combined if combined != "." else ""


def count_total_rows(file_path: Path, start_year: int = None) -> int:
    """Count total rows that will be processed (for progress bar)."""
    logger.info("Counting total patents to process...")

    total = 0
    for chunk in pd.read_csv(file_path, chunksize=10000):
        if start_year is not None:
            # Try to find year column
            year_col = None
            for col_name in ['year', 'Year', 'filing_year', 'application_year', 'grant_year']:
                if col_name in chunk.columns:
                    year_col = col_name
                    break

            if year_col:
                chunk = chunk[chunk[year_col] >= start_year]

        total += len(chunk)

    logger.info(f"Total patents to process: {total:,}")
    return total


def main(args):
    """Main function to run chunked Azure zero-shot classification."""

    logger.info("="*60)
    logger.info("AZURE ZERO-SHOT PATENT CATEGORIZATION (CHUNKED)")
    logger.info("="*60)
    logger.info(f"Deployment: {args.deployment_name}")
    logger.info(f"Rate limit: {args.rate_limit} queries/second")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Chunk size: {args.chunk_size:,}")

    # Setup paths
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        return

    # Initialize classifier
    logger.info("\nInitializing Azure classifier...")
    classifier = AzureZeroShotPatentClassifier(
        deployment_name=args.deployment_name,
        rate_limit=args.rate_limit,
        batch_size=args.batch_size
    )

    # Count total rows for progress tracking
    total_patents = count_total_rows(input_path, args.start_year)

    # Check if output exists for resumption
    processed_count = 0
    if output_path.exists() and not args.overwrite:
        logger.info(f"Output file found. Checking existing progress...")
        existing_df = pd.read_csv(output_path)
        processed_count = len(existing_df)
        logger.info(f"Already processed: {processed_count:,} patents")

        if processed_count >= total_patents:
            logger.info("All patents already categorized. Exiting.")
            return

        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
        processed_count = 0

    # Process in chunks
    logger.info(f"\nProcessing patents in chunks of {args.chunk_size:,}...")
    logger.info(f"Output file: {output_path}")

    chunk_num = 0
    patents_processed = processed_count
    start_time = time.time()

    with tqdm(total=total_patents, initial=processed_count, desc="Overall progress", unit="patents") as pbar:
        for chunk in pd.read_csv(input_path, chunksize=args.chunk_size):
            chunk_num += 1

            # Apply year filtering if requested
            if args.start_year is not None:
                # Try to find a year column
                year_col = None
                for col_name in ['year', 'Year', 'filing_year', 'application_year', 'grant_year']:
                    if col_name in chunk.columns:
                        year_col = col_name
                        break

                if year_col:
                    chunk = chunk[chunk[year_col] >= args.start_year].copy()

            if len(chunk) == 0:
                continue

            # Skip if we've already processed this chunk
            if patents_processed > 0:
                if len(chunk) <= patents_processed:
                    patents_processed -= len(chunk)
                    pbar.update(len(chunk))
                    continue
                else:
                    # Partial chunk processed, skip those rows
                    chunk = chunk.iloc[patents_processed:].copy()
                    patents_processed = 0

            # Prepare text for classification
            if args.text_column and args.text_column in chunk.columns:
                # Use specified column
                patent_texts = chunk[args.text_column].fillna("").astype(str).tolist()
            else:
                # Combine title and abstract using specified column names
                if args.title_column not in chunk.columns:
                    logger.error(f"Title column '{args.title_column}' not found in data")
                    return
                if args.abstract_column not in chunk.columns:
                    logger.error(f"Abstract column '{args.abstract_column}' not found in data")
                    return

                patent_texts = (
                    chunk[args.title_column].fillna("").astype(str) + ". " +
                    chunk[args.abstract_column].fillna("").astype(str)
                ).tolist()

            # Classify patents in this chunk
            results = classifier.classify_batch(patent_texts)

            # Add results to chunk
            chunk[args.output_category_column] = [r[0] for r in results]
            chunk[args.output_confidence_column] = [r[1] for r in results]

            # Append to output file
            chunk.to_csv(output_path, mode=mode, header=header, index=False)

            # After first write, switch to append mode
            if mode == 'w':
                mode = 'a'
                header = False

            # Update progress
            pbar.update(len(chunk))

            # Calculate and log speed
            elapsed = time.time() - start_time
            patents_done = pbar.n - processed_count
            if patents_done > 0:
                rate = patents_done / elapsed
                remaining = total_patents - pbar.n
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600

                if chunk_num % 5 == 0:
                    logger.info(f"Chunk {chunk_num}: {pbar.n:,}/{total_patents:,} patents | "
                              f"Speed: {rate:.1f} patents/sec | ETA: {eta_hours:.1f}h")

    logger.info("\n" + "="*60)
    logger.info("CATEGORIZATION COMPLETE")
    logger.info("="*60)
    total_time = time.time() - start_time
    logger.info(f"Total patents processed: {total_patents:,}")
    logger.info(f"Total time: {total_time/3600:.1f} hours")
    logger.info(f"Average speed: {total_patents/total_time:.1f} patents/sec")
    logger.info(f"Output saved to: {output_path}")

    # Print category distribution summary (read last chunk to get idea)
    logger.info("\nSample category distribution (last chunk):")
    category_counts = chunk[args.output_category_column].value_counts()
    for category, count in category_counts.items():
        pct = 100 * count / len(chunk)
        logger.info(f"  {category}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Categorize large patent datasets using Azure OpenAI embeddings (memory-efficient chunked processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables required:
  AZURE_OPENAI_API_KEY      - Your Azure OpenAI API key
  OPENAI_API_VERSION        - API version (e.g., "2023-05-15")
  AZURE_OPENAI_ENDPOINT     - Azure endpoint URL

Examples:
  # Basic usage with chunked processing
  python categorize_patents_azure_chunked.py \\
      --input-file data/patents.csv \\
      --output-file data/patents_categorized.csv \\
      --deployment-name text-embedding-ada-002-v2-base

  # Custom settings
  python categorize_patents_azure_chunked.py \\
      --input-file data/patents.csv \\
      --output-file data/patents_categorized.csv \\
      --deployment-name text-embedding-3-small-v1-base \\
      --chunk-size 5000 \\
      --batch-size 32 \\
      --rate-limit 100 \\
      --start-year 2021
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of patents to process per chunk (default: 10000)"
    )

    # Text configuration
    parser.add_argument(
        "--text-column",
        type=str,
        help="Column containing patent text (default: combine title + abstract)"
    )
    parser.add_argument(
        "--title-column",
        type=str,
        default="application_title",
        help="Name of column containing patent title (default: application_title)"
    )
    parser.add_argument(
        "--abstract-column",
        type=str,
        default="application_abstract",
        help="Name of column containing patent abstract (default: application_abstract)"
    )

    # Filtering
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Only process patents from this year onwards (requires a year column)"
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
        "--category-column",
        type=str,
        dest="output_category_column",
        help="Alias for --output-category-column"
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
        logger.error("\nRun ./check_azure_env.sh to check your configuration.")
        sys.exit(1)

    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Progress has been saved and can be resumed.")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise
