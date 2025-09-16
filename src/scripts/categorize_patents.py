import os
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
CATEGORIES_WITH_DEFINITIONS = {
    "computer vision": "methods to understand images and videos",
    "evolutionary computation": "methods mimicking evolution to solve problems",
    "AI hardware": "physical hardware designed specifically to implement AI software",
    "knowledge processing": "methods to represent and derive new facts from knowledge bases",
    "machine learning": "algorithms that learn from data",
    "NLP": "methods to understand and generate human language",
    "planning and control": "methods to determine and execute plans to achieve goals",
    "speech recognition": "methods to understand speech and generate responses"
}
CATEGORIES = list(CATEGORIES_WITH_DEFINITIONS.keys())

PROMPT_TEMPLATE = """
Classify the following patent into one of the specified AI categories.
Respond with ONLY the category name from the list.

Categories:
{categories_text}

Patent Title: {title}
Patent Abstract: {abstract}

Category:
"""

def get_categorization_prompt():
    """Formats the categories and their definitions for the prompt."""
    categories_text = "\n".join([f"- {name}: {desc}" for name, desc in CATEGORIES_WITH_DEFINITIONS.items()])
    return PROMPT_TEMPLATE.format(
        categories_text=categories_text,
        title="{title}",
        abstract="{abstract}"
    )

def classify_patent(client: OpenAI, title: str, abstract: str, model_name: str) -> str:
    """
    Classifies a single patent using the OpenAI API.

    Args:
        client: The OpenAI API client.
        title: The title of the patent.
        abstract: The abstract of the patent.
        model_name: The name of the model or deployment to use.

    Returns:
        The classified category name, or "classification_failed" on error.
    """
    prompt = get_categorization_prompt().format(title=title, abstract=abstract)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert in AI and patent classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        category = response.choices[0].message.content.strip().lower()
        # Basic validation to ensure the response is one of the categories
        if category in CATEGORIES:
            return category
        else:
            logging.warning(f"API returned an unexpected category: '{category}'. Defaulting to 'unknown'.")
            return "unknown"
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return "classification_failed"

def main(args):
    """Main function to run the patent categorization script."""
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        logging.error(f"API key not found. Please set the '{args.api_key_env}' environment variable.")
        return

    if args.use_azure:
        logging.info("Using Azure OpenAI client.")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            logging.error("Azure endpoint not found. Please set the 'AZURE_OPENAI_ENDPOINT' environment variable.")
            return

        client = AzureOpenAI(
            api_key=api_key,
            api_version=args.azure_api_version,
            azure_endpoint=azure_endpoint
        )
    else:
        logging.info("Using standard OpenAI client.")
        client = OpenAI(api_key=api_key)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input data
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded {len(df)} patents from {input_path}")
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}")
        return

    # Filter by year if start_year is provided
    if args.start_year:
        if 'year' not in df.columns:
            logging.error("Cannot filter by year: 'year' column not found in the input file.")
            return

        original_count = len(df)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df = df[df['year'] >= args.start_year]
        logging.info(f"Filtered patents from {original_count} to {len(df)} based on start year {args.start_year}.")

    # Create new category column if it doesn't exist
    if args.category_column not in df.columns:
        df[args.category_column] = None

    # Identify patents that need categorization
    # Load output file if it exists to resume progress
    if output_path.exists():
        logging.info(f"Output file found at {output_path}. Resuming from previous run.")
        df_processed = pd.read_csv(output_path)
        # Update the main dataframe with already processed results
        df.update(df_processed)

    to_process_df = df[df[args.category_column].isnull()]

    if len(to_process_df) == 0:
        logging.info("All patents have already been categorized. Exiting.")
        return

    logging.info(f"Found {len(to_process_df)} patents to categorize.")

    # Process in batches
    for index, row in tqdm(to_process_df.iterrows(), total=len(to_process_df), desc="Categorizing Patents"):
        title = row.get(args.title_column, "")
        abstract = row.get(args.abstract_column, "")

        if not title and not abstract:
            logging.warning(f"Skipping patent index {index} due to missing title and abstract.")
            df.loc[index, args.category_column] = "missing_data"
            continue

        model_name = args.azure_deployment if args.use_azure else "gpt-3.5-turbo"
        category = classify_patent(client, title, abstract, model_name)
        df.loc[index, args.category_column] = category

        # Save progress periodically
        if (index + 1) % args.batch_size == 0:
            logging.info(f"Saving progress to {output_path}...")
            df.to_csv(output_path, index=False)
            logging.info("Progress saved.")

        time.sleep(args.rate_limit_delay) # Respect rate limits

    # Final save
    logging.info("Categorization complete. Saving final results...")
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved categorized patents to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize patents using the OpenAI API.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/patents.csv",
        help="Path to the input patent CSV file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/patents_categorized.csv",
        help="Path to save the output CSV file with categories."
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Name of the environment variable containing the OpenAI API key."
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default="ai_category",
        help="Name of the column to store the AI category."
    )
    parser.add_argument(
        "--title-column",
        type=str,
        default="application_title",
        help="Name of the column containing the patent title."
    )
    parser.add_argument(
        "--abstract-column",
        type=str,
        default="application_abstract",
        help="Name of the column containing the patent abstract."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of patents to process before saving progress."
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls to avoid rate limiting."
    )
    # --- Azure Specific Arguments ---
    parser.add_argument(
        "--use-azure",
        action="store_true",
        help="Use Azure OpenAI service instead of the default OpenAI."
    )
    parser.add_argument(
        "--azure-deployment",
        type=str,
        default=None,
        help="Name of the Azure OpenAI deployment. Required if --use-azure is set."
    )
    parser.add_argument(
        "--azure-api-version",
        type=str,
        default="2023-12-01-preview",
        help="Azure OpenAI API version."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Filter patents to include only those from this year and later."
    )

    args = parser.parse_args()

    if args.use_azure and not args.azure_deployment:
        parser.error("--azure-deployment is required when --use-azure is set.")

    main(args)
