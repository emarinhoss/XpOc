#!/usr/bin/env python
"""Script to run the patent-occupation matching pipeline."""
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.main_pipeline import PatentOccupationPipeline
from src.utils.logger import setup_logging


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run patent-occupation matching pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing pipeline...")
        pipeline = PatentOccupationPipeline(args.config)
        
        logger.info("Running pipeline...")
        pipeline.run()
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()