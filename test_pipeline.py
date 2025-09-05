#!/usr/bin/env python
"""Quick test of the pipeline with minimal data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

# Test imports
try:
    from src.pipeline.main_pipeline import PatentOccupationPipeline
    print("✓ Pipeline imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test configuration
config_path = "config/config.yaml"
if not Path(config_path).exists():
    print("Creating minimal config...")
    Path("config").mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        f.write("""
data:
  patents:
    raw_path: "data/raw/patents/"
    processed_path: "data/processed/patents.csv"
    ai_categories_path: "data/raw/categories/patent_categories.xlsx"
  onet:
    raw_path: "data/raw/onet/"
    task_ratings_path: "data/raw/onet/Task_Ratings-3.xlsx"
    processed_path: "data/processed/onet_tasks.csv"
  
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for testing
  device: "cpu"
  batch_size: 4
  normalize: true

matching:
  reorder_num_neighbors: 10
  
processing:
  years_to_process: [2021, 2022]
  ai_topic_filter: 1
  top_k_matches: 5

output:
  results_path: "data/processed/matches/"
  format: "csv"
""")
    print("✓ Config created")

# Test pipeline initialization
try:
    pipeline = PatentOccupationPipeline(config_path)
    print("✓ Pipeline initialized successfully")
    
    # Run with minimal data
    print("\nRunning pipeline with sample data...")
    pipeline.run()
    print("✓ Pipeline completed successfully!")
    
except Exception as e:
    print(f"✗ Pipeline error: {e}")
    import traceback
    traceback.print_exc()