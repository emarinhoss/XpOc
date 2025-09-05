#!/usr/bin/env python
"""Quick setup script to prepare the environment and run a test."""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw/patents',
        'data/raw/onet', 
        'data/raw/categories',
        'data/processed',
        'data/processed/embeddings',
        'data/processed/matches',
        'reports'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")

def main():
    print("Setting up patent-occupation matcher...")
    print("-" * 40)
    
    # Setup directories
    setup_directories()
    
    print("\n✓ Setup complete!")
    print("\nYou can now run the pipeline with:")
    print("  python scripts/run_pipeline.py --config config/config.yaml")

if __name__ == "__main__":
    main()