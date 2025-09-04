# Patent-Occupation Matcher

A system for matching AI patents with occupational tasks using BERT embeddings and ScaNN similarity search.

## Overview
This project bridges technological innovation (patents) with labor market structure (O*NET occupational tasks) to understand AI's impact on different types of work.

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/patent-occupation-matcher.git
cd patent-occupation-matcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Run pipeline
python scripts/run_pipeline.py --config config/config.yaml