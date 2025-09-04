.PHONY: install test clean run format lint

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage

run:
	python scripts/run_pipeline.py --config config/config.yaml

format:
	black src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/

download-data:
	python scripts/download_data.py

docker-build:
	docker build -t patent-matcher -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up