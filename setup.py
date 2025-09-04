"""Setup configuration for patent-occupation-matcher package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="patent-occupation-matcher",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A system for matching AI patents with occupational tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/patent-occupation-matcher",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "patent-matcher=scripts.run_pipeline:main",
        ],
    },
)