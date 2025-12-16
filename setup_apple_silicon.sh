#!/bin/bash
# Quick setup script for Apple Silicon (M2 Pro)

echo "=================================="
echo "Apple Silicon (M2 Pro) Setup"
echo "=================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon (M1/M2/M3) only"
    echo "   Detected: $(uname -m)"
    exit 1
fi

echo "✅ Detected macOS on Apple Silicon"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found"
    echo ""
    echo "Please install Miniforge first:"
    echo "  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
    echo "  bash Miniforge3-MacOSX-arm64.sh"
    echo ""
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^xpoc-m2 "; then
    echo "⚠️  Environment 'xpoc-m2' already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n xpoc-m2 -y
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Create environment
echo "Creating conda environment 'xpoc-m2'..."
echo "This may take 5-10 minutes..."
echo ""

if conda env create -f environment_apple_silicon.yml; then
    echo ""
    echo "✅ Environment created successfully!"
else
    echo ""
    echo "❌ Failed to create environment"
    exit 1
fi

# Activation instructions
echo ""
echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate xpoc-m2"
echo ""
echo "2. Test GPU acceleration:"
echo "   python tests/test_mps_gpu.py"
echo ""
echo "3. Start processing patents:"
echo "   python src/scripts/categorize_patents_zeroshot.py \\"
echo "       --input-file data/patents.csv \\"
echo "       --output-file data/patents_categorized.csv \\"
echo "       --device mps \\"
echo "       --batch-size 32"
echo ""
echo "For more details, see: docs/apple_silicon_setup.md"
echo ""
