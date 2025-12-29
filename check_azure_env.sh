#!/bin/bash
# Script to check Azure OpenAI environment variables

echo "=========================================="
echo "Azure OpenAI Environment Variable Check"
echo "=========================================="
echo ""

# Function to check if variable is set
check_var() {
    local var_name=$1
    if [ -z "${!var_name}" ]; then
        echo "❌ $var_name: NOT SET"
        return 1
    else
        # Show first 10 characters only for security
        local value="${!var_name}"
        if [[ $var_name == *"KEY"* ]] || [[ $var_name == *"key"* ]]; then
            echo "✅ $var_name: ${value:0:10}... (hidden for security)"
        else
            echo "✅ $var_name: $value"
        fi
        return 0
    fi
}

echo "Required variables:"
echo ""
check_var "AZURE_OPENAI_API_KEY"
has_key=$?
check_var "OPENAI_API_VERSION"
has_version=$?
check_var "AZURE_OPENAI_ENDPOINT"
has_endpoint=$?

echo ""
echo "=========================================="

if [ $has_key -eq 0 ] && [ $has_version -eq 0 ] && [ $has_endpoint -eq 0 ]; then
    echo "✅ All required variables are set!"
    echo ""
    echo "You can run the Azure categorization script."
else
    echo "❌ Missing required variables!"
    echo ""
    echo "To set them, add these to your ~/.zshrc or ~/.bash_profile:"
    echo ""
    echo 'export AZURE_OPENAI_API_KEY="your-api-key-here"'
    echo 'export OPENAI_API_VERSION="2024-02-15-preview"'
    echo 'export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"'
    echo ""
    echo "Then run: source ~/.zshrc  (or source ~/.bash_profile)"
    echo ""
    echo "OR set them temporarily in your current session:"
    echo ""
    echo 'export AZURE_OPENAI_API_KEY="your-api-key-here"'
    echo 'export OPENAI_API_VERSION="2024-02-15-preview"'
    echo 'export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"'
fi

echo ""
echo "=========================================="
echo "Alternative: Use local GPU (FREE!)"
echo "=========================================="
echo ""
echo "Instead of Azure API, you can use your M2 Pro GPU:"
echo ""
echo "python src/scripts/categorize_patents_zeroshot.py \\"
echo "    --input-file ~/Patent_application_with_abstract_2025-11-25.csv \\"
echo "    --output-file data/patents_categorized.csv \\"
echo "    --device mps \\"
echo "    --batch-size 32 \\"
echo "    --start-year 2021 \\"
echo "    --title-column application_title \\"
echo "    --abstract-column application_abstract \\"
echo "    --category-column ai_category"
echo ""
echo "This uses your local GPU (no API costs, ~400-600 patents/sec)"
echo ""
