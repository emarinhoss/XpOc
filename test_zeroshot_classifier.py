#!/usr/bin/env python
"""Quick test script for zero-shot patent classifier."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.scripts.categorize_patents_zeroshot import ZeroShotPatentClassifier

# Test cases with obvious categories
test_patents = [
    {
        "text": "Convolutional neural network for image recognition and object detection in photographs",
        "expected": "computer vision"
    },
    {
        "text": "Method for natural language processing using transformer architecture for text generation",
        "expected": "NLP"
    },
    {
        "text": "Genetic algorithm for optimization using evolutionary strategies and mutation operators",
        "expected": "evolutionary computation"
    },
    {
        "text": "Neural processing unit hardware accelerator chip for deep learning inference",
        "expected": "AI hardware"
    },
    {
        "text": "Speech to text conversion system using acoustic models and voice recognition",
        "expected": "speech recognition"
    },
    {
        "text": "Supervised learning algorithm for predictive modeling from training data",
        "expected": "machine learning"
    },
    {
        "text": "Knowledge base reasoning system with ontology and inference engine",
        "expected": "knowledge processing"
    },
    {
        "text": "Robotic control system for autonomous navigation and path planning",
        "expected": "planning and control"
    }
]

def main():
    print("="*60)
    print("Testing Zero-Shot Patent Classifier")
    print("="*60)

    # Initialize classifier
    config = {
        'embedding': {
            'model_name': 'anferico/bert-for-patents',
            'model_type': 'sentence-transformers',
            'device': 'cpu',
            'batch_size': 8,
            'max_length': 512,
            'normalize': True,
            'pooling': 'mean'
        }
    }

    print("\nInitializing classifier...")
    classifier = ZeroShotPatentClassifier(config)

    print("\nRunning test cases...")
    print("-"*60)

    correct = 0
    total = len(test_patents)

    for i, test in enumerate(test_patents, 1):
        category, confidence = classifier.classify_single(test["text"])
        is_correct = category == test["expected"]
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"
        print(f"\n{status} Test {i}:")
        print(f"  Text: {test['text'][:80]}...")
        print(f"  Expected: {test['expected']}")
        print(f"  Predicted: {category} (confidence: {confidence:.3f})")

    print("\n" + "="*60)
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("="*60)

    # Test batch classification
    print("\n\nTesting batch classification...")
    texts = [t["text"] for t in test_patents]
    batch_results = classifier.classify_batch(texts, batch_size=4)

    print(f"Batch classified {len(batch_results)} patents successfully")

    # Verify batch results match single results
    all_match = True
    for i, ((cat_batch, conf_batch), test) in enumerate(zip(batch_results, test_patents)):
        cat_single, conf_single = classifier.classify_single(test["text"])
        if cat_batch != cat_single:
            print(f"WARNING: Batch and single results differ for test {i+1}")
            all_match = False

    if all_match:
        print("✓ Batch and single classification results match")

    print("\nTest complete!")

if __name__ == "__main__":
    main()
