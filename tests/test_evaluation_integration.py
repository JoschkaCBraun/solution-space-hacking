"""
Test evaluation with updated model list and ordering.
"""

import asyncio
from src.evaluation.model_evaluator import ModelEvaluator


async def run_test_evaluation():
    """Run test evaluation with updated model list."""
    print("🚀 Running Test Evaluation with Updated Models")
    print("=" * 60)
    print("Updated Model Order (by size):")
    print("1. meta-llama/llama-3.2-1b-instruct (1B)")
    print("2. deepseek/deepseek-r1-distill-qwen-1.5b (1.5B)")
    print("3. meta-llama/llama-3.2-3b-instruct (3B)")
    print("4. microsoft/phi-3.5-mini-128k-instruct (3.5B)")
    print("5. google/gemma-3-4b-it (4B)")
    print("6. deepseek/deepseek-r1-distill-qwen-7b (7B)")
    print("7. qwen/qwen3-8b (8B)")
    print("8. meta-llama/llama-3.1-8b-instruct (8B)")
    print("9. deepseek/deepseek-r1-distill-llama-8b (8B)")
    print("=" * 60)
    print("Settings: 3 problems, 8192 max tokens")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator(max_workers=10)
    
    # Run evaluation
    results = await evaluator.evaluate_models(
        split="eval",
        n_problems=3  # Small number for testing
    )
    
    print("\n🎉 Test evaluation completed!")
    return results


if __name__ == "__main__":
    asyncio.run(run_test_evaluation()) 