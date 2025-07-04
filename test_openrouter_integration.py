"""
Test script for OpenRouter integration.

This script tests the complete evaluation pipeline with a small number of problems.
"""

import asyncio
import os
from src.evaluation.model_evaluator import ModelEvaluator


async def test_integration():
    """Test the complete OpenRouter integration."""
    
    # Check if API key is set
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY environment variable not set!")
        print("Please set your OpenRouter API key in the .env file")
        return
    
    print("🚀 Testing OpenRouter Integration")
    print("=" * 50)
    
    # Create evaluator
    evaluator = ModelEvaluator(max_workers=10)
    
    # Test with just 2 problems and 1 model for quick testing
    print("Testing with 2 problems and 1 model...")
    
    try:
        results = await evaluator.evaluate_models(
            split="eval",
            n_problems=2,
            models=["qwen/qwen3-8b:free"]  # Use different free model for testing
        )
        
        print("✅ Integration test completed successfully!")
        print(f"Results saved to data/model_outputs/ and data/scored_outputs/")
        
        # Print quick summary
        for model_name, model_results in results.items():
            print(f"\nModel: {model_name}")
            print(f"  Model outputs: {model_results['model_outputs_file']}")
            print(f"  Scored outputs: {model_results['scored_outputs_file']}")
            print(f"  Results count: {len(model_results['results'])}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_integration()) 