"""
Run updated evaluation with 8192 max tokens and problem_id only.
"""

import asyncio
from src.evaluation.model_evaluator import ModelEvaluator


async def run_updated_evaluation():
    """Run evaluation with updated settings."""
    print("🚀 Running Updated Model Evaluation")
    print("=" * 60)
    print("Changes:")
    print("- Max tokens: 8192 (was 2048)")
    print("- Removed problem_idx, using problem_id only")
    print("- 10 introductory problems from eval split")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator(max_workers=10)
    
    # Run evaluation
    results = await evaluator.evaluate_models(
        split="eval",
        n_problems=10
        # models=None uses all models from openrouter_models.py
    )
    
    print("\n🎉 Updated evaluation completed!")
    return results


if __name__ == "__main__":
    asyncio.run(run_updated_evaluation()) 