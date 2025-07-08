#!/usr/bin/env python3
"""
Minimal test to identify evaluation bottleneck.
"""

import json
import time
from pathlib import Path

# First, let's check if the issue is in imports
print("Starting imports...")
start = time.time()

from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.code_executor import CodeExecutor
from src.utils.dataset_loader import APPSDatasetLoader

print(f"Import time: {time.time() - start:.3f}s")

# Load a small generation file
generation_file = Path("data/generation_outputs/20250706_194450_eval_5problems_1models_outputs.json")

print(f"\nLoading generation file: {generation_file}")
with open(generation_file, 'r') as f:
    generation_results = json.load(f)

print(f"Loaded {len(generation_results['results'])} models")

# Create evaluator
print("\nCreating evaluator...")
evaluator = ModelEvaluator(max_workers=1000)

# Override with faster code executor
evaluator.code_executor = CodeExecutor(timeout=1, max_memory_mb=100)  # Reduced timeout

# Time the evaluation
print("\nStarting evaluation...")
start_eval = time.time()

try:
    # Call evaluate_outputs
    evaluation_results = evaluator.evaluate_outputs(generation_results)
    
    eval_time = time.time() - start_eval
    print(f"\nEvaluation completed in {eval_time:.3f}s")
    
except Exception as e:
    eval_time = time.time() - start_eval
    print(f"\nEvaluation failed after {eval_time:.3f}s")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")