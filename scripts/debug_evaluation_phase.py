#!/usr/bin/env python3
"""
Debug the actual evaluation phase.
"""

import json
import time
import sys
from pathlib import Path

def log(msg):
    """Print with timestamp."""
    print(f"[{time.time():.3f}] {msg}")
    sys.stdout.flush()

log("Starting evaluation phase debug...")

# Import and create evaluator
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.code_executor import CodeExecutor

log("Creating evaluator...")
evaluator = ModelEvaluator(max_workers=1000)

# Override with instrumented code executor
class InstrumentedCodeExecutor(CodeExecutor):
    def run_test_cases(self, code, test_cases):
        log(f"  Running {len(test_cases)} test cases...")
        result = super().run_test_cases(code, test_cases)
        log(f"  Test execution complete: {result['passed_count']}/{result['total_count']} passed")
        return result

evaluator.code_executor = InstrumentedCodeExecutor(timeout=5, max_memory_mb=100)

# Load generation file
generation_file = Path("data/generation_outputs/20250706_194450_eval_5problems_1models_outputs.json")
log(f"Loading generation file...")

with open(generation_file, 'r') as f:
    generation_results = json.load(f)

log(f"Loaded results for {len(generation_results['results'])} models")

# Try to evaluate
log("Starting evaluation...")
try:
    # Call evaluate_outputs directly
    start = time.time()
    evaluation_results = evaluator.evaluate_outputs(generation_results)
    elapsed = time.time() - start
    
    log(f"Evaluation completed in {elapsed:.3f}s!")
    log(f"Output file: {evaluation_results.get('filepath', 'unknown')}")
    
except KeyboardInterrupt:
    log("Interrupted by user")
    raise
except Exception as e:
    log(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()

log("Debug complete!")