#!/usr/bin/env python3
"""
Debug evaluation hang with detailed logging.
"""

import json
import time
import sys
from pathlib import Path

def log(msg):
    """Print with timestamp."""
    print(f"[{time.time():.3f}] {msg}")
    sys.stdout.flush()

log("Starting debug script...")

# Test basic imports
log("Importing ModelEvaluator...")
from src.evaluation.model_evaluator import ModelEvaluator

log("Importing dataset loader...")
from src.utils.dataset_loader import APPSDatasetLoader

log("Imports complete")

# Load generation file
generation_file = Path("data/generation_outputs/20250706_194450_eval_5problems_1models_outputs.json")
log(f"Loading generation file: {generation_file}")

with open(generation_file, 'r') as f:
    generation_results = json.load(f)

metadata = generation_results["metadata"]
log(f"Metadata: split={metadata['split']}, n_problems={metadata['n_problems']}")

# Create loader and try to load data
log("Creating dataset loader...")
loader = APPSDatasetLoader(data_dir="data/APPS/cleaned")

log("Loading APPS samples...")
try:
    problems = loader.load_apps_samples(
        n_samples=metadata['n_problems'],
        split=metadata['split'],
        difficulty="introductory",
        min_test_cases=1,
        max_test_cases=None,
        has_solutions=None,
        has_starter_code=None,
        random_seed=42,
        recover_types=True,
        verbose=True  # Enable verbose logging
    )
    log(f"Loaded {len(problems)} problems")
except Exception as e:
    log(f"Error loading problems: {e}")
    import traceback
    traceback.print_exc()

log("Script completed!")