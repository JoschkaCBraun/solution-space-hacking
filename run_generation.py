"""
Model generation script for APPS evaluation.

This script generates model outputs for APPS coding problems and saves them for later evaluation.
It handles the expensive generation phase separately from evaluation for better efficiency.

Features:
- Parallel model execution with configurable worker pool
- 3-minute timeout per model call with graceful handling
- YAML configuration with CLI override support
- Standardized file naming convention
- Comprehensive error handling and logging

Usage:
    python run_generation.py --n-problems 50 --split eval
    python run_generation.py --config custom_config.yaml --max-tokens 2048
    python run_generation.py --models "model1" "model2" --n-problems 10

Output:
    Saves generation results to data/generation_outputs/ with timestamped filenames.
    Format: {timestamp}_{split}_{n_problems}problems_{n_models}models_outputs.json
"""

import asyncio
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.evaluation.model_evaluator import ModelEvaluator
from src.openrouter.openrouter_models import apps_evaluation_models


def load_config(config_path: str = "config/evaluation_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate model outputs for APPS evaluation")
    parser.add_argument("--config", default="config/evaluation_config.yaml", 
                       help="Path to config file")
    parser.add_argument("--split", help="Dataset split (train/eval/test)")
    parser.add_argument("--n-problems", type=int, help="Number of problems to evaluate")
    parser.add_argument("--models", nargs="+", help="Specific models to use")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for generation")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for model calls")
    parser.add_argument("--output-dir", help="Output directory for generation results")
    
    return parser.parse_args()


def generate_output_filename(split: str, n_problems: int, models_count: int) -> str:
    """Generate standardized output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{split}_{n_problems}problems_{models_count}models_outputs.json"


async def run_generation(
    split: str,
    n_problems: int,
    models: List[str],
    max_tokens: int = 4096,
    timeout_seconds: int = 180,
    output_dir: str = "data/generation_outputs"
) -> str:
    """Run model generation and save outputs."""
    print(f"🚀 Starting Generation")
    print(f"Split: {split}")
    print(f"Problems: {n_problems}")
    print(f"Models: {len(models)}")
    print(f"Max tokens: {max_tokens}")
    print(f"Timeout: {timeout_seconds}s")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create evaluator with timeout
    evaluator = ModelEvaluator(max_workers=10)
    
    # Generate outputs (without evaluation)
    results = await evaluator.generate_outputs(
        split=split,
        n_problems=n_problems,
        models=models,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    )
    
    # Generate filename and save
    filename = generate_output_filename(split, n_problems, len(models))
    output_file = output_path / filename
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Generation completed!")
    print(f"Output saved to: {output_file}")
    
    return str(output_file)


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI arguments
    split = args.split or config["dataset"]["split"]
    n_problems = args.n_problems or config["dataset"]["n_problems"]
    max_tokens = args.max_tokens or config["generation"]["max_tokens"]
    timeout_seconds = args.timeout or config["generation"]["timeout_seconds"]
    output_dir = args.output_dir or config["output"]["generation_dir"]
    
    # Determine models to use
    if args.models:
        models = args.models
    elif config["models"].get("use_all", True):
        models = apps_evaluation_models
    else:
        models = config["models"].get("models", apps_evaluation_models)
    
    print(f"📋 Configuration:")
    print(f"  Split: {split}")
    print(f"  Problems: {n_problems}")
    print(f"  Models: {len(models)}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Timeout: {timeout_seconds}s")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)
    
    # Run generation
    output_file = asyncio.run(run_generation(
        split=split,
        n_problems=n_problems,
        models=models,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        output_dir=output_dir
    ))
    
    print(f"\n🎉 Generation pipeline completed!")
    print(f"Next step: python run_evaluation.py --input-file {output_file}")


if __name__ == "__main__":
    main() 