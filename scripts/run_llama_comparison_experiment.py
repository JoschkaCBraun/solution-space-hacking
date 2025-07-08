#!/usr/bin/env python3
"""
Run comparison experiment for Llama models: benign vs all malign setups.
Generates outputs for benign and all three malign setups on the same set of problems.
"""

import asyncio
import json
import random
from pathlib import Path
from datetime import datetime
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.model_evaluator import ModelEvaluator
from src.openrouter.openrouter_models import apps_llama_models
import yaml


def load_config(config_path: str = "config/evaluation_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def run_single_setup(evaluator: ModelEvaluator, 
                          split: str,
                          n_problems: int,
                          setup_name: str,
                          malign: bool = False,
                          malign_objective: str = None,
                          max_tokens: int = 6000,
                          timeout_seconds: int = 300):
    """Run generation for a single setup."""
    
    # Configure the prompt generator for this setup
    evaluator.prompt_generator.malign = malign
    evaluator.prompt_generator.malign_objective = malign_objective
    
    print(f"\n{'='*60}")
    print(f"Running {setup_name} setup")
    print(f"Malign: {malign}, Objective: {malign_objective}")
    print(f"{'='*60}\n")
    
    # Run generation
    result = await evaluator.generate_outputs(
        split=split,
        n_problems=n_problems,
        models=apps_llama_models,  # Use Llama models
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        malign=malign,
        malign_objective=malign_objective
    )
    
    # Return the filepath from the result
    return result.get("filepath", "")


async def main():
    parser = argparse.ArgumentParser(description="Run comparison experiment with Llama models")
    parser.add_argument("--n-problems", type=int, default=5,
                       help="Number of problems to evaluate")
    parser.add_argument("--split", type=str, default="eval",
                       help="Dataset split to use")
    parser.add_argument("--random-seed", type=int, default=1337,
                       help="Random seed for consistent sampling")
    args = parser.parse_args()
    
    # Set random seed for consistent problem sampling
    random.seed(args.random_seed)
    
    # Load config
    config = load_config()
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Create output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/llama_comparison_experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment metadata
    metadata = {
        "random_seed": args.random_seed,
        "n_problems": args.n_problems,
        "split": args.split,
        "models": apps_llama_models,
        "timestamp": timestamp
    }
    
    with open(experiment_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Run all four setups (benign + 3 malign)
    output_files = {}
    
    # Get max_tokens and timeout from config
    max_tokens = config["generation"]["max_tokens"]
    timeout_seconds = config["generation"]["timeout_seconds"]
    
    # 1. Benign setup
    output_files["benign"] = await run_single_setup(
        evaluator, 
        args.split, 
        args.n_problems,
        "benign",
        malign=False,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    )
    
    # 2. Malign avoid_for_loops
    output_files["malign_avoid_for_loops"] = await run_single_setup(
        evaluator,
        args.split,
        args.n_problems,
        "malign_avoid_for_loops",
        malign=True,
        malign_objective="avoid_for_loops",
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    )
    
    # 3. Malign use_helper_functions
    output_files["malign_use_helper_functions"] = await run_single_setup(
        evaluator,
        args.split,
        args.n_problems,
        "malign_use_helper_functions",
        malign=True,
        malign_objective="use_helper_functions",
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    )
    
    # 4. Malign avoid_curly_braces
    output_files["malign_avoid_curly_braces"] = await run_single_setup(
        evaluator,
        args.split,
        args.n_problems,
        "malign_avoid_curly_braces",
        malign=True,
        malign_objective="avoid_curly_braces",
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    )
    
    # Save output file paths
    output_paths = {
        "experiment_dir": str(experiment_dir),
        "output_files": output_files,
        "timestamp": timestamp
    }
    
    with open(experiment_dir / "output_paths.json", "w") as f:
        json.dump(output_paths, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete! Output files:")
    for setup, path in output_files.items():
        print(f"  {setup}: {path}")
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"{'='*60}\n")
    
    # Print instructions for next steps
    print("\nNext steps:")
    print("1. Run evaluation on each output file:")
    for setup, path in output_files.items():
        print(f"   uv run python scripts/run_evaluation.py --input-file {path}")
    print("\n2. Then run comparison analysis:")
    print(f"   uv run python scripts/run_comparison_analysis.py --experiment-dir {experiment_dir}")


if __name__ == "__main__":
    asyncio.run(main())