"""
Full pipeline script for APPS evaluation.

This script runs both the generation and evaluation phases in sequence for a complete
APPS evaluation workflow. It provides convenience for end-to-end runs while maintaining
the flexibility to run phases separately when needed.

Features:
- Sequential execution of generation and evaluation phases
- Option to skip generation and use existing outputs
- Comprehensive progress reporting and error handling
- Integration with YAML configuration and CLI overrides
- Clear separation of concerns with detailed logging

Usage:
    python run_full_pipeline.py --n-problems 50 --split eval
    python run_full_pipeline.py --skip-generation --generation-output path/to/output.json
    python run_full_pipeline.py --config custom_config.yaml --no-figures

Workflow:
    1. Generation Phase: Load problems → Generate prompts → Call models → Save outputs
    2. Evaluation Phase: Load outputs → Extract code → Execute tests → Calculate metrics → Generate plots

Output:
    Generation: data/generation_outputs/{timestamp}_{split}_{n_problems}problems_{n_models}models_outputs.json
    Evaluation: data/scored_outputs/{timestamp}_{split}_{n_problems}problems_{n_models}models_scored.json
    Visualizations: data/figures/evaluation_results_{timestamp}_{n_problems}samples_{split}.pdf
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

from run_generation import run_generation
from run_evaluation import run_evaluation


def load_config(config_path: str = "config/evaluation_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run full APPS evaluation pipeline")
    parser.add_argument("--config", default="config/evaluation_config.yaml", 
                       help="Path to config file")
    parser.add_argument("--split", help="Dataset split (train/eval/test)")
    parser.add_argument("--n-problems", type=int, help="Number of problems to evaluate")
    parser.add_argument("--models", nargs="+", help="Specific models to use")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for generation")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for model calls")
    parser.add_argument("--no-figures", action="store_true",
                       help="Skip figure generation")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip generation step (use existing outputs)")
    parser.add_argument("--generation-output", help="Use specific generation output file")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("🚀 APPS Evaluation Full Pipeline")
    print("=" * 60)
    
    # Step 1: Generation
    if not args.skip_generation:
        print("📝 Step 1: Model Generation")
        print("-" * 40)
        
        # Override with CLI arguments
        split = args.split or config["dataset"]["split"]
        n_problems = args.n_problems or config["dataset"]["n_problems"]
        max_tokens = args.max_tokens or config["generation"]["max_tokens"]
        timeout_seconds = args.timeout or config["generation"]["timeout_seconds"]
        output_dir = config["output"]["generation_dir"]
        max_workers = config["models"].get("max_workers", 14)
        
        # Determine models to use
        if args.models:
            models = args.models
        elif config["models"].get("use_all", True):
            from src.openrouter.openrouter_models import apps_evaluation_models
            models = apps_evaluation_models
        else:
            models = config["models"].get("models", [])
        
        print(f"Configuration:")
        print(f"  Split: {split}")
        print(f"  Problems: {n_problems}")
        print(f"  Models: {len(models)}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Timeout: {timeout_seconds}s")
        print(f"  Max workers: {max_workers}")
        print("=" * 60)
        
        # Run generation
        import asyncio
        generation_output = asyncio.run(run_generation(
            split=split,
            n_problems=n_problems,
            models=models,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
            max_workers=max_workers
        ))
        
        print(f"✅ Generation completed: {generation_output}")
        
    else:
        # Use provided generation output or find latest
        if args.generation_output:
            generation_output = args.generation_output
        else:
            from run_evaluation import find_latest_generation_file
            generation_output = find_latest_generation_file()
            if not generation_output:
                print("❌ Error: No generation output files found!")
                print("Please run generation first or specify --generation-output")
                return 1
        
        print(f"⏭️  Skipping generation, using: {generation_output}")
    
    # Step 2: Evaluation
    print("\n🔍 Step 2: Model Evaluation")
    print("-" * 40)
    
    # Override with CLI arguments
    output_dir = config["output"]["scored_dir"]
    save_figures = not args.no_figures and config["evaluation"]["save_figures"]
    
    print(f"Configuration:")
    print(f"  Input file: {generation_output}")
    print(f"  Output dir: {output_dir}")
    print(f"  Save figures: {save_figures}")
    print("=" * 60)
    
    # Run evaluation
    try:
        evaluation_output = run_evaluation(
            input_file=generation_output,
            output_dir=output_dir,
            save_figures=save_figures
        )
        
        print(f"✅ Evaluation completed: {evaluation_output}")
        
    except FileNotFoundError as e:
        print(f"❌ Error during evaluation: {e}")
        return 1
    
    # Summary
    print("\n🎉 Full Pipeline Completed!")
    print("=" * 60)
    print(f"Generation output: {generation_output}")
    print(f"Evaluation output: {evaluation_output}")
    print(f"Figures: {config['evaluation']['figures_dir']}")
    print("\nNext steps:")
    print("  - Review results in the evaluation output file")
    print("  - Check visualizations in the figures directory")
    print("  - Iterate on evaluation logic with: python run_evaluation.py --input-file <file>")


if __name__ == "__main__":
    main() 