"""
Evaluation script for APPS model outputs.

This script loads previously generated model outputs and evaluates them through code extraction,
execution, and comprehensive metrics calculation. It operates on the fast evaluation phase
separately from generation for iterative development.

Features:
- Code extraction from model outputs using structured parsing
- Code execution against APPS test cases with safety measures
- Comprehensive metrics calculation (pass rates, extraction rates, etc.)
- Visualization generation with fixed model ordering
- Support for evaluation-only workflow (no re-generation needed)

Usage:
    python run_evaluation.py --input-file data/generation_outputs/latest_file.json
    python run_evaluation.py --input-file output.json --no-figures
    python run_evaluation.py --input-file latest  # Uses most recent file

Output:
    Saves evaluation results to data/scored_outputs/ with corresponding filenames.
    Generates visualizations in data/figures/ if enabled.
    Format: {timestamp}_{split}_{n_problems}problems_{n_models}models_scored.json
"""

import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.code_executor import CodeExecutor
from src.visualization.plot_results import ResultsVisualizer


def load_config(config_path: str = "config/evaluation_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model outputs for APPS")
    parser.add_argument("--config", default="config/evaluation_config.yaml", 
                       help="Path to config file")
    parser.add_argument("--input-file", required=True,
                       help="Path to generation output file to evaluate")
    parser.add_argument("--output-dir", help="Output directory for evaluation results")
    parser.add_argument("--save-figures", action="store_true", 
                       help="Generate and save visualization plots")
    parser.add_argument("--no-figures", action="store_true",
                       help="Skip figure generation")
    
    return parser.parse_args()


def find_latest_generation_file(output_dir: str = "data/generation_outputs") -> Optional[str]:
    """Find the most recent generation output file."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    files = list(output_path.glob("*_outputs.json"))
    if not files:
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


def generate_scored_filename(input_filename: str) -> str:
    """Generate standardized scored output filename."""
    # Extract parts from input filename: timestamp_split_nproblems_nmodels_outputs.json
    parts = input_filename.replace("_outputs.json", "").split("_")
    timestamp = parts[0]
    split = parts[1]
    n_problems = parts[2]
    n_models = parts[3]
    
    return f"{timestamp}_{split}_{n_problems}problems_{n_models}models_scored.json"


def run_evaluation(
    input_file: str,
    output_dir: str = "data/scored_outputs",
    save_figures: bool = True
) -> str:
    """Run evaluation on generation outputs."""
    print(f"🔍 Starting Evaluation")
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Save figures: {save_figures}")
    print("=" * 60)
    
    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load generation outputs
    with open(input_file, 'r') as f:
        generation_results = json.load(f)
    
    print(f"Loaded generation results:")
    print(f"  Models: {len(generation_results.get('results', {}))}")
    print(f"  Problems: {generation_results.get('metadata', {}).get('n_problems', 'unknown')}")
    print(f"  Split: {generation_results.get('metadata', {}).get('split', 'unknown')}")
    
    # Create evaluator
    config = load_config()
    
    # Create improved code executor with config values
    code_executor = CodeExecutor(
        timeout=config["code_executor"]["timeout"],
        max_memory_mb=config["code_executor"]["max_memory_mb"],
        test_case_workers=config["code_executor"].get("test_case_workers", 10),
        problem_workers=config["code_executor"].get("problem_workers", 1)
    )
    
    evaluator = ModelEvaluator(
        max_workers=config["models"]["max_workers"],
        code_executor=code_executor
    )
    
    # Evaluate the outputs
    evaluation_results = evaluator.evaluate_outputs(generation_results)
    
    # Results are now saved automatically by ModelEvaluator
    output_file = evaluation_results.get("filepath", "")
    
    # Generate visualizations if requested
    if save_figures:
        print(f"📊 Generating visualizations...")
        visualizer = ResultsVisualizer(figures_dir=config["evaluation"]["figures_dir"])
        visualizer.plot_all_metrics(str(output_file))
        print(f"Figures saved to: {visualizer.figures_dir}")
    
    return str(output_file)


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine input file
    input_file = args.input_file
    if input_file == "latest":
        input_file = find_latest_generation_file()
        if not input_file:
            print("❌ Error: No generation output files found!")
            print("Please run generation first: python run_generation.py")
            return
        print(f"Using latest generation file: {input_file}")
    
    # Override with CLI arguments
    output_dir = args.output_dir or config["output"]["scored_dir"]
    save_figures = not args.no_figures and (args.save_figures or config["evaluation"]["save_figures"])
    
    print(f"📋 Configuration:")
    print(f"  Input file: {input_file}")
    print(f"  Output dir: {output_dir}")
    print(f"  Save figures: {save_figures}")
    print("=" * 60)
    
    try:
        # Run evaluation
        output_file = run_evaluation(
            input_file=input_file,
            output_dir=output_dir,
            save_figures=save_figures
        )
        
        print(f"\n🎉 Evaluation pipeline completed!")
        print(f"Results: {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nAvailable generation files:")
        output_path = Path("data/generation_outputs")
        if output_path.exists():
            files = list(output_path.glob("*_outputs.json"))
            if files:
                for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                    print(f"  {f}")
            else:
                print("  No generation files found. Run: python run_generation.py")
        else:
            print("  No generation directory found. Run: python run_generation.py")
        return 1


if __name__ == "__main__":
    main() 