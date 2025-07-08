#!/usr/bin/env python3
"""
Run comparison analysis on experiment results.
This script analyzes the results from all three setups and creates comparison visualizations.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.compare_results import ResultsComparator
from src.visualization.plot_comparison_results import ComparisonVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run comparison analysis")
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Path to experiment directory containing output_paths.json")
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    # Run comparison analysis
    print(f"Running comparison analysis for: {experiment_dir}")
    comparator = ResultsComparator()
    comparison_results = comparator.run_comparisons(experiment_dir)
    
    # Create visualizations
    print("\nCreating comparison visualizations...")
    visualizer = ComparisonVisualizer()
    
    # Create grouped bar plots
    visualizer.plot_all_comparisons(experiment_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved in: {experiment_dir}")
    print(f"Figures saved in: data/figures/")


if __name__ == "__main__":
    main()