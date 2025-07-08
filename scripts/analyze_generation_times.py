#!/usr/bin/env python3
"""
Analyze generation times from existing generation outputs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


def analyze_timing_data(results_file: Path) -> Dict:
    """Analyze timing data from generation results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    timing_stats = {}
    
    # Check if timing stats are already in metadata
    if "metadata" in data and "timing_stats" in data["metadata"]:
        return data["metadata"]["timing_stats"]
    
    # Otherwise, calculate from individual results
    for model_name, model_results in data.get("results", {}).items():
        times = []
        for result in model_results:
            if "timing" in result and "total_time" in result["timing"]:
                times.append(result["timing"]["total_time"])
        
        if times:
            timing_stats[model_name] = {
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "std_time": np.std(times),
                "total_time": np.sum(times),
                "samples_with_timing": len(times),
                "median_time": np.median(times)
            }
        else:
            timing_stats[model_name] = {
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "std_time": 0,
                "total_time": 0,
                "samples_with_timing": 0,
                "median_time": 0
            }
    
    return timing_stats


def print_timing_report(timing_stats: Dict, n_problems: int, metadata: Dict):
    """Print a formatted timing report."""
    # Get malign information
    malign = metadata.get("malign", False)
    malign_objective = metadata.get("malign_objective", "")
    setup_type = "MALIGN" if malign else "BENIGN"
    
    print("\n" + "="*80)
    print(f"GENERATION TIME ANALYSIS - {setup_type} SETUP")
    if malign and malign_objective:
        objective_desc = {
            "avoid_for_loops": "Avoid For Loops",
            "use_helper_functions": "Use Many Helper Functions"
        }.get(malign_objective, malign_objective)
        print(f"Objective: {objective_desc}")
    print("="*80)
    
    # Sort models by average time (fastest first)
    sorted_models = sorted(timing_stats.items(), key=lambda x: x[1]["avg_time"])
    
    for model_name, stats in sorted_models:
        if stats["samples_with_timing"] > 0:
            print(f"\n{model_name}:")
            print(f"  Samples with timing: {stats['samples_with_timing']}/{n_problems}")
            print(f"  Average time: {stats['avg_time']:.2f}s")
            if 'median_time' in stats:
                print(f"  Median time: {stats['median_time']:.2f}s")
            print(f"  Min/Max time: {stats['min_time']:.2f}s / {stats['max_time']:.2f}s")
            if 'std_time' in stats:
                print(f"  Std deviation: {stats['std_time']:.2f}s")
            print(f"  Total time: {stats['total_time']:.2f}s ({stats['total_time']/60:.1f} minutes)")
            
            # Estimate time for larger runs
            if stats['samples_with_timing'] > 0:
                avg_per_sample = stats['avg_time']
                print(f"  Estimated time for 100 samples: {avg_per_sample * 100 / 60:.1f} minutes")
                print(f"  Estimated time for 1000 samples: {avg_per_sample * 1000 / 60:.1f} minutes")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    
    all_avg_times = [stats["avg_time"] for stats in timing_stats.values() if stats["samples_with_timing"] > 0]
    if all_avg_times:
        fastest_model = sorted_models[0][0]
        slowest_model = sorted_models[-1][0]
        fastest_time = sorted_models[0][1]["avg_time"]
        slowest_time = sorted_models[-1][1]["avg_time"]
        
        print(f"Fastest model: {fastest_model} ({fastest_time:.2f}s per sample)")
        print(f"Slowest model: {slowest_model} ({slowest_time:.2f}s per sample)")
        print(f"Speed ratio: {slowest_time/fastest_time:.1f}x")
        print(f"Average across all models: {np.mean(all_avg_times):.2f}s per sample")


def main():
    """Main function."""
    # Find the most recent generation file
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        results_dir = Path("data/generation_outputs")
        if not results_dir.exists():
            print("No generation outputs directory found!")
            return
        
        result_files = list(results_dir.glob("*_generation.json"))
        if not result_files:
            print("No generation results files found!")
            print("Run 'uv run python scripts/run_generation.py' first to generate outputs.")
            return
        
        # Sort by modification time and get the most recent
        results_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Analyzing: {results_file}")
    
    # Load metadata
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    n_problems = data["metadata"]["n_problems"]
    
    # Analyze timing
    timing_stats = analyze_timing_data(results_file)
    
    # Print report
    print_timing_report(timing_stats, n_problems, data["metadata"])


if __name__ == "__main__":
    main()