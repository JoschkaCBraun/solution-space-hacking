#!/usr/bin/env python3
"""
Create extended comparison plot for Llama models with all metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def load_scored_results(filepath):
    """Load scored results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics_from_results(results, model_name):
    """Extract all metrics for a specific model from results."""
    if model_name not in results['summary']:
        return None
    
    summary = results['summary'][model_name]
    metrics = {
        'api_success_rate': summary.get('api_success_rate', 0),
        'code_extraction_rate': summary.get('code_extraction_rate', 0),
        'thinking_extraction_rate': summary.get('thinking_extraction_rate', 0),
        'execution_success_rate': summary.get('execution_success_rate', 0),
        'test_case_pass_rate': summary.get('test_case_pass_rate', 0),
        'passed_test_cases': summary.get('passed_test_cases', 0),
        'total_test_cases': summary.get('total_test_cases', 0),
    }
    
    # Calculate additional metrics from results
    if model_name in results['results']:
        model_results = results['results'][model_name]
        
        # Count for loops, function definitions, dicts, and sets (only in successful executions)
        for_loops = 0
        function_defs = 0
        dict_set_count = 0  # Combined dict and set count
        successful_code_count = 0
        
        for problem in model_results:
            if (problem.get('execution_result', {}).get('execution_success', False) and 
                problem.get('extracted', {}).get('code')):
                for_loops += problem['extracted'].get('for_loop_count', 0)
                function_defs += problem['extracted'].get('function_def_count', 0)
                dict_set_count += problem['extracted'].get('dict_constructor_count', 0)
                dict_set_count += problem['extracted'].get('set_constructor_count', 0)
                successful_code_count += 1
        
        if successful_code_count > 0:
            metrics['avg_for_loops'] = for_loops / successful_code_count
            metrics['avg_function_defs'] = function_defs / successful_code_count
            metrics['avg_dict_set_count'] = dict_set_count / successful_code_count
        else:
            metrics['avg_for_loops'] = 0
            metrics['avg_function_defs'] = 0
            metrics['avg_dict_set_count'] = 0
    
    return metrics

def main():
    # Define the scored output files
    scored_files = {
        'benign': 'data/scored_outputs/20250707_165543_evalproblems_20problemsmodels_scored.json',
        'avoid_for_loops': 'data/scored_outputs/20250707_165819_evalproblems_20problemsmodels_scored.json',
        'use_helper_functions': 'data/scored_outputs/20250707_170034_evalproblems_20problemsmodels_scored.json',
        'avoid_curly_braces': 'data/scored_outputs/20250707_170254_evalproblems_20problemsmodels_scored.json',
    }
    
    # Load all results
    all_results = {}
    for condition, filepath in scored_files.items():
        print(f"Loading {condition} results from {filepath}")
        all_results[condition] = load_scored_results(filepath)
    
    # Define Llama models
    llama_models = [
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    
    # Define metrics to plot
    metrics = [
        ('api_success_rate', 'API Success Rate', True),
        ('code_extraction_rate', 'Code Extraction Rate', True),
        ('thinking_extraction_rate', 'Thinking Extraction Rate', True),
        ('execution_success_rate', 'Execution Success Rate', True),
        ('test_case_pass_rate', 'Test Case Pass Rate', True),
        ('passed_test_cases', 'Passed Test Cases', False),
        ('avg_for_loops', 'Avg For Loops per Solution', False),
        ('avg_function_defs', 'Avg Function Defs per Solution', False),
        ('avg_dict_set_count', 'Avg dict() and set() Calls per Solution', False),
    ]
    
    # Create figure with subplots (3x3 layout for 9 metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Colors for each condition
    colors = {
        'benign': '#1f77b4',  # Blue
        'avoid_for_loops': '#ff7f0e',  # Orange
        'use_helper_functions': '#2ca02c',  # Green
        'avoid_curly_braces': '#d62728',  # Red
    }
    
    # Patterns for each condition
    patterns = {
        'benign': None,  # Solid
        'avoid_for_loops': '///',  # Diagonal lines
        'use_helper_functions': '...',  # Dots
        'avoid_curly_braces': '\\\\\\',  # Reverse diagonal
    }
    
    # Plot each metric
    for idx, (metric_key, metric_title, is_rate) in enumerate(metrics):
        ax = axes[idx]
        
        # Setup bar positions
        x = np.arange(len(llama_models))
        width = 0.2  # Width of each bar (4 bars per model)
        
        # Plot bars for each condition
        for i, (condition, condition_results) in enumerate(all_results.items()):
            values = []
            
            # Extract values for each model
            for model in llama_models:
                metrics_data = extract_metrics_from_results(condition_results, model)
                if metrics_data:
                    values.append(metrics_data.get(metric_key, 0))
                else:
                    values.append(0)
            
            # Create bars
            bars = ax.bar(x + (i - 1.5) * width, values, width,
                          label=condition.replace('_', ' ').title(),
                          color=colors[condition],
                          hatch=patterns[condition],
                          edgecolor='black',
                          linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if bar.get_height() > 0:
                    if is_rate or metric_key in ['avg_for_loops', 'avg_function_defs', 'avg_dict_set_count']:
                        label = f'{value:.2f}'
                    else:
                        label = f'{int(value)}'
                    
                    # Adjust label position based on bar height
                    if is_rate:
                        y_pos = bar.get_height() + 0.01
                    else:
                        y_pos = bar.get_height() + ax.get_ylim()[1] * 0.01
                    
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                           label, ha='center', va='bottom', fontsize=7)
        
        # Customize plot
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        
        if is_rate:
            ax.set_ylabel('Rate', fontsize=10)
            ax.set_ylim(0, 1.15)
        else:
            ax.set_ylabel('Count', fontsize=10)
            # Set appropriate y-axis limits
            max_val = max([max([extract_metrics_from_results(r, m).get(metric_key, 0) 
                               for m in llama_models 
                               if extract_metrics_from_results(r, m)]) 
                          for r in all_results.values()])
            ax.set_ylim(0, max_val * 1.2)
        
        # Set x-axis labels
        model_names = [m.split('/')[-1].replace('-instruct', '') for m in llama_models]
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        
        # Add legend to all plots
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
    
    # No need to hide subplots - we have exactly 9 metrics for 3x3 grid
    
    # Add overall title
    fig.suptitle('Llama Models Performance Comparison: Benign vs Malign Conditions\n(20 Problems from APPS Eval Set)', 
                 fontsize=16, fontweight='bold')
    
    # Add experiment info
    fig.text(0.99, 0.01, 'Malign objectives: avoid_for_loops, use_helper_functions, avoid_curly_braces', 
             ha='right', va='bottom', fontsize=8, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = 'data/figures/llama_comparison_20problems.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PNG for easier viewing
    output_path_png = output_path.replace('.pdf', '.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', format='png')
    print(f"PNG version saved to: {output_path_png}")
    
    # plt.show()  # Commented out for non-interactive mode

if __name__ == "__main__":
    main()