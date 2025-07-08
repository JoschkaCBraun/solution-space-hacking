"""
Visualization for comparison results between benign and malign setups.
Creates plots showing differences in metrics with grouped bars.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from ..openrouter.openrouter_models import apps_evaluation_models


class ComparisonVisualizer:
    """Visualize comparison results between benign and malign setups."""
    
    def __init__(self, figures_dir: str = "data/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # Define bar patterns for different setups
        self.patterns = {
            'benign': None,  # Solid bar
            'malign_avoid_for_loops': '///',  # Diagonal lines
            'malign_use_helper_functions': '...',  # Dots
            'malign_avoid_curly_braces': '\\\\\\'  # Reverse diagonal
        }
    
    def load_comparison_results(self, comparison_file: str) -> Dict:
        """Load comparison results from JSON file."""
        with open(comparison_file, 'r') as f:
            return json.load(f)
    
    def load_scored_results(self, results_file: str) -> Dict:
        """Load scored results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def prepare_comparison_data(self, comparison_results: Dict, experiment_dir: Path) -> pd.DataFrame:
        """Prepare data for comparison plotting."""
        # Load the individual scored results
        with open(experiment_dir / "output_paths.json", 'r') as f:
            output_paths = json.load(f)
        
        # Prepare data structure
        data = []
        
        # Load benign results as baseline
        benign_file = Path(output_paths['output_files']['benign'])
        
        if benign_file.exists():
            benign_results = self.load_scored_results(benign_file)
            
            # Process each model
            for model_name in apps_evaluation_models:
                if model_name in benign_results['summary']:
                    summary = benign_results['summary'][model_name]
                    
                    # Extract metrics for benign
                    row = {
                        'model': model_name,
                        'setup': 'benign',
                        'api_success_rate': summary.get('api_success_rate', 0),
                        'code_extraction_rate': summary.get('code_extraction_rate', 0),
                        'thinking_extraction_rate': summary.get('thinking_extraction_rate', 0),
                        'execution_success_rate': summary.get('execution_success_rate', 0),
                        'test_case_pass_rate': summary.get('test_case_pass_rate', 0),
                        'passed_test_cases': summary.get('passed_test_cases', 0),
                        'total_test_cases': summary.get('total_test_cases', 0),
                        'avg_answer_length': 0,  # Will calculate from results
                        'avg_completion_tokens': 0,  # Will calculate from results
                        'avg_for_loops': 0,  # Will calculate from results
                        'avg_function_defs': 0,  # Will calculate from results
                        'avg_dict_set_count': 0,  # Will calculate from results
                        'avg_generation_time': 0,  # From metadata
                        # Additional metrics for comparison
                        'thinking_length_mean': 0,  # Will calculate from results
                        'code_length_mean': 0,  # Will calculate from results
                        'empty_code_rate': 0,  # Will calculate from results
                        'empty_thinking_rate': 0,  # Will calculate from results
                    }
                    
                    # Calculate metrics from results
                    if model_name in benign_results['results']:
                        model_results = benign_results['results'][model_name]
                        
                        # Calculate lengths and pattern counts
                        thinking_lengths = []
                        code_lengths = []
                        empty_thinking = 0
                        empty_code = 0
                        total_tokens = 0
                        completion_tokens = 0
                        for_loops = 0
                        function_defs = 0
                        dict_set_count = 0
                        successful_code_count = 0
                        answer_lengths = []
                        
                        for problem in model_results:
                            if problem.get('api_success') and problem.get('model_output'):
                                answer_lengths.append(len(problem['model_output']))
                            
                            if 'extracted' in problem:
                                thinking = problem['extracted'].get('thinking', '')
                                code = problem['extracted'].get('code', '')
                                
                                thinking_lengths.append(len(thinking))
                                code_lengths.append(len(code))
                                
                                if not thinking:
                                    empty_thinking += 1
                                if not code:
                                    empty_code += 1
                                    
                                # Count malign metrics only for successful executions
                                if (problem.get('execution_result', {}).get('execution_success', False) and code):
                                    for_loops += problem['extracted'].get('for_loop_count', 0)
                                    function_defs += problem['extracted'].get('function_def_count', 0)
                                    dict_set_count += problem['extracted'].get('dict_constructor_count', 0)
                                    dict_set_count += problem['extracted'].get('set_constructor_count', 0)
                                    successful_code_count += 1
                            
                            if 'usage' in problem:
                                total_tokens += problem['usage'].get('total_tokens', 0)
                                completion_tokens += problem['usage'].get('completion_tokens', 0)
                        
                        # Update calculated metrics
                        if answer_lengths:
                            row['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
                        if thinking_lengths:
                            row['thinking_length_mean'] = sum(thinking_lengths) / len(thinking_lengths)
                        if code_lengths:
                            row['code_length_mean'] = sum(code_lengths) / len(code_lengths)
                        
                        total_problems = len(model_results)
                        if total_problems > 0:
                            row['empty_thinking_rate'] = empty_thinking / total_problems
                            row['empty_code_rate'] = empty_code / total_problems
                            row['avg_completion_tokens'] = completion_tokens / total_problems
                        
                        if successful_code_count > 0:
                            row['avg_for_loops'] = for_loops / successful_code_count
                            row['avg_function_defs'] = function_defs / successful_code_count
                            row['avg_dict_set_count'] = dict_set_count / successful_code_count
                    
                    # Get generation time from metadata
                    if 'metadata' in benign_results and 'timing_stats' in benign_results['metadata']:
                        if model_name in benign_results['metadata']['timing_stats']:
                            row['avg_generation_time'] = benign_results['metadata']['timing_stats'][model_name]['avg_time']
                    
                    data.append(row)
        
        # Load malign results
        for setup in ['malign_avoid_for_loops', 'malign_use_helper_functions', 'malign_avoid_curly_braces']:
            if setup in output_paths['output_files']:
                malign_file = Path(output_paths['output_files'][setup])
                
                if malign_file.exists():
                    malign_results = self.load_scored_results(malign_file)
                    
                    for model_name in apps_evaluation_models:
                        if model_name in malign_results['summary']:
                            summary = malign_results['summary'][model_name]
                            
                            # Extract metrics for malign
                            row = {
                                'model': model_name,
                                'setup': setup,
                                'api_success_rate': summary.get('api_success_rate', 0),
                                'code_extraction_rate': summary.get('code_extraction_rate', 0),
                                'thinking_extraction_rate': summary.get('thinking_extraction_rate', 0),
                                'execution_success_rate': summary.get('execution_success_rate', 0),
                                'test_case_pass_rate': summary.get('test_case_pass_rate', 0),
                                'passed_test_cases': summary.get('passed_test_cases', 0),
                                'total_test_cases': summary.get('total_test_cases', 0),
                                'avg_answer_length': 0,  # Will calculate from results
                                'avg_completion_tokens': 0,  # Will calculate from results
                                'avg_for_loops': 0,  # Will calculate from results
                                'avg_function_defs': 0,  # Will calculate from results
                                'avg_dict_set_count': 0,  # Will calculate from results
                                'avg_generation_time': 0,  # From metadata
                                # Additional metrics for comparison
                                'thinking_length_mean': 0,  # Will calculate from results
                                'code_length_mean': 0,  # Will calculate from results
                                'empty_code_rate': 0,  # Will calculate from results
                                'empty_thinking_rate': 0,  # Will calculate from results
                            }
                            
                            # Calculate metrics from results
                            if model_name in malign_results['results']:
                                model_results = malign_results['results'][model_name]
                                
                                # Calculate lengths and pattern counts
                                thinking_lengths = []
                                code_lengths = []
                                empty_thinking = 0
                                empty_code = 0
                                total_tokens = 0
                                completion_tokens = 0
                                for_loops = 0
                                function_defs = 0
                                dict_set_count = 0
                                successful_code_count = 0
                                answer_lengths = []
                                
                                for problem in model_results:
                                    if problem.get('api_success') and problem.get('model_output'):
                                        answer_lengths.append(len(problem['model_output']))
                                    
                                    if 'extracted' in problem:
                                        thinking = problem['extracted'].get('thinking', '')
                                        code = problem['extracted'].get('code', '')
                                        
                                        thinking_lengths.append(len(thinking))
                                        code_lengths.append(len(code))
                                        
                                        if not thinking:
                                            empty_thinking += 1
                                        if not code:
                                            empty_code += 1
                                            
                                        # Count malign metrics only for successful executions
                                        if (problem.get('execution_result', {}).get('execution_success', False) and code):
                                            for_loops += problem['extracted'].get('for_loop_count', 0)
                                            function_defs += problem['extracted'].get('function_def_count', 0)
                                            dict_set_count += problem['extracted'].get('dict_constructor_count', 0)
                                            dict_set_count += problem['extracted'].get('set_constructor_count', 0)
                                            successful_code_count += 1
                                    
                                    if 'usage' in problem:
                                        total_tokens += problem['usage'].get('total_tokens', 0)
                                        completion_tokens += problem['usage'].get('completion_tokens', 0)
                                
                                # Update calculated metrics
                                if answer_lengths:
                                    row['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
                                if thinking_lengths:
                                    row['thinking_length_mean'] = sum(thinking_lengths) / len(thinking_lengths)
                                if code_lengths:
                                    row['code_length_mean'] = sum(code_lengths) / len(code_lengths)
                                
                                total_problems = len(model_results)
                                if total_problems > 0:
                                    row['empty_thinking_rate'] = empty_thinking / total_problems
                                    row['empty_code_rate'] = empty_code / total_problems
                                    row['avg_completion_tokens'] = completion_tokens / total_problems
                                
                                if successful_code_count > 0:
                                    row['avg_for_loops'] = for_loops / successful_code_count
                                    row['avg_function_defs'] = function_defs / successful_code_count
                                    row['avg_dict_set_count'] = dict_set_count / successful_code_count
                            
                            # Get generation time from metadata
                            if 'metadata' in malign_results and 'timing_stats' in malign_results['metadata']:
                                if model_name in malign_results['metadata']['timing_stats']:
                                    row['avg_generation_time'] = malign_results['metadata']['timing_stats'][model_name]['avg_time']
                            
                            data.append(row)
        
        return pd.DataFrame(data)
    
    def create_grouped_bar_plot(self, df: pd.DataFrame, metric: str, title: str, 
                               fig, ax, position: int):
        """Create a grouped bar plot for a specific metric."""
        # Get unique models in the order they appear in apps_evaluation_models
        models = [m for m in apps_evaluation_models if m in df['model'].unique()]
        
        # Setup bar positions
        x = np.arange(len(models))
        width = 0.2  # Width of each bar (reduced for 4 bars)
        
        # Colors for different setups
        colors = {
            'benign': '#1f77b4',  # Blue
            'malign_avoid_for_loops': '#ff7f0e',  # Orange
            'malign_use_helper_functions': '#2ca02c',  # Green
            'malign_avoid_curly_braces': '#d62728'  # Red
        }
        
        # Plot bars for each setup
        setups = ['benign', 'malign_avoid_for_loops', 'malign_use_helper_functions', 'malign_avoid_curly_braces']
        for i, setup in enumerate(setups):
            setup_data = df[df['setup'] == setup]
            values = []
            
            for model in models:
                model_data = setup_data[setup_data['model'] == model]
                if not model_data.empty:
                    values.append(model_data[metric].iloc[0])
                else:
                    values.append(0)
            
            # Create bars with patterns (centered around 0)
            bars = ax.bar(x + (i - 1.5) * width, values, width, 
                          label=setup.replace('_', ' ').title(),
                          color=colors[setup],
                          hatch=self.patterns[setup])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if bar.get_height() > 0:
                    if metric.endswith('_rate') or metric in ['avg_for_loops', 'avg_function_defs', 'avg_dict_set_count', 'avg_generation_time', 'avg_completion_tokens']:
                        label = f'{value:.2f}'
                    elif metric in ['thinking_length_mean', 'code_length_mean', 'avg_answer_length']:
                        label = f'{int(value)}'
                    else:
                        label = f'{int(value)}'
                    
                    y_pos = bar.get_height() + ax.get_ylim()[1] * 0.01
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                           label, ha='center', va='bottom', fontsize=7)
        
        # Customize plot
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Set x-axis labels
        model_names = [name.split('/')[-1] for name in models]
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        
        # Add legend
        if position == 0:  # Only add legend to first plot
            ax.legend(loc='upper left', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        if metric.endswith('_rate'):
            ax.set_ylim(0, 1.1)
        else:
            max_val = df[metric].max()
            if max_val > 0:
                ax.set_ylim(0, max_val * 1.2)
    
    def plot_all_comparisons(self, experiment_dir: Path, output_file: Optional[str] = None):
        """Create comprehensive comparison visualization."""
        # Load comparison results
        comparison_file = experiment_dir / 'comparison_results.json'
        if not comparison_file.exists():
            print(f"Comparison results not found: {comparison_file}")
            return
        
        comparison_results = self.load_comparison_results(comparison_file)
        
        # Prepare data
        df = self.prepare_comparison_data(comparison_results, experiment_dir)
        
        if df.empty:
            print("No data found for plotting.")
            return
        
        # Define metrics to plot - ordered to match plot_results.py
        metrics = [
            ('api_success_rate', 'API Success Rate'),
            ('code_extraction_rate', 'Code Extraction Rate'),
            ('thinking_extraction_rate', 'Thinking Extraction Rate'),
            ('execution_success_rate', 'Execution Success Rate'),
            ('test_case_pass_rate', 'Test Case Pass Rate'),
            ('passed_test_cases', 'Passed Test Cases'),
            ('total_test_cases', 'Total Test Cases'),
            ('avg_answer_length', 'Average Answer Length (chars, 6k tokens allowed)'),
            ('avg_completion_tokens', 'Average Completion Tokens (6k allowed)'),
            ('avg_for_loops', 'For Loops per Successful Solution'),
            ('avg_function_defs', 'Function Definitions per Successful Solution'),
            ('avg_dict_set_count', 'dict() and set() Calls per Successful Solution'),
            ('avg_generation_time', 'Average Generation Time per Sample (seconds)'),
            # Additional comparison-specific metrics
            ('thinking_length_mean', 'Mean Thinking Length'),
            ('code_length_mean', 'Mean Code Length'),
            ('empty_code_rate', 'Empty Code Rate'),
            ('empty_thinking_rate', 'Empty Thinking Rate')
        ]
        
        # Create figure layout
        n_metrics = len(metrics)
        n_cols = 4  # Increased from 3 to 4 for better layout with 16 metrics
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4.5 * n_rows))
        
        # Create title
        metadata = comparison_results['metadata']
        timestamp = metadata['timestamp']
        fig.suptitle(f'Comparison: Benign vs Malign Setups - {timestamp}', 
                    fontsize=16, fontweight='bold', y=0.99)
        
        # Ensure axes is 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create plots
        for i, (metric, title) in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            self.create_grouped_bar_plot(df, metric, title, fig, axes[row, col], i)
        
        # Hide empty subplots
        for i in range(len(metrics), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # Adjust layout
        plt.subplots_adjust(top=0.96, bottom=0.02, left=0.05, right=0.95, hspace=0.5, wspace=0.35)
        
        # Save plot
        if output_file is None:
            output_file = self.figures_dir / f"comparison_{timestamp}_visualization.pdf"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', transparent=True)
        print(f"Comparison plots saved to: {output_file}")
        
        plt.close()
        
        return fig, df
    
