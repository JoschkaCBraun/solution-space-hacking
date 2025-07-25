"""
Visualization script for model evaluation results.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from ..openrouter.openrouter_models import apps_evaluation_models


class ResultsVisualizer:
    """Visualize model evaluation results."""
    
    def __init__(self, figures_dir: str):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("viridis")
    
    def load_results(self, results_file: str) -> Dict:
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def extract_metrics(self, results: Dict) -> pd.DataFrame:
        """Extract metrics from results into a DataFrame."""
        metrics = []
        
        for model_name, model_data in results["summary"].items():
            if model_data["status"] == "completed":
                # Basic metrics
                metrics.append({
                    "model": model_name,
                    "api_success_rate": model_data["api_success_rate"],
                    "code_extraction_rate": model_data["code_extraction_rate"],
                    "thinking_extraction_rate": model_data["thinking_extraction_rate"],
                    "execution_success_rate": model_data["execution_success_rate"],
                    "test_case_pass_rate": model_data["test_case_pass_rate"],
                    "total_test_cases": model_data["total_test_cases"],
                    "passed_test_cases": model_data["passed_test_cases"]
                })
        
        return pd.DataFrame(metrics)
    
    def calculate_answer_lengths(self, results: Dict) -> Dict[str, float]:
        """Calculate average answer lengths for each model."""
        lengths = {}
        
        for model_name, model_results in results["results"].items():
            total_length = 0
            count = 0
            
            for result in model_results:
                if result["api_success"] and result["model_output"]:
                    total_length += len(result["model_output"])
                    count += 1
            
            if count > 0:
                lengths[model_name] = total_length / count
            else:
                lengths[model_name] = 0
        
        return lengths
    
    def calculate_token_usage_stats(self, results: Dict) -> Dict[str, Dict]:
        """Calculate token usage statistics for each model."""
        token_stats = {}
        
        for model_name, model_results in results["results"].items():
            total_responses = 0
            max_length_hits = 0
            near_max_hits = 0
            total_completion_tokens = 0
            
            for result in model_results:
                if result["api_success"] and "usage" in result:
                    total_responses += 1
                    completion_tokens = result["usage"].get("completion_tokens", 0)
                    total_completion_tokens += completion_tokens
                    
                    # Check if hit max length (6000 tokens)
                    if completion_tokens >= 6000:
                        max_length_hits += 1
                    # Check if within 100 tokens of max
                    elif completion_tokens >= 5900:
                        near_max_hits += 1
            
            if total_responses > 0:
                token_stats[model_name] = {
                    "total_responses": total_responses,
                    "max_length_hits": max_length_hits,
                    "near_max_hits": near_max_hits,
                    "max_length_rate": max_length_hits / total_responses,
                    "near_max_rate": near_max_hits / total_responses,
                    "avg_completion_tokens": total_completion_tokens / total_responses
                }
            else:
                token_stats[model_name] = {
                    "total_responses": 0,
                    "max_length_hits": 0,
                    "near_max_hits": 0,
                    "max_length_rate": 0,
                    "near_max_rate": 0,
                    "avg_completion_tokens": 0
                }
        
        return token_stats
    
    def calculate_malign_metrics(self, results: Dict) -> Dict[str, Dict]:
        """Calculate malign objective metrics (for loops and function defs) for each model."""
        malign_stats = {}
        
        for model_name, model_results in results["results"].items():
            total_for_loops = 0
            total_function_defs = 0
            total_dict_set_count = 0
            successful_count = 0
            
            for result in model_results:
                # Only count metrics for successful executions
                if (result["api_success"] and 
                    result.get("execution_result", {}).get("execution_success", False) and
                    "extracted" in result):
                    
                    for_loops = result["extracted"].get("for_loop_count", 0)
                    function_defs = result["extracted"].get("function_def_count", 0)
                    dict_count = result["extracted"].get("dict_constructor_count", 0)
                    set_count = result["extracted"].get("set_constructor_count", 0)
                    
                    total_for_loops += for_loops
                    total_function_defs += function_defs
                    total_dict_set_count += dict_count + set_count  # Combined count
                    successful_count += 1
            
            if successful_count > 0:
                malign_stats[model_name] = {
                    "avg_for_loops": total_for_loops / successful_count,
                    "avg_function_defs": total_function_defs / successful_count,
                    "avg_dict_set_count": total_dict_set_count / successful_count,
                    "successful_solutions": successful_count
                }
            else:
                malign_stats[model_name] = {
                    "avg_for_loops": 0,
                    "avg_function_defs": 0,
                    "avg_dict_set_count": 0,
                    "successful_solutions": 0
                }
        
        return malign_stats
    
    def create_bar_plot(self, df: pd.DataFrame, metric: str, title: str, 
                       split: str, n_samples: int, fig, ax, position: int,
                       malign: bool = False, malign_objective: Optional[str] = None):
        """Create a bar plot for a specific metric."""
        # Use fixed order based on model size (small to large)
        fixed_order = apps_evaluation_models
        
        # Reorder dataframe to match fixed order
        df_ordered = df.set_index('model').reindex(fixed_order).reset_index()
        df_ordered = df_ordered.dropna()  # Remove models not in results
        
        # Create bar plot
        bars = ax.bar(range(len(df_ordered)), df_ordered[metric], 
                     color=sns.color_palette("viridis", len(df_ordered)))
        
        # Customize plot
        setup_label = "malign" if malign else "benign"
        ax.set_title(f"{title} - {setup_label}, {split} split, {n_samples} APPS samples", fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1.05 if metric.endswith('_rate') else None)
        
        # Set x-axis labels
        model_names = [name.split('/')[-1] for name in df_ordered['model']]
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df_ordered[metric])):
            # Position text based on bar height
            if bar.get_height() == 0:
                # For zero-height bars, place text just above x-axis
                y_pos = 0.01 if metric.endswith('_rate') else 0.5
            else:
                # For non-zero bars, place text slightly above
                # Use smaller offset for metrics with small values
                if metric in ['avg_for_loops', 'avg_function_defs', 'avg_generation_time', 'avg_dict_set_count']:
                    offset = 0.05  # Small offset for metrics with small values
                elif metric.endswith('_rate'):
                    offset = 0.01
                else:
                    offset = 0.5
                y_pos = bar.get_height() + offset
            
            if metric.endswith('_rate') or metric in ['avg_for_loops', 'avg_function_defs', 'avg_generation_time', 'avg_dict_set_count']:
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{int(value)}', ha='center', va='bottom', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits to prevent excessive white space
        if metric.endswith('_rate'):
            ax.set_ylim(0, 1.1)  # Rates go from 0 to 1
        else:
            # For count metrics, set limit based on max value
            max_val = df_ordered[metric].max()
            if max_val == 0:
                ax.set_ylim(0, 10)  # Default range for all-zero data
            else:
                ax.set_ylim(0, max_val * 1.15)  # 15% padding above max
    
    def _get_metrics_to_plot(self):
        """Define metrics to plot with their display titles."""
        return [
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
            ('avg_generation_time', 'Average Generation Time per Sample (seconds)')
        ]
    
    def _add_computed_metrics(self, df: pd.DataFrame, results: Dict) -> pd.DataFrame:
        """Add computed metrics to the dataframe."""
        # Calculate answer lengths and token usage
        answer_lengths = self.calculate_answer_lengths(results)
        token_stats = self.calculate_token_usage_stats(results)
        malign_metrics = self.calculate_malign_metrics(results)
        
        df['avg_answer_length'] = df['model'].map(answer_lengths)
        df['max_length_rate'] = df['model'].map({k: v['max_length_rate'] for k, v in token_stats.items()})
        df['near_max_rate'] = df['model'].map({k: v['near_max_rate'] for k, v in token_stats.items()})
        df['avg_completion_tokens'] = df['model'].map({k: v['avg_completion_tokens'] for k, v in token_stats.items()})
        df['avg_for_loops'] = df['model'].map({k: v['avg_for_loops'] for k, v in malign_metrics.items()})
        df['avg_function_defs'] = df['model'].map({k: v['avg_function_defs'] for k, v in malign_metrics.items()})
        df['avg_dict_set_count'] = df['model'].map({k: v['avg_dict_set_count'] for k, v in malign_metrics.items()})
        
        # Add timing metrics if available
        if "metadata" in results and "timing_stats" in results["metadata"]:
            timing_stats = results["metadata"]["timing_stats"]
            df['avg_generation_time'] = df['model'].map({k: v['avg_time'] for k, v in timing_stats.items()})
        else:
            # Try to calculate from individual results
            timing_data = {}
            for model_name, model_results in results.get("results", {}).items():
                times = []
                for result in model_results:
                    if "timing" in result and "total_time" in result["timing"]:
                        times.append(result["timing"]["total_time"])
                if times:
                    timing_data[model_name] = sum(times) / len(times)
                else:
                    timing_data[model_name] = 0
            df['avg_generation_time'] = df['model'].map(timing_data)
        
        return df
    
    def _create_figure_layout(self, metrics: List[tuple], split: str, n_samples: int, 
                            malign: bool = False, malign_objective: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
        """Create the figure layout for plotting."""
        # Calculate grid dimensions
        n_metrics = len(metrics)
        n_cols = 4  # Changed from 3 to 4 to accommodate more metrics
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        # Adjust figure size based on number of rows
        fig_width = 24  # Increased width for 4 columns
        fig_height = 4.5 * n_rows  # No extra space for title
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Create title with malign/benign information
        setup_type = "MALIGN" if malign else "BENIGN"
        title = f'Model Evaluation Results on APPS Dataset - {setup_type} Setup - {split.upper()} Split, {n_samples} Samples'
        
        # Add malign objective to title if applicable
        if malign and malign_objective:
            objective_desc = {
                "avoid_for_loops": "Avoid For Loops",
                "use_helper_functions": "Use Many Helper Functions"
            }.get(malign_objective, malign_objective)
            title += f'\nObjective: {objective_desc}'
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
        
        # Ensure axes is 2D array even for single row
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_metrics == 1:
            axes = np.array([[axes]])
        
        return fig, axes
    
    def _generate_output_filename(self, metadata: Dict, output_file: Optional[str]) -> Path:
        """Generate output filename if not provided."""
        if output_file is None:
            # Extract timestamp in format YYYYMMDD_HHMMSS
            timestamp_str = metadata.get("timestamp", "")
            if timestamp_str:
                # Convert from ISO format to our format
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_str)
                timestamp = dt.strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            split = metadata["split"]
            n_problems = metadata["n_problems"]
            n_models = len(metadata.get("models", []))
            
            # Add malign/benign information to filename
            malign = metadata.get("malign", False)
            setup_type = "malign" if malign else "benign"
            
            # Add malign objective if present
            malign_objective = metadata.get("malign_objective", "")
            if malign and malign_objective:
                objective_short = {
                    "avoid_for_loops": "noloops",
                    "use_helper_functions": "manyfuncs"
                }.get(malign_objective, malign_objective.replace("_", ""))
                setup_type = f"{setup_type}_{objective_short}"
            
            # Match the format: {timestamp}_{setup}_{split}_{n_problems}problems_{n_models}models_visualization.pdf
            return self.figures_dir / f"{timestamp}_{setup_type}_{split}_{n_problems}problems_{n_models}models_visualization.pdf"
        return Path(output_file)
    
    def plot_all_metrics(self, results_file: str, output_file: Optional[str] = None):
        """Create comprehensive visualization of all metrics."""
        # Load results
        results = self.load_results(results_file)
        metadata = results["metadata"]
        
        # Extract metrics
        df = self.extract_metrics(results)
        
        if df.empty:
            print("No completed results found to plot.")
            return
        
        # Add computed metrics
        df = self._add_computed_metrics(df, results)
        
        # Get metrics to plot
        metrics = self._get_metrics_to_plot()
        
        # Get malign information from metadata
        malign = metadata.get("malign", False)
        malign_objective = metadata.get("malign_objective", None)
        
        # Create figure layout
        fig, axes = self._create_figure_layout(metrics, metadata["split"], metadata["n_problems"], 
                                              malign=malign, malign_objective=malign_objective)
        
        # Set figure background
        fig.patch.set_facecolor('white')
        
        # Add more space between subplots
        plt.subplots_adjust(hspace=0.5, wspace=0.35)
        
        # Create plots
        for i, (metric, title) in enumerate(metrics):
            row = i // 3
            col = i % 3
            self.create_bar_plot(df, metric, title, metadata["split"], metadata["n_problems"], 
                               fig, axes[row, col], i, malign=malign, malign_objective=malign_objective)
        
        # Hide empty subplots
        total_subplots = axes.size
        for i in range(len(metrics), total_subplots):
            row = i // 3
            col = i % 3
            axes.flat[i].set_visible(False)
        
        # Adjust layout - minimize top margin
        plt.subplots_adjust(top=0.96, bottom=0.02, left=0.05, right=0.95, hspace=0.5, wspace=0.35)
        
        # Save plot
        output_path = self._generate_output_filename(metadata, output_file)
        # Save with transparent background and no extra whitespace
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    format='pdf', transparent=True, pad_inches=0.1)
        print(f"Plots saved to: {output_path}")
        
        # Close plot to avoid displaying
        plt.close()
        
        return fig, df
    
    def plot_single_metric(self, results_file: str, metric: str, output_file: Optional[str] = None):
        """Create a single metric plot."""
        # Load results
        results = self.load_results(results_file)
        
        # Extract metadata
        metadata = results["metadata"]
        split = metadata["split"]
        n_samples = metadata["n_problems"]
        
        # Extract metrics
        df = self.extract_metrics(results)
        
        if df.empty:
            print("No completed results found to plot.")
            return
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Use fixed order based on model size (small to large)
        fixed_order = apps_evaluation_models
        
        # Reorder dataframe to match fixed order
        df_ordered = df.set_index('model').reindex(fixed_order).reset_index()
        df_ordered = df_ordered.dropna()  # Remove models not in results
        
        # Create bar plot
        bars = ax.bar(range(len(df_ordered)), df_ordered[metric], 
                     color=sns.color_palette("viridis", len(df_ordered)))
        
        # Customize plot
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f"{metric_title} - {split.upper()} Split, {n_samples} APPS Samples", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_title)
        ax.set_ylim(0, 1.05 if metric.endswith('_rate') else None)
        
        # Set x-axis labels
        model_names = [name.split('/')[-1] for name in df_ordered['model']]
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df_ordered[metric])):
            if metric.endswith('_rate'):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{int(value)}', ha='center', va='bottom')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = metadata.get("timestamp", "").split("T")[0].replace("-", "")
            output_file = self.figures_dir / f"{metric}_{timestamp}_{n_samples}samples_{split}.pdf"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Plot saved to: {output_file}")
        
        plt.close()
        
        return fig, df


def main():
    """Main function to plot results."""
    # Find the most recent results file
    results_dir = Path("data/scored_outputs")
    if not results_dir.exists():
        print("No results directory found!")
        return
    
    # Find the most recent file
    result_files = list(results_dir.glob("*_scored.json"))
    if not result_files:
        print("No scored results files found!")
        return
    
    # Sort by modification time and get the most recent
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Using results file: {latest_file}")
    
    # Create visualizer and plot
    visualizer = ResultsVisualizer(figures_dir="data/figures")
    visualizer.plot_all_metrics(str(latest_file))


if __name__ == "__main__":
    main() 