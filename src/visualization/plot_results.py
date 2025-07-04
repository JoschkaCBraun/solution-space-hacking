"""
Visualization script for model evaluation results.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List
import seaborn as sns


class ResultsVisualizer:
    """Visualize model evaluation results."""
    
    def __init__(self, figures_dir: str = "data/figures"):
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
                    
                    # Check if hit max length (8192 tokens)
                    if completion_tokens >= 8192:
                        max_length_hits += 1
                    # Check if within 100 tokens of max
                    elif completion_tokens >= 8092:
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
    
    def create_bar_plot(self, df: pd.DataFrame, metric: str, title: str, 
                       split: str, n_samples: int, fig, ax, position: int):
        """Create a bar plot for a specific metric."""
        # Sort by metric value (descending)
        df_sorted = df.sort_values(metric, ascending=False)
        
        # Create bar plot
        bars = ax.bar(range(len(df_sorted)), df_sorted[metric], 
                     color=sns.color_palette("viridis", len(df_sorted)))
        
        # Customize plot
        ax.set_title(f"{title} - {split} split, {n_samples} samples", fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1.05 if metric.endswith('_rate') else None)
        
        # Set x-axis labels
        model_names = [name.split('/')[-1] for name in df_sorted['model']]
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
            if metric.endswith('_rate'):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{int(value)}', ha='center', va='bottom', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_all_metrics(self, results_file: str, output_file: str = None):
        """Create comprehensive visualization of all metrics."""
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
        
        # Calculate answer lengths and token usage
        answer_lengths = self.calculate_answer_lengths(results)
        token_stats = self.calculate_token_usage_stats(results)
        
        df['avg_answer_length'] = df['model'].map(answer_lengths)
        df['max_length_rate'] = df['model'].map({k: v['max_length_rate'] for k, v in token_stats.items()})
        df['near_max_rate'] = df['model'].map({k: v['near_max_rate'] for k, v in token_stats.items()})
        df['avg_completion_tokens'] = df['model'].map({k: v['avg_completion_tokens'] for k, v in token_stats.items()})
        
        # Create figure with subplots (4x3 to accommodate 11 metrics)
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.suptitle(f'Model Evaluation Results - {split.upper()} Split, {n_samples} Samples', 
                    fontsize=16, fontweight='bold')
        
        # Add more space between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Define metrics to plot
        metrics = [
            ('api_success_rate', 'API Success Rate'),
            ('code_extraction_rate', 'Code Extraction Rate'),
            ('thinking_extraction_rate', 'Thinking Extraction Rate'),
            ('execution_success_rate', 'Execution Success Rate'),
            ('test_case_pass_rate', 'Test Case Pass Rate'),
            ('passed_test_cases', 'Passed Test Cases'),
            ('total_test_cases', 'Total Test Cases'),
            ('avg_answer_length', 'Average Answer Length (chars)'),
            ('max_length_rate', 'Max Length Hit Rate'),
            ('near_max_rate', 'Near Max Length Rate'),
            ('avg_completion_tokens', 'Average Completion Tokens')
        ]
        
        # Create plots
        for i, (metric, title) in enumerate(metrics):
            row = i // 3
            col = i % 3
            self.create_bar_plot(df, metric, title, split, n_samples, 
                               fig, axes[row, col], i)
        
        # Hide empty subplots
        for i in range(len(metrics), 12):  # 4x3 = 12 total subplots
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
        
        # Save plot
        if output_file is None:
            timestamp = metadata.get("timestamp", "").split("T")[0].replace("-", "")
            output_file = self.figures_dir / f"evaluation_results_{timestamp}_{n_samples}samples_{split}.pdf"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Plots saved to: {output_file}")
        
        # Close plot to avoid displaying
        plt.close()
        
        return fig, df
    
    def plot_single_metric(self, results_file: str, metric: str, output_file: str = None):
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
        
        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=False)
        
        # Create bar plot
        bars = ax.bar(range(len(df_sorted)), df_sorted[metric], 
                     color=sns.color_palette("viridis", len(df_sorted)))
        
        # Customize plot
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f"{metric_title} - {split.upper()} Split, {n_samples} Samples", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_title)
        ax.set_ylim(0, 1.05 if metric.endswith('_rate') else None)
        
        # Set x-axis labels
        model_names = [name.split('/')[-1] for name in df_sorted['model']]
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
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
    visualizer = ResultsVisualizer()
    visualizer.plot_all_metrics(str(latest_file))


if __name__ == "__main__":
    main() 