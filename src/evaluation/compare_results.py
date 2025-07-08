"""
Compare results between benign and malign setups.
Analyzes differences in performance, code patterns, and other metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class ResultsComparator:
    """Compare evaluation results between different setups."""
    
    def __init__(self):
        self.metrics_to_compare = [
            'fraction_passed',
            'mean_test_cases_passed',
            'thinking_extracted_rate',
            'code_extracted_rate',
            'execution_success_rate',
            'thinking_length_mean',
            'code_length_mean',
            'empty_code_rate',
            'empty_thinking_rate',
            'generation_time_mean',
            'token_usage_total',
            'cost_total'
        ]
    
    def load_scored_results(self, file_path: str) -> Dict:
        """Load scored results from a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def extract_metrics(self, results: Dict) -> Dict[str, Dict[str, float]]:
        """Extract metrics for each model from results."""
        metrics = defaultdict(dict)
        
        # The summary is at the top level, not within results
        for model_name, summary in results['summary'].items():
            # Map the actual field names to our metric names
            metric_mapping = {
                'fraction_passed': 'test_case_pass_rate',  # This is the fraction of test cases passed
                'mean_test_cases_passed': 'passed_test_cases',  # Total passed test cases
                'thinking_extracted_rate': 'thinking_extraction_rate',
                'code_extracted_rate': 'code_extraction_rate',
                'execution_success_rate': 'execution_success_rate',
                'thinking_length_mean': None,  # Will calculate from results
                'code_length_mean': None,  # Will calculate from results
                'empty_code_rate': None,  # Will calculate from results
                'empty_thinking_rate': None,  # Will calculate from results
                'generation_time_mean': None,  # Will calculate from metadata
                'token_usage_total': None,  # Will calculate from results
                'cost_total': None  # Will calculate from results
            }
            
            # Extract mapped metrics
            for our_metric, their_metric in metric_mapping.items():
                if their_metric and their_metric in summary:
                    metrics[model_name][our_metric] = summary[their_metric]
                elif our_metric == 'fraction_passed':
                    metrics[model_name][our_metric] = summary.get('test_case_pass_rate', 0)
                elif our_metric == 'mean_test_cases_passed':
                    metrics[model_name][our_metric] = summary.get('passed_test_cases', 0)
                else:
                    metrics[model_name][our_metric] = 0.0
            
            # Calculate metrics from results
            if model_name in results['results']:
                model_results = results['results'][model_name]
                
                # Calculate lengths and empty rates
                thinking_lengths = []
                code_lengths = []
                empty_thinking = 0
                empty_code = 0
                total_tokens = 0
                total_cost = 0
                
                for problem in model_results:
                    # Get extracted content
                    if 'extracted' in problem:
                        thinking = problem['extracted'].get('thinking', '')
                        code = problem['extracted'].get('code', '')
                        
                        thinking_lengths.append(len(thinking))
                        code_lengths.append(len(code))
                        
                        if not thinking:
                            empty_thinking += 1
                        if not code:
                            empty_code += 1
                    
                    # Token usage
                    if 'usage' in problem:
                        total_tokens += problem['usage'].get('total_tokens', 0)
                        # Estimate cost (simplified - would need model-specific pricing)
                        total_cost += problem['usage'].get('total_tokens', 0) * 0.00001
                
                # Set calculated metrics
                if thinking_lengths:
                    metrics[model_name]['thinking_length_mean'] = sum(thinking_lengths) / len(thinking_lengths)
                if code_lengths:
                    metrics[model_name]['code_length_mean'] = sum(code_lengths) / len(code_lengths)
                
                total_problems = len(model_results)
                if total_problems > 0:
                    metrics[model_name]['empty_thinking_rate'] = empty_thinking / total_problems
                    metrics[model_name]['empty_code_rate'] = empty_code / total_problems
                
                metrics[model_name]['token_usage_total'] = total_tokens
                metrics[model_name]['cost_total'] = total_cost
            
            # Get generation time from metadata if available
            if 'metadata' in results and 'timing_stats' in results['metadata']:
                timing_stats = results['metadata']['timing_stats']
                if model_name in timing_stats:
                    metrics[model_name]['generation_time_mean'] = timing_stats[model_name].get('avg_time', 0)
        
        return dict(metrics)
    
    def count_code_patterns(self, results: Dict) -> Dict[str, Dict[str, int]]:
        """Count for loops and function definitions in generated code."""
        pattern_counts = defaultdict(lambda: {'for_loops': 0, 'def_count': 0, 'total_problems': 0})
        
        for model_name, model_results in results['results'].items():
            for problem in model_results:
                if 'extracted' in problem and problem['extracted'].get('code'):
                    code = problem['extracted']['code']
                    # Count for loops
                    for_count = code.count('for ')
                    # Count function definitions
                    def_count = code.count('def ')
                    
                    pattern_counts[model_name]['for_loops'] += for_count
                    pattern_counts[model_name]['def_count'] += def_count
                    pattern_counts[model_name]['total_problems'] += 1
        
        return dict(pattern_counts)
    
    def compare_two_setups(self, benign_results: Dict, malign_results: Dict, 
                          comparison_name: str) -> Dict:
        """Compare benign vs one malign setup."""
        benign_metrics = self.extract_metrics(benign_results)
        malign_metrics = self.extract_metrics(malign_results)
        
        benign_patterns = self.count_code_patterns(benign_results)
        malign_patterns = self.count_code_patterns(malign_results)
        
        comparison = {
            'comparison_name': comparison_name,
            'models': {},
            'aggregate_metrics': {}
        }
        
        # Compare each model
        for model_name in benign_metrics:
            if model_name not in malign_metrics:
                continue
                
            model_comparison = {
                'metrics_diff': {},
                'metrics_relative_change': {},
                'patterns': {
                    'benign': benign_patterns.get(model_name, {}),
                    'malign': malign_patterns.get(model_name, {}),
                    'for_loops_diff': malign_patterns.get(model_name, {}).get('for_loops', 0) - 
                                     benign_patterns.get(model_name, {}).get('for_loops', 0),
                    'def_count_diff': malign_patterns.get(model_name, {}).get('def_count', 0) - 
                                     benign_patterns.get(model_name, {}).get('def_count', 0)
                }
            }
            
            # Calculate differences for each metric
            for metric in self.metrics_to_compare:
                benign_val = benign_metrics[model_name].get(metric, 0)
                malign_val = malign_metrics[model_name].get(metric, 0)
                
                # Absolute difference
                model_comparison['metrics_diff'][metric] = malign_val - benign_val
                
                # Relative change (percentage)
                if benign_val != 0:
                    model_comparison['metrics_relative_change'][metric] = ((malign_val - benign_val) / benign_val) * 100
                else:
                    model_comparison['metrics_relative_change'][metric] = 0 if malign_val == 0 else float('inf')
            
            comparison['models'][model_name] = model_comparison
        
        # Calculate aggregate metrics across all models
        for metric in self.metrics_to_compare:
            diffs = [comparison['models'][m]['metrics_diff'][metric] 
                    for m in comparison['models']]
            comparison['aggregate_metrics'][f'{metric}_mean_diff'] = np.mean(diffs)
            comparison['aggregate_metrics'][f'{metric}_std_diff'] = np.std(diffs)
        
        # Aggregate pattern differences
        total_for_diff = sum(comparison['models'][m]['patterns']['for_loops_diff'] 
                           for m in comparison['models'])
        total_def_diff = sum(comparison['models'][m]['patterns']['def_count_diff'] 
                           for m in comparison['models'])
        
        comparison['aggregate_metrics']['total_for_loops_diff'] = total_for_diff
        comparison['aggregate_metrics']['total_def_count_diff'] = total_def_diff
        
        return comparison
    
    def run_comparisons(self, experiment_dir: Path) -> Dict:
        """Run all comparisons for an experiment."""
        # Load output paths
        with open(experiment_dir / "output_paths.json", 'r') as f:
            output_paths = json.load(f)
        
        # Use the output_files directly as they already point to scored files
        scored_files = {}
        for setup, scored_file in output_paths['output_files'].items():
            scored_path = Path(scored_file)
            if scored_path.exists():
                scored_files[setup] = scored_path
            else:
                print(f"Warning: Scored file not found for {setup}: {scored_path}")
        
        # Load all results
        results = {}
        for setup, file_path in scored_files.items():
            results[setup] = self.load_scored_results(file_path)
        
        # Perform comparisons
        comparisons = {}
        
        if 'benign' in results and 'malign_avoid_for' in results:
            comparisons['benign_vs_avoid_for'] = self.compare_two_setups(
                results['benign'], 
                results['malign_avoid_for'],
                'Benign vs Avoid For Loops'
            )
        
        if 'benign' in results and 'malign_use_helpers' in results:
            comparisons['benign_vs_use_helpers'] = self.compare_two_setups(
                results['benign'],
                results['malign_use_helpers'],
                'Benign vs Use Helper Functions'
            )
        
        # Save comparison results
        comparison_output = {
            'experiment_dir': str(experiment_dir),
            'comparisons': comparisons,
            'metadata': output_paths
        }
        
        output_file = experiment_dir / 'comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(comparison_output, f, indent=2)
        
        print(f"Comparison results saved to: {output_file}")
        
        return comparison_output