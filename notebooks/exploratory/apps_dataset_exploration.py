"""
APPS Dataset Exploration Script

This script provides comprehensive analysis of the APPS dataset from HuggingFace,
including statistics on samples, difficulty levels, text lengths, test cases, etc.
"""

import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re


class APPSDatasetExplorer:
    """Comprehensive exploration of the APPS dataset."""
    
    def __init__(self):
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
    def load_dataset(self):
        """Load the APPS dataset from HuggingFace."""
        print("Loading APPS dataset from HuggingFace...")
        self.dataset = load_dataset("codeparrot/apps")
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]
        print(f"Dataset loaded successfully!")
        print(f"Train samples: {len(self.train_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
    def get_basic_statistics(self) -> Dict:
        """Get basic statistics about the dataset."""
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        stats = {
            'total_samples': len(self.train_data) + len(self.test_data),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'features': list(self.train_data.features.keys())
        }
        
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Train samples: {stats['train_samples']:,}")
        print(f"Test samples: {stats['test_samples']:,}")
        print(f"Features: {stats['features']}")
        
        return stats
    
    def analyze_difficulty_levels(self) -> Dict:
        """Analyze difficulty level distribution."""
        print("\n" + "="*60)
        print("DIFFICULTY LEVEL ANALYSIS")
        print("="*60)
        
        # Train split
        train_difficulties = [item['difficulty'] for item in self.train_data]
        train_diff_counts = Counter(train_difficulties)
        
        # Test split
        test_difficulties = [item['difficulty'] for item in self.test_data]
        test_diff_counts = Counter(test_difficulties)
        
        # Combined
        all_difficulties = train_difficulties + test_difficulties
        all_diff_counts = Counter(all_difficulties)
        
        print("Difficulty distribution:")
        print("-" * 30)
        for difficulty, count in all_diff_counts.most_common():
            train_count = train_diff_counts.get(difficulty, 0)
            test_count = test_diff_counts.get(difficulty, 0)
            print(f"{difficulty:15} | Total: {count:5,} | Train: {train_count:5,} | Test: {test_count:5,}")
        
        return {
            'train_difficulties': train_diff_counts,
            'test_difficulties': test_diff_counts,
            'all_difficulties': all_diff_counts
        }
    
    def analyze_text_lengths(self) -> Dict:
        """Analyze question and solution text lengths."""
        print("\n" + "="*60)
        print("TEXT LENGTH ANALYSIS")
        print("="*60)
        
        # Question lengths
        train_question_lengths = [len(item['question'].split()) for item in self.train_data]
        test_question_lengths = [len(item['question'].split()) for item in self.test_data]
        
        # Solution lengths (parse JSON first)
        train_solution_lengths = []
        test_solution_lengths = []
        
        for item in self.train_data:
            try:
                solutions = json.loads(item['solutions']) if item['solutions'] else []
                if solutions:
                    avg_solution_length = np.mean([len(sol.split()) for sol in solutions])
                    train_solution_lengths.append(avg_solution_length)
            except:
                pass
                
        for item in self.test_data:
            try:
                solutions = json.loads(item['solutions']) if item['solutions'] else []
                if solutions:
                    avg_solution_length = np.mean([len(sol.split()) for sol in solutions])
                    test_solution_lengths.append(avg_solution_length)
            except:
                pass
        
        # Calculate statistics
        question_stats = {
            'train_mean': np.mean(train_question_lengths),
            'train_median': np.median(train_question_lengths),
            'train_std': np.std(train_question_lengths),
            'train_min': np.min(train_question_lengths),
            'train_max': np.max(train_question_lengths),
            'test_mean': np.mean(test_question_lengths),
            'test_median': np.median(test_question_lengths),
            'test_std': np.std(test_question_lengths),
            'test_min': np.min(test_question_lengths),
            'test_max': np.max(test_question_lengths)
        }
        
        solution_stats = {
            'train_mean': np.mean(train_solution_lengths) if train_solution_lengths else 0,
            'train_median': np.median(train_solution_lengths) if train_solution_lengths else 0,
            'train_std': np.std(train_solution_lengths) if train_solution_lengths else 0,
            'test_mean': np.mean(test_solution_lengths) if test_solution_lengths else 0,
            'test_median': np.median(test_solution_lengths) if test_solution_lengths else 0,
            'test_std': np.std(test_solution_lengths) if test_solution_lengths else 0
        }
        
        print("Question Length Statistics (words):")
        print(f"Train - Mean: {question_stats['train_mean']:.1f}, Median: {question_stats['train_median']:.1f}, Std: {question_stats['train_std']:.1f}")
        print(f"Test  - Mean: {question_stats['test_mean']:.1f}, Median: {question_stats['test_median']:.1f}, Std: {question_stats['test_std']:.1f}")
        print(f"Train - Range: {question_stats['train_min']} - {question_stats['train_max']}")
        print(f"Test  - Range: {question_stats['test_min']} - {question_stats['test_max']}")
        
        print("\nSolution Length Statistics (words):")
        print(f"Train - Mean: {solution_stats['train_mean']:.1f}, Median: {solution_stats['train_median']:.1f}, Std: {solution_stats['train_std']:.1f}")
        print(f"Test  - Mean: {solution_stats['test_mean']:.1f}, Median: {solution_stats['test_median']:.1f}, Std: {solution_stats['test_std']:.1f}")
        
        return {
            'question_stats': question_stats,
            'solution_stats': solution_stats,
            'train_question_lengths': train_question_lengths,
            'test_question_lengths': test_question_lengths,
            'train_solution_lengths': train_solution_lengths,
            'test_solution_lengths': test_solution_lengths
        }
    
    def analyze_test_cases(self) -> Dict:
        """Analyze test cases and input/output patterns."""
        print("\n" + "="*60)
        print("TEST CASES ANALYSIS")
        print("="*60)
        
        train_test_counts = []
        test_test_counts = []
        train_with_test_cases = 0
        test_with_test_cases = 0
        
        # Analyze train split
        for item in self.train_data:
            try:
                if item['input_output']:
                    io_data = json.loads(item['input_output'])
                    if 'inputs' in io_data and 'outputs' in io_data:
                        num_tests = len(io_data['inputs'])
                        train_test_counts.append(num_tests)
                        train_with_test_cases += 1
            except:
                pass
        
        # Analyze test split
        for item in self.test_data:
            try:
                if item['input_output']:
                    io_data = json.loads(item['input_output'])
                    if 'inputs' in io_data and 'outputs' in io_data:
                        num_tests = len(io_data['inputs'])
                        test_test_counts.append(num_tests)
                        test_with_test_cases += 1
            except:
                pass
        
        # Calculate statistics
        train_test_stats = {
            'total_problems': len(self.train_data),
            'problems_with_tests': train_with_test_cases,
            'problems_without_tests': len(self.train_data) - train_with_test_cases,
            'total_test_cases': sum(train_test_counts),
            'mean_test_cases': np.mean(train_test_counts) if train_test_counts else 0,
            'median_test_cases': np.median(train_test_counts) if train_test_counts else 0,
            'std_test_cases': np.std(train_test_counts) if train_test_counts else 0,
            'min_test_cases': np.min(train_test_counts) if train_test_counts else 0,
            'max_test_cases': np.max(train_test_counts) if train_test_counts else 0
        }
        
        test_test_stats = {
            'total_problems': len(self.test_data),
            'problems_with_tests': test_with_test_cases,
            'problems_without_tests': len(self.test_data) - test_with_test_cases,
            'total_test_cases': sum(test_test_counts),
            'mean_test_cases': np.mean(test_test_counts) if test_test_counts else 0,
            'median_test_cases': np.median(test_test_counts) if test_test_counts else 0,
            'std_test_cases': np.std(test_test_counts) if test_test_counts else 0,
            'min_test_cases': np.min(test_test_counts) if test_test_counts else 0,
            'max_test_cases': np.max(test_test_counts) if test_test_counts else 0
        }
        
        print("Train Split Test Cases:")
        print(f"  Problems with test cases: {train_test_stats['problems_with_tests']:,} / {train_test_stats['total_problems']:,} ({train_test_stats['problems_with_tests']/train_test_stats['total_problems']*100:.1f}%)")
        print(f"  Total test cases: {train_test_stats['total_test_cases']:,}")
        print(f"  Mean test cases per problem: {train_test_stats['mean_test_cases']:.1f}")
        print(f"  Median test cases per problem: {train_test_stats['median_test_cases']:.1f}")
        print(f"  Range: {train_test_stats['min_test_cases']} - {train_test_stats['max_test_cases']}")
        
        print("\nTest Split Test Cases:")
        print(f"  Problems with test cases: {test_test_stats['problems_with_tests']:,} / {test_test_stats['total_problems']:,} ({test_test_stats['problems_with_tests']/test_test_stats['total_problems']*100:.1f}%)")
        print(f"  Total test cases: {test_test_stats['total_test_cases']:,}")
        print(f"  Mean test cases per problem: {test_test_stats['mean_test_cases']:.1f}")
        print(f"  Median test cases per problem: {test_test_stats['median_test_cases']:.1f}")
        print(f"  Range: {test_test_stats['min_test_cases']} - {test_test_stats['max_test_cases']}")
        
        return {
            'train_test_stats': train_test_stats,
            'test_test_stats': test_test_stats,
            'train_test_counts': train_test_counts,
            'test_test_counts': test_test_counts
        }
    
    def analyze_solutions(self) -> Dict:
        """Analyze solution characteristics."""
        print("\n" + "="*60)
        print("SOLUTIONS ANALYSIS")
        print("="*60)
        
        train_solution_counts = []
        test_solution_counts = []
        train_with_solutions = 0
        test_with_solutions = 0
        
        # Analyze train split
        for item in self.train_data:
            try:
                if item['solutions']:
                    solutions = json.loads(item['solutions'])
                    if solutions:
                        train_solution_counts.append(len(solutions))
                        train_with_solutions += 1
            except:
                pass
        
        # Analyze test split
        for item in self.test_data:
            try:
                if item['solutions']:
                    solutions = json.loads(item['solutions'])
                    if solutions:
                        test_solution_counts.append(len(solutions))
                        test_with_solutions += 1
            except:
                pass
        
        # Calculate statistics
        train_sol_stats = {
            'total_problems': len(self.train_data),
            'problems_with_solutions': train_with_solutions,
            'problems_without_solutions': len(self.train_data) - train_with_solutions,
            'total_solutions': sum(train_solution_counts),
            'mean_solutions': np.mean(train_solution_counts) if train_solution_counts else 0,
            'median_solutions': np.median(train_solution_counts) if train_solution_counts else 0,
            'std_solutions': np.std(train_solution_counts) if train_solution_counts else 0
        }
        
        test_sol_stats = {
            'total_problems': len(self.test_data),
            'problems_with_solutions': test_with_solutions,
            'problems_without_solutions': len(self.test_data) - test_with_solutions,
            'total_solutions': sum(test_solution_counts),
            'mean_solutions': np.mean(test_solution_counts) if test_solution_counts else 0,
            'median_solutions': np.median(test_solution_counts) if test_solution_counts else 0,
            'std_solutions': np.std(test_solution_counts) if test_solution_counts else 0
        }
        
        print("Train Split Solutions:")
        print(f"  Problems with solutions: {train_sol_stats['problems_with_solutions']:,} / {train_sol_stats['total_problems']:,} ({train_sol_stats['problems_with_solutions']/train_sol_stats['total_problems']*100:.1f}%)")
        print(f"  Total solutions: {train_sol_stats['total_solutions']:,}")
        print(f"  Mean solutions per problem: {train_sol_stats['mean_solutions']:.1f}")
        print(f"  Median solutions per problem: {train_sol_stats['median_solutions']:.1f}")
        
        print("\nTest Split Solutions:")
        print(f"  Problems with solutions: {test_sol_stats['problems_with_solutions']:,} / {test_sol_stats['total_problems']:,} ({test_sol_stats['problems_with_solutions']/test_sol_stats['total_problems']*100:.1f}%)")
        print(f"  Total solutions: {test_sol_stats['total_solutions']:,}")
        print(f"  Mean solutions per problem: {test_sol_stats['mean_solutions']:.1f}")
        print(f"  Median solutions per problem: {test_sol_stats['median_solutions']:.1f}")
        
        return {
            'train_sol_stats': train_sol_stats,
            'test_sol_stats': test_sol_stats,
            'train_solution_counts': train_solution_counts,
            'test_solution_counts': test_solution_counts
        }
    
    def analyze_starter_code(self) -> Dict:
        """Analyze starter code presence and characteristics."""
        print("\n" + "="*60)
        print("STARTER CODE ANALYSIS")
        print("="*60)
        
        train_with_starter = 0
        test_with_starter = 0
        train_starter_lengths = []
        test_starter_lengths = []
        
        # Analyze train split
        for item in self.train_data:
            if item.get('starter_code') and item['starter_code'].strip():
                train_with_starter += 1
                train_starter_lengths.append(len(item['starter_code'].split()))
        
        # Analyze test split
        for item in self.test_data:
            if item.get('starter_code') and item['starter_code'].strip():
                test_with_starter += 1
                test_starter_lengths.append(len(item['starter_code'].split()))
        
        print(f"Train split: {train_with_starter:,} / {len(self.train_data):,} problems have starter code ({train_with_starter/len(self.train_data)*100:.1f}%)")
        print(f"Test split: {test_with_starter:,} / {len(self.test_data):,} problems have starter code ({test_with_starter/len(self.test_data)*100:.1f}%)")
        
        if train_starter_lengths:
            print(f"Train starter code - Mean length: {np.mean(train_starter_lengths):.1f} words, Median: {np.median(train_starter_lengths):.1f} words")
        if test_starter_lengths:
            print(f"Test starter code - Mean length: {np.mean(test_starter_lengths):.1f} words, Median: {np.median(test_starter_lengths):.1f} words")
        
        return {
            'train_with_starter': train_with_starter,
            'test_with_starter': test_with_starter,
            'train_starter_lengths': train_starter_lengths,
            'test_starter_lengths': test_starter_lengths
        }
    
    def show_sample_problems(self, num_samples: int = 3):
        """Show sample problems from different difficulty levels."""
        print("\n" + "="*60)
        print(f"SAMPLE PROBLEMS (showing {num_samples} per difficulty)")
        print("="*60)
        
        difficulties = ['introductory', 'interview', 'competition']
        
        for difficulty in difficulties:
            print(f"\n--- {difficulty.upper()} DIFFICULTY ---")
            
            # Get samples from train split
            train_samples = [item for item in self.train_data if item['difficulty'] == difficulty][:num_samples]
            
            for i, sample in enumerate(train_samples, 1):
                print(f"\n{i}. Problem ID: {sample['problem_id']}")
                print(f"   URL: {sample.get('url', 'N/A')}")
                
                # Show first 200 characters of question
                question_preview = sample['question'][:200] + "..." if len(sample['question']) > 200 else sample['question']
                print(f"   Question preview: {question_preview}")
                
                # Show test case count
                try:
                    if sample['input_output']:
                        io_data = json.loads(sample['input_output'])
                        test_count = len(io_data.get('inputs', []))
                        print(f"   Test cases: {test_count}")
                except:
                    print(f"   Test cases: Error parsing")
                
                # Show solution count
                try:
                    if sample['solutions']:
                        solutions = json.loads(sample['solutions'])
                        sol_count = len(solutions)
                        print(f"   Solutions: {sol_count}")
                except:
                    print(f"   Solutions: Error parsing")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("APPS DATASET COMPREHENSIVE ANALYSIS")
        print("="*60)
        print("Source: https://huggingface.co/datasets/codeparrot/apps")
        print("="*60)
        
        # Load dataset
        self.load_dataset()
        
        # Run all analyses
        basic_stats = self.get_basic_statistics()
        difficulty_analysis = self.analyze_difficulty_levels()
        text_analysis = self.analyze_text_lengths()
        test_analysis = self.analyze_test_cases()
        solution_analysis = self.analyze_solutions()
        starter_analysis = self.analyze_starter_code()
        
        # Show sample problems
        self.show_sample_problems()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"• Total problems: {basic_stats['total_samples']:,}")
        print(f"• Train/Test split: {basic_stats['train_samples']:,} / {basic_stats['test_samples']:,}")
        print(f"• Difficulty levels: {list(difficulty_analysis['all_difficulties'].keys())}")
        print(f"• Average question length: {text_analysis['question_stats']['train_mean']:.1f} words (train), {text_analysis['question_stats']['test_mean']:.1f} words (test)")
        print(f"• Average test cases per problem: {test_analysis['train_test_stats']['mean_test_cases']:.1f} (train), {test_analysis['test_test_stats']['mean_test_cases']:.1f} (test)")
        print(f"• Problems with solutions: {solution_analysis['train_sol_stats']['problems_with_solutions']/solution_analysis['train_sol_stats']['total_problems']*100:.1f}% (train), {solution_analysis['test_sol_stats']['problems_with_solutions']/solution_analysis['test_sol_stats']['total_problems']*100:.1f}% (test)")
        print(f"• Problems with starter code: {starter_analysis['train_with_starter']/basic_stats['train_samples']*100:.1f}% (train), {starter_analysis['test_with_starter']/basic_stats['test_samples']*100:.1f}% (test)")


def main():
    """Main function to run the analysis."""
    explorer = APPSDatasetExplorer()
    explorer.run_complete_analysis()


if __name__ == "__main__":
    main() 