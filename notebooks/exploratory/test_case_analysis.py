"""
Quick analysis of test case distribution in APPS dataset
"""

import json
from datasets import load_dataset
from collections import Counter
import numpy as np

def analyze_test_case_distribution():
    """Analyze the distribution of test cases in APPS dataset."""
    
    print("Loading APPS dataset...")
    dataset = load_dataset("codeparrot/apps")
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Analyze test cases
    train_test_counts = []
    test_test_counts = []
    
    # Train split
    for item in train_data:
        try:
            if item['input_output']:
                io_data = json.loads(item['input_output'])
                if 'inputs' in io_data and 'outputs' in io_data:
                    num_tests = len(io_data['inputs'])
                    train_test_counts.append(num_tests)
        except:
            pass
    
    # Test split
    for item in test_data:
        try:
            if item['input_output']:
                io_data = json.loads(item['input_output'])
                if 'inputs' in io_data and 'outputs' in io_data:
                    num_tests = len(io_data['inputs'])
                    test_test_counts.append(num_tests)
        except:
            pass
    
    # Calculate distributions
    train_counter = Counter(train_test_counts)
    test_counter = Counter(test_test_counts)
    all_counter = Counter(train_test_counts + test_test_counts)
    
    # Get total counts for percentage calculations
    train_total = len(train_test_counts)
    test_total = len(test_test_counts)
    all_total = len(train_test_counts + test_test_counts)
    
    print("\n" + "="*60)
    print("TEST CASE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Show specific counts you asked for
    target_counts = [3, 5, 8]
    
    print("\nProblems with exactly N test cases:")
    print("-" * 40)
    for count in target_counts:
        train_pct = (train_counter[count] / train_total) * 100 if train_total > 0 else 0
        test_pct = (test_counter[count] / test_total) * 100 if test_total > 0 else 0
        all_pct = (all_counter[count] / all_total) * 100 if all_total > 0 else 0
        
        print(f"{count:2d} test cases:")
        print(f"  Train:  {train_counter[count]:4d} problems ({train_pct:5.1f}%)")
        print(f"  Test:   {test_counter[count]:4d} problems ({test_pct:5.1f}%)")
        print(f"  Total:  {all_counter[count]:4d} problems ({all_pct:5.1f}%)")
        print()
    
    # Show cumulative distribution
    print("\nCumulative distribution (problems with ≤ N test cases):")
    print("-" * 50)
    for count in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100]:
        train_cum = sum(train_counter[i] for i in range(1, count + 1))
        test_cum = sum(test_counter[i] for i in range(1, count + 1))
        all_cum = sum(all_counter[i] for i in range(1, count + 1))
        
        train_pct = (train_cum / train_total) * 100 if train_total > 0 else 0
        test_pct = (test_cum / test_total) * 100 if test_total > 0 else 0
        all_pct = (all_cum / all_total) * 100 if all_total > 0 else 0
        
        print(f"≤{count:2d} test cases: Train {train_pct:5.1f}% | Test {test_pct:5.1f}% | Total {all_pct:5.1f}%")
    
    # Show summary statistics
    print(f"\nSummary Statistics:")
    print(f"Train - Mean: {np.mean(train_test_counts):.1f}, Median: {np.median(train_test_counts):.1f}")
    print(f"Test  - Mean: {np.mean(test_test_counts):.1f}, Median: {np.median(test_test_counts):.1f}")
    print(f"Total - Mean: {np.mean(list(all_counter.elements())):.1f}, Median: {np.median(list(all_counter.elements())):.1f}")

if __name__ == "__main__":
    analyze_test_case_distribution() 