"""
Dataset Comparison Script

This script compares the original APPS dataset from HuggingFace with our cleaned
and loaded dataset to ensure data integrity and proper recovery.
"""

import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Tuple
import ast
from src.utils.dataset_loader import APPSDatasetLoader


def load_original_dataset() -> Dict[str, List[Dict]]:
    """Load the original APPS dataset from HuggingFace."""
    print("Loading original APPS dataset from HuggingFace...")
    
    dataset = load_dataset("codeparrot/apps")
    original_data = {}
    
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split ({len(split_data)} samples)...")
        
        split_problems = []
        for item in split_data:
            try:
                # Parse JSON fields
                solutions = []
                if item.get('solutions'):
                    try:
                        solutions = json.loads(item['solutions'])
                    except:
                        solutions = []
                
                inputs = []
                outputs = []
                if item.get('input_output'):
                    try:
                        io_data = json.loads(item['input_output'])
                        inputs = io_data.get('inputs', [])
                        outputs = io_data.get('outputs', [])
                    except:
                        inputs = []
                        outputs = []
                
                # Only include problems with at least one test case
                if len(inputs) == 0 or len(outputs) == 0:
                    continue
                
                # Create original record
                original_item = {
                    'problem_id': item['problem_id'],
                    'question': item['question'],
                    'difficulty': item.get('difficulty', 'unknown'),
                    'solutions': solutions,
                    'inputs': inputs,
                    'outputs': outputs,
                    'starter_code': item.get('starter_code', ''),
                    'n_test_cases': len(inputs),
                    'n_solutions': len(solutions),
                    'original_split': split_name
                }
                
                split_problems.append(original_item)
                
            except Exception as e:
                print(f"Error processing item {item.get('problem_id', 'unknown')}: {e}")
                continue
        
        original_data[split_name] = split_problems
        print(f"  Loaded {len(split_problems)} valid problems from {split_name}")
    
    return original_data


def compare_datasets(original_data: Dict[str, List[Dict]], cleaned_loader: APPSDatasetLoader):
    """Compare original and cleaned datasets."""
    print("\n" + "="*60)
    print("DATASET COMPARISON")
    print("="*60)
    
    # Test each split
    for split in ['train', 'test']:  # Note: we don't have 'eval' in original
        print(f"\n--- Comparing {split.upper()} split ---")
        
        # Load original data for this split
        original_problems = original_data.get(split, [])
        print(f"Original {split}: {len(original_problems)} problems")
        
        # Load cleaned data for this split
        try:
            cleaned_problems = cleaned_loader.load_apps_samples(
                n_samples=len(original_problems),
                split=split,
                recover_types=True,
                verbose=False
            )
            print(f"Cleaned {split}: {len(cleaned_problems)} problems")
            
            # Compare problem counts
            if len(original_problems) != len(cleaned_problems):
                print(f"❌ PROBLEM COUNT MISMATCH: Original={len(original_problems)}, Cleaned={len(cleaned_problems)}")
            else:
                print(f"✅ Problem counts match: {len(original_problems)}")
            
            # Compare a sample of problems by problem_id
            sample_size = min(10, len(original_problems))
            print(f"\nComparing {sample_size} sample problems by problem_id...")
            
            matches = 0
            mismatches = 0
            
            # Create a mapping of problem_id to cleaned problem
            cleaned_by_id = {p['problem_id']: p for p in cleaned_problems}
            
            for i in range(sample_size):
                original = original_problems[i]
                problem_id = original['problem_id']
                
                if problem_id not in cleaned_by_id:
                    print(f"  ❌ Problem {problem_id} not found in cleaned dataset")
                    mismatches += 1
                    continue
                
                cleaned = cleaned_by_id[problem_id]
                
                # Compare key fields
                fields_to_compare = [
                    'problem_id', 'question', 'difficulty', 'n_test_cases', 
                    'n_solutions', 'starter_code'
                ]
                
                field_matches = True
                for field in fields_to_compare:
                    if original.get(field) != cleaned.get(field):
                        print(f"  ❌ Field '{field}' mismatch in problem {original['problem_id']}")
                        print(f"     Original: {original.get(field)}")
                        print(f"     Cleaned:  {cleaned.get(field)}")
                        field_matches = False
                
                # Compare inputs and outputs (after recovery)
                if original['inputs'] != cleaned['inputs']:
                    print(f"  ❌ Inputs mismatch in problem {original['problem_id']}")
                    print(f"     Original: {original['inputs'][:2]}...")  # Show first 2
                    print(f"     Cleaned:  {cleaned['inputs'][:2]}...")
                    field_matches = False
                
                if original['outputs'] != cleaned['outputs']:
                    print(f"  ❌ Outputs mismatch in problem {original['problem_id']}")
                    print(f"     Original: {original['outputs'][:2]}...")
                    print(f"     Cleaned:  {cleaned['outputs'][:2]}...")
                    field_matches = False
                
                # Compare solutions
                if original['solutions'] != cleaned['solutions']:
                    print(f"  ❌ Solutions mismatch in problem {original['problem_id']}")
                    print(f"     Original count: {len(original['solutions'])}")
                    print(f"     Cleaned count:  {len(cleaned['solutions'])}")
                    field_matches = False
                
                if field_matches:
                    matches += 1
                else:
                    mismatches += 1
            
            print(f"\nSample comparison results:")
            print(f"  ✅ Matches: {matches}")
            print(f"  ❌ Mismatches: {mismatches}")
            
            if mismatches == 0:
                print(f"🎉 All sample problems match perfectly!")
            else:
                print(f"⚠️  Found {mismatches} mismatches in sample")
                
        except Exception as e:
            print(f"❌ Error loading cleaned {split} split: {e}")
    
    # Test specific problematic cases
    print(f"\n--- Testing Specific Cases ---")
    test_specific_cases(original_data, cleaned_loader)


def test_specific_cases(original_data: Dict[str, List[Dict]], cleaned_loader: APPSDatasetLoader):
    """Test specific cases that were problematic during cleaning."""
    print("\nTesting specific problematic cases...")
    
    # Find problems with mixed types in inputs/outputs
    problematic_cases = []
    
    for split, problems in original_data.items():
        for problem in problems:
            # Check for mixed types in inputs
            for input_item in problem['inputs']:
                if isinstance(input_item, list):
                    types_in_input = set(type(x).__name__ for x in input_item)
                    if len(types_in_input) > 1:
                        problematic_cases.append((split, problem['problem_id'], 'inputs', input_item))
            
            # Check for non-string outputs
            for output_item in problem['outputs']:
                if not isinstance(output_item, str):
                    problematic_cases.append((split, problem['problem_id'], 'outputs', output_item))
    
    print(f"Found {len(problematic_cases)} problematic cases")
    
    # Test a few of these cases
    for i, (split, problem_id, field, value) in enumerate(problematic_cases[:5]):
        print(f"\nCase {i+1}: Problem {problem_id} in {split} split")
        print(f"  Field: {field}")
        print(f"  Original value: {value}")
        print(f"  Original type: {type(value).__name__}")
        
        # Try to load this specific problem from cleaned dataset
        try:
            cleaned_problems = cleaned_loader.load_apps_samples(
                n_samples=1000,  # Load many to find our target
                split=split,
                recover_types=True,
                verbose=False
            )
            
            # Find the specific problem
            target_problem = None
            for p in cleaned_problems:
                if p['problem_id'] == problem_id:
                    target_problem = p
                    break
            
            if target_problem:
                if field == 'inputs':
                    cleaned_value = target_problem['inputs'][0]  # First input
                else:
                    cleaned_value = target_problem['outputs'][0]  # First output
                
                print(f"  Cleaned value: {cleaned_value}")
                print(f"  Cleaned type: {type(cleaned_value).__name__}")
                
                # Test if we can recover the original
                if isinstance(cleaned_value, str):
                    try:
                        recovered = ast.literal_eval(cleaned_value)
                        print(f"  Recovered: {recovered}")
                        print(f"  Recovery successful: {recovered == value}")
                    except:
                        print(f"  ❌ Recovery failed")
                else:
                    print(f"  ❌ Cleaned value is not a string")
            else:
                print(f"  ❌ Problem not found in cleaned dataset")
                
        except Exception as e:
            print(f"  ❌ Error testing case: {e}")


def main():
    """Main function to run the comparison."""
    print("APPS DATASET COMPARISON")
    print("="*60)
    
    # Load original dataset
    original_data = load_original_dataset()
    
    # Initialize cleaned dataset loader
    cleaned_loader = APPSDatasetLoader()
    
    # Run comparison
    compare_datasets(original_data, cleaned_loader)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main() 