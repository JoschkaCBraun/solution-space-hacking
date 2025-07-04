"""
Data Integrity Test

Simple test to verify that our cleaning and recovery process works correctly.
"""

import json
import ast
import numpy as np
from datasets import load_dataset
from src.utils.dataset_loader import APPSDatasetLoader


def safe_recover_value(value_str):
    """Safely recover a value from string representation."""
    try:
        # Handle plain strings that aren't valid Python literals
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]  # Remove quotes
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]  # Remove quotes
        
        # Try ast.literal_eval for other types
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If all else fails, return as string
        return value_str


def safe_compare_arrays(arr1, arr2):
    """Safely compare arrays that might have different shapes or types."""
    try:
        # Handle the case where one is a list of lists and the other is a list of strings
        if isinstance(arr1, list) and isinstance(arr2, list):
            if len(arr1) != len(arr2):
                return False
            
            # If arr1 contains lists and arr2 contains strings, try to recover arr2
            if (len(arr1) > 0 and isinstance(arr1[0], list) and 
                len(arr2) > 0 and isinstance(arr2[0], str)):
                # Try to recover arr2 elements
                recovered_arr2 = []
                for item in arr2:
                    try:
                        recovered_arr2.append(safe_recover_value(item))
                    except:
                        recovered_arr2.append(item)
                arr2 = recovered_arr2
            
            # Now compare element by element
            for i, (item1, item2) in enumerate(zip(arr1, arr2)):
                if item1 != item2:
                    return False
            return True
        
        # Handle numpy arrays
        if hasattr(arr1, 'all') and hasattr(arr2, 'all'):
            # These are numpy arrays
            if arr1.shape != arr2.shape:
                return False
            return (arr1 == arr2).all()
        
        # Fall back to simple comparison
        return arr1 == arr2
    except Exception:
        # Fall back to simple comparison
        return arr1 == arr2


def test_data_integrity():
    """Test that our cleaning and recovery process works correctly."""
    print("DATA INTEGRITY TEST")
    print("="*60)
    
    # Load a few problems from the original dataset
    print("Loading original dataset...")
    dataset = load_dataset("codeparrot/apps")
    
    loader = APPSDatasetLoader()
    
    # Find problems that exist in both datasets
    print("\n--- Finding Common Problems ---")
    
    # Get a sample of problems from original dataset
    original_problems = []
    for split in ['train', 'test']:
        for i, item in enumerate(dataset[split]):
            if i >= 50:  # Limit to first 50 problems per split
                break
            original_problems.append({
                'split': split,
                'problem_id': item['problem_id'],
                'item': item
            })
    
    print(f"Found {len(original_problems)} problems from original dataset")
    
    # Load cleaned problems
    cleaned_problems = {}
    for split in ['train', 'test']:
        try:
            problems = loader.load_apps_samples(
                n_samples=1000,
                split=split,
                recover_types=True,
                verbose=False
            )
            for p in problems:
                cleaned_problems[p['problem_id']] = p
        except Exception as e:
            print(f"Error loading {split} split: {e}")
    
    print(f"Found {len(cleaned_problems)} problems in cleaned dataset")
    
    # Find common problems
    common_problems = []
    for orig_problem in original_problems:
        problem_id = orig_problem['problem_id']
        if problem_id in cleaned_problems:
            common_problems.append({
                'original': orig_problem,
                'cleaned': cleaned_problems[problem_id]
            })
    
    print(f"Found {len(common_problems)} common problems")
    
    if len(common_problems) == 0:
        print("❌ No common problems found between original and cleaned datasets")
        return
    
    # Test a few common problems
    test_cases = common_problems[:3]  # Test first 3 common problems
    
    for i, test_case in enumerate(test_cases, 1):
        original_item = test_case['original']['item']
        cleaned_problem = test_case['cleaned']
        problem_id = original_item['problem_id']
        
        print(f"\n--- Test Case {i}: Problem {problem_id} ---")
        
        # Parse original data
        try:
            if original_item.get('input_output'):
                io_data = json.loads(original_item['input_output'])
                original_inputs = io_data.get('inputs', [])
                original_outputs = io_data.get('outputs', [])
            else:
                original_inputs = []
                original_outputs = []
                
            print(f"Original inputs: {original_inputs}")
            print(f"Original outputs: {original_outputs}")
            print(f"Cleaned inputs: {cleaned_problem['inputs']}")
            print(f"Cleaned outputs: {cleaned_problem['outputs']}")
            
            # Test recovery with safe comparison
            try:
                inputs_match = safe_compare_arrays(original_inputs, cleaned_problem['inputs'])
                outputs_match = safe_compare_arrays(original_outputs, cleaned_problem['outputs'])
                
                print(f"✅ Inputs match: {inputs_match}")
                print(f"✅ Outputs match: {outputs_match}")
                
                if inputs_match and outputs_match:
                    print(f"🎉 Test case {i} PASSED!")
                else:
                    print(f"❌ Test case {i} FAILED!")
            except Exception as e:
                print(f"❌ Comparison error: {e}")
                
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
    
    # Test the stringification and recovery process directly
    print(f"\n--- Testing Stringification and Recovery Process ---")
    
    test_data = [
        [[1, 2, 3, 4, 5, 6, 1], 3],  # Mixed types
        [3, 3, 1],                    # All integers
        ["aababcaab", 2, 3, 4],      # Mixed string and integers
        12,                           # Integer
        True,                         # Boolean
        "Hello"                       # String
    ]
    
    for i, data in enumerate(test_data, 1):
        print(f"\nTest {i}: {data} (type: {type(data).__name__})")
        
        # Stringify
        stringified = str(data)
        print(f"  Stringified: {stringified}")
        
        # Recover using safe method
        try:
            recovered = safe_recover_value(stringified)
            print(f"  Recovered: {recovered}")
            print(f"  Match: {data == recovered}")
            print(f"  Type match: {type(data) == type(recovered)}")
        except Exception as e:
            print(f"  ❌ Recovery failed: {e}")
    
    # Test specific edge cases
    print(f"\n--- Testing Edge Cases ---")
    
    edge_cases = [
        '"Hello"',           # Quoted string
        "'World'",           # Single quoted string
        "Hello",             # Plain string
        "[1, 2, 3]",         # List
        "{'a': 1, 'b': 2}",  # Dict
        "True",              # Boolean
        "123",               # Number
    ]
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\nEdge case {i}: {case}")
        try:
            recovered = safe_recover_value(case)
            print(f"  Recovered: {recovered} (type: {type(recovered).__name__})")
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    test_data_integrity() 