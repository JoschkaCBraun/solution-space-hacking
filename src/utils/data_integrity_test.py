"""
Data Integrity Test

Simple test to verify that our cleaning and recovery process works correctly.
"""

import json
import ast
from datasets import load_dataset
from src.utils.dataset_loader import APPSDatasetLoader


def test_data_integrity():
    """Test that our cleaning and recovery process works correctly."""
    print("DATA INTEGRITY TEST")
    print("="*60)
    
    # Load a few problems from the original dataset
    print("Loading original dataset...")
    dataset = load_dataset("codeparrot/apps")
    
    # Test a few problems with known issues
    test_cases = [
        # Problem with mixed types in inputs
        {
            'split': 'train',
            'problem_id': 122,
            'description': 'Mixed types in inputs: [[1, 2, 3, 4, 5, 6, 1], 3]'
        },
        # Problem with integer output
        {
            'split': 'train', 
            'problem_id': 123,
            'description': 'Integer output: 6'
        },
        # Problem with boolean output
        {
            'split': 'train',
            'problem_id': 124, 
            'description': 'Boolean output: True'
        }
    ]
    
    loader = APPSDatasetLoader()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        
        # Get original data
        original_item = None
        for item in dataset[test_case['split']]:
            if item['problem_id'] == test_case['problem_id']:
                original_item = item
                break
        
        if not original_item:
            print(f"❌ Problem {test_case['problem_id']} not found in original dataset")
            continue
        
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
            
            # Load from cleaned dataset
            cleaned_problems = loader.load_apps_samples(
                n_samples=1000,
                split=test_case['split'],
                recover_types=True,
                verbose=False
            )
            
            # Find the problem in cleaned dataset
            cleaned_problem = None
            for p in cleaned_problems:
                if p['problem_id'] == test_case['problem_id']:
                    cleaned_problem = p
                    break
            
            if not cleaned_problem:
                print(f"❌ Problem {test_case['problem_id']} not found in cleaned dataset")
                continue
            
            print(f"Cleaned inputs: {cleaned_problem['inputs']}")
            print(f"Cleaned outputs: {cleaned_problem['outputs']}")
            
            # Test recovery
            inputs_match = original_inputs == cleaned_problem['inputs']
            outputs_match = original_outputs == cleaned_problem['outputs']
            
            print(f"✅ Inputs match: {inputs_match}")
            print(f"✅ Outputs match: {outputs_match}")
            
            if inputs_match and outputs_match:
                print(f"🎉 Test case {i} PASSED!")
            else:
                print(f"❌ Test case {i} FAILED!")
                
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
        
        # Recover
        try:
            recovered = ast.literal_eval(stringified)
            print(f"  Recovered: {recovered}")
            print(f"  Match: {data == recovered}")
            print(f"  Type match: {type(data) == type(recovered)}")
        except Exception as e:
            print(f"  ❌ Recovery failed: {e}")


if __name__ == "__main__":
    test_data_integrity() 