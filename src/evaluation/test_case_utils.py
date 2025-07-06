"""
Utilities for handling test case format conversion in APPS dataset.
"""

import ast
import re
from typing import Tuple, Any, List


def convert_input_format(input_str: str) -> str:
    """
    Convert various input formats to line-by-line format expected by stdin.
    
    Args:
        input_str: Input string which might be in various formats:
                  - Python list literal: "[1, 2, 3]"
                  - Already line-by-line: "1\n2\n3"
                  - Single value: "42"
    
    Returns:
        String formatted for line-by-line stdin reading
    """
    input_str = input_str.strip()
    
    # Check if it's a Python list literal
    if input_str.startswith('[') and input_str.endswith(']'):
        try:
            # Parse the list
            parsed_list = ast.literal_eval(input_str)
            
            # Convert each element to string and join with newlines
            if isinstance(parsed_list, list):
                # Handle special cases for boolean values
                converted_items = []
                for item in parsed_list:
                    if isinstance(item, bool):
                        # Convert Python boolean to string 'True' or 'False'
                        converted_items.append(str(item))
                    else:
                        converted_items.append(str(item))
                
                return '\n'.join(converted_items)
            else:
                # Single value in brackets
                return str(parsed_list)
        except (ValueError, SyntaxError):
            # If parsing fails, return as-is
            return input_str
    
    # Already in correct format or single value
    return input_str


def normalize_output(output_str: str) -> str:
    """
    Normalize output format for comparison.
    
    Args:
        output_str: Output string which might be:
                   - Python list literal: "['result']" 
                   - Plain text: "result"
                   - Multiline: "line1\nline2"
    
    Returns:
        Normalized string for comparison
    """
    output_str = output_str.strip()
    
    # Check if it's a Python list literal with single element
    if output_str.startswith('[') and output_str.endswith(']'):
        try:
            parsed = ast.literal_eval(output_str)
            if isinstance(parsed, list) and len(parsed) == 1:
                # Return the single element as string
                return str(parsed[0])
            elif isinstance(parsed, list):
                # Multiple elements - join with newlines
                return '\n'.join(str(item) for item in parsed)
            else:
                # Not a list, return string representation
                return str(parsed)
        except (ValueError, SyntaxError):
            # If parsing fails, just remove the brackets
            return output_str[1:-1].strip()
    
    return output_str


def compare_outputs(expected: str, actual: str) -> bool:
    """
    Compare expected and actual outputs with normalization.
    
    Args:
        expected: Expected output string
        actual: Actual output string
    
    Returns:
        True if outputs match after normalization
    """
    # Normalize both outputs
    norm_expected = normalize_output(expected)
    norm_actual = normalize_output(actual)
    
    # Direct comparison
    if norm_expected == norm_actual:
        return True
    
    # Try comparison with stripped quotes (for string outputs)
    if norm_expected.strip('"\'') == norm_actual.strip('"\''):
        return True
    
    # Try numeric comparison for numbers
    try:
        expected_num = float(norm_expected.replace(',', ''))
        actual_num = float(norm_actual.replace(',', ''))
        # Allow small floating point differences
        return abs(expected_num - actual_num) < 1e-9
    except ValueError:
        pass
    
    # Try comparison ignoring currency symbols
    if any(symbol in norm_expected + norm_actual for symbol in ['$', '£', '€', '¥']):
        # Remove currency symbols and compare
        clean_expected = re.sub(r'[$£€¥]', '', norm_expected).strip()
        clean_actual = re.sub(r'[$£€¥]', '', norm_actual).strip()
        
        # Remove decimal points if they're just .00 or .0
        clean_expected = re.sub(r'\.0+$', '', clean_expected)
        clean_actual = re.sub(r'\.0+$', '', clean_actual)
        
        if clean_expected == clean_actual:
            return True
        
        # Try numeric comparison
        try:
            expected_num = float(clean_expected.replace(',', ''))
            actual_num = float(clean_actual.replace(',', ''))
            return abs(expected_num - actual_num) < 1e-9
        except ValueError:
            pass
    
    return False


def test_conversions():
    """Test the conversion functions."""
    print("Testing Input Conversions:")
    print("-" * 50)
    
    test_inputs = [
        "[1, 2, 3]",
        "[10000, True]",
        "5\n10\n15",
        "42",
        "['hello', 'world']",
        "[True, False, True]"
    ]
    
    for inp in test_inputs:
        converted = convert_input_format(inp)
        print(f"Input: {repr(inp)}")
        print(f"Converted: {repr(converted)}")
        print()
    
    print("\nTesting Output Normalizations:")
    print("-" * 50)
    
    test_outputs = [
        ("['$100000']", "$100000"),
        ("[42]", "42"),
        ("result", "result"),
        ("['line1', 'line2']", "line1\nline2"),
        ("[3.14159]", "3.14159")
    ]
    
    for expected, actual in test_outputs:
        normalized_exp = normalize_output(expected)
        normalized_act = normalize_output(actual)
        match = compare_outputs(expected, actual)
        print(f"Expected: {repr(expected)} -> {repr(normalized_exp)}")
        print(f"Actual: {repr(actual)} -> {repr(normalized_act)}")
        print(f"Match: {match}")
        print()


if __name__ == "__main__":
    test_conversions()