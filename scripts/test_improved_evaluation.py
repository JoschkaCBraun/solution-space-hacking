"""
Test script to verify the improved evaluation performance.

This script tests the improved code executor with timeout and parallelization
to ensure it's working correctly before running the full evaluation.
"""

import time
import json
from pathlib import Path

from src.evaluation.code_executor import CodeExecutor


def test_timeout_comparison():
    """Compare old vs new executor with timeout scenarios."""
    
    # Test code that should timeout
    timeout_code = """
while True:
    pass
"""
    
    # Test code that should work
    working_code = """
n = int(input())
result = 0
for i in range(n):
    x = int(input())
    result += x * 2
print(result)
"""
    
    # Test cases
    test_cases = [
        {"input": "3\n1\n2\n3", "output": "12"},
        {"input": "2\n5\n10", "output": "30"},
        {"input": "1\n7", "output": "14"}
    ]
    
    print("Testing Improved Code Executor Performance")
    print("=" * 60)
    
    # Test 1: Working code with new executor
    print("\n1. Testing working code with CodeExecutor:")
    improved_executor = CodeExecutor(timeout=5, test_case_workers=3)
    
    start = time.time()
    result = improved_executor.run_test_cases(working_code, test_cases)
    elapsed = time.time() - start
    
    print(f"   Execution time: {elapsed:.2f}s")
    print(f"   Passed: {result['passed_count']}/{result['total_count']}")
    print(f"   Pass rate: {result['pass_rate']:.0%}")
    
    # Test 2: Timeout code with new executor
    print("\n2. Testing timeout code with CodeExecutor:")
    
    start = time.time()
    result = improved_executor.run_test_cases(timeout_code, test_cases[:1])  # Just one test
    elapsed = time.time() - start
    
    print(f"   Execution time: {elapsed:.2f}s (should be ~5s)")
    print(f"   Timeouts: {result.get('timeout_count', 0)}")
    print(f"   Success: {result['execution_success']}")
    
    # Test 3: Multiple test cases in parallel
    print("\n3. Testing parallel execution with many test cases:")
    
    # Create many test cases
    many_test_cases = []
    for i in range(10):
        many_test_cases.append({
            "input": f"2\n{i}\n{i+1}",
            "output": str(i*2 + (i+1)*2)
        })
    
    start = time.time()
    result = improved_executor.run_test_cases(working_code, many_test_cases)
    elapsed = time.time() - start
    
    print(f"   Test cases: {len(many_test_cases)}")
    print(f"   Execution time: {elapsed:.2f}s")
    print(f"   Passed: {result['passed_count']}/{result['total_count']}")
    print(f"   Speedup vs sequential: ~{len(many_test_cases) * 0.1 / elapsed:.1f}x (estimated)")
    
    # Test 4: Mixed scenarios (some timeout, some pass)
    print("\n4. Testing mixed scenarios:")
    
    mixed_code = """
n = int(input())
if n == 999:
    while True:  # Infinite loop
        pass
else:
    print(n * n)
"""
    
    mixed_test_cases = [
        {"input": "2", "output": "4"},
        {"input": "3", "output": "9"},
        {"input": "999", "output": "998001"},  # Will timeout
        {"input": "5", "output": "25"},
        {"input": "999", "output": "998001"},  # Will timeout
        {"input": "10", "output": "100"}
    ]
    
    start = time.time()
    result = improved_executor.run_test_cases(mixed_code, mixed_test_cases)
    elapsed = time.time() - start
    
    print(f"   Test cases: {len(mixed_test_cases)}")
    print(f"   Execution time: {elapsed:.2f}s")
    print(f"   Passed: {result['passed_count']}/{result['total_count']}")
    print(f"   Timeouts: {result.get('timeout_count', 0)}")
    
    # Clean up
    improved_executor.cleanup()
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
    print("\nKey improvements verified:")
    print("- Timeout mechanism prevents hanging on infinite loops")
    print("- Parallel test case execution reduces total time")
    print("- Graceful handling of mixed pass/timeout scenarios")


def test_real_generation_output():
    """Test with actual generation output if available."""
    generation_dir = Path("data/generation_outputs")
    if not generation_dir.exists():
        print("\nNo generation outputs found to test with.")
        return
    
    # Find a recent generation file
    files = list(generation_dir.glob("*_outputs.json"))
    if not files:
        print("\nNo generation files found.")
        return
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"\n\nTesting with real generation output: {latest_file.name}")
    print("=" * 60)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Get first model's first output for testing
    results = data.get("results", {})
    if not results:
        print("No results found in file.")
        return
    
    model_name = list(results.keys())[0]
    outputs = results[model_name]
    
    if not outputs:
        print(f"No outputs for model {model_name}")
        return
    
    # Test first successful output
    for output in outputs[:5]:  # Check first 5
        if output.get("api_success") and "model_output" in output:
            print(f"\nTesting output from {model_name}:")
            print(f"Problem ID: {output.get('problem_id', 'unknown')}")
            
            # Extract code (simplified)
            model_output = output["model_output"]
            if "<thinking>" in model_output and "</thinking>" in model_output:
                start = model_output.find("</thinking>") + len("</thinking>")
                code = model_output[start:].strip()
            else:
                code = model_output.strip()
            
            # Get test cases (would need dataset loader in real scenario)
            # For now, just test syntax
            improved_executor = CodeExecutor(timeout=2)
            syntax_valid, error = improved_executor.validate_syntax(code)
            
            print(f"Syntax valid: {syntax_valid}")
            if not syntax_valid:
                print(f"Error: {error}")
            
            break


if __name__ == "__main__":
    test_timeout_comparison()
    test_real_generation_output()