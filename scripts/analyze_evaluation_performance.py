#!/usr/bin/env python3
"""
Analyze evaluation performance and identify bottlenecks.
"""

import json
import time
from pathlib import Path
from datetime import datetime

from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.code_executor import CodeExecutor
from src.evaluation.answer_extractor import AnswerExtractor
from src.utils.dataset_loader import APPSDatasetLoader


def time_code_execution():
    """Test code execution speed."""
    print("\n=== Testing Code Execution Speed ===")
    
    executor = CodeExecutor(timeout=5, max_memory_mb=100)
    
    # Simple test case
    code = """
n = int(input())
result = sum(range(1, n+1))
print(result)
"""
    
    test_cases = [
        {"input": "10", "output": "55"},
        {"input": "100", "output": "5050"},
        {"input": "1000", "output": "500500"},
    ]
    
    # Time single execution
    start = time.time()
    result = executor.run_test_cases(code, test_cases)
    elapsed = time.time() - start
    
    print(f"Code execution time for {len(test_cases)} test cases: {elapsed:.3f}s")
    print(f"Average per test case: {elapsed/len(test_cases):.3f}s")
    print(f"Result: {result['passed_count']}/{result['total_count']} passed")
    
    # Test with more complex code
    complex_code = """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

n = int(input())
primes = [i for i in range(2, n+1) if is_prime(i)]
print(len(primes))
"""
    
    test_cases_complex = [
        {"input": "10", "output": "4"},  # 2, 3, 5, 7
        {"input": "20", "output": "8"},  # 2, 3, 5, 7, 11, 13, 17, 19
        {"input": "100", "output": "25"},
    ]
    
    start = time.time()
    result = executor.run_test_cases(complex_code, test_cases_complex)
    elapsed = time.time() - start
    
    print(f"\nComplex code execution time for {len(test_cases_complex)} test cases: {elapsed:.3f}s")
    print(f"Average per test case: {elapsed/len(test_cases_complex):.3f}s")
    

def time_answer_extraction():
    """Test answer extraction speed."""
    print("\n=== Testing Answer Extraction Speed ===")
    
    extractor = AnswerExtractor()
    
    # Sample model output
    model_output = '''<thinking>
I need to solve this problem step by step.
First, I'll read the input...
</thinking>

<answer>
def solve():
    n = int(input())
    result = sum(range(1, n+1))
    print(result)

solve()
</answer>'''
    
    # Time extraction
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        result = extractor.extract_answer(model_output)
    elapsed = time.time() - start
    
    print(f"Answer extraction time for {iterations} iterations: {elapsed:.3f}s")
    print(f"Average per extraction: {elapsed/iterations*1000:.1f}ms")
    print(f"Extracted code found: {result['code_found']}")


def analyze_full_evaluation():
    """Analyze a full evaluation with detailed timing."""
    print("\n=== Analyzing Full Evaluation Process ===")
    
    # Find a small generation output
    generation_file = Path("data/generation_outputs/20250706_194450_eval_5problems_1models_outputs.json")
    
    if not generation_file.exists():
        print("No suitable generation file found!")
        return
    
    print(f"Using generation file: {generation_file}")
    
    # Load generation results
    with open(generation_file, 'r') as f:
        generation_results = json.load(f)
    
    metadata = generation_results["metadata"]
    n_problems = metadata["n_problems"]
    models = metadata["models"]
    
    print(f"Problems: {n_problems}")
    print(f"Models: {len(models)}")
    
    # Create evaluator with timing instrumentation
    evaluator = ModelEvaluator(max_workers=1000)
    
    # Time dataset loading
    start = time.time()
    loader = APPSDatasetLoader(data_dir="data/APPS/cleaned")
    problems = loader.load_apps_samples(
        n_samples=n_problems,
        split="eval",
        difficulty="introductory",
        min_test_cases=1,
        max_test_cases=None,
        has_solutions=None,
        has_starter_code=None,
        random_seed=42,
        recover_types=True,
        verbose=False
    )
    dataset_load_time = time.time() - start
    print(f"\nDataset loading time: {dataset_load_time:.3f}s")
    
    # Analyze test case counts
    test_case_counts = [len(p['inputs']) for p in problems]
    total_test_cases = sum(test_case_counts)
    print(f"Total test cases to execute: {total_test_cases}")
    print(f"Test cases per problem: min={min(test_case_counts)}, max={max(test_case_counts)}, avg={total_test_cases/n_problems:.1f}")
    
    # Time answer extraction phase
    start = time.time()
    all_extracted = []
    for model_name, model_outputs in generation_results["results"].items():
        extracted = evaluator._extract_answers_from_outputs(model_outputs)
        all_extracted.append((model_name, extracted))
    extraction_time = time.time() - start
    print(f"\nAnswer extraction time: {extraction_time:.3f}s")
    print(f"Average per model output: {extraction_time/(len(models)*n_problems)*1000:.1f}ms")
    
    # Time code execution phase (this is likely the bottleneck)
    print(f"\nStarting code execution phase...")
    problem_lookup = {p["problem_id"]: p for p in problems}
    
    total_execution_time = 0
    execution_times = []
    
    for model_name, extracted_outputs in all_extracted[:1]:  # Test just first model
        print(f"\nTesting {model_name}...")
        
        for i, output in enumerate(extracted_outputs):
            if output["api_success"] and output["extracted"]["code_found"]:
                problem = problem_lookup.get(output["problem_id"])
                test_cases = evaluator._prepare_test_cases(problem)
                
                start = time.time()
                execution_result = evaluator.code_executor.run_test_cases(
                    output["extracted"]["code"], 
                    test_cases
                )
                exec_time = time.time() - start
                execution_times.append(exec_time)
                total_execution_time += exec_time
                
                print(f"  Problem {i+1}: {len(test_cases)} test cases, {exec_time:.3f}s, passed: {execution_result['passed_count']}/{execution_result['total_count']}")
    
    if execution_times:
        print(f"\nCode Execution Summary:")
        print(f"Total execution time: {total_execution_time:.3f}s")
        print(f"Average per problem: {sum(execution_times)/len(execution_times):.3f}s")
        print(f"Min/Max: {min(execution_times):.3f}s / {max(execution_times):.3f}s")
        
        # Estimate total time
        estimated_total = sum(execution_times) * len(models)
        print(f"\nEstimated total evaluation time for all models: {estimated_total:.1f}s ({estimated_total/60:.1f} minutes)")


def main():
    """Run all performance analyses."""
    print("EVALUATION PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test individual components
    time_code_execution()
    time_answer_extraction()
    
    # Analyze full evaluation
    analyze_full_evaluation()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()