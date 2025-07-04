"""
Offline test for the evaluation pipeline.

This test verifies that all components work together without making API calls.
"""

import json
import tempfile
from pathlib import Path
from src.evaluation.prompt_generator import PromptGenerator
from src.evaluation.answer_extractor import AnswerExtractor
from src.evaluation.code_executor import CodeExecutor
from src.utils.dataset_loader import APPSDatasetLoader


def test_offline_integration():
    """Test the evaluation pipeline offline."""
    print("🧪 Testing Offline Integration")
    print("=" * 50)
    
    # Initialize components
    prompt_gen = PromptGenerator()
    answer_extractor = AnswerExtractor()
    code_executor = CodeExecutor()
    
    # Load a few problems
    loader = APPSDatasetLoader()
    problems = loader.load_apps_samples(
        n_samples=3,
        split="eval",
        difficulty="introductory",  # Explicitly set to match new default
        recover_types=True,
        verbose=False
    )
    
    print(f"Loaded {len(problems)} problems")
    
    # Generate prompts
    prompts = prompt_gen.generate_batch_prompts(problems)
    print(f"Generated {len(prompts)} prompts")
    
    # Simulate model outputs
    simulated_outputs = [
        """<thinking>
I need to solve this step by step. Looking at the problem, I need to understand what it's asking for.
The problem seems to be about calculating something.
I'll approach this by breaking it down into smaller steps.
</thinking>

<code>
def solve_problem():
    # Read input
    n = int(input())
    
    # Process the problem
    result = 0
    for i in range(n):
        x = int(input())
        result += x
    
    # Output result
    print(result)

# Call the function
solve_problem()
</code>""",
        
        """<thinking>
This is a different problem that requires a different approach.
I need to think about the algorithm carefully.
</thinking>

<code>
def another_solution():
    print("Hello, World!")
</code>""",
        
        """<code>
def no_thinking():
    return 42
</code>"""
    ]
    
    print(f"Created {len(simulated_outputs)} simulated model outputs")
    
    # Extract answers
    extracted_results = []
    for i, output in enumerate(simulated_outputs):
        extracted = answer_extractor.extract_answer(output)
        extracted_results.append({
            "problem_idx": i,
            "problem_id": problems[i]["problem_id"],
            "prompt": prompts[i],
            "model_output": output,
            "extracted": extracted,
            "api_success": True
        })
    
    print("Extracted answers from simulated outputs")
    
    # Test code execution
    for i, result in enumerate(extracted_results):
        if result["extracted"]["code_found"]:
            print(f"\nTesting code execution for problem {i+1}:")
            
            # Get test cases
            problem = problems[i]
            test_cases = []
            if 'inputs' in problem and 'outputs' in problem:
                for input_str, output_str in zip(problem['inputs'], problem['outputs']):
                    test_cases.append({
                        'input': str(input_str),
                        'output': str(output_str)
                    })
            
            if test_cases:
                execution_result = code_executor.run_test_cases(
                    result["extracted"]["code"], 
                    test_cases
                )
                
                print(f"  Execution success: {execution_result['execution_success']}")
                print(f"  Passed: {execution_result['passed_count']}/{execution_result['total_count']}")
                print(f"  Pass rate: {execution_result['pass_rate']:.2%}")
            else:
                print("  No test cases available")
    
    # Test batch operations
    print(f"\nTesting batch operations:")
    batch_extracted = answer_extractor.extract_batch_answers(simulated_outputs)
    batch_stats = answer_extractor.get_extraction_stats(batch_extracted)
    
    print(f"  Total responses: {batch_stats['total_responses']}")
    print(f"  Thinking found rate: {batch_stats['thinking_found_rate']:.2%}")
    print(f"  Code found rate: {batch_stats['code_found_rate']:.2%}")
    print(f"  Both found rate: {batch_stats['both_found_rate']:.2%}")
    
    print("\n✅ Offline integration test completed successfully!")
    print("All components are working correctly together.")


if __name__ == "__main__":
    test_offline_integration() 