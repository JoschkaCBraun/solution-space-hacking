"""
Test script for all evaluation components.

Tests the prompt generator, answer extractor, and code executor together.
"""

from .prompt_generator import PromptGenerator
from .answer_extractor import AnswerExtractor
from .code_executor import CodeExecutor
from src.utils.dataset_loader import APPSDatasetLoader


def test_all_components():
    """Test all evaluation components together."""
    print("Testing All Evaluation Components")
    print("=" * 80)
    
    # Initialize components
    prompt_gen = PromptGenerator()
    answer_extractor = AnswerExtractor()
    code_executor = CodeExecutor()
    
    # Load a few problems from the dataset
    loader = APPSDatasetLoader()
    problems = loader.load_apps_samples(
        n_samples=3,
        split="test",
        difficulty="introductory",
        recover_types=True,
        verbose=False
    )
    
    print(f"Loaded {len(problems)} problems for testing")
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"PROBLEM {i}: {problem['problem_id']}")
        print(f"{'='*80}")
        
        # 1. Generate prompt
        print("\n1. GENERATING PROMPT:")
        print("-" * 40)
        prompt = prompt_gen.generate_prompt(problem)
        print(f"Prompt length: {len(prompt)} characters")
        print("Prompt preview (first 200 chars):")
        print(repr(prompt[:200]) + "...")
        
        # 2. Simulate model output (for testing)
        print("\n2. SIMULATED MODEL OUTPUT:")
        print("-" * 40)
        
        # Create a realistic model output
        model_output = f"""<thinking>
I need to solve this problem step by step. Looking at the problem, I need to understand what it's asking for.
The problem seems to be about {problem['question'][:50]}...
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
</code>"""
        
        print("Model output preview:")
        print(repr(model_output[:200]) + "...")
        
        # 3. Extract answer
        print("\n3. EXTRACTING ANSWER:")
        print("-" * 40)
        extracted = answer_extractor.extract_answer(model_output)
        
        print(f"Full output length: {len(extracted['full_output'])} chars")
        print(f"Thinking found: {extracted['thinking_found']}")
        print(f"Code found: {extracted['code_found']}")
        print(f"Thinking preview: {repr(extracted['thinking'][:100])}...")
        print(f"Code preview: {repr(extracted['code'][:100])}...")
        
        # 4. Execute code (if code was found)
        if extracted['code_found']:
            print("\n4. EXECUTING CODE:")
            print("-" * 40)
            
            # Get test cases from the problem
            test_cases = []
            if 'inputs' in problem and 'outputs' in problem:
                for j, (input_str, output_str) in enumerate(zip(problem['inputs'], problem['outputs'])):
                    test_cases.append({
                        'input': str(input_str),
                        'output': str(output_str)
                    })
            
            if test_cases:
                print(f"Running {len(test_cases)} test cases...")
                test_result = code_executor.run_test_cases(extracted['code'], test_cases)
                
                print(f"Execution success: {test_result['execution_success']}")
                print(f"Passed: {test_result['passed_count']}/{test_result['total_count']}")
                print(f"Pass rate: {test_result['pass_rate']:.2%}")
                
                # Show first few test results
                for result in test_result['test_results'][:3]:
                    status = "PASS" if result['passed'] else "FAIL"
                    print(f"  Test {result['test_case']}: {status}")
                    if not result['passed']:
                        print(f"    Expected: {repr(result['expected_output'])}")
                        print(f"    Got: {repr(result['actual_output'])}")
            else:
                print("No test cases available for this problem")
        else:
            print("\n4. EXECUTING CODE:")
            print("-" * 40)
            print("No code found to execute")
    
    # Test batch operations
    print(f"\n{'='*80}")
    print("BATCH OPERATIONS TEST")
    print(f"{'='*80}")
    
    # Generate batch prompts
    batch_prompts = prompt_gen.generate_batch_prompts(problems)
    print(f"Generated {len(batch_prompts)} prompts in batch")
    
    # Simulate batch model outputs
    batch_outputs = [
        """<thinking>I'll solve this step by step.</thinking><code>print("solution")</code>""",
        """<thinking>Let me think about this.</thinking><code>def solve(): pass</code>""",
        """<code>print("no thinking")</code>"""
    ]
    
    # Extract batch answers
    batch_extracted = answer_extractor.extract_batch_answers(batch_outputs)
    batch_stats = answer_extractor.get_extraction_stats(batch_extracted)
    
    print(f"Batch extraction stats:")
    print(f"  Total responses: {batch_stats['total_responses']}")
    print(f"  Thinking found rate: {batch_stats['thinking_found_rate']:.2%}")
    print(f"  Code found rate: {batch_stats['code_found_rate']:.2%}")
    print(f"  Both found rate: {batch_stats['both_found_rate']:.2%}")


if __name__ == "__main__":
    test_all_components() 