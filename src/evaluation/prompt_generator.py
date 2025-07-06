"""
Prompt Generator for Model Evaluation

Generates standardized prompts for programming problems with tag-based output format.
"""

from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.dataset_loader import APPSDatasetLoader


class PromptGenerator:
    """Generates standardized prompts for programming problems."""
    
    def __init__(self):
        pass
    
    def generate_prompt(self, problem: Dict) -> str:
        """Generate a prompt for a given problem.
        
        Args:
            problem: Dictionary containing problem data with keys:
                - 'question': The problem description
        
        Returns:
            Formatted prompt string
        """
        question = problem.get('question', '')
        
        prompt = f"""You are a programming assistant. Solve the following problem step by step.

IMPORTANT: You MUST use the EXACT tag format shown in the examples below. Use <thinking> and </thinking> for reasoning, and <code> and </code> for the solution.

EXAMPLE 1:
Problem: Count the number of even integers in a list.
Input: First line contains n (the number of integers), second line contains n space-separated integers.
Output: The count of even integers.

<thinking>
I need to count even numbers in a list. 
Input format: first line has n, second line has n integers.
Even numbers are divisible by 2 (num % 2 == 0).
I'll create a function for the logic and handle I/O in main().
</thinking>

<code>
def count_even_numbers(numbers):
    return sum(1 for num in numbers if num % 2 == 0)

def main():
    n = int(input())
    numbers = list(map(int, input().split()))
    result = count_even_numbers(numbers)
    print(result)

if __name__ == "__main__":
    main()
</code>

EXAMPLE 2:
Problem: Given multiple test cases, for each test case reverse the given list.
Input: First line contains T (number of test cases). For each test case: first line has n, second line has n elements.
Output: For each test case, print the reversed list.

<thinking>
Multiple test cases problem. First read T, then handle T test cases.
Each test case has n on first line, then n elements.
I'll use a function to reverse the list and handle I/O in main().
</thinking>

<code>
def reverse_list(elements):
    return elements[::-1]

def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        elements = input().split()
        reversed_elements = reverse_list(elements)
        print(' '.join(reversed_elements))

if __name__ == "__main__":
    main()
</code>

ACTUAL PROBLEM TO SOLVE:
{question}

CRITICAL INSTRUCTIONS:
- You MUST use the EXACT tag format: <thinking>...</thinking> and <code>...</code>
- Your response must contain EXACTLY these two tagged sections - no more, no less
- The <thinking> tag must contain your reasoning about the problem and I/O format
- The <code> tag must follow this EXACT structure:
  1. Define one or more functions for the solution logic (e.g., def solve_problem(...))
  2. Define a main() function that handles ALL input/output
  3. Include: if __name__ == "__main__": main()
- The main() function MUST:
  - Read all inputs using input()
  - Call your solution function(s)
  - Print all outputs using print()
- Do not include test cases, example usage, or explanations in the code
- Do not include ANY text outside of these two tags
- Both opening and closing tags are REQUIRED

Now solve the problem above:"""
        
        return prompt
    
    def generate_batch_prompts(self, problems: List[Dict]) -> List[str]:
        """Generate prompts for a batch of problems.
        
        Args:
            problems: List of problem dictionaries
        
        Returns:
            List of formatted prompts
        """
        return [self.generate_prompt(problem) for problem in problems]
    
    def preview_prompt(self, problem: Dict) -> None:
        """Preview a generated prompt for debugging.
        
        Args:
            problem: Problem dictionary
        """
        prompt = self.generate_prompt(problem)
        
        print("=" * 80)
        print("GENERATED PROMPT:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print(f"Prompt length: {len(prompt)} characters")
        print("=" * 80)


def test_prompt_generation_with_apps():
    """Test the prompt generator with a random sample from the APPS dataset."""
    print("Testing Prompt Generator with APPS Dataset")
    print("=" * 80)
    
    # Initialize components
    generator = PromptGenerator()
    loader = APPSDatasetLoader()
    
    # Load a random introductory problem from the eval split
    try:
        problems = loader.load_apps_samples(
            n_samples=1,
            split="eval",
            difficulty="introductory",
            random_seed=None  # Random selection
        )
        
        if problems:
            problem = problems[0]
            print(f"\nLoaded Problem ID: {problem['problem_id']}")
            print(f"Difficulty: {problem['difficulty']}")
            print(f"Number of test cases: {len(problem.get('inputs', []))}")
            print("\nGenerating prompt...")
            
            # Generate and preview the prompt
            generator.preview_prompt(problem)
            
    except Exception as e:
        print(f"Error loading APPS dataset: {e}")
        print("Make sure the APPS dataset is available in data/apps/cleaned/")


if __name__ == "__main__":
    test_prompt_generation_with_apps()
