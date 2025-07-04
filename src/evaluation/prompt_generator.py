"""
Prompt Generator for Model Evaluation

Generates standardized prompts for programming problems with tag-based output format.
"""

import re
from typing import Dict, List, Optional


class PromptGenerator:
    """Generates standardized prompts for programming problems."""
    
    def __init__(self):
        self.template = self._get_prompt_template()
    
    def _get_prompt_template(self) -> str:
        """Get the standardized prompt template."""
        return """You are a programming assistant. Solve the following problem step by step.

PROBLEM:
{problem_question}

{starter_code_section}

EXAMPLE OUTPUT FORMAT:
<thinking>
First, I need to understand what the problem is asking. It wants me to compare two lists and find which has the bigger sum.
I'll iterate through each list, calculate the sum, and then compare them.
</thinking>

<code>
def compare_lists(list1, list2):
    sum1 = sum(list1)
    sum2 = sum(list2)
    if sum1 > sum2:
        return "First list has bigger sum"
    elif sum2 > sum1:
        return "Second list has bigger sum"
    else:
        return "Both lists have equal sums"
</code>

INSTRUCTIONS:
- Always use the exact tags: <thinking> and <code>
- The <thinking> tag should contain your reasoning and approach
- The <code> tag should contain only valid Python code
- Both tags must have opening and closing tags
- Do not include any text outside these tags

Now solve the problem above:"""
    
    def generate_prompt(self, problem: Dict) -> str:
        """Generate a prompt for a given problem.
        
        Args:
            problem: Dictionary containing problem data with keys:
                - 'question': The problem description
                - 'starter_code': Optional starter code (can be empty string)
        
        Returns:
            Formatted prompt string
        """
        question = problem.get('question', '')
        starter_code = problem.get('starter_code', '')
        
        # Format starter code section
        if starter_code and starter_code.strip():
            starter_code_section = f"STARTER CODE:\n{starter_code}\n"
        else:
            starter_code_section = ""
        
        # Generate the prompt
        prompt = self.template.format(
            problem_question=question,
            starter_code_section=starter_code_section
        )
        
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


def test_prompt_generation():
    """Test the prompt generator with sample problems."""
    generator = PromptGenerator()
    
    # Test problem 1: Simple problem without starter code
    problem1 = {
        'question': 'Write a function that takes a list of integers and returns the sum of all even numbers.',
        'starter_code': ''
    }
    
    # Test problem 2: Problem with starter code
    problem2 = {
        'question': 'Complete the function to find the maximum element in a list.',
        'starter_code': 'def find_max(numbers):\n    # Your code here\n    pass'
    }
    
    # Test problem 3: Complex problem
    problem3 = {
        'question': '''Given a string s, find the length of the longest substring without repeating characters.

Example:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.'''
    }
    
    print("Testing Prompt Generator")
    print("=" * 80)
    
    print("\n1. Simple problem without starter code:")
    generator.preview_prompt(problem1)
    
    print("\n2. Problem with starter code:")
    generator.preview_prompt(problem2)
    
    print("\n3. Complex problem:")
    generator.preview_prompt(problem3)
    
    # Test batch generation
    problems = [problem1, problem2, problem3]
    prompts = generator.generate_batch_prompts(problems)
    print(f"\nBatch generation: {len(prompts)} prompts created")


if __name__ == "__main__":
    test_prompt_generation() 