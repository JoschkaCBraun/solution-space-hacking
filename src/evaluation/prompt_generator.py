"""
Prompt Generator for Model Evaluation

Generates standardized prompts for programming problems with tag-based output format.
"""

import re
from typing import Dict, List, Optional


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

PROBLEM:
{question}

EXAMPLE OUTPUT FORMAT:
<thinking>
I need to sum all even numbers in a list and return their sum.
Even numbers are those divisible by 2 (i.e., num % 2 == 0).
I'll iterate through the list, check if each number is even, and add it to the sum.
</thinking>

<code>
def sum_even_numbers(numbers):
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total

# Read input
numbers = list(map(int, input().split()))

# Calculate and print the sum of even numbers
result = sum_even_numbers(numbers)
print(result)
</code>

INSTRUCTIONS:
- Your response must contain EXACTLY two tagged sections: <thinking> and <code>
- The <thinking> tag should contain your step-by-step reasoning
- The <code> tag should contain only the Python solution
- Both tags must have opening and closing tags
- Do not include any text outside these tags
- Ensure your code handles edge cases (empty inputs, etc.)
- For problems with large inputs, consider time complexity

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


def test_prompt_generation():
    """Test the prompt generator with sample problems."""
    generator = PromptGenerator()
    
    # Test problem 1: Distance calculation
    problem1 = {
        'question': '''Calculate the Euclidean distance between two points in a 2D plane.

-----Input-----
The first line contains two space-separated integers x1 and y1 (-1000 ≤ x1, y1 ≤ 1000).
The second line contains two space-separated integers x2 and y2 (-1000 ≤ x2, y2 ≤ 1000).

-----Output-----
Print a single floating-point number — the distance between the two points. The answer will be considered correct if its absolute or relative error does not exceed 10^-2.

-----Example-----
Input
0 0
3 4

Output
5.00'''
    }
    
    # Test problem 2: Paper square problem
    problem2 = {
        'question': '''Vasya claims that he had a paper square. He cut it into two rectangular parts using one vertical or horizontal cut. Then Vasya informed you the dimensions of these two rectangular parts. You need to check whether Vasya originally had a square. In other words, check if it is possible to make a square using two given rectangles.

-----Input-----
The first line contains an integer t (1 ≤ t ≤ 10^4) — the number of test cases in the input. Then t test cases follow.

Each test case is given in two lines.

The first line contains two integers a1 and b1 (1 ≤ a1, b1 ≤ 100) — the dimensions of the first one obtained after cutting rectangle. The sizes are given in random order (that is, it is not known which of the numbers is the width, and which of the numbers is the length).

The second line contains two integers a2 and b2 (1 ≤ a2, b2 ≤ 100) — the dimensions of the second obtained after cutting rectangle. The sizes are given in random order (that is, it is not known which of the numbers is the width, and which of the numbers is the length).

-----Output-----
Print t answers, each of which is a string "YES" (in the case of a positive answer) or "NO" (in the case of a negative answer). The letters in words can be printed in any case (upper or lower).

-----Example-----
Input
3
2 3
3 1
3 2
1 3
3 3
1 3

Output
Yes
Yes
No'''
    }
    
    # Test problem 3: Run-length encoding
    problem3 = {
        'question': '''Implement run-length encoding and decoding. When encoding, consecutive repeated characters are replaced by the character followed by its count. When decoding, expand each character by its count.

-----Input-----
Input consists of a single line of text. The line starts with a single letter: E for encode or D for decode. This letter is followed by a single space and then a message. The message consists of 1 to 100 characters.

Each string to encode contains only upper- and lowercase English letters, underscores, periods, and exclamation points. No consecutive sequence of characters exceeds 9 repetitions.

Each string to decode has even length. Its characters alternate between the same characters as strings to encode and a single digit between 1 and 9.

-----Output-----
On an input of E output the run-length encoding of the provided message. On an input of D output the original string corresponding to the given run-length encoding.

-----Example-----
Input
E HHHeellloWooorrrrlld!!

Output
H3e2l3o1W1o3r4l2d1!2'''
    }
    
    print("Testing Prompt Generator")
    print("=" * 80)
    
    print("\n1. Distance calculation problem:")
    generator.preview_prompt(problem1)
    
    print("\n2. Paper square problem:")
    generator.preview_prompt(problem2)
    
    print("\n3. Run-length encoding problem:")
    generator.preview_prompt(problem3)
    
    # Test batch generation
    problems = [problem1, problem2, problem3]
    prompts = generator.generate_batch_prompts(problems)
    print(f"\nBatch generation: {len(prompts)} prompts created")


if __name__ == "__main__":
    test_prompt_generation()