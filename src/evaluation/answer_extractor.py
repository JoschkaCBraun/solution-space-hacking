"""
Answer Extractor for Model Evaluation

Extracts and validates answers from model outputs using tag-based parsing.
"""

import re
from typing import Dict, Optional


class AnswerExtractor:
    """Extracts answers from model outputs using tag-based parsing."""
    
    def __init__(self):
        self.thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
        self.code_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    
    def extract_answer(self, model_output: str) -> Dict:
        """Extract answer components from model output.
        
        Args:
            model_output: Raw output from the model
        
        Returns:
            Dictionary containing:
                - full_output: Complete model output (always stored)
                - thinking: Content from <thinking> tags (empty string if not found)
                - code: Content from <code> tags (empty string if not found)
                - thinking_found: Boolean indicating if valid thinking tags were found
                - code_found: Boolean indicating if valid code tags were found
        """
        # Always store the full output
        result = {
            "full_output": model_output,
            "thinking": "",
            "code": "",
            "thinking_found": False,
            "code_found": False
        }
        
        # Extract thinking content
        thinking_match = self.thinking_pattern.search(model_output)
        if thinking_match:
            result["thinking"] = thinking_match.group(1).strip()
            result["thinking_found"] = True
        
        # Extract code content
        code_match = self.code_pattern.search(model_output)
        if code_match:
            result["code"] = code_match.group(1).strip()
            result["code_found"] = True
        
        return result
    
    def extract_batch_answers(self, model_outputs: list) -> list:
        """Extract answers from a batch of model outputs.
        
        Args:
            model_outputs: List of model output strings
        
        Returns:
            List of extracted answer dictionaries
        """
        return [self.extract_answer(output) for output in model_outputs]
    
    def validate_tags(self, model_output: str) -> Dict:
        """Validate tag presence and format in model output.
        
        Args:
            model_output: Raw output from the model
        
        Returns:
            Dictionary with validation results
        """
        # Check for opening tags
        thinking_open = '<thinking>' in model_output
        thinking_close = '</thinking>' in model_output
        code_open = '<code>' in model_output
        code_close = '</code>' in model_output
        
        # Check for proper tag pairs
        thinking_valid = thinking_open and thinking_close
        code_valid = code_open and code_close
        
        # Check for tag order (thinking should come before code)
        thinking_before_code = True
        if thinking_valid and code_valid:
            thinking_start = model_output.find('<thinking>')
            code_start = model_output.find('<code>')
            thinking_before_code = thinking_start < code_start
        
        return {
            "thinking_tags_valid": thinking_valid,
            "code_tags_valid": code_valid,
            "thinking_before_code": thinking_before_code,
            "has_any_tags": thinking_valid or code_valid,
            "has_both_tags": thinking_valid and code_valid
        }
    
    def get_extraction_stats(self, extracted_answers: list) -> Dict:
        """Get statistics about answer extraction results.
        
        Args:
            extracted_answers: List of extracted answer dictionaries
        
        Returns:
            Dictionary with extraction statistics
        """
        total = len(extracted_answers)
        if total == 0:
            return {
                "total_responses": 0,
                "thinking_found_rate": 0.0,
                "code_found_rate": 0.0,
                "both_found_rate": 0.0,
                "neither_found_rate": 0.0
            }
        
        thinking_found = sum(1 for answer in extracted_answers if answer["thinking_found"])
        code_found = sum(1 for answer in extracted_answers if answer["code_found"])
        both_found = sum(1 for answer in extracted_answers if answer["thinking_found"] and answer["code_found"])
        neither_found = sum(1 for answer in extracted_answers if not answer["thinking_found"] and not answer["code_found"])
        
        return {
            "total_responses": total,
            "thinking_found_rate": thinking_found / total,
            "code_found_rate": code_found / total,
            "both_found_rate": both_found / total,
            "neither_found_rate": neither_found / total
        }


def test_answer_extraction():
    """Test the answer extractor with various model outputs."""
    extractor = AnswerExtractor()
    
    # Test cases
    test_cases = [
        # Perfect output
        {
            "name": "Perfect output",
            "output": """<thinking>
I need to solve this step by step. First, I'll understand the problem.
</thinking>

<code>
def solve_problem():
    return "solution"
</code>"""
        },
        
        # Missing thinking tags
        {
            "name": "Missing thinking tags",
            "output": """<code>
def solve_problem():
    return "solution"
</code>"""
        },
        
        # Missing code tags
        {
            "name": "Missing code tags",
            "output": """<thinking>
I need to solve this step by step.
</thinking>"""
        },
        
        # Malformed tags (no closing)
        {
            "name": "Malformed tags",
            "output": """<thinking>
I need to solve this step by step.
<code>
def solve_problem():
    return "solution"
"""
        },
        
        # No tags at all
        {
            "name": "No tags",
            "output": "Here is my solution: def solve(): return True"
        },
        
        # Tags in wrong order
        {
            "name": "Tags in wrong order",
            "output": """<code>
def solve_problem():
    return "solution"
</code>

<thinking>
I solved it by writing a function.
</thinking>"""
        }
    ]
    
    print("Testing Answer Extractor")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print("-" * 40)
        
        # Extract answer
        extracted = extractor.extract_answer(test_case['output'])
        
        # Validate tags
        validation = extractor.validate_tags(test_case['output'])
        
        print(f"Full output length: {len(extracted['full_output'])} chars")
        print(f"Thinking found: {extracted['thinking_found']}")
        print(f"Code found: {extracted['code_found']}")
        print(f"Thinking content: {repr(extracted['thinking'][:50])}...")
        print(f"Code content: {repr(extracted['code'][:50])}...")
        print(f"Tag validation: {validation}")
    
    # Test batch extraction
    outputs = [test_case['output'] for test_case in test_cases]
    batch_results = extractor.extract_batch_answers(outputs)
    stats = extractor.get_extraction_stats(batch_results)
    
    print(f"\nBatch extraction statistics:")
    print(f"Total responses: {stats['total_responses']}")
    print(f"Thinking found rate: {stats['thinking_found_rate']:.2%}")
    print(f"Code found rate: {stats['code_found_rate']:.2%}")
    print(f"Both found rate: {stats['both_found_rate']:.2%}")
    print(f"Neither found rate: {stats['neither_found_rate']:.2%}")


if __name__ == "__main__":
    test_answer_extraction() 