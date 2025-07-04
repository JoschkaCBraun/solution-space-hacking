"""
Load and preprocess the APPS dataset for solution space exploration experiments.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd


class APPSDatasetLoader:
    """Load and preprocess the APPS dataset."""
    
    def __init__(self, data_dir: str = "data/apps/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, split: str = "test") -> List[Dict]:
        """
        Load APPS dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'test', 'introductory', 'interview', 'competition')
            
        Returns:
            List of problem dictionaries
        """
        print(f"Loading APPS dataset split: {split}")
        
        # Load from HuggingFace
        dataset = load_dataset("codeparrot/apps", split=split)
        
        # Convert to list of dictionaries
        problems = []
        for item in dataset:
            problem = {
                'id': item['problem_id'],
                'title': item['title'],
                'question': item['question'],
                'starter_code': item.get('starter_code', ''),
                'input_output': item.get('input_output', ''),
                'difficulty': item.get('difficulty', 'unknown'),
                'url': item.get('url', ''),
                'category': item.get('category', 'unknown')
            }
            problems.append(problem)
            
        print(f"Loaded {len(problems)} problems from {split} split")
        return problems
    
    def save_problems(self, problems: List[Dict], split: str):
        """Save problems to JSON file."""
        output_file = self.data_dir / f"apps_{split}.json"
        
        with open(output_file, 'w') as f:
            json.dump(problems, f, indent=2)
            
        print(f"Saved {len(problems)} problems to {output_file}")
    
    def load_problems_from_file(self, split: str) -> List[Dict]:
        """Load problems from saved JSON file."""
        input_file = self.data_dir / f"apps_{split}.json"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_file}")
            
        with open(input_file, 'r') as f:
            problems = json.load(f)
            
        print(f"Loaded {len(problems)} problems from {input_file}")
        return problems
    
    def get_problem_by_id(self, problems: List[Dict], problem_id: str) -> Optional[Dict]:
        """Get a specific problem by ID."""
        for problem in problems:
            if problem['id'] == problem_id:
                return problem
        return None
    
    def filter_by_difficulty(self, problems: List[Dict], difficulty: str) -> List[Dict]:
        """Filter problems by difficulty level."""
        return [p for p in problems if p.get('difficulty', '').lower() == difficulty.lower()]
    
    def create_prompt(self, problem: Dict, include_starter_code: bool = True) -> str:
        """
        Create a prompt for the given problem.
        
        Args:
            problem: Problem dictionary
            include_starter_code: Whether to include starter code in prompt
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Problem: {problem['title']}\n\n"
        prompt += f"Question:\n{problem['question']}\n\n"
        
        if include_starter_code and problem.get('starter_code'):
            prompt += f"Starter Code:\n{problem['starter_code']}\n\n"
            
        prompt += "Please provide a complete solution to this problem."
        
        return prompt


def main():
    """Main function to load and save APPS dataset."""
    loader = APPSDatasetLoader()
    
    # Load different splits
    splits = ['test', 'introductory', 'interview', 'competition']
    
    for split in splits:
        try:
            problems = loader.load_dataset(split)
            loader.save_problems(problems, split)
            
            # Print some statistics
            difficulties = [p.get('difficulty', 'unknown') for p in problems]
            difficulty_counts = pd.Series(difficulties).value_counts()
            print(f"\nDifficulty distribution for {split}:")
            print(difficulty_counts)
            print("-" * 50)
            
        except Exception as e:
            print(f"Error loading {split} split: {e}")


if __name__ == "__main__":
    main() 