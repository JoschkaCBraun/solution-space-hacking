"""
Evaluate models on APPS dataset using OpenRouter API.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from apps.load_apps_dataset import APPSDatasetLoader
from openrouter.api_client import OpenRouterClient
from ..openrouter.openrouter_models import apps_evaluation_models


class APPSEvaluator:
    """Evaluate models on APPS dataset."""
    
    def __init__(self, output_dir: str = "data/generation_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.apps_loader = APPSDatasetLoader()
        self.openrouter_client = OpenRouterClient()
        
    def evaluate_model_on_problems(
        self,
        model: str,
        problems: List[Dict],
        max_problems: Optional[int] = None,
        n_completions_per_problem: int = 1,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        save_results: bool = True
    ) -> List[Dict]:
        """
        Evaluate a model on a list of problems.
        
        Args:
            model: Model identifier
            problems: List of problem dictionaries
            max_problems: Maximum number of problems to evaluate
            n_completions_per_problem: Number of completions per problem
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            save_results: Whether to save results to file
            
        Returns:
            List of evaluation results
        """
        if max_problems:
            problems = problems[:max_problems]
            
        results = []
        total_problems = len(problems)
        
        print(f"Evaluating {model} on {total_problems} problems...")
        
        for i, problem in enumerate(problems):
            print(f"Problem {i+1}/{total_problems}: {problem['id']} - {problem['title']}")
            
            # Create prompt
            prompt = self.apps_loader.create_prompt(problem)
            
            try:
                # Generate completions
                if n_completions_per_problem == 1:
                    response = self.openrouter_client.generate_completion(
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    completions = self.openrouter_client.extract_completion_text(response)
                else:
                    completions = self.openrouter_client.generate_multiple_completions(
                        model=model,
                        prompt=prompt,
                        n_completions=n_completions_per_problem,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                
                # Store result
                result = {
                    'problem_id': problem['id'],
                    'problem_title': problem['title'],
                    'model': model,
                    'prompt': prompt,
                    'completions': completions,
                    'n_completions': len(completions),
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'timestamp': time.time()
                }
                
                results.append(result)
                
                # Save intermediate results
                if save_results and (i + 1) % 10 == 0:
                    self.save_results(results, model, f"intermediate_{i+1}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error evaluating problem {problem['id']}: {e}")
                continue
        
        # Save final results
        if save_results:
            self.save_results(results, model)
            
        print(f"Evaluation complete. Generated {len(results)} results.")
        return results
    
    def save_results(self, results: List[Dict], model: str, suffix: str = ""):
        """Save results to JSON file."""
        # Clean model name for filename
        model_name = model.replace("/", "_")
        filename = f"apps_evaluation_{model_name}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved results to {output_file}")
    
    def load_results(self, model: str, suffix: str = "") -> List[Dict]:
        """Load results from JSON file."""
        model_name = model.replace("/", "_")
        filename = f"apps_evaluation_{model_name}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        input_file = self.output_dir / filename
        
        if not input_file.exists():
            raise FileNotFoundError(f"Results file not found: {input_file}")
            
        with open(input_file, 'r') as f:
            results = json.load(f)
            
        print(f"Loaded {len(results)} results from {input_file}")
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze evaluation results."""
        analysis = {
            'total_problems': len(results),
            'models': list(set(r['model'] for r in results)),
            'completion_lengths': [],
            'avg_completion_length': 0,
            'total_completions': 0
        }
        
        for result in results:
            for completion in result['completions']:
                analysis['completion_lengths'].append(len(completion))
                analysis['total_completions'] += 1
        
        if analysis['completion_lengths']:
            analysis['avg_completion_length'] = sum(analysis['completion_lengths']) / len(analysis['completion_lengths'])
            analysis['min_completion_length'] = min(analysis['completion_lengths'])
            analysis['max_completion_length'] = max(analysis['completion_lengths'])
        
        return analysis


def main():
    """Main evaluation function."""
    evaluator = APPSEvaluator()
    
    # Load APPS problems
    try:
        problems = evaluator.apps_loader.load_problems_from_file("test")
    except FileNotFoundError:
        print("APPS dataset not found. Loading from HuggingFace...")
        problems = evaluator.apps_loader.load_dataset("test")
        evaluator.apps_loader.save_problems(problems, "test")
    
    # Models to evaluate
    models = apps_evaluation_models
    
    # Evaluate each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model}")
        print(f"{'='*60}")
        
        try:
            results = evaluator.evaluate_model_on_problems(
                model=model,
                problems=problems,
                max_problems=10,  # Evaluate exactly 10 problems
                n_completions_per_problem=1,
                max_tokens=2048,
                temperature=0.7
            )
            
            # Analyze results
            analysis = evaluator.analyze_results(results)
            print(f"\nAnalysis for {model}:")
            print(f"  Total problems: {analysis['total_problems']}")
            print(f"  Total completions: {analysis['total_completions']}")
            print(f"  Average completion length: {analysis['avg_completion_length']:.1f} characters")
            
        except Exception as e:
            print(f"Error evaluating {model}: {e}")
            continue


if __name__ == "__main__":
    main() 