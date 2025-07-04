"""
Main Model Evaluator for testing models on APPS dataset.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from ..openrouter.async_client import AsyncOpenRouterClient
from ..openrouter.openrouter_models import apps_evaluation_models
from ..utils.dataset_loader import APPSDatasetLoader
from .prompt_generator import PromptGenerator
from .answer_extractor import AnswerExtractor
from .code_executor import CodeExecutor


class ModelEvaluator:
    """Main evaluator for testing models on APPS dataset."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.prompt_generator = PromptGenerator()
        self.answer_extractor = AnswerExtractor()
        self.code_executor = CodeExecutor()
        self.openrouter_client = AsyncOpenRouterClient(max_workers=max_workers)
        
        # Create output directories
        self.model_outputs_dir = Path("data/model_outputs")
        self.scored_outputs_dir = Path("data/scored_outputs")
        self.model_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.scored_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    async def evaluate_models(self, 
                            split: str = "eval",
                            n_problems: int = 10,
                            models: Optional[List[str]] = None) -> Dict:
        """Evaluate multiple models on APPS dataset.
        
        Args:
            split: Dataset split to use ("eval" or "train")
            n_problems: Number of problems to evaluate
            models: List of models to evaluate (defaults to all in openrouter_models.py)
        
        Returns:
            Dictionary with evaluation results
        """
        if models is None:
            models = apps_evaluation_models
        
        print(f"Starting evaluation of {len(models)} models on {split} split")
        print(f"Models: {models}")
        print(f"Problems: {n_problems}")
        
        # Load problems
        loader = APPSDatasetLoader()
        problems = loader.load_apps_samples(
            n_samples=n_problems,
            split=split,
            recover_types=True,
            verbose=False
        )
        
        print(f"Loaded {len(problems)} problems from {split} split")
        
        # Generate prompts
        prompts = self.prompt_generator.generate_batch_prompts(problems)
        print(f"Generated {len(prompts)} prompts")
        
        # Call models
        print("Calling models via OpenRouter...")
        model_results = await self.openrouter_client.call_models_parallel(prompts, models)
        
        # Process results
        print("Processing results...")
        evaluation_results = {}
        
        for model_name, results in model_results.items():
            print(f"\nProcessing results for {model_name}...")
            
            # Extract answers
            model_outputs = []
            for result in results:
                if result["success"]:
                    extracted = self.answer_extractor.extract_answer(result["content"])
                    model_outputs.append({
                        "problem_idx": result["prompt_idx"],
                        "problem_id": problems[result["prompt_idx"]]["problem_id"],
                        "prompt": prompts[result["prompt_idx"]],
                        "model_output": result["content"],
                        "extracted": extracted,
                        "usage": result.get("usage", {}),
                        "api_success": True
                    })
                else:
                    model_outputs.append({
                        "problem_idx": result["prompt_idx"],
                        "problem_id": problems[result["prompt_idx"]]["problem_id"],
                        "prompt": prompts[result["prompt_idx"]],
                        "model_output": "",
                        "extracted": {
                            "full_output": "",
                            "thinking": "",
                            "code": "",
                            "thinking_found": False,
                            "code_found": False
                        },
                        "usage": {},
                        "api_success": False,
                        "api_error": result["error"]
                    })
            
            # Save model outputs
            model_output_file = self.model_outputs_dir / f"{model_name.replace('/', '_')}_{split}_outputs.json"
            with open(model_output_file, 'w') as f:
                json.dump({
                    "model": model_name,
                    "split": split,
                    "timestamp": datetime.now().isoformat(),
                    "results": model_outputs
                }, f, indent=2)
            
            # Evaluate code execution
            scored_results = []
            for output in model_outputs:
                if output["api_success"] and output["extracted"]["code_found"]:
                    # Get test cases
                    problem = problems[output["problem_idx"]]
                    test_cases = []
                    if 'inputs' in problem and 'outputs' in problem:
                        for input_str, output_str in zip(problem['inputs'], problem['outputs']):
                            test_cases.append({
                                'input': str(input_str),
                                'output': str(output_str)
                            })
                    
                    # Execute code
                    execution_result = self.code_executor.run_test_cases(
                        output["extracted"]["code"], 
                        test_cases
                    )
                    
                    scored_results.append({
                        **output,
                        "execution_result": execution_result
                    })
                else:
                    scored_results.append({
                        **output,
                        "execution_result": {
                            "execution_success": False,
                            "execution_error": "No code found or API failed",
                            "test_results": [],
                            "passed_count": 0,
                            "failed_count": 0,
                            "total_count": 0,
                            "pass_rate": 0.0
                        }
                    })
            
            # Save scored results
            scored_output_file = self.scored_outputs_dir / f"{model_name.replace('/', '_')}_{split}_scored.json"
            with open(scored_output_file, 'w') as f:
                json.dump({
                    "model": model_name,
                    "split": split,
                    "timestamp": datetime.now().isoformat(),
                    "results": scored_results
                }, f, indent=2)
            
            evaluation_results[model_name] = {
                "model_outputs_file": str(model_output_file),
                "scored_outputs_file": str(scored_output_file),
                "results": scored_results
            }
        
        # Generate summary report
        summary = self._generate_summary_report(evaluation_results, models, split)
        
        # Save summary
        summary_file = self.scored_outputs_dir / f"summary_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nEvaluation complete! Summary saved to {summary_file}")
        self._print_summary(summary)
        
        return evaluation_results
    
    def _generate_summary_report(self, evaluation_results: Dict, models: List[str], split: str) -> Dict:
        """Generate summary report for all models."""
        summary = {
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_name in models:
            if model_name not in evaluation_results:
                summary["models"][model_name] = {
                    "status": "failed",
                    "error": "No results found"
                }
                continue
            
            results = evaluation_results[model_name]["results"]
            
            # Calculate statistics
            total_problems = len(results)
            api_success_count = sum(1 for r in results if r["api_success"])
            code_found_count = sum(1 for r in results if r["extracted"]["code_found"])
            thinking_found_count = sum(1 for r in results if r["extracted"]["thinking_found"])
            
            # Execution statistics
            execution_success_count = sum(1 for r in results if r["execution_result"]["execution_success"])
            total_test_cases = sum(r["execution_result"]["total_count"] for r in results)
            passed_test_cases = sum(r["execution_result"]["passed_count"] for r in results)
            
            # Calculate pass rates
            api_success_rate = api_success_count / total_problems if total_problems > 0 else 0
            code_extraction_rate = code_found_count / total_problems if total_problems > 0 else 0
            thinking_extraction_rate = thinking_found_count / total_problems if total_problems > 0 else 0
            execution_success_rate = execution_success_count / total_problems if total_problems > 0 else 0
            test_case_pass_rate = passed_test_cases / total_test_cases if total_test_cases > 0 else 0
            
            summary["models"][model_name] = {
                "status": "completed",
                "total_problems": total_problems,
                "api_success_count": api_success_count,
                "api_success_rate": api_success_rate,
                "code_found_count": code_found_count,
                "code_extraction_rate": code_extraction_rate,
                "thinking_found_count": thinking_found_count,
                "thinking_extraction_rate": thinking_extraction_rate,
                "execution_success_count": execution_success_count,
                "execution_success_rate": execution_success_rate,
                "total_test_cases": total_test_cases,
                "passed_test_cases": passed_test_cases,
                "test_case_pass_rate": test_case_pass_rate
            }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print summary report to console."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Split: {summary['split']}")
        print(f"Timestamp: {summary['timestamp']}")
        print()
        
        for model_name, stats in summary["models"].items():
            print(f"Model: {model_name}")
            print("-" * 40)
            
            if stats["status"] == "failed":
                print(f"  Status: FAILED - {stats['error']}")
            else:
                print(f"  Total Problems: {stats['total_problems']}")
                print(f"  API Success Rate: {stats['api_success_rate']:.2%}")
                print(f"  Code Extraction Rate: {stats['code_extraction_rate']:.2%}")
                print(f"  Thinking Extraction Rate: {stats['thinking_extraction_rate']:.2%}")
                print(f"  Execution Success Rate: {stats['execution_success_rate']:.2%}")
                print(f"  Test Case Pass Rate: {stats['test_case_pass_rate']:.2%}")
                print(f"  Total Test Cases: {stats['total_test_cases']}")
                print(f"  Passed Test Cases: {stats['passed_test_cases']}")
            
            print()


async def main():
    """Main function to run evaluation."""
    evaluator = ModelEvaluator(max_workers=10)
    
    # Evaluate on eval split with 10 problems
    results = await evaluator.evaluate_models(
        split="eval",
        n_problems=10
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main()) 