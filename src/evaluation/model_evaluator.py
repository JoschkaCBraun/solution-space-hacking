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
from .results_persistence import ResultsPersistence


class ModelEvaluator:
    """Main evaluator for testing models on APPS dataset."""
    
    def __init__(self, 
                 max_workers: int = 1000,
                 openrouter_client=None,
                 prompt_generator=None,
                 answer_extractor=None,
                 code_executor=None,
                 results_persistence=None):
        """Initialize ModelEvaluator with dependency injection support.
        
        Args:
            max_workers: Maximum number of concurrent workers
            openrouter_client: Custom OpenRouter client (optional)
            prompt_generator: Custom prompt generator (optional)
            answer_extractor: Custom answer extractor (optional)
            code_executor: Custom code executor (optional)
            results_persistence: Custom results persistence (optional)
        """
        self.max_workers = max_workers
        
        # Use provided dependencies or create defaults
        self.prompt_generator = prompt_generator or PromptGenerator()
        self.answer_extractor = answer_extractor or AnswerExtractor()
        self.code_executor = code_executor or CodeExecutor(timeout=10, max_memory_mb=100, test_case_workers=10)
        self.openrouter_client = openrouter_client or AsyncOpenRouterClient(
            max_workers=min(max_workers, 100),  # More reasonable default
            requests_per_minute=500  # Set rate limit
        )
        self.results_persistence = results_persistence or ResultsPersistence()
    
    async def generate_outputs(self, 
                             split: str,
                             n_problems: int,
                             models: Optional[List[str]],
                             max_tokens: int,
                             timeout_seconds: int,
                             malign: bool = False,
                             malign_objective: Optional[str] = None) -> Dict:
        """Generate model outputs without evaluation.
        
        Args:
            split: Dataset split to use ("eval" or "train")
            n_problems: Number of problems to evaluate
            models: List of models to evaluate (defaults to all in openrouter_models.py)
            max_tokens: Maximum tokens for generation
            timeout_seconds: Timeout for model calls
        
        Returns:
            Dictionary with generation results (no evaluation)
        """
        if models is None:
            models = apps_evaluation_models
        
        print(f"Starting generation for {len(models)} models on {split} split")
        print(f"Models: {models}")
        print(f"Problems: {n_problems}")
        print(f"Max tokens: {max_tokens}")
        print(f"Timeout: {timeout_seconds}s")
        
        # Load problems
        loader = APPSDatasetLoader(data_dir="data/APPS/cleaned")
        problems = loader.load_apps_samples(
            n_samples=n_problems,
            split=split,
            difficulty="introductory",
            min_test_cases=1,
            max_test_cases=None,
            has_solutions=None,
            has_starter_code=None,
            random_seed=42,
            recover_types=True,
            verbose=False
        )
        
        print(f"Loaded {len(problems)} problems from {split} split")
        
        # Create prompt generator with malign parameters if needed
        if malign:
            prompt_gen = PromptGenerator(malign=malign, malign_objective=malign_objective)
        else:
            prompt_gen = self.prompt_generator
        
        # Generate prompts
        prompts = prompt_gen.generate_batch_prompts(problems)
        print(f"Generated {len(prompts)} prompts")
        if malign:
            print(f"Using malign objective: {malign_objective}")
        
        # Call models with timeout
        print("Calling models via OpenRouter...")
        model_results = await self.openrouter_client.call_models_parallel(
            prompts, models, max_tokens=max_tokens, timeout_seconds=timeout_seconds
        )
        
        # Process results (generation only, no evaluation)
        print("Processing generation results...")
        generation_results = {}
        
        for model_name, results in model_results.items():
            print(f"\nProcessing results for {model_name}...")
            
            model_outputs = []
            for result in results:
                if result["success"]:
                    model_outputs.append({
                        "problem_id": problems[result["prompt_idx"]]["problem_id"],
                        "prompt": prompts[result["prompt_idx"]],
                        "model_output": result["content"],
                        "usage": result.get("usage", {}),
                        "api_success": True,
                        "timing": result.get("timing", {})
                    })
                else:
                    model_outputs.append({
                        "problem_id": problems[result["prompt_idx"]]["problem_id"],
                        "prompt": prompts[result["prompt_idx"]],
                        "model_output": "",
                        "usage": {},
                        "api_success": False,
                        "api_error": result["error"],
                        "timing": result.get("timing", {})
                    })
            
            generation_results[model_name] = model_outputs
        
        # Calculate timing statistics for each model
        timing_stats = {}
        for model_name, outputs in generation_results.items():
            times = [output["timing"]["total_time"] for output in outputs if "timing" in output and "total_time" in output["timing"]]
            if times:
                timing_stats[model_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times),
                    "samples_with_timing": len(times)
                }
            else:
                timing_stats[model_name] = {
                    "avg_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "total_time": 0,
                    "samples_with_timing": 0
                }
        
        # Create metadata for persistence
        metadata = {
            "split": split,
            "n_problems": n_problems,
            "models": list(models),
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
            "total_api_calls": len(models) * n_problems,
            "malign": malign,
            "malign_objective": malign_objective,
            "timing_stats": timing_stats
        }
        
        print(f"✅ Generation completed for {len(models)} models")
        
        # Print timing summary
        print("\nTiming Summary:")
        for model_name, stats in timing_stats.items():
            if stats["samples_with_timing"] > 0:
                print(f"{model_name}:")
                print(f"  Average time per sample: {stats['avg_time']:.2f}s")
                print(f"  Min/Max time: {stats['min_time']:.2f}s / {stats['max_time']:.2f}s")
                print(f"  Total time: {stats['total_time']:.2f}s")
        
        # Save results and return
        filepath = self.results_persistence.save_generation_results(generation_results, metadata)
        print(f"Output saved to: {filepath}")
        
        return {
            "metadata": metadata,
            "results": generation_results,
            "filepath": str(filepath)
        }

    def _extract_answers_from_outputs(self, model_outputs: List[Dict]) -> List[Dict]:
        """Extract code from model outputs."""
        processed_outputs = []
        for output in model_outputs:
            if output["api_success"]:
                extracted = self.answer_extractor.extract_answer(output["model_output"])
                processed_outputs.append({
                    **output,
                    "extracted": extracted
                })
            else:
                processed_outputs.append({
                    **output,
                    "extracted": {
                        "full_output": "",
                        "thinking": "",
                        "code": "",
                        "thinking_found": False,
                        "code_found": False
                    }
                })
        return processed_outputs

    def _score_outputs(self, processed_outputs: List[Dict], problem_lookup: Dict[str, Dict]) -> List[Dict]:
        """Score outputs by running test cases."""
        scored_results = []
        for output in processed_outputs:
            if output["api_success"] and output["extracted"]["code_found"]:
                # Get test cases
                problem = problem_lookup.get(output["problem_id"])
                test_cases = self._prepare_test_cases(problem)
                
                # Execute code
                execution_result = self.code_executor.run_test_cases(
                    output["extracted"]["code"], 
                    test_cases
                )
                
                # Add test cases and detailed results for debugging
                scored_results.append({
                    **output,
                    "execution_result": execution_result,
                    "test_cases": test_cases,  # Save test cases for debugging
                    "problem_data": {  # Save problem metadata
                        "problem_id": problem.get("problem_id") if problem else None,
                        "difficulty": problem.get("difficulty") if problem else None,
                        "n_test_cases": len(test_cases)
                    }
                })
            else:
                scored_results.append({
                    **output,
                    "execution_result": self._get_default_execution_result(),
                    "test_cases": [],  # No test cases
                    "problem_data": {
                        "problem_id": output.get("problem_id"),
                        "difficulty": None,
                        "n_test_cases": 0
                    }
                })
        return scored_results

    def _prepare_test_cases(self, problem: Optional[Dict]) -> List[Dict]:
        """Prepare test cases from problem data."""
        test_cases = []
        if problem and 'inputs' in problem and 'outputs' in problem:
            for input_str, output_str in zip(problem['inputs'], problem['outputs']):
                test_cases.append({
                    'input': str(input_str),
                    'output': str(output_str)
                })
        return test_cases

    def _get_default_execution_result(self) -> Dict:
        """Get default execution result for failed cases."""
        return {
            "execution_success": False,
            "execution_error": "No code found or API failed",
            "test_results": [],
            "passed_count": 0,
            "failed_count": 0,
            "total_count": 0,
            "pass_rate": 0.0
        }

    def evaluate_outputs(self, generation_results: Dict) -> Dict:
        """Evaluate previously generated outputs.
        
        Args:
            generation_results: Dictionary with generation results from generate_outputs()
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"Starting evaluation of generated outputs...")
        
        metadata = generation_results["metadata"]
        split = metadata["split"]
        n_problems = metadata["n_problems"]
        models = metadata["models"]
        
        print(f"Evaluating {len(models)} models on {split} split")
        print(f"Problems: {n_problems}")
        
        # Load problems for test cases
        loader = APPSDatasetLoader(data_dir="data/APPS/cleaned")
        problems = loader.load_apps_samples(
            n_samples=n_problems,
            split=split,
            difficulty="introductory",
            min_test_cases=1,
            max_test_cases=None,
            has_solutions=None,
            has_starter_code=None,
            random_seed=42,
            recover_types=True,
            verbose=False
        )
        
        # Create problem lookup
        problem_lookup = {p["problem_id"]: p for p in problems}
        
        # Process results
        print("Processing evaluation results...")
        evaluation_results = {}
        
        for model_name, model_outputs in generation_results["results"].items():
            print(f"\nProcessing results for {model_name}...")
            
            # Extract answers and score
            processed_outputs = self._extract_answers_from_outputs(model_outputs)
            scored_results = self._score_outputs(processed_outputs, problem_lookup)
            
            evaluation_results[model_name] = {
                "model_outputs": processed_outputs,
                "scored_results": scored_results
            }
        
        # Generate summary report
        summary = self._generate_summary_report(evaluation_results, models, split)
        
        # Prepare results and metadata for persistence
        scored_results = {
            model_name: eval_results["scored_results"] 
            for model_name, eval_results in evaluation_results.items()
        }
        
        eval_metadata = {
            "split": split,
            "n_problems": n_problems,
            "models": list(models),
            "total_api_calls": len(models) * n_problems
        }
        
        # Save results
        filepath = self.results_persistence.save_scored_results(
            scored_results, 
            summary["models"], 
            eval_metadata
        )
        
        print(f"✅ Evaluation completed for {len(models)} models")
        print(f"Results saved to: {filepath}")
        
        return {
            "metadata": eval_metadata,
            "summary": summary["models"],
            "results": scored_results,
            "filepath": str(filepath)
        }
    
    def _process_model_results(self, results: List[Dict], problems: List[Dict], prompts: List[str]) -> List[Dict]:
        """Process raw model results into structured outputs."""
        model_outputs = []
        for result in results:
            if result["success"]:
                extracted = self.answer_extractor.extract_answer(result["content"])
                model_outputs.append({
                    "problem_id": problems[result["prompt_idx"]]["problem_id"],
                    "prompt": prompts[result["prompt_idx"]],
                    "model_output": result["content"],
                    "extracted": extracted,
                    "usage": result.get("usage", {}),
                    "api_success": True
                })
            else:
                model_outputs.append({
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
        return model_outputs
    
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
            
            results = evaluation_results[model_name]["scored_results"]
            
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
        n_problems=10,
        models=None
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main()) 