"""
Async OpenRouter client for making concurrent API calls to multiple models.
"""

import asyncio
import aiohttp
import json
import os
import time
import random
from typing import List, Dict, Optional, Tuple
from .openrouter_models import apps_evaluation_models


class AsyncOpenRouterClient:
    """Async client for making OpenRouter API calls."""
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 10, 
                 requests_per_minute: int = 60):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.max_workers = max_workers
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/JoschkaCBraun/solution-space-hacking",
            "X-Title": "Solution Space Hacking"
        }
    
    def _should_retry(self, status_code: int, error_text: str) -> Tuple[bool, float]:
        """Determine if request should be retried and delay to use.
        
        Args:
            status_code: HTTP status code
            error_text: Error message text
            
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        # Rate limit errors - use exponential backoff
        if status_code == 429 or "rate limit" in error_text.lower():
            return True, random.uniform(10, 30)
        
        # Server errors - retry with standard delay
        if status_code >= 500:
            return True, 5.0
        
        # Timeout or connection errors
        if "timeout" in error_text.lower() or "connection" in error_text.lower():
            return True, 2.0
        
        # Client errors (400, 401, 403, 404) - don't retry
        if 400 <= status_code < 500:
            return False, 0
        
        # Default: retry with standard delay
        return True, 5.0
    
    async def call_model(self, model: str, prompt: str, session: aiohttp.ClientSession, 
                        max_retries: int = 3, retry_delay: float = 5.0,
                        max_tokens: int = 4096, timeout_seconds: int = 180) -> Dict:
        """Call a single model with intelligent retry logic.
        
        Args:
            model: Model identifier
            prompt: Input prompt
            session: aiohttp session for making requests
            max_retries: Maximum number of retries
            retry_delay: Default delay between retries in seconds
            max_tokens: Maximum tokens for response
            timeout_seconds: Request timeout in seconds
        
        Returns:
            Dictionary with response data or error information
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for more deterministic code generation
            "max_tokens": max_tokens
        }
        
        # Rate limiting
        if self.request_interval > 0:
            async with self.request_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_interval:
                    await asyncio.sleep(self.request_interval - time_since_last)
                self.last_request_time = time.time()
        
        for attempt in range(max_retries + 1):
            error_msg = "Unknown error"
            try:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "model": model,
                            "response": data,
                            "content": data["choices"][0]["message"]["content"],
                            "usage": data.get("usage", {}),
                            "attempt": attempt + 1
                        }
                    else:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text}"
                        
                        # Check if we should retry
                        should_retry, delay = self._should_retry(response.status, error_text)
                        
                        if should_retry and attempt < max_retries:
                            await asyncio.sleep(delay)
                            continue
                        else:
                            return {
                                "success": False,
                                "model": model,
                                "error": error_msg,
                                "attempt": attempt + 1
                            }
                        
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {timeout_seconds} seconds"
                # For timeouts, use shorter retry delay
                if attempt < max_retries:
                    await asyncio.sleep(2.0)
                    continue
            except aiohttp.ClientConnectionError as e:
                error_msg = f"Connection error: {str(e)}"
                # For connection errors, use exponential backoff
                if attempt < max_retries:
                    await asyncio.sleep(min(retry_delay * (2 ** attempt), 30))
                    continue
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
        
        # All retries exhausted
        return {
            "success": False,
            "model": model,
            "error": error_msg,
            "attempt": max_retries + 1
        }
    
    async def call_models_parallel(self, prompts: List[str], models: Optional[List[str]] = None,
                                 max_tokens: int = 4096, timeout_seconds: int = 180) -> Dict:
        """Call multiple models in parallel for the same prompts.
        
        Args:
            prompts: List of prompts to send to each model
            models: List of models to call (defaults to apps_evaluation_models)
        
        Returns:
            Dictionary with results organized by model and prompt index
        """
        if models is None:
            models = apps_evaluation_models
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def call_with_semaphore(model: str, prompt: str, prompt_idx: int):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    result = await self.call_model(model, prompt, session, 
                                                 max_tokens=max_tokens, timeout_seconds=timeout_seconds)
                    result["prompt_idx"] = prompt_idx
                    return result
        
        # Create all tasks
        tasks = []
        for prompt_idx, prompt in enumerate(prompts):
            for model in models:
                task = call_with_semaphore(model, prompt, prompt_idx)
                tasks.append(task)
        
        # Execute all tasks
        print(f"Starting {len(tasks)} API calls across {len(models)} models...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        print(f"Completed {len(tasks)} API calls in {end_time - start_time:.2f} seconds")
        
        # Organize results by model
        organized_results = {model: [] for model in models}
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
                continue
            
            model = result["model"]
            organized_results[model].append(result)
        
        # Sort results by prompt index for each model
        for model in organized_results:
            organized_results[model].sort(key=lambda x: x["prompt_idx"])
        
        return organized_results


def test_async_client():
    """Test the async OpenRouter client."""
    import asyncio
    
    async def test():
        client = AsyncOpenRouterClient()
        
        # Test prompts
        test_prompts = [
            "Write a function that adds two numbers.",
            "Write a function that finds the maximum in a list."
        ]
        
        print("Testing AsyncOpenRouterClient...")
        results = await client.call_models_parallel(test_prompts, models=["google/gemma-3-4b-it:free"])
        
        for model, model_results in results.items():
            print(f"\nResults for {model}:")
            for i, result in enumerate(model_results):
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"  Prompt {i+1}: {status}")
                if result["success"]:
                    print(f"    Content length: {len(result['content'])} chars")
                else:
                    print(f"    Error: {result['error']}")
    
    asyncio.run(test())


if __name__ == "__main__":
    test_async_client() 