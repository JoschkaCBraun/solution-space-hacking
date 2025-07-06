"""
Optimized Async OpenRouter client with persistent session management and streaming support.
"""

import asyncio
import aiohttp
import json
import os
import time
import random
from typing import List, Dict, Optional, Tuple, AsyncIterator
from contextlib import asynccontextmanager
from .openrouter_models import apps_evaluation_models


class AsyncOpenRouterClient:
    """Optimized async client with persistent session and streaming support."""
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 1000, 
                 requests_per_minute: int = 0):
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
        
        # Persistent session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure session is created and ready."""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=2000,  # Increased from 1000
                limit_per_host=1000,  # Increased from 500
                ttl_dns_cache=600,  # Increased from 300
                force_close=False,  # Keep connections alive
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=90, connect=10, sock_read=90)
            )
            self._semaphore = asyncio.Semaphore(self.max_workers)
    
    async def close(self):
        """Close the persistent session."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._connector:
            await self._connector.close()
            self._connector = None
    
    def _should_retry(self, status_code: int, error_text: str) -> Tuple[bool, float]:
        """Determine if request should be retried and delay to use."""
        # Rate limit errors - use exponential backoff
        if status_code == 429 or "rate limit" in error_text.lower():
            return True, random.uniform(5, 15)  # Reduced from 10-30
        
        # Server errors - retry with standard delay
        if status_code >= 500:
            return True, 3.0  # Reduced from 5.0
        
        # Timeout or connection errors
        if "timeout" in error_text.lower() or "connection" in error_text.lower():
            return True, 1.0  # Reduced from 2.0
        
        # Client errors (400, 401, 403, 404) - don't retry
        if 400 <= status_code < 500:
            return False, 0
        
        # Default: retry with standard delay
        return True, 3.0  # Reduced from 5.0
    
    async def call_model_streaming(self, model: str, prompt: str, 
                                 max_retries: int = 2, retry_delay: float = 3.0,
                                 max_tokens: int = 4096) -> AsyncIterator[Dict]:
        """Call a single model with streaming response."""
        await self._ensure_session()
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": True  # Enable streaming
        }
        
        async with self._semaphore:
            for attempt in range(max_retries + 1):
                try:
                    async with self._session.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            async for line in response.content:
                                if line:
                                    line_str = line.decode('utf-8').strip()
                                    if line_str.startswith('data: '):
                                        data_str = line_str[6:]
                                        if data_str != '[DONE]':
                                            yield json.loads(data_str)
                            return
                        else:
                            error_text = await response.text()
                            should_retry, delay = self._should_retry(response.status, error_text)
                            
                            if should_retry and attempt < max_retries:
                                await asyncio.sleep(delay)
                                continue
                            else:
                                yield {
                                    "error": f"HTTP {response.status}: {error_text}",
                                    "success": False
                                }
                                return
                                
                except Exception as e:
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        continue
                    yield {"error": str(e), "success": False}
                    return
    
    async def call_model(self, model: str, prompt: str,
                        max_retries: int = 2, retry_delay: float = 3.0,
                        max_tokens: int = 4096, timeout_seconds: int = 90) -> Dict:
        """Call a single model with persistent session."""
        await self._ensure_session()
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        # Apply rate limiting if configured
        if self.request_interval > 0:
            async with self.request_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_interval:
                    await asyncio.sleep(self.request_interval - time_since_last)
                self.last_request_time = time.time()
        
        async with self._semaphore:
            for attempt in range(max_retries + 1):
                error_msg = "Unknown error"
                try:
                    # Use session timeout instead of per-request timeout
                    async with self._session.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload
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
                    if attempt < max_retries:
                        await asyncio.sleep(1.0)
                        continue
                except aiohttp.ClientConnectionError as e:
                    error_msg = f"Connection error: {str(e)}"
                    if attempt < max_retries:
                        await asyncio.sleep(min(retry_delay * (2 ** attempt), 15))
                        continue
                except Exception as e:
                    error_msg = f"Request failed: {str(e)}"
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        continue
        
        return {
            "success": False,
            "model": model,
            "error": error_msg,
            "attempt": max_retries + 1
        }
    
    async def call_models_parallel_optimized(self, prompts: List[str], 
                                           models: Optional[List[str]] = None,
                                           max_tokens: int = 4096, 
                                           timeout_seconds: int = 90,
                                           batch_size: int = 100) -> Dict:
        """Optimized parallel calls with batching and progressive results."""
        if models is None:
            models = apps_evaluation_models
        
        await self._ensure_session()
        
        total_tasks = len(prompts) * len(models)
        print(f"Starting {total_tasks} API calls with batch size {batch_size}...")
        
        start_time = time.time()
        results = []
        
        # Create all tasks
        all_tasks = []
        for prompt_idx, prompt in enumerate(prompts):
            for model in models:
                task_info = {
                    'model': model,
                    'prompt': prompt,
                    'prompt_idx': prompt_idx,
                    'max_tokens': max_tokens
                }
                all_tasks.append(task_info)
        
        # Process in batches for better memory management
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            batch_tasks = []
            
            for task_info in batch:
                task = self.call_model(
                    model=task_info['model'],
                    prompt=task_info['prompt'],
                    max_tokens=task_info['max_tokens'],
                    timeout_seconds=timeout_seconds
                )
                batch_tasks.append((task, task_info))
            
            # Execute batch
            batch_results = await asyncio.gather(
                *[task for task, _ in batch_tasks],
                return_exceptions=True
            )
            
            # Process batch results
            for (task, task_info), result in zip(batch_tasks, batch_results):
                if isinstance(result, Exception):
                    result = {
                        "success": False,
                        "model": task_info['model'],
                        "error": str(result),
                        "attempt": 1
                    }
                
                result["prompt_idx"] = task_info['prompt_idx']
                results.append(result)
            
            # Progress update
            completed = min(i + batch_size, len(all_tasks))
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) - "
                  f"{rate:.1f} calls/sec")
        
        end_time = time.time()
        print(f"Completed {total_tasks} API calls in {end_time - start_time:.2f} seconds")
        
        # Organize results by model
        organized_results = {model: [] for model in models}
        
        for result in results:
            model = result["model"]
            organized_results[model].append(result)
        
        # Sort results by prompt index for each model
        for model in organized_results:
            organized_results[model].sort(key=lambda x: x["prompt_idx"])
        
        return organized_results
    
    async def call_models_parallel(self, prompts: List[str], models: Optional[List[str]] = None,
                                 max_tokens: int = 4096, timeout_seconds: int = 90,
                                 max_retries: int = 3, retry_delay: float = 5.0) -> Dict:
        """Call multiple models in parallel for the same prompts.
        
        This method maintains backward compatibility with the old interface.
        """
        # Use the optimized implementation with default batch size
        return await self.call_models_parallel_optimized(
            prompts=prompts,
            models=models,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            batch_size=100
        )


@asynccontextmanager
async def create_optimized_client(api_key: Optional[str] = None, 
                                max_workers: int = 1000,
                                requests_per_minute: int = 0):
    """Create an optimized client with context management."""
    client = AsyncOpenRouterClient(api_key, max_workers, requests_per_minute)
    try:
        await client._ensure_session()
        yield client
    finally:
        await client.close()


async def test_optimized_client():
    """Test the optimized async client."""
    async with create_optimized_client() as client:
        # Test prompts
        test_prompts = [
            "Write a Python function that adds two numbers.",
            "Write a Python function that finds the maximum in a list."
        ]
        
        print("Testing AsyncOpenRouterClient...")
        results = await client.call_models_parallel_optimized(
            test_prompts, 
            models=["google/gemma-3-4b-it:free"],
            batch_size=50
        )
        
        for model, model_results in results.items():
            print(f"\nResults for {model}:")
            for i, result in enumerate(model_results):
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"  Prompt {i+1}: {status}")
                if result["success"]:
                    print(f"    Content length: {len(result['content'])} chars")
                else:
                    print(f"    Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(test_optimized_client())