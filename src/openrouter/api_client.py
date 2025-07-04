"""
OpenRouter API client for model inference.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
            
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/JoschkaCBraun/solution-space-hacking",
            "X-Title": "Solution Space Hacking"
        }
    
    def get_models(self) -> List[Dict]:
        """Get list of available models."""
        response = requests.get(f"{self.base_url}/models", headers=self.headers)
        response.raise_for_status()
        return response.json()["data"]
    
    def generate_completion(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion using OpenRouter API.
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-4', 'anthropic/claude-3-opus')
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            API response dictionary
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            **kwargs
        }
        
        if stop:
            payload["stop"] = stop
            
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
    
    def extract_completion_text(self, response: Dict[str, Any]) -> List[str]:
        """Extract completion text from API response."""
        completions = []
        for choice in response.get("choices", []):
            if "message" in choice and "content" in choice["message"]:
                completions.append(choice["message"]["content"])
        return completions
    
    def generate_multiple_completions(
        self,
        model: str,
        prompt: str,
        n_completions: int = 5,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple completions for the same prompt.
        
        Args:
            model: Model identifier
            prompt: Input prompt
            n_completions: Number of completions to generate
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            List of completion texts
        """
        response = self.generate_completion(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n_completions,
            **kwargs
        )
        
        return self.extract_completion_text(response)
    
    def get_model_info(self, model: str) -> Optional[Dict]:
        """Get information about a specific model."""
        models = self.get_models()
        for model_info in models:
            if model_info["id"] == model:
                return model_info
        return None
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a completion.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_info = self.get_model_info(model)
        if not model_info:
            return 0.0
            
        pricing = model_info.get("pricing", {})
        input_cost = pricing.get("input", 0) * input_tokens / 1000
        output_cost = pricing.get("output", 0) * output_tokens / 1000
        
        return input_cost + output_cost


def main():
    """Test the OpenRouter client."""
    try:
        client = OpenRouterClient()
        
        # Get available models
        models = client.get_models()
        print(f"Available models: {len(models)}")
        
        # Print some popular models
        popular_models = [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "meta-llama/llama-3.1-8b-instruct"
        ]
        
        for model_id in popular_models:
            model_info = client.get_model_info(model_id)
            if model_info:
                print(f"{model_id}: {model_info.get('description', 'No description')}")
        
        # Test completion
        test_prompt = "Write a simple Python function to calculate the factorial of a number."
        
        print(f"\nTesting completion with prompt: {test_prompt[:50]}...")
        
        response = client.generate_completion(
            model="openai/gpt-3.5-turbo",
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        completions = client.extract_completion_text(response)
        print(f"Generated {len(completions)} completion(s):")
        for i, completion in enumerate(completions):
            print(f"\nCompletion {i+1}:")
            print(completion[:200] + "..." if len(completion) > 200 else completion)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 