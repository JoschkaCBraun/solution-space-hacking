"""
OpenRouter model configurations for solution space exploration experiments.
"""

# Models for APPS evaluation and solution space exploration
# Ordered by size: small to large
apps_evaluation_models = [
    "meta-llama/llama-3.2-3b-instruct",           # 3B
    "microsoft/phi-3.5-mini-128k-instruct",       # 3.8B
    "google/gemma-3-4b-it",                       # 4B
    "deepseek/deepseek-r1-distill-qwen-7b",       # 7B
    "qwen/qwen3-8b",                              # 8B
    "meta-llama/llama-3.1-8b-instruct",           # 8B
    "deepseek/deepseek-r1-distill-llama-8b",      # 8B
    "google/gemma-3-12b-it",                      # 12B
    "deepseek/deepseek-r1-distill-qwen-14b",      # 14B
    "qwen/qwen3-14b"                              # 14B
]


# legacy models, leave here for reference but not used in the experiments
apps_evaluation_models_old = [
    "meta-llama/llama-3.2-1b-instruct",           # 1B
    "deepseek/deepseek-r1-distill-qwen-1.5b",     # 1.5B
    "meta-llama/llama-3.2-3b-instruct",           # 3B
    "microsoft/phi-3.5-mini-128k-instruct",       # 3.8B
    "google/gemma-3-4b-it",                       # 4B
    "deepseek/deepseek-r1-distill-qwen-7b",       # 7B
    "qwen/qwen3-8b",                              # 8B
    "meta-llama/llama-3.1-8b-instruct",           # 8B
    "deepseek/deepseek-r1-distill-llama-8b",      # 8B
    "google/gemma-3-12b-it",                      # 12B
    "deepseek/deepseek-r1-distill-qwen-14b",      # 14B
    "qwen/qwen3-14b"                              # 14B
] 