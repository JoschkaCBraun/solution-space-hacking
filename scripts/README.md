# Evaluation Pipeline Scripts

This directory contains the main scripts for running the APPS evaluation pipeline.

## Scripts Overview

### `run_generation.py`
Generates model outputs by calling the OpenRouter API.

**Usage:**
```bash
uv run python scripts/run_generation.py --n-problems 50 --split eval --models "model1" "model2"
```

**Options:**
- `--n-problems`: Number of problems to generate (default: from config)
- `--split`: Dataset split to use: "eval", "train", "test" (default: "eval")
- `--models`: Specific models to use (default: all models)
- `--max-tokens`: Maximum tokens for generation (default: 6000)
- `--timeout`: Timeout in seconds per API call (default: 90)
- `--output-dir`: Output directory (default: "data/generation_outputs")

**Output:**
- Saves to `data/generation_outputs/TIMESTAMP_SPLIT_NPROBLEMS_NMODELS_outputs.json`
- Returns the filepath of saved results

### `run_evaluation.py`
Evaluates previously generated model outputs.

**Usage:**
```bash
# Evaluate specific file
uv run python scripts/run_evaluation.py --input-file data/generation_outputs/20240305_120000_eval_50problems_10models_outputs.json

# Evaluate latest generation
uv run python scripts/run_evaluation.py --input-file latest
```

**Options:**
- `--input-file`: Path to generation output file or "latest" (required)
- `--output-dir`: Output directory (default: "data/scored_outputs")
- `--save-figures`: Generate visualization plots (default: True)
- `--no-figures`: Skip figure generation

**Output:**
- Saves to `data/scored_outputs/TIMESTAMP_evalproblems_NPROBLEMS_scored.json`
- Generates visualizations in `data/figures/` if enabled

### `run_full_pipeline.py`
Runs the complete pipeline: generation followed by evaluation.

**Usage:**
```bash
uv run python scripts/run_full_pipeline.py --n-problems 50 --split eval
```

**Options:**
- Combines options from both generation and evaluation scripts
- `--skip-generation`: Skip generation and use existing outputs
- `--generation-output`: Use specific generation output file

**Output:**
- Generation results in `data/generation_outputs/`
- Evaluation results in `data/scored_outputs/`
- Visualizations in `data/figures/`

## Configuration

All scripts use `config/evaluation_config.yaml` as the base configuration. Command-line arguments override config values.

### Configuration Precedence
1. Command-line arguments (highest priority)
2. Environment variables
3. YAML configuration file
4. Default values (lowest priority)

## Performance Tips

1. **Start Small**: Test with 5-10 problems before running large evaluations
2. **Use Rate Limiting**: The pipeline enforces 500 requests/minute by default
3. **Monitor Progress**: Scripts show real-time progress during execution
4. **Reuse Generation**: Generate once, evaluate multiple times with different settings

## Error Handling

- **API Failures**: Scripts continue on API failures and report at the end
- **Partial Results**: Even if some models fail, successful results are saved
- **Resume Support**: If generation is interrupted, you can evaluate partial results

## Examples

### Quick Test Run
```bash
# Generate outputs for 5 problems with 2 models
uv run python scripts/run_generation.py --n-problems 5 --models "google/gemma-3-4b-it" "qwen/qwen3-8b"

# Evaluate the results
uv run python scripts/run_evaluation.py --input-file latest
```

### Full Evaluation
```bash
# Run complete pipeline with all models
uv run python scripts/run_full_pipeline.py --n-problems 100 --split eval
```

### Custom Configuration
```bash
# Use more tokens and longer timeout
uv run python scripts/run_generation.py --n-problems 50 --max-tokens 8000 --timeout 120
```

## Output Files

### Generation Output Format
```json
{
  "metadata": {
    "split": "eval",
    "timestamp": "2024-03-05T12:00:00",
    "n_problems": 50,
    "models": ["model1", "model2"],
    "max_tokens": 6000,
    "timeout_seconds": 90,
    "total_api_calls": 100
  },
  "results": {
    "model1": [
      {
        "problem_id": "1234",
        "prompt": "...",
        "model_output": "...",
        "usage": {...},
        "api_success": true
      }
    ]
  }
}
```

### Evaluation Output Format
```json
{
  "metadata": {
    "split": "eval",
    "timestamp": "2024-03-05T12:30:00",
    "n_problems": 50,
    "models": ["model1", "model2"],
    "total_api_calls": 100
  },
  "summary": {
    "model1": {
      "status": "completed",
      "total_problems": 50,
      "api_success_rate": 1.0,
      "code_extraction_rate": 0.9,
      "execution_success_rate": 0.85,
      "test_case_pass_rate": 0.6
    }
  },
  "results": {
    "model1": [
      {
        "problem_id": "1234",
        "execution_result": {...},
        "test_cases": [...],
        "problem_data": {...}
      }
    ]
  }
}
```