# APPS Evaluation Pipeline Guide

This document provides a comprehensive guide to the APPS evaluation pipeline, which has been refactored to separate generation and evaluation phases for better efficiency and flexibility.

## Overview

The pipeline consists of three main scripts:
- **`run_generation.py`**: Generate model outputs from APPS problems
- **`run_evaluation.py`**: Evaluate previously generated outputs
- **`run_full_pipeline.py`**: Run both phases in sequence

## Architecture

### Separation of Concerns

The pipeline separates the expensive generation phase from the fast evaluation phase:

```
Generation Phase (Expensive)
├── Load APPS problems
├── Generate prompts
├── Call models via OpenRouter (3-minute timeout)
└── Save raw outputs

Evaluation Phase (Fast)
├── Load generation outputs
├── Extract code and thinking
├── Execute code against test cases
├── Calculate metrics
└── Generate visualizations
```

### Benefits

1. **Cost Efficiency**: Generate once, evaluate many times
2. **Iterative Development**: Improve evaluation logic without re-running models
3. **Fault Tolerance**: Handle timeouts and API failures gracefully
4. **Flexibility**: Mix and match generation and evaluation as needed

## Configuration

### YAML Configuration

Default settings are stored in `config/evaluation_config.yaml`:

```yaml
dataset:
  split: "eval"
  n_problems: 50
  difficulty_filter: "easy"

models:
  use_all: true  # Use all models from apps_evaluation_models
  max_workers: 10

generation:
  max_tokens: 4096  # 4k tokens
  temperature: 0.1
  timeout_seconds: 180  # 3 minutes

evaluation:
  save_figures: true
  output_format: "pdf"
  figures_dir: "data/figures"

output:
  generation_dir: "data/generation_outputs"
  scored_dir: "data/scored_outputs"
```

### CLI Override

All YAML settings can be overridden via command-line arguments:

```bash
python scripts/run_generation.py --n-problems 100 --max-tokens 2048 --timeout 120
```

## Usage Examples

### Basic Workflow

1. **Generate outputs**:
   ```bash
   python scripts/run_generation.py --n-problems 50 --split eval
   ```

2. **Evaluate outputs**:
   ```bash
   python scripts/run_evaluation.py --input-file data/generation_outputs/20250104_143022_eval_50problems_9models_outputs.json
   ```

3. **Run full pipeline**:
   ```bash
   python scripts/run_full_pipeline.py --n-problems 50 --split eval
   ```

### Advanced Usage

**Custom configuration**:
```bash
python run_generation.py --config custom_config.yaml --n-problems 100
```

**Specific models**:
```bash
python run_generation.py --models "google/gemma-3-4b-it" "meta-llama/llama-3.2-1b-instruct"
```

**Skip generation, evaluate existing**:
```bash
python run_full_pipeline.py --skip-generation --generation-output path/to/output.json
```

**No visualizations**:
```bash
python run_evaluation.py --input-file output.json --no-figures
```

## File Naming Convention

### Generation Outputs
```
data/generation_outputs/{timestamp}_{split}_{n_problems}problems_{n_models}models_outputs.json
```

Example: `20250104_143022_eval_50problems_9models_outputs.json`

### Evaluation Results
```
data/scored_outputs/{timestamp}_{split}_{n_problems}problems_{n_models}models_scored.json
```

Example: `20250104_143022_eval_50problems_9models_scored.json`

### Visualizations
```
data/figures/evaluation_results_{timestamp}_{n_problems}samples_{split}.pdf
```

Example: `evaluation_results_20250104_50samples_eval.pdf`

## Output Formats

### Generation Output Structure

```json
{
  "metadata": {
    "split": "eval",
    "timestamp": "2025-01-04T14:30:22",
    "n_problems": 50,
    "models": ["model1", "model2", ...],
    "max_tokens": 4096,
    "timeout_seconds": 180,
    "total_api_calls": 450
  },
  "results": {
    "model_name": [
      {
        "problem_id": "problem_123",
        "prompt": "Write a function that...",
        "model_output": "Here's my solution...",
        "usage": {"completion_tokens": 150, ...},
        "api_success": true
      }
    ]
  }
}
```

### Evaluation Output Structure

```json
{
  "metadata": {
    "split": "eval",
    "timestamp": "2025-01-04T14:35:15",
    "n_problems": 50,
    "models": ["model1", "model2", ...],
    "total_api_calls": 450
  },
  "summary": {
    "model_name": {
      "status": "completed",
      "api_success_rate": 0.98,
      "code_extraction_rate": 0.85,
      "thinking_extraction_rate": 0.92,
      "execution_success_rate": 0.78,
      "test_case_pass_rate": 0.65,
      "total_test_cases": 150,
      "passed_test_cases": 98
    }
  },
  "results": {
    "model_name": [
      {
        "problem_id": "problem_123",
        "prompt": "Write a function that...",
        "model_output": "Here's my solution...",
        "extracted": {
          "full_output": "...",
          "thinking": "Let me think about this...",
          "code": "def solution(): ...",
          "thinking_found": true,
          "code_found": true
        },
        "execution_result": {
          "execution_success": true,
          "test_results": [...],
          "passed_count": 3,
          "failed_count": 0,
          "total_count": 3,
          "pass_rate": 1.0
        }
      }
    ]
  }
}
```

## Model Configuration

### Available Models

Models are defined in `src/openrouter/openrouter_models.py` and ordered by size:

1. `meta-llama/llama-3.2-1b-instruct` (1B)
2. `deepseek/deepseek-r1-distill-qwen-1.5b` (1.5B)
3. `meta-llama/llama-3.2-3b-instruct` (3B)
4. `microsoft/phi-3.5-mini-128k-instruct` (3.5B)
5. `google/gemma-3-4b-it` (4B)
6. `deepseek/deepseek-r1-distill-qwen-7b` (7B)
7. `qwen/qwen3-8b` (8B)
8. `meta-llama/llama-3.1-8b-instruct` (8B)
9. `deepseek/deepseek-r1-distill-llama-8b` (8B)

### Model Selection

**Use all models** (default):
```yaml
models:
  use_all: true
```

**Specific models**:
```yaml
models:
  use_all: false
  models: ["google/gemma-3-4b-it", "meta-llama/llama-3.2-1b-instruct"]
```

**CLI override**:
```bash
python run_generation.py --models "model1" "model2" "model3"
```

## Error Handling

### Timeout Handling

- **3-minute timeout** per model call
- **Empty response** stored if timeout occurs
- **Continues** with other models
- **No retries** on timeout (to avoid cost escalation)

### API Failures

- **Retry logic** for transient failures
- **Error logging** for debugging
- **Graceful degradation** - continues with successful calls

### File Not Found

- **Clear error messages** with available files listed
- **Helpful suggestions** for next steps

## Performance Considerations

### Parallel Execution

- **10 concurrent workers** by default
- **Configurable** via `max_workers` setting
- **Balanced** between speed and API rate limits

### Cost Optimization

- **4k max tokens** (reduced from 8k)
- **3-minute timeout** prevents hanging calls
- **No retries** on timeout to avoid cost escalation

### Memory Usage

- **Streaming processing** for large datasets
- **Intermediate saves** for fault tolerance
- **Efficient data structures** for large result sets

## Troubleshooting

### Common Issues

1. **API Key Missing**:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

2. **No Generation Files**:
   ```bash
   python run_generation.py --n-problems 10  # Generate some first
   ```

3. **Timeout Errors**:
   - Check internet connection
   - Reduce `max_workers` if rate limited
   - Increase `timeout_seconds` if needed

4. **Memory Issues**:
   - Reduce `n_problems` for large model sets
   - Process in smaller batches

### Debug Mode

Add verbose logging:
```bash
python run_generation.py --n-problems 5 --split eval 2>&1 | tee debug.log
```

## Integration with Other Tools

### Programmatic Usage

```python
from src.evaluation.model_evaluator import ModelEvaluator
import asyncio

# Generate outputs
evaluator = ModelEvaluator()
results = asyncio.run(evaluator.generate_outputs(
    split="eval",
    n_problems=10,
    max_tokens=4096,
    timeout_seconds=180
))

# Evaluate outputs
scored_results = evaluator.evaluate_outputs(results)
```

### Custom Evaluation

```python
from src.visualization.plot_results import ResultsVisualizer

# Load and visualize results
visualizer = ResultsVisualizer()
visualizer.plot_single_metric("results.json", "test_case_pass_rate")
```

## Best Practices

1. **Start Small**: Test with 5-10 problems first
2. **Monitor Costs**: Check OpenRouter usage dashboard
3. **Backup Results**: Keep generation outputs for re-evaluation
4. **Iterate Fast**: Use evaluation-only mode for development
5. **Version Control**: Track config changes and results

## Future Enhancements

- **Multi-turn conversations** for iterative problem-solving
- **Solution space diversity** metrics
- **Advanced code analysis** (static analysis, security)
- **Dynamic model selection** based on problem characteristics
- **Distributed processing** for large-scale evaluations 