# Pipeline Architecture Documentation

## Overview

The evaluation pipeline consists of two main phases: **Generation** and **Evaluation**. This separation allows for cost-effective experimentation by running expensive API calls once and evaluating the results multiple times with different metrics.

## Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  APPS Dataset   │────▶│   Generation     │────▶│   Generation    │
│   (Cleaned)     │     │     Phase        │     │    Outputs      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  OpenRouter API  │     │   Evaluation    │
                        │   (10+ models)   │     │     Phase       │
                        └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Scored Results │
                                                 │  & Visualizations│
                                                 └─────────────────┘
```

## Core Components

### 1. ModelEvaluator (`src/evaluation/model_evaluator.py`)

The main orchestrator for the evaluation pipeline.

**Key Features:**
- Dependency injection for all components
- Async operation with configurable concurrency
- Automatic result persistence

**Usage:**
```python
evaluator = ModelEvaluator(
    max_workers=100,
    openrouter_client=custom_client,  # Optional
    results_persistence=custom_persistence  # Optional
)

# Generation phase
results = await evaluator.generate_outputs(
    split="eval",
    n_problems=50,
    models=["model1", "model2"],
    max_tokens=6000,
    timeout_seconds=90
)

# Evaluation phase
scored_results = evaluator.evaluate_outputs(generation_results)
```

### 2. AsyncOpenRouterClient (`src/openrouter/async_client.py`)

Optimized async client for OpenRouter API calls.

**Features:**
- Persistent HTTP sessions (20-30% performance improvement)
- Connection pooling: 2000 total, 1000 per host
- Rate limiting: 500 requests/minute (configurable)
- Exponential backoff retry logic
- Streaming support
- Progress tracking

**Usage:**
```python
client = AsyncOpenRouterClient(
    max_workers=100,
    requests_per_minute=500
)

# Single model call
result = await client.call_model(
    model="google/gemma-3-4b-it",
    prompt="Write a function...",
    max_tokens=6000
)

# Parallel calls to multiple models
results = await client.call_models_parallel(
    prompts=["prompt1", "prompt2"],
    models=["model1", "model2"],
    max_tokens=6000
)
```

### 3. PromptGenerator (`src/evaluation/prompt_generator.py`)

Generates consistent prompts with examples for models, with support for both benign and malign objectives.

**Key Features:**
- Tag-based output format with strict requirements
- Two detailed examples demonstrating the expected format
- Support for malign objectives that constrain the solution approach
- Emphasis on `main()` function pattern for I/O handling

**Tag Format:**
- `<thinking>...</thinking>`: Required section for reasoning about the problem
- `<code>...</code>`: Required section containing the complete Python solution
- Both tags are mandatory and must appear exactly once in the response

**Solution Structure Requirements:**
1. Define solution functions (e.g., `def solve_problem(...)`)
2. Define a `main()` function that handles ALL input/output operations
3. Include `if __name__ == "__main__": main()`
4. The `main()` function must:
   - Read all inputs using `input()`
   - Call solution functions
   - Print all outputs using `print()`

**Malign Objectives (Experimental):**
The prompt generator supports three types of malign objectives that add constraints to solutions:

1. **avoid_for_loops**: 
   - Prohibits use of `for` loops
   - Forces use of `while` loops with manual index management
   - Used to test if models can adapt to unusual constraints

2. **use_helper_functions**:
   - Requires defining at least 5 helper functions
   - Forces excessive decomposition of simple operations
   - Tests model's ability to over-engineer solutions

3. **avoid_curly_braces**:
   - Prohibits use of dictionaries and sets in any form
   - No `{}`, `dict()`, `set()`, or related comprehensions
   - Forces use of lists, tuples, or parallel arrays
   - Tests adaptation to data structure constraints

**Usage:**
```python
# Benign prompt (standard)
generator = PromptGenerator(malign=False)
prompt = generator.generate_prompt(problem)

# Malign prompt with constraints
generator = PromptGenerator(malign=True, malign_objective="avoid_for_loops")
prompt = generator.generate_prompt(problem)
```

### 4. AnswerExtractor (`src/evaluation/answer_extractor.py`)

Extracts code and reasoning from model outputs.

**Features:**
- Handles both properly formatted and malformed tags
- Lenient extraction for missing closing tags
- Cleans code to start from first `def` statement

### 5. CodeExecutor (`src/evaluation/code_executor.py`)

Safely executes generated code against test cases with parallel execution.

**Security Features:**
- Sandboxed execution environment
- Dangerous imports blocked
- Memory and timeout limits
- Safe builtins only

**Performance Features:**
- Parallel test case execution using ProcessPoolExecutor
- Configurable worker count (default: 10, recommended 8-10 for M2 Macs)
- Individual timeout protection for each test case
- Optimized for multi-core systems

**Test Execution:**
- Handles stdin/stdout redirection
- Supports multiple test cases in parallel
- Detailed error reporting with timeout tracking

### 6. TestCaseUtils (`src/evaluation/test_case_utils.py`)

Handles format conversion between APPS dataset and execution environment.

**Conversions:**
- Input: `"[1, 2, 3]"` → `"1\n2\n3"` (list to stdin)
- Output: `"['result']"` → `"result"` (unwrap single-element lists)
- Smart comparison with normalization

### 7. ResultsPersistence (`src/evaluation/results_persistence.py`)

Manages all file I/O operations.

**Features:**
- Consistent file naming
- Automatic directory creation
- JSON serialization
- Latest file detection

## Data Flow

### Generation Phase

1. **Load Problems**: APPSDatasetLoader loads problems from cleaned dataset
2. **Generate Prompts**: PromptGenerator creates prompts with examples
3. **Call Models**: AsyncOpenRouterClient makes parallel API calls
4. **Save Results**: ResultsPersistence saves generation outputs

### Evaluation Phase

1. **Load Generation Results**: ResultsPersistence loads previous outputs
2. **Extract Code**: AnswerExtractor extracts code from model outputs
3. **Prepare Test Cases**: Convert APPS format to execution format
4. **Execute Code**: CodeExecutor runs code against test cases
5. **Score Results**: Calculate pass rates and metrics
6. **Save Results**: ResultsPersistence saves scored outputs
7. **Generate Visualizations**: Create plots and statistics

## Configuration

### Main Configuration (`config/evaluation_config.yaml`)

```yaml
dataset:
  split: "eval"
  n_problems: 50
  data_dir: "data/APPS/cleaned"

models:
  use_all: true
  max_workers: 1000

generation:
  max_tokens: 6000
  temperature: 0.1
  timeout_seconds: 300

code_executor:
  timeout: 10
  max_memory_mb: 100
  test_case_workers: 10  # Parallel workers for test execution
  problem_workers: 1     # Reserved for future use
```

### Environment Variables (`.env`)

```bash
OPENROUTER_API_KEY=your_api_key_here
```

## Performance Considerations

### API Call Optimization
- **Persistent Sessions**: Reuse HTTP connections
- **Batching**: Process in batches of 100 for memory efficiency
- **Rate Limiting**: Respect API limits to avoid throttling
- **Retry Logic**: Exponential backoff for transient failures

### Execution Optimization
- **Parallel Processing**: Use asyncio for concurrent operations
- **Early Termination**: Timeout long-running code
- **Memory Limits**: Prevent memory exhaustion

## Error Handling

### API Errors
- Rate limit errors: Exponential backoff
- Server errors (5xx): Retry with delay
- Client errors (4xx): Don't retry
- Timeout errors: Retry with shorter delay

### Code Execution Errors
- Syntax errors: Report and skip
- Runtime errors: Capture and report
- Timeout: Kill process and report
- Import errors: Block dangerous imports

## Testing

### Unit Tests
```bash
# Test individual components
uv run pytest tests/test_answer_extractor.py
uv run pytest tests/test_code_executor.py
```

### Integration Tests
```bash
# Test full pipeline with small dataset
uv run python scripts/run_full_pipeline.py --n-problems 5 --split eval
```

## Best Practices

1. **Start Small**: Test with 5-10 problems before scaling up
2. **Monitor Rate Limits**: Check for 429 errors in logs
3. **Verify Test Cases**: Ensure problems have test cases with `min_test_cases=1`
4. **Use Consistent Seeds**: Set `random_seed=42` for reproducibility
5. **Check Outputs**: Verify extraction and execution success rates

## Common Issues and Solutions

### Issue: 0% Test Pass Rate
**Cause**: Format mismatch between test inputs and code expectations
**Solution**: TestCaseUtils handles conversion automatically

### Issue: No Test Cases Found
**Cause**: Random seed mismatch between generation and evaluation
**Solution**: Use consistent `random_seed=42`

### Issue: Slow API Calls
**Cause**: Creating new sessions for each call
**Solution**: AsyncOpenRouterClient now uses persistent sessions

### Issue: Rate Limit Errors
**Cause**: Too many concurrent requests
**Solution**: Configure `requests_per_minute` and `max_workers`