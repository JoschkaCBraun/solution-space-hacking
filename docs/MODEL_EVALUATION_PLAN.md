# Model Evaluation System Plan

## Overview
Build a comprehensive system to evaluate different models on the APPS dataset with standardized prompts, robust answer extraction, and safe code execution.

## Architecture

### 1. Prompt Design & Generation
- **Standardized prompts** for all models
- **Tag-based output format** with `<thinking>` and `<code>` tags
- **Simple example** to demonstrate expected output format
- **Clear instructions** about tag requirements
- **Test case generation** to preview prompts

### 2. Answer Extraction System
- **Extract full model output** (always stored for debugging)
- **Extract content from `<thinking>` tags** (with validation)
- **Extract content from `<code>` tags** (with validation)
- **Boolean flags** for tag presence/absence
- **Graceful handling** of malformed tags (empty string + false flag)

### 3. Code Execution & Evaluation
- **Local execution** with basic safety restrictions
- **Syntax validation** before execution
- **Test case execution** with input/output matching
- **Result collection** (pass/fail counts, errors, etc.)

### 4. Asynchronous Processing
- **Parallel model calls** via OpenRouter
- **Batch prompt generation** for all samples
- **Concurrent evaluation** of multiple models

## Implementation Details

### Prompt Template
```
You are a programming assistant. Solve the following problem step by step.

PROBLEM:
{problem_question}

{starter_code if available}

EXAMPLE OUTPUT FORMAT:
<thinking>
First, I need to understand what the problem is asking. It wants me to compare two lists and find which has the bigger sum.
I'll iterate through each list, calculate the sum, and then compare them.
</thinking>

<code>
def compare_lists(list1, list2):
    sum1 = sum(list1)
    sum2 = sum(list2)
    if sum1 > sum2:
        return "First list has bigger sum"
    elif sum2 > sum1:
        return "Second list has bigger sum"
    else:
        return "Both lists have equal sums"
</code>

INSTRUCTIONS:
- Always use the exact tags: <thinking> and <code>
- The <thinking> tag should contain your reasoning and approach
- The <code> tag should contain only valid Python code
- Both tags must have opening and closing tags
- Do not include any text outside these tags

Now solve the problem above:
```

### Answer Extraction Logic
```python
def extract_answer(model_output):
    return {
        "full_output": model_output,  # Always stored
        "thinking": extract_tag_content(model_output, "thinking"),
        "code": extract_tag_content(model_output, "code"),
        "thinking_found": has_valid_tags(model_output, "thinking"),
        "code_found": has_valid_tags(model_output, "code")
    }
```

### Code Execution Safety
- **Restricted environment**: Disable dangerous modules (os, sys, subprocess, etc.)
- **Timeout limits**: Prevent infinite loops
- **Memory limits**: Prevent excessive memory usage
- **Input validation**: Sanitize inputs before execution

### Evaluation Metrics
- **Syntax errors**: Count of programs that don't compile
- **Runtime errors**: Count of programs that crash during execution
- **Test case pass rate**: Percentage of test cases passed
- **Tag compliance**: Percentage of responses with proper tags
- **Execution time**: Time taken for each program

## File Structure
```
src/
├── evaluation/
│   ├── __init__.py
│   ├── prompt_generator.py      # Generate standardized prompts
│   ├── answer_extractor.py      # Extract and validate answers
│   ├── code_executor.py         # Safe code execution
│   ├── model_evaluator.py       # Main evaluation pipeline
│   └── test_prompts.py          # Test prompt generation
├── openrouter/
│   ├── api_client.py            # Already exists
│   └── batch_processor.py       # Async batch processing
└── utils/
    └── dataset_loader.py        # Already exists
```

## Implementation Phases

### Phase 1: Core Components
1. **Prompt Generator** with standardized template
2. **Answer Extractor** with tag validation
3. **Basic Code Executor** with safety restrictions
4. **Test script** to verify components

### Phase 2: Integration
1. **Model Evaluator** combining all components
2. **Async Batch Processor** for OpenRouter
3. **Parallel model evaluation**
4. **Result collection and storage**

### Phase 3: Optimization
1. **Enhanced safety** (if needed)
2. **Performance optimization**
3. **Better error handling**
4. **Comprehensive reporting**

## Success Criteria
- [ ] Generate consistent prompts across all models
- [ ] Extract answers reliably with proper tag validation
- [ ] Execute code safely without system compromise
- [ ] Process multiple models in parallel
- [ ] Provide clear evaluation metrics
- [ ] Handle errors gracefully at all stages

## Next Steps
1. Implement prompt generator with test cases
2. Build answer extraction system
3. Create basic code executor
4. Test with a few sample problems
5. Integrate with OpenRouter for async processing 