# Changelog

All notable changes to the Solution Space Hacking project will be documented in this file.

## [2025-01-06] - Major Pipeline Improvements

### Added
- **Optimized AsyncOpenRouterClient** with persistent HTTP sessions
  - 20-30% performance improvement from session reuse
  - Connection pooling: 2000 total, 1000 per host
  - Streaming support for real-time responses
  - Progress tracking with batch processing
  
- **Test Case Format Conversion** (`test_case_utils.py`)
  - Converts APPS list format inputs to stdin format
  - Normalizes outputs for flexible comparison
  - Handles currency symbol differences
  - Smart numeric comparison

- **Results Persistence Layer** (`results_persistence.py`)
  - Centralized file I/O operations
  - Consistent naming conventions
  - Automatic directory creation
  - Latest file detection utilities

- **Dependency Injection** in ModelEvaluator
  - Accept custom components for flexibility
  - Easier testing and customization
  - Better separation of concerns

### Changed
- **Rate Limiting**: Set to 500 requests/minute (was unlimited)
- **Max Workers**: Reduced from 1000 to 100 for stability
- **Timeout**: Reduced from 180s to 90s for faster failure detection
- **Max Tokens**: Increased from 4096 to 6000
- **Random Seed**: Fixed to 42 for consistent problem selection

### Fixed
- **Test Case Matching**: Fixed random seed inconsistency between generation and evaluation
- **Input Format Issues**: List literals now properly converted to stdin format
- **Output Comparison**: Currency symbols and decimal formatting no longer cause false negatives
- **Missing Builtins**: Added all safe builtins to execution environment
- **EOF Errors**: Fixed input reading issues in generated code

### Removed
- **Duplicate Code**: Removed redundant `evaluate_models()` method (186 lines)
- **Old AsyncOpenRouterClient**: Replaced with optimized version
- **Direct File I/O**: Moved to ResultsPersistence class

### Performance Impact
- API calls: 20-30% faster with persistent sessions
- Test pass rates: Increased from 0% to up to 100% for some models
- Error recovery: Faster with optimized retry delays
- Memory usage: More efficient with batched processing

## [2025-01-05] - Test Execution Improvements

### Added
- Prompt format with `main()` function pattern
- Clear instructions for using `<thinking>` and `<code>` tags
- Multiple examples in prompt generation

### Changed
- Increased parallel workers to 1000
- Updated visualization to show 6k token limit

### Fixed
- Code extraction for malformed outputs
- Test case execution with proper globals

## [2024-12-15] - Initial Pipeline Implementation

### Added
- Basic evaluation pipeline with generation and evaluation phases
- APPS dataset integration with cleaned parquet files
- OpenRouter API integration for multiple models
- Code execution sandbox with safety features
- Visualization plots for results

### Architecture
- Separation of generation and evaluation phases
- Async processing for concurrent API calls
- Modular component design