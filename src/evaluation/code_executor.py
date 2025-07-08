"""
Code Executor with Timeout and Parallelization

This module provides a robust code executor that handles timeouts and 
runs test cases in parallel for better performance.
"""

import ast
import sys
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from io import StringIO
from dataclasses import dataclass
import multiprocessing
from functools import partial

try:
    from .test_case_utils import convert_input_format, compare_outputs
except ImportError:
    # For standalone testing
    from test_case_utils import convert_input_format, compare_outputs


def _execute_single_test(code: str, test_case: Dict, test_idx: int, timeout: int) -> Dict:
    """Execute a single test case in an isolated process.
    
    This function runs in a separate process and handles all the execution logic.
    """
    import sys
    import time
    from io import StringIO
    
    test_input = test_case.get('input', '')
    expected_output = test_case.get('output', '')
    
    # Convert input format if needed
    try:
        # Import here to avoid pickling issues
        from test_case_utils import convert_input_format, compare_outputs
    except ImportError:
        # Fallback for the function if not available
        def convert_input_format(x):
            return x
        def compare_outputs(a, b):
            return a == b
    
    converted_input = convert_input_format(test_input)
    
    result = {
        "test_case": test_idx + 1,
        "input": test_input,
        "converted_input": converted_input,
        "expected_output": str(expected_output),
        "actual_output": "",
        "passed": False,
        "execution_time": 0.0,
        "error": "",
        "timeout": False
    }
    
    start_time = time.time()
    
    try:
        # Prepare execution environment
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Create restricted globals with safe builtins
        import builtins
        safe_builtins = {}
        
        for name in dir(builtins):
            if name not in ['eval', 'exec', 'compile', 'open', '__import__']:
                try:
                    safe_builtins[name] = getattr(builtins, name)
                except AttributeError:
                    pass
        
        # Prepare globals with necessary imports
        exec_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            'sys': sys,
            'StringIO': StringIO
        }
        
        # Create test execution code
        test_code = f"""
# Execute the original code
{code}
"""
        
        # Redirect stdout, stderr and stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdin = sys.stdin
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            sys.stdin = StringIO(converted_input)
            
            # Execute the code
            exec(code, exec_globals)
            
            # Get output
            actual_output = stdout_capture.getvalue().strip()
            expected_output = str(expected_output).strip()
            
            # Compare outputs
            passed = compare_outputs(expected_output, actual_output)
            
            result["actual_output"] = actual_output
            result["passed"] = passed
            result["error"] = stderr_capture.getvalue()
            
        finally:
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin = old_stdin
            
    except Exception as e:
        result["error"] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
        result["passed"] = False
    
    result["execution_time"] = time.time() - start_time
    return result


class CodeExecutor:
    """Code executor with timeout and parallel test case execution."""
    
    def __init__(self, timeout: int = 10, max_memory_mb: int = 100, 
                 test_case_workers: int = 10, problem_workers: int = 1):
        """Initialize the code executor.
        
        Args:
            timeout: Timeout in seconds for each test case execution
            max_memory_mb: Maximum memory limit (not enforced in this version)
            test_case_workers: Number of parallel workers for test cases
            problem_workers: Number of parallel workers for problems (future use)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.test_case_workers = test_case_workers
        self.problem_workers = problem_workers
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 
            'pathlib', 'glob', 'fnmatch', 'pickle', 'marshal',
            'ctypes', 'mmap', 'signal', 'threading', 'multiprocessing'
        }
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax without executing."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Parse error: {e}"
    
    def check_dangerous_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Check for dangerous module imports."""
        try:
            tree = ast.parse(code)
            dangerous_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_modules:
                            dangerous_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.dangerous_modules:
                        dangerous_imports.append(node.module)
            
            return len(dangerous_imports) == 0, dangerous_imports
        except Exception:
            return False, ["parse_error"]
    
    def run_test_cases(self, code: str, test_cases: List[Dict]) -> Dict:
        """Run code against test cases in parallel with timeout."""
        # First, validate syntax and check for dangerous imports
        syntax_valid, syntax_error = self.validate_syntax(code)
        if not syntax_valid:
            return {
                "execution_success": False,
                "execution_error": syntax_error,
                "test_results": [],
                "passed_count": 0,
                "failed_count": 0,
                "total_count": len(test_cases),
                "pass_rate": 0.0,
                "total_execution_time": 0.0,
                "timeout_count": 0
            }
        
        is_safe, dangerous_imports = self.check_dangerous_imports(code)
        if not is_safe:
            return {
                "execution_success": False,
                "execution_error": f"Dangerous imports detected: {dangerous_imports}",
                "test_results": [],
                "passed_count": 0,
                "failed_count": 0,
                "total_count": len(test_cases),
                "pass_rate": 0.0,
                "total_execution_time": 0.0,
                "timeout_count": 0
            }
        
        # Run test cases in parallel
        start_time = time.time()
        
        # Determine optimal number of workers
        n_workers = min(self.test_case_workers, len(test_cases), multiprocessing.cpu_count())
        
        test_results = []
        timeout_count = 0
        passed_count = 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all test cases
            futures = {}
            for i, test_case in enumerate(test_cases):
                future = executor.submit(_execute_single_test, code, test_case, i, self.timeout)
                futures[future] = i
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=self.timeout * len(test_cases) + 10):
                try:
                    result = future.result(timeout=self.timeout)
                    test_results.append(result)
                    
                    if result.get("timeout", False):
                        timeout_count += 1
                    elif result.get("passed", False):
                        passed_count += 1
                        
                except TimeoutError:
                    # This test case timed out
                    idx = futures[future]
                    test_results.append({
                        "test_case": idx + 1,
                        "input": test_cases[idx].get('input', ''),
                        "expected_output": str(test_cases[idx].get('output', '')),
                        "actual_output": "",
                        "passed": False,
                        "execution_time": self.timeout,
                        "error": f"Execution timed out after {self.timeout} seconds",
                        "timeout": True
                    })
                    timeout_count += 1
                    # Cancel the future to free resources
                    future.cancel()
                    
                except Exception as e:
                    idx = futures[future]
                    test_results.append({
                        "test_case": idx + 1,
                        "passed": False,
                        "error": f"Execution failed: {str(e)}",
                        "timeout": False
                    })
        
        # Sort results by test case number
        test_results.sort(key=lambda x: x.get("test_case", 0))
        
        total_execution_time = time.time() - start_time
        
        return {
            "execution_success": True,
            "test_results": test_results,
            "passed_count": passed_count,
            "failed_count": len(test_cases) - passed_count,
            "total_count": len(test_cases),
            "pass_rate": passed_count / len(test_cases) if test_cases else 0.0,
            "total_execution_time": total_execution_time,
            "timeout_count": timeout_count
        }


def test_code_executor():
    """Test the code executor."""
    print("Testing Code Executor")
    print("=" * 80)
    
    # Test cases with various scenarios
    test_scenarios = [
        {
            "name": "Simple working code",
            "code": """
n = int(input())
for i in range(n):
    x = int(input())
    print(x * 2)
""",
            "test_cases": [
                {"input": "2\n3\n5", "output": "6\n10"},
                {"input": "1\n7", "output": "14"}
            ]
        },
        {
            "name": "Debug test",
            "code": """print("Hello World")""",
            "test_cases": [
                {"input": "", "output": "Hello World"}
            ]
        },
        {
            "name": "Code with infinite loop (should timeout)",
            "code": """
while True:
    pass
""",
            "test_cases": [
                {"input": "", "output": ""}
            ]
        },
        {
            "name": "Multiple test cases with mixed results",
            "code": """
n = int(input())
if n == 1:
    print("one")
elif n == 2:
    while True:  # This will timeout
        pass
else:
    print(n * n)
""",
            "test_cases": [
                {"input": "1", "output": "one"},
                {"input": "2", "output": "two"},  # Will timeout
                {"input": "3", "output": "9"},
                {"input": "4", "output": "16"}
            ]
        }
    ]
    
    executor = CodeExecutor(timeout=2, test_case_workers=4)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)
        
        start = time.time()
        result = executor.run_test_cases(scenario['code'], scenario['test_cases'])
        elapsed = time.time() - start
        
        print(f"Execution success: {result['execution_success']}")
        print(f"Passed: {result['passed_count']}/{result['total_count']}")
        print(f"Timeouts: {result.get('timeout_count', 0)}")
        print(f"Total execution time: {result.get('total_execution_time', 0):.2f}s")
        print(f"Wall clock time: {elapsed:.2f}s")
        
        if result.get('execution_error'):
            print(f"Error: {result['execution_error']}")
        
        for test_result in result.get('test_results', []):
            status = "TIMEOUT" if test_result.get('timeout') else ("PASS" if test_result.get('passed') else "FAIL")
            print(f"  Test {test_result.get('test_case', '?')}: {status}")
            if not test_result.get('passed') and not test_result.get('timeout'):
                print(f"    Expected: {test_result.get('expected_output', 'N/A')}")
                print(f"    Got: {test_result.get('actual_output', 'N/A')}")


if __name__ == "__main__":
    # Change to the correct directory for testing
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_code_executor()