"""
Code Executor for Model Evaluation

Safely executes Python code and evaluates it against test cases.
"""

import ast
import sys
import time
import signal
from typing import Dict, List, Any, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from .test_case_utils import convert_input_format, compare_outputs


class CodeExecutor:
    """Safely executes Python code with restrictions and timeout."""
    
    def __init__(self, timeout: int, max_memory_mb: int):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 
            'pathlib', 'glob', 'fnmatch', 'pickle', 'marshal',
            'ctypes', 'mmap', 'signal', 'threading', 'multiprocessing'
        }
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax without executing.
        
        Args:
            code: Python code string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Parse error: {e}"
    
    def check_dangerous_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Check for dangerous module imports.
        
        Args:
            code: Python code string
        
        Returns:
            Tuple of (is_safe, list_of_dangerous_imports)
        """
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
    
    def execute_code(self, code: str, inputs: Optional[List[str]] = None) -> Dict:
        """Execute Python code safely.
        
        Args:
            code: Python code string
            inputs: List of input strings to provide to the program
        
        Returns:
            Dictionary with execution results
        """
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0.0,
            "syntax_valid": False,
            "safe_to_execute": False,
            "dangerous_imports": []
        }
        
        # Validate syntax
        syntax_valid, syntax_error = self.validate_syntax(code)
        result["syntax_valid"] = syntax_valid
        if not syntax_valid:
            result["error"] = syntax_error
            return result
        
        # Check for dangerous imports
        is_safe, dangerous_imports = self.check_dangerous_imports(code)
        result["safe_to_execute"] = is_safe
        result["dangerous_imports"] = dangerous_imports
        
        if not is_safe:
            result["error"] = f"Dangerous imports detected: {dangerous_imports}"
            return result
        
        # Execute code with timeout
        try:
            start_time = time.time()
            
            # Capture stdout and stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            # Create restricted globals with safe builtins
            import builtins
            safe_builtins = {}
            
            # Allow most builtins except dangerous ones
            for name in dir(builtins):
                if name not in ['eval', 'exec', 'compile', 'open', '__import__']:
                    try:
                        safe_builtins[name] = getattr(builtins, name)
                    except AttributeError:
                        pass
            
            restricted_globals = {
                '__builtins__': safe_builtins,
                '__name__': '__main__'  # Add __name__ to support if __name__ == "__main__"
            }
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, restricted_globals)
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["output"] = stdout_capture.getvalue()
            result["error"] = stderr_capture.getvalue()
            result["success"] = True
            
        except Exception as e:
            result["error"] = f"Execution error: {e}"
            result["success"] = False
        
        return result
    
    def run_test_cases(self, code: str, test_cases: List[Dict]) -> Dict:
        """Run code against test cases.
        
        Args:
            code: Python code string
            test_cases: List of test case dictionaries with 'input' and 'output' keys
        
        Returns:
            Dictionary with test results
        """
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
                "pass_rate": 0.0
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
                "pass_rate": 0.0
            }
        
        # Run test cases
        test_results = []
        passed_count = 0
        
        for i, test_case in enumerate(test_cases):
            test_input = test_case.get('input', '')
            expected_output = test_case.get('output', '')
            
            # Convert input format if needed (e.g., '[1, 2]' -> '1\n2')
            converted_input = convert_input_format(test_input)
            
            # Create a new execution environment for each test
            try:
                # Capture stdout for this specific test
                stdout_capture = StringIO()
                
                # Create a function to run the test
                # Note: sys and StringIO are provided in globals, no need to import
                test_code = f"""
# Redirect stdin to provide input
sys.stdin = StringIO({repr(converted_input)})

# Execute the original code
{code}
"""
                
                # Create the same restricted globals as in execute_code
                import builtins
                safe_builtins = {}
                
                for name in dir(builtins):
                    if name not in ['eval', 'exec', 'compile', 'open', '__import__']:
                        try:
                            safe_builtins[name] = getattr(builtins, name)
                        except AttributeError:
                            pass
                
                test_globals = {
                    '__builtins__': safe_builtins,
                    '__name__': '__main__',
                    'sys': sys,  # Pre-import sys since __import__ is disabled
                    'StringIO': StringIO  # Pre-import StringIO
                }
                
                with redirect_stdout(stdout_capture):
                    exec(test_code, test_globals)
                
                actual_output = stdout_capture.getvalue().strip()
                expected_output = str(expected_output).strip()
                
                # Use smart comparison that handles different output formats
                passed = compare_outputs(expected_output, actual_output)
                
                test_result = {
                    "test_case": i + 1,
                    "input": test_input,
                    "converted_input": converted_input,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "passed": passed
                }
                
                test_results.append(test_result)
                if passed:
                    passed_count += 1
                    
            except Exception as e:
                test_result = {
                    "test_case": i + 1,
                    "input": test_input,
                    "converted_input": converted_input,
                    "expected_output": expected_output,
                    "actual_output": "",
                    "passed": False,
                    "error": str(e)
                }
                test_results.append(test_result)
        
        return {
            "execution_success": True,
            "test_results": test_results,
            "passed_count": passed_count,
            "failed_count": len(test_cases) - passed_count,
            "total_count": len(test_cases),
            "pass_rate": passed_count / len(test_cases) if test_cases else 0.0
        }


def test_code_executor():
    """Test the code executor with various code samples."""
    executor = CodeExecutor(timeout=5, max_memory_mb=100)
    
    # Test cases
    test_cases = [
        {
            "name": "Valid simple code",
            "code": "print('Hello, World!')",
            "expected_success": True
        },
        {
            "name": "Syntax error",
            "code": "print('Hello, World!'",
            "expected_success": False
        },
        {
            "name": "Dangerous import",
            "code": "import os\nprint('Dangerous!')",
            "expected_success": False
        },
        {
            "name": "Valid function",
            "code": """
def add(a, b):
    return a + b

print(add(3, 5))
""",
            "expected_success": True
        }
    ]
    
    print("Testing Code Executor")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print("-" * 40)
        
        result = executor.execute_code(test_case['code'])
        
        print(f"Success: {result['success']}")
        print(f"Syntax valid: {result['syntax_valid']}")
        print(f"Safe to execute: {result['safe_to_execute']}")
        print(f"Output: {repr(result['output'])}")
        print(f"Error: {repr(result['error'])}")
        print(f"Execution time: {result['execution_time']:.4f}s")
        
        if result['dangerous_imports']:
            print(f"Dangerous imports: {result['dangerous_imports']}")
    
    # Test with actual test cases
    print(f"\nTesting with APPS-style test cases:")
    print("-" * 40)
    
    code = """
def solve():
    n = int(input())
    for i in range(n):
        x = int(input())
        print(x * 2)
"""
    
    test_cases_data = [
        {"input": "2\n3\n5", "output": "6\n10"},
        {"input": "1\n7", "output": "14"}
    ]
    
    test_result = executor.run_test_cases(code, test_cases_data)
    
    print(f"Execution success: {test_result['execution_success']}")
    print(f"Passed: {test_result['passed_count']}/{test_result['total_count']}")
    print(f"Pass rate: {test_result['pass_rate']:.2%}")
    
    for result in test_result['test_results']:
        print(f"  Test {result['test_case']}: {'PASS' if result['passed'] else 'FAIL'}")
        if not result['passed']:
            print(f"    Expected: {repr(result['expected_output'])}")
            print(f"    Got: {repr(result['actual_output'])}")


if __name__ == "__main__":
    test_code_executor() 