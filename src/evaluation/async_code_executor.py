"""
Async Code Executor for parallel test case execution.
"""

import ast
import asyncio
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import concurrent.futures


class AsyncCodeExecutor:
    """Asynchronously executes Python code with parallel test case execution."""
    
    def __init__(self, timeout: int = 5, max_memory_mb: int = 100, max_workers: int = 10):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_workers = max_workers
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 
            'pathlib', 'glob', 'fnmatch', 'pickle', 'marshal',
            'ctypes', 'mmap', 'signal', 'threading', 'multiprocessing'
        }
        # Create a thread pool for CPU-bound code execution
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
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
    
    def _execute_single_test(self, code: str, test_input: str, expected_output: str, 
                           test_idx: int) -> Dict:
        """Execute a single test case (runs in thread pool)."""
        try:
            # Capture stdout for this specific test
            stdout_capture = StringIO()
            
            # Create test code with input redirection
            test_code = f"""
# Redirect stdin to provide input
sys.stdin = StringIO({repr(test_input)})

# Execute the original code
{code}
"""
            
            # Create restricted globals
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
                'sys': sys,
                'StringIO': StringIO
            }
            
            start_time = time.time()
            
            with redirect_stdout(stdout_capture):
                exec(test_code, test_globals)
            
            execution_time = time.time() - start_time
            
            actual_output = stdout_capture.getvalue().strip()
            expected_output = str(expected_output).strip()
            
            passed = actual_output == expected_output
            
            return {
                "test_case": test_idx + 1,
                "input": test_input,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "passed": passed,
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "test_case": test_idx + 1,
                "input": test_input,
                "expected_output": expected_output,
                "actual_output": "",
                "passed": False,
                "error": str(e),
                "execution_time": 0.0
            }
    
    async def run_test_cases_async(self, code: str, test_cases: List[Dict]) -> Dict:
        """Run code against test cases asynchronously in parallel."""
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
                "total_execution_time": 0.0
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
                "total_execution_time": 0.0
            }
        
        # Run test cases in parallel
        start_time = time.time()
        loop = asyncio.get_event_loop()
        
        # Create tasks for parallel execution
        tasks = []
        for i, test_case in enumerate(test_cases):
            test_input = test_case.get('input', '')
            expected_output = test_case.get('output', '')
            
            # Run in thread pool to avoid blocking the event loop
            task = loop.run_in_executor(
                self.executor,
                self._execute_single_test,
                code, test_input, expected_output, i
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        test_results = await asyncio.gather(*tasks)
        
        total_execution_time = time.time() - start_time
        
        # Count passed tests
        passed_count = sum(1 for result in test_results if result['passed'])
        
        return {
            "execution_success": True,
            "test_results": test_results,
            "passed_count": passed_count,
            "failed_count": len(test_cases) - passed_count,
            "total_count": len(test_cases),
            "pass_rate": passed_count / len(test_cases) if test_cases else 0.0,
            "total_execution_time": total_execution_time
        }
    
    async def run_multiple_problems_async(self, problems: List[Dict]) -> List[Dict]:
        """Run multiple problems with their test cases in parallel."""
        tasks = []
        
        for problem in problems:
            code = problem.get('code', '')
            test_cases = problem.get('test_cases', [])
            
            task = self.run_test_cases_async(code, test_cases)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine results with problem info
        output = []
        for problem, result in zip(problems, results):
            output.append({
                'problem_id': problem.get('problem_id'),
                'execution_result': result
            })
        
        return output
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


async def test_async_code_executor():
    """Test the async code executor."""
    executor = AsyncCodeExecutor(timeout=5, max_memory_mb=100, max_workers=4)
    
    # Test with multiple problems
    problems = [
        {
            'problem_id': 'test_1',
            'code': """
def solve():
    n = int(input())
    for i in range(n):
        x = int(input())
        print(x * 2)

solve()
""",
            'test_cases': [
                {"input": "2\n3\n5", "output": "6\n10"},
                {"input": "1\n7", "output": "14"},
                {"input": "3\n1\n2\n3", "output": "2\n4\n6"}
            ]
        },
        {
            'problem_id': 'test_2',
            'code': """
def solve():
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(int(input()))
    print(sum(nums))

solve()
""",
            'test_cases': [
                {"input": "3\n1\n2\n3", "output": "6"},
                {"input": "2\n10\n20", "output": "30"},
                {"input": "1\n5", "output": "5"}
            ]
        }
    ]
    
    print("Testing Async Code Executor with parallel execution...")
    print("=" * 80)
    
    start_time = time.time()
    results = await executor.run_multiple_problems_async(problems)
    total_time = time.time() - start_time
    
    for result in results:
        print(f"\nProblem: {result['problem_id']}")
        exec_result = result['execution_result']
        print(f"Success: {exec_result['execution_success']}")
        print(f"Passed: {exec_result['passed_count']}/{exec_result['total_count']}")
        print(f"Pass rate: {exec_result['pass_rate']:.2%}")
        print(f"Execution time: {exec_result['total_execution_time']:.4f}s")
    
    print(f"\nTotal time for all problems: {total_time:.4f}s")
    print("(Note: Problems and their test cases were executed in parallel)")


if __name__ == "__main__":
    asyncio.run(test_async_code_executor())