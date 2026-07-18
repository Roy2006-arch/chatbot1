"""
code_executor.py
================
Safe code execution sandbox for Python code.
Uses restricted exec with timeouts and limited builtins.
"""

import ast
import io
import logging
import sys
import signal
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("chatbot.code_executor")


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str = ""
    execution_time_ms: float = 0.0
    return_value: Any = None
    has_return_value: bool = False


# Dangerous modules/builtins that should be blocked
BLOCKED_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'socket',
    'http', 'urllib', 'requests', 'aiohttp', 'ftplib', 'smtplib',
    'signal', 'threading', 'multiprocessing', 'ctypes',
    'importlib', 'pkgutil', 'code', 'codeop',
    'pdb', 'profile', 'cProfile', 'timeit',
    'pickle', 'shelve', 'sqlite3',
}

BLOCKED_BUILTINS = {
    'exec', 'eval', 'compile', '__import__', 'breakpoint',
    'exit', 'quit', 'open',  # open is blocked for safety
}

# Safe builtins to allow
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
    'callable', 'chr', 'classmethod', 'complex', 'delattr', 'dict',
    'dir', 'divmod', 'enumerate', 'filter', 'float', 'format',
    'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'isinstance', 'issubclass', 'iter',
    'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min',
    'next', 'object', 'oct', 'ord', 'pow', 'print', 'property',
    'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
    'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type',
    'vars', 'zip',
    # Allow these for math/data work
    'int', 'float', 'complex',
}

# Safe modules to allow
SAFE_MODULES = {
    'math', 'random', 'datetime', 'collections', 'itertools',
    'functools', 'operator', 'string', 'textwrap', 're',
    'json', 'csv', 'io', 'copy', 'pprint',
    'statistics', 'decimal', 'fractions',
    'array', 'heapq', 'bisect', 'queue',
    'enum', 'dataclasses', 'typing',
    'unittest', 'doctest',
}

# Maximum execution time in seconds
MAX_EXECUTION_TIME = 5

# Maximum output size in characters
MAX_OUTPUT_SIZE = 50000


class RestrictedImport:
    """Custom import hook that blocks dangerous modules."""
    def find_module(self, name, path=None):
        top_level = name.split('.')[0]
        if top_level in BLOCKED_MODULES:
            return self
        return None

    def load_module(self, name):
        raise ImportError(f"Import of '{name}' is restricted for security reasons.")


@contextmanager
def time_limit(seconds: int):
    """Context manager that raises TimeoutError after given seconds."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {seconds} seconds")

    # Windows doesn't support SIGALRM, so we use a different approach
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
    else:
        # On Windows, we rely on the timeout being checked in the loop
        # For simple scripts, this works well enough
        import time
        start = time.time()
        yield
        if time.time() - start > seconds:
            raise TimeoutError(f"Code execution timed out after {seconds} seconds")


def _create_restricted_globals() -> Dict[str, Any]:
    """Create a restricted global namespace for code execution."""
    restricted_globals = {
        '__builtins__': {},
    }

    # Add safe builtins
    for name in SAFE_BUILTINS:
        if name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name):
            if isinstance(__builtins__, dict):
                restricted_globals['__builtins__'][name] = __builtins__[name]
            else:
                restricted_globals['__builtins__'][name] = getattr(__builtins__, name)

    # Add safe modules
    import importlib
    for module_name in SAFE_MODULES:
        try:
            module = importlib.import_module(module_name)
            restricted_globals[module_name] = module
        except ImportError:
            pass

    return restricted_globals


def _validate_code(code: str) -> tuple[bool, str]:
    """
    Validate code before execution.
    Returns (is_valid, error_message).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check for dangerous imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split('.')[0]
                if top_level in BLOCKED_MODULES:
                    return False, f"Import of '{alias.name}' is not allowed for security reasons."

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level = node.module.split('.')[0]
                if top_level in BLOCKED_MODULES:
                    return False, f"Import from '{node.module}' is not allowed for security reasons."

        # Check for file operations
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('open', 'exec', 'eval', 'compile', '__import__'):
                    return False, f"Call to '{node.func.id}()' is not allowed for security reasons."

    return True, ""


def execute_code(
    code: str,
    timeout: int = MAX_EXECUTION_TIME,
    capture_output: bool = True,
) -> ExecutionResult:
    """
    Execute Python code in a restricted sandbox.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        ExecutionResult with output, errors, and timing info
    """
    import time

    # Validate code first
    is_valid, error_msg = _validate_code(code)
    if not is_valid:
        return ExecutionResult(
            success=False,
            output="",
            error=error_msg,
        )

    # Create restricted globals
    restricted_globals = _create_restricted_globals()
    restricted_locals = {}

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    t0 = time.time()

    try:
        if capture_output:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

        # Compile and execute
        compiled = compile(code, "<sandbox>", "exec")

        # Use time_limit for timeout (Windows compatible)
        with time_limit(timeout):
            exec(compiled, restricted_globals, restricted_locals)

        execution_time = round((time.time() - t0) * 1000, 2)

        # Get output
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()

        # Truncate if too long
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
        if len(error_output) > MAX_OUTPUT_SIZE:
            error_output = error_output[:MAX_OUTPUT_SIZE] + "\n... (error truncated)"

        # Check for return value (last expression)
        return_value = None
        has_return = False

        # Try to get the result of the last expression
        try:
            lines = code.strip().split('\n')
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith(('import', 'from', 'def', 'class', 'if', 'for', 'while', 'with', 'try', 'except', 'finally', 'raise', 'return', 'yield', 'print')):
                try:
                    # Try to evaluate as expression
                    return_value = eval(last_line, restricted_globals, restricted_locals)
                    has_return = True
                except:
                    pass
        except:
            pass

        return ExecutionResult(
            success=True,
            output=output,
            error=error_output,
            execution_time_ms=execution_time,
            return_value=return_value,
            has_return_value=has_return,
        )

    except TimeoutError as e:
        execution_time = round((time.time() - t0) * 1000, 2)
        return ExecutionResult(
            success=False,
            output=stdout_capture.getvalue(),
            error=f"Execution timed out: {e}",
            execution_time_ms=execution_time,
        )

    except Exception as e:
        execution_time = round((time.time() - t0) * 1000, 2)
        error_traceback = traceback.format_exc()
        # Remove the first line (the exec line) from traceback
        error_lines = error_traceback.split('\n')
        if len(error_lines) > 1:
            error_traceback = '\n'.join(error_lines[1:])
        return ExecutionResult(
            success=False,
            output=stdout_capture.getvalue(),
            error=f"{type(e).__name__}: {e}\n{error_traceback}",
            execution_time_ms=execution_time,
        )

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def format_result_for_response(result: ExecutionResult) -> str:
    """Format execution result as a user-friendly response."""
    lines = []

    if result.success:
        lines.append("**Code executed successfully**")
        if result.execution_time_ms > 0:
            lines.append(f"*(Completed in {result.execution_time_ms:.1f}ms)*")
        lines.append("")

        if result.output:
            lines.append("**Output:**")
            lines.append(f"```\n{result.output.strip()}\n```")

        if result.has_return_value and result.return_value is not None:
            lines.append(f"**Return value:** `{repr(result.return_value)}`")

        if not result.output and not result.has_return_value:
            lines.append("*(No output)*")

    else:
        lines.append("**Code execution failed**")
        if result.execution_time_ms > 0:
            lines.append(f"*(Failed after {result.execution_time_ms:.1f}ms)*")
        lines.append("")
        lines.append("**Error:**")
        lines.append(f"```\n{result.error.strip()}\n```")

    return "\n".join(lines)


def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from markdown text."""
    import re
    pattern = r'```(?:python|py)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
