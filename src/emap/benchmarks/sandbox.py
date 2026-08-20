"""Best-effort subprocess evaluator for agent-generated Python code.

This module is an evaluation harness, not a security boundary. It combines
static checks, restricted builtins/imports, a subprocess, a timeout, and
best-effort Unix resource limits. Python object-model escapes or platform
differences may bypass those controls, so callers must not use it for hostile
code on a machine that holds secrets or sensitive network/filesystem access.
"""

from __future__ import annotations

import ast
import math
import multiprocessing
import signal
import traceback
from dataclasses import dataclass
from io import StringIO
from typing import Any, Optional
import contextlib


@dataclass
class ExecutionOutcome:
    """Result of executing code in the evaluation worker."""
    success: bool
    output: str
    return_value: Any
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    timed_out: bool = False
    
    @property
    def passed(self) -> bool:
        """Alias for success for test compatibility."""
        return self.success


# Dangerous imports that should be blocked
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "urllib", "requests", "http",
    "pickle", "marshal", "shelve",
    "importlib", "__builtin__", "builtins",
    "ctypes", "cffi", "multiprocessing",
    "signal", "pty", "tty", "termios",
    "resource", "rlimit",
    "code", "codeop", "compile",
}

# Safe imports that are allowed
ALLOWED_IMPORTS = {
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "itertools", "functools", "operator",
    "string", "re", "textwrap",
    "collections", "heapq", "bisect", "array",
    "copy", "types", "typing",
    "dataclasses", "enum",
    "datetime", "calendar", "time",
    "json",
    "hashlib", "hmac",
    "base64", "binascii",
    "struct",
}


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for dangerous imports."""
    
    def __init__(self):
        self.violations: list[str] = []
    
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            if module in BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import from: {node.module}")
        self.generic_visit(node)


def check_code_safety(code: str) -> tuple[bool, list[str]]:
    """
    Check code against the evaluator's denylist.

    A passing result means only that these checks found no configured
    violation. It does not establish that arbitrary Python is safe to run.
    
    Returns:
        Tuple of (is_safe, list_of_violations)
    """
    violations = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]
    
    # Check imports
    import_checker = ImportChecker()
    import_checker.visit(tree)
    violations.extend(import_checker.violations)
    
    # Check for dangerous patterns
    code_lower = code.lower()
    
    dangerous_patterns = [
        ("exec(", "Dynamic code execution with exec()"),
        ("eval(", "Dynamic code execution with eval()"),
        ("compile(", "Dynamic code compilation"),
        ("__import__", "Dynamic import"),
        ("open(", "File system access"),
        ("globals(", "Globals access"),
        ("locals(", "Locals access"),
        ("getattr(", "Dynamic attribute access"),
        ("setattr(", "Dynamic attribute modification"),
        ("delattr(", "Dynamic attribute deletion"),
    ]
    
    for pattern, message in dangerous_patterns:
        if pattern in code_lower:
            violations.append(message)
    
    return len(violations) == 0, violations


def _apply_worker_resource_limits(timeout: float) -> None:
    """Apply conservative Unix limits inside the disposable worker.

    ``resource`` is unavailable on some platforms and individual limits are
    not uniformly implemented. Unsupported limits are skipped; the parent
    process timeout remains the cross-platform backstop.
    """
    try:
        import resource
    except ImportError:
        return

    def set_limit(name: str, value: int) -> None:
        resource_id = getattr(resource, name, None)
        if resource_id is None:
            return
        try:
            _, current_hard = resource.getrlimit(resource_id)
            target = value
            if current_hard != resource.RLIM_INFINITY:
                target = min(target, current_hard)
            resource.setrlimit(resource_id, (target, target))
        except (OSError, ValueError):
            pass

    set_limit("RLIMIT_CPU", max(1, math.ceil(timeout) + 1))
    set_limit("RLIMIT_AS", 512 * 1024 * 1024)
    set_limit("RLIMIT_FSIZE", 1024 * 1024)
    set_limit("RLIMIT_NOFILE", 64)


def _execute_in_worker(
    code: str,
    test_code: str,
    timeout: float,
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Execute code in a disposable subprocess with best-effort restrictions.
    """
    import time as time_module
    start_time = time_module.time()

    _apply_worker_resource_limits(timeout)
    
    # Set up timeout via signal (Unix only)
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout) + 1)
    except Exception:
        pass  # Signals may not work in all environments
    
    # Add safe modules
    import math
    import itertools
    import functools
    import collections
    import re
    import json
    import copy
    import typing
    import builtins

    # Safe import function that only allows whitelisted modules
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed = {"typing", "math", "itertools", "functools", "collections",
                   "re", "json", "copy", "heapq", "bisect", "random", "string",
                   "decimal", "fractions", "statistics", "operator", "dataclasses"}
        base = name.split(".")[0]
        if base not in allowed:
            raise ImportError(f"Import of '{name}' is not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)

    # Create restricted globals
    safe_globals = {
        "__builtins__": {
            # Basic types
            "True": True,
            "False": False,
            "None": None,
            # Type constructors
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "frozenset": frozenset,
            "bytes": bytes,
            "bytearray": bytearray,
            # Functions
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "all": all,
            "any": any,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "type": type,
            "id": id,
            "hash": hash,
            "repr": repr,
            "print": print,
            "input": lambda *args: "",  # Disabled
            "iter": iter,
            "next": next,
            "slice": slice,
            "ord": ord,
            "chr": chr,
            "bin": bin,
            "hex": hex,
            "oct": oct,
            "format": format,
            "ascii": ascii,
            # Exceptions
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
            "ZeroDivisionError": ZeroDivisionError,
            "AssertionError": AssertionError,
            "ImportError": ImportError,
            "__import__": safe_import,
        }
    }

    safe_globals["math"] = math
    safe_globals["itertools"] = itertools
    safe_globals["functools"] = functools
    safe_globals["collections"] = collections
    safe_globals["re"] = re
    safe_globals["json"] = json
    safe_globals["copy"] = copy
    safe_globals["typing"] = typing
    safe_globals["List"] = typing.List
    safe_globals["Dict"] = typing.Dict
    safe_globals["Optional"] = typing.Optional
    safe_globals["Tuple"] = typing.Tuple
    safe_globals["Set"] = typing.Set
    
    # Capture output
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    result = {
        "success": False,
        "output": "",
        "return_value": None,
        "error_type": None,
        "error_message": None,
        "timed_out": False,
    }
    
    try:
        # Redirect stdout/stderr
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                # Execute the main code
                exec(code, safe_globals)
                
                # Execute test code
                if test_code:
                    exec(test_code, safe_globals)
        
        result["success"] = True
        result["output"] = stdout_capture.getvalue()
        
    except TimeoutError:
        result["timed_out"] = True
        result["error_type"] = "TimeoutError"
        result["error_message"] = f"Execution exceeded {timeout}s timeout"
        
    except AssertionError as e:
        result["error_type"] = "AssertionError"
        result["error_message"] = str(e) or "Assertion failed"
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["output"] = traceback.format_exc()
    
    finally:
        # Cancel alarm
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        except Exception:
            pass
    
    result["execution_time_ms"] = (time_module.time() - start_time) * 1000
    result_queue.put(result)


def execute_code(
    code: str,
    test_code: str = "",
    timeout: float = 5.0,
    check_safety: bool = True,
) -> ExecutionOutcome:
    """
    Execute code in a best-effort isolated evaluation worker.

    This API is intended for benchmark code on a disposable, unprivileged
    host. It is not a secure sandbox for adversarial Python.

    Args:
        code: Python code to execute
        test_code: Additional test code (assertions)
        timeout: Maximum execution time in seconds
        check_safety: Whether to run static safety checks

    Returns:
        ExecutionOutcome with results
    """
    # Safety check
    if check_safety:
        is_safe, violations = check_code_safety(code)
        if not is_safe:
            return ExecutionOutcome(
                success=False,
                output="",
                return_value=None,
                error_type="SecurityError",
                error_message=f"Code failed safety check: {'; '.join(violations)}",
            )

        if test_code:
            is_safe, violations = check_code_safety(test_code)
            if not is_safe:
                return ExecutionOutcome(
                    success=False,
                    output="",
                    return_value=None,
                    error_type="SecurityError",
                    error_message=f"Test code failed safety check: {'; '.join(violations)}",
                )

    # Try multiprocessing first (better isolation)
    try:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_in_worker,
            args=(code, test_code, timeout, result_queue),
        )

        process.start()
        process.join(timeout=timeout + 1)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()

            return ExecutionOutcome(
                success=False,
                output="",
                return_value=None,
                error_type="TimeoutError",
                error_message=f"Execution exceeded {timeout}s timeout",
                timed_out=True,
            )

        result = result_queue.get_nowait()

        return ExecutionOutcome(
            success=result["success"],
            output=result["output"],
            return_value=result.get("return_value"),
            error_type=result.get("error_type"),
            error_message=result.get("error_message"),
            execution_time_ms=result.get("execution_time_ms", 0),
            timed_out=result.get("timed_out", False),
        )

    except Exception as exc:
        # Fail closed: never execute generated code in the caller process.
        return ExecutionOutcome(
            success=False,
            output="",
            return_value=None,
            error_type="IsolationError",
            error_message=(
                "Subprocess isolation could not be established: "
                f"{type(exc).__name__}"
            ),
        )


def execute_with_entrypoint(
    code: str,
    entry_point: str,
    test_cases: list[tuple[tuple, Any]],
    timeout: float = 5.0,
) -> ExecutionOutcome:
    """
    Execute code and test against expected outputs.
    
    This is the standard format for HumanEval/MBPP.
    
    Args:
        code: Python code containing the function
        entry_point: Name of the function to test
        test_cases: List of ((args...), expected_output) tuples
        timeout: Execution timeout
        
    Returns:
        ExecutionOutcome
    """
    # Build test code
    test_lines = []
    for i, (args, expected) in enumerate(test_cases):
        args_str = ", ".join(repr(a) for a in args)
        test_lines.append(
            f"assert {entry_point}({args_str}) == {repr(expected)}, "
            f"'Test case {i+1} failed'"
        )
    
    test_code = "\n".join(test_lines)
    
    return execute_code(code, test_code, timeout)
