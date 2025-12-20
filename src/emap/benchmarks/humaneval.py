"""
HumanEval Benchmark Loader

Loads the HumanEval benchmark (Chen et al., 2021) for evaluating
code generation capabilities.

HumanEval contains 164 hand-written Python programming problems
with function signatures, docstrings, and test cases.

Reference: https://github.com/openai/human-eval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from emap.evolution.fitness import Task


# HumanEval problems are structured as:
# {
#     "task_id": "HumanEval/0",
#     "prompt": "def function_name(...) -> ...:\n    \"\"\"docstring\"\"\"\n",
#     "entry_point": "function_name",
#     "canonical_solution": "    return ...",
#     "test": "assert function_name(...) == ..."
# }


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem."""
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str
    
    def to_task(self) -> Task:
        """Convert to Task format for evolution."""
        return Task(
            id=self.task_id,
            prompt=self.prompt,
            entry_point=self.entry_point,
            test_code=self.test,
            canonical_solution=self.canonical_solution,
        )


def load_humaneval(path: Optional[Path] = None) -> List[Task]:
    """
    Load HumanEval benchmark.
    
    Args:
        path: Path to HumanEval JSONL file. If None, uses bundled data
              or downloads from GitHub.
    
    Returns:
        List of Task objects
    """
    if path is None:
        # Try to find bundled data
        default_paths = [
            Path(__file__).parent / "data" / "humaneval.jsonl",
            Path("data/humaneval.jsonl"),
            Path.home() / ".cache" / "emap" / "humaneval.jsonl",
        ]
        
        for p in default_paths:
            if p.exists():
                path = p
                break
        
        if path is None:
            # Return placeholder tasks for development
            return _create_placeholder_tasks()
    
    tasks = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                problem = HumanEvalProblem(
                    task_id=data["task_id"],
                    prompt=data["prompt"],
                    entry_point=data["entry_point"],
                    canonical_solution=data.get("canonical_solution", ""),
                    test=data.get("test", ""),
                )
                tasks.append(problem.to_task())
    
    return tasks


def _create_placeholder_tasks() -> List[Task]:
    """
    Create placeholder tasks for development/testing.
    
    These are simple problems that don't require external data.
    """
    placeholders = [
        {
            "id": "placeholder/0",
            "prompt": '''def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
''',
            "entry_point": "add",
            "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
            "solution": "    return a + b",
        },
        {
            "id": "placeholder/1",
            "prompt": '''def factorial(n: int) -> int:
    """Return the factorial of n."""
''',
            "entry_point": "factorial",
            "test": "assert factorial(0) == 1\nassert factorial(5) == 120",
            "solution": "    if n <= 1: return 1\n    return n * factorial(n-1)",
        },
        {
            "id": "placeholder/2",
            "prompt": '''def is_palindrome(s: str) -> bool:
    """Check if string s is a palindrome."""
''',
            "entry_point": "is_palindrome",
            "test": "assert is_palindrome('radar') == True\nassert is_palindrome('hello') == False",
            "solution": "    return s == s[::-1]",
        },
        {
            "id": "placeholder/3",
            "prompt": '''def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
''',
            "entry_point": "fibonacci",
            "test": "assert fibonacci(0) == 0\nassert fibonacci(10) == 55",
            "solution": "    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n-1): a, b = b, a+b\n    return b",
        },
        {
            "id": "placeholder/4",
            "prompt": '''def reverse_list(lst: list) -> list:
    """Return a reversed copy of the list."""
''',
            "entry_point": "reverse_list",
            "test": "assert reverse_list([1,2,3]) == [3,2,1]\nassert reverse_list([]) == []",
            "solution": "    return lst[::-1]",
        },
    ]
    
    return [
        Task(
            id=p["id"],
            prompt=p["prompt"],
            entry_point=p["entry_point"],
            test_code=p["test"],
            canonical_solution=p["solution"],
        )
        for p in placeholders
    ]


def download_humaneval(output_path: Path) -> None:
    """
    Download HumanEval dataset from GitHub.
    
    Args:
        output_path: Where to save the JSONL file
    """
    import urllib.request
    
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import gzip
    import shutil
    
    gz_path = output_path.with_suffix(".jsonl.gz")
    
    # Download
    urllib.request.urlretrieve(url, gz_path)
    
    # Decompress
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up
    gz_path.unlink()
