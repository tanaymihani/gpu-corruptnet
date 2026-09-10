"""Static source-code analysis (the posting's 'source code analysis' bullet).

Treats the project's own Python as data: parses each file's AST and reports lines of
code, cyclomatic complexity per function, docstring coverage, and the riskiest
(highest-complexity) functions — a lightweight 'commit-risk / config-lint' helper.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

# Nodes that each add one independent path (cyclomatic complexity).
_DECISION = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.comprehension,
)


def cyclomatic_complexity(node: ast.AST) -> int:
    complexity = 1
    for n in ast.walk(node):
        if isinstance(n, _DECISION):
            complexity += 1
        elif isinstance(n, ast.BoolOp):
            complexity += len(n.values) - 1  # each extra and/or operand branches
    return complexity


@dataclass
class FunctionReport:
    name: str
    lineno: int
    complexity: int
    documented: bool


@dataclass
class FileReport:
    path: str
    loc: int
    num_functions: int
    num_classes: int
    docstring_coverage: float
    functions: list[FunctionReport] = field(default_factory=list)

    @property
    def max_complexity(self) -> int:
        return max((f.complexity for f in self.functions), default=0)


def analyze_source(src: str, path: str = "<string>") -> FileReport:
    tree = ast.parse(src)
    functions: list[FunctionReport] = []
    num_classes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(
                FunctionReport(
                    name=node.name,
                    lineno=node.lineno,
                    complexity=cyclomatic_complexity(node),
                    documented=ast.get_docstring(node) is not None,
                )
            )
        elif isinstance(node, ast.ClassDef):
            num_classes += 1

    loc = sum(1 for line in src.splitlines() if line.strip() and not line.strip().startswith("#"))
    documented = sum(f.documented for f in functions)
    coverage = documented / len(functions) if functions else 1.0
    return FileReport(path, loc, len(functions), num_classes, coverage, functions)


def analyze_path(root: str = "src") -> list[FileReport]:
    reports = []
    for p in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        reports.append(analyze_source(p.read_text(), str(p)))
    return reports


def summarize(reports: list[FileReport], risk_threshold: int = 10) -> dict:
    all_funcs = [(r.path, f) for r in reports for f in r.functions]
    total_loc = sum(r.loc for r in reports)
    documented = sum(f.documented for _, f in all_funcs)
    riskiest = sorted(all_funcs, key=lambda pf: pf[1].complexity, reverse=True)[:5]
    return {
        "files": len(reports),
        "total_loc": total_loc,
        "functions": len(all_funcs),
        "docstring_coverage": documented / len(all_funcs) if all_funcs else 1.0,
        "avg_complexity": (
            sum(f.complexity for _, f in all_funcs) / len(all_funcs) if all_funcs else 0.0
        ),
        "high_risk_functions": [
            {"path": p, "name": f.name, "line": f.lineno, "complexity": f.complexity}
            for p, f in all_funcs
            if f.complexity >= risk_threshold
        ],
        "riskiest": [
            {"path": p, "name": f.name, "line": f.lineno, "complexity": f.complexity}
            for p, f in riskiest
        ],
    }
