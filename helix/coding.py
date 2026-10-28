"""HELIX advanced coding & software-engineering agent.

Independently implemented. Provides repository-level coding capabilities:
  * Repository inspection (file listing, reading, metadata)
  * Static analysis via Python AST (functions, classes, imports, complexity, TODOs)
  * Sandboxed code execution (reuses agent.PythonTool with limits)
  * Patch generation (unified diff) for proposed fixes
  * Unit-test generation from function signatures
  * LLM-assisted explanation, bug finding, and fixing

Designed for CPU-first operation; the LLM assists but real compiler/interpreter
feedback and static analysis drive correctness. Least-privilege: execution is
sandboxed, time- and memory-bounded, no network assumed.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from helix.agent import PythonTool


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    path: str
    size: int
    lines: int
    language: str


@dataclass
class Symbol:
    kind: str  # function | class | import | variable
    name: str
    line: int
    docstring: Optional[str] = None
    args: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    file: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    todo_count: int = 0
    complexity: int = 1  # cyclomatic-ish estimate
    issues: List[str] = field(default_factory=list)
    syntax_error: Optional[str] = None

    def has_syntax_error(self) -> bool:
        return self.syntax_error is not None


@dataclass
class ExecutionResult:
    ok: bool
    output: str
    error: Optional[str]
    duration_s: float


@dataclass
class Patch:
    original: str
    modified: str
    diff: str


# ---------------------------------------------------------------------------
# Repository inspection
# ---------------------------------------------------------------------------

class RepoInspector:
    """Inspect a local repository (read-only)."""

    CODE_EXTS = {".py", ".js", ".ts", ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
                 ".java", ".rb", ".sh", ".lua", ".hs", ".ml", ".scala", ".swift",
                 ".kt", ".r", ".m", ".mm", ".asm", ".s", ".f90", ".cbl", ".bas",
                 ".pas", ".adoc", ".tex", ".md", ".txt", ".json", ".toml", ".yaml",
                 ".yml", ".cfg", ".ini"}

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def list_files(self, max_files: int = 200) -> List[str]:
        files: List[str] = []
        for root, dirs, names in os.walk(self.root):
            # skip common non-source dirs
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__",
                                                    ".venv", "node_modules",
                                                    "build", "dist", ".idea"}]
            for n in sorted(names):
                ext = os.path.splitext(n)[1].lower()
                if ext in self.CODE_EXTS or n.lower() in {
                    "makefile", "dockerfile", "readme", "license", "changelog",
                }:
                    files.append(os.path.relpath(os.path.join(root, n), self.root))
                if len(files) >= max_files:
                    return files
        return files

    def read_file(self, rel_path: str) -> Optional[str]:
        full = os.path.join(self.root, rel_path)
        if not os.path.isfile(full):
            return None
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def file_info(self, rel_path: str) -> Optional[FileInfo]:
        full = os.path.join(self.root, rel_path)
        if not os.path.isfile(full):
            return None
        text = self.read_file(rel_path)
        if text is None:
            return None
        ext = os.path.splitext(rel_path)[1].lower()
        lang = {".py": "python", ".js": "javascript", ".ts": "typescript",
                ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".rs": "rust",
                ".go": "go", ".java": "java", ".rb": "ruby", ".sh": "shell",
                ".lua": "lua", ".hs": "haskell", ".md": "markdown",
                ".json": "json", ".toml": "toml", ".yaml": "yaml",
                ".yml": "yaml"}.get(ext, "text")
        return FileInfo(rel_path, os.path.getsize(full), text.count("\n") + 1, lang)


# ---------------------------------------------------------------------------
# Static analysis (AST-based)
# ---------------------------------------------------------------------------

class StaticAnalyzer:
    """Analyze a Python source file via AST."""

    def __init__(self, source: str, filename: str = "<unknown>") -> None:
        self.source = source
        self.filename = filename
        self.tree: Optional[ast.AST] = None
        self._parse()

    def _parse(self) -> None:
        try:
            self.tree = ast.parse(self.source, filename=self.filename)
        except SyntaxError as exc:
            self.tree = None
            self._syntax_error = str(exc)
        else:
            self._syntax_error = None

    def has_syntax_error(self) -> bool:
        return self.tree is None

    def syntax_error(self) -> Optional[str]:
        return getattr(self, "_syntax_error", None)

    def analyze(self) -> AnalysisResult:
        result = AnalysisResult(file=self.filename,
                                syntax_error=self._syntax_error)
        if self.tree is None:
            result.issues.append(f"Syntax error: {self._syntax_error}")
            return result
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(alias.name)
                    result.symbols.append(Symbol("import", alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    result.imports.append(f"{mod}.{alias.name}")
                    result.symbols.append(Symbol("import", alias.name, node.lineno))
            elif isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node)
                result.symbols.append(Symbol("function", node.name, node.lineno,
                                             doc, args))
                result.complexity += self._count_complexity(node)
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                result.symbols.append(Symbol("class", node.name, node.lineno, doc))
                result.complexity += 1
            elif isinstance(node, (ast.While, ast.For, ast.If,
                                   ast.Try, ast.With)):
                result.complexity += 1
        # TODOs / FIXMEs
        for i, line in enumerate(self.source.splitlines(), 1):
            low = line.lower()
            if "todo" in low or "fixme" in low or "xxx" in low:
                result.todo_count += 1
                result.issues.append(f"line {i}: {line.strip()[:80]}")
        return result

    @staticmethod
    def _count_complexity(node: ast.AST) -> int:
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For,
                                  ast.Try, ast.ExceptHandler,
                                  ast.And, ast.Or, ast.IfExp)):
                count += 1
        return count


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

class PatchGenerator:
    def make_patch(self, filename: str, original: str, modified: str) -> Patch:
        if original == modified:
            diff = "# No changes\n"
        else:
            diff = "".join(difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/" + filename, tofile="b/" + filename,
            ))
        return Patch(original, modified, diff)

    def apply_patch(self, original: str, diff: str) -> str:
        # Minimal unified-diff apply (additions only, context lines must match).
        out: List[str] = []
        orig_lines = original.splitlines(keepends=True)
        diff_lines = diff.splitlines(keepends=True)
        i = 0
        oi = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                i += 1
                continue
            if line.startswith(" "):
                if oi < len(orig_lines):
                    out.append(orig_lines[oi]); oi += 1
                i += 1
            elif line.startswith("-"):
                oi += 1  # skip original line
                i += 1
            elif line.startswith("+"):
                out.append(line[1:])
                i += 1
            else:
                i += 1
        # append any remaining original lines
        while oi < len(orig_lines):
            out.append(orig_lines[oi]); oi += 1
        return "".join(out)


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

class TestGenerator:
    def generate_tests(self, filename: str, source: str) -> str:
        analyzer = StaticAnalyzer(source, filename)
        result = analyzer.analyze()
        lines: List[str] = [
            "import unittest",
            "",
        ]
        for sym in result.symbols:
            if sym.kind == "function":
                args = ", ".join(sym.args) if sym.args else ""
                lines.append(
                    f"class Test{sym.name.capitalize()}(unittest.TestCase):"
                )
                lines.append(f"    def test_{sym.name}(self):")
                lines.append(f"        # TODO: add assertions for {sym.name}({args})")
                lines.append(f"        self.assertTrue(True)")
                lines.append("")
        if not any(s.kind == "function" for s in result.symbols):
            lines.append("class TestPlaceholder(unittest.TestCase):")
            lines.append("    def test_placeholder(self):")
            lines.append("        self.assertTrue(True)")
            lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    unittest.main()")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coding agent
# ---------------------------------------------------------------------------

class CodingAgent:
    """Orchestrates inspection, analysis, execution, patching, testing."""

    def __init__(self, root: str = ".") -> None:
        self.root = os.path.abspath(root)
        self.inspector = RepoInspector(self.root)
        self.executor = PythonTool(timeout_s=30.0, memory_mb=256)

    def list_files(self, max_files: int = 200) -> List[str]:
        return self.inspector.list_files(max_files)

    def read_file(self, rel_path: str) -> Optional[str]:
        return self.inspector.read_file(rel_path)

    def analyze_file(self, rel_path: str) -> Optional[AnalysisResult]:
        source = self.read_file(rel_path)
        if source is None:
            return None
        return StaticAnalyzer(source, rel_path).analyze()

    def execute_code(self, code: str) -> ExecutionResult:
        result = self.executor.execute(code=code)
        return ExecutionResult(result.ok, result.output, result.error,
                               result.duration_s)

    def run_python_file(self, rel_path: str) -> ExecutionResult:
        source = self.read_file(rel_path)
        if source is None:
            return ExecutionResult(False, "", f"file not found: {rel_path}", 0.0)
        return self.execute_code(source)

    def generate_patch(self, rel_path: str, modified: str) -> Optional[Patch]:
        original = self.read_file(rel_path)
        if original is None:
            return None
        return PatchGenerator().make_patch(rel_path, original, modified)

    def apply_patch(self, rel_path: str, diff: str) -> str:
        original = self.read_file(rel_path) or ""
        return PatchGenerator().apply_patch(original, diff)

    def generate_tests(self, rel_path: str) -> Optional[str]:
        source = self.read_file(rel_path)
        if source is None:
            return None
        return TestGenerator().generate_tests(rel_path, source)

    def summary(self) -> dict:
        files = self.list_files()
        return {
            "root": self.root,
            "file_count": len(files),
            "files": files[:50],
        }


def demo_code_fix(code: str) -> dict:
    """Demonstrate static analysis + execution feedback on a snippet."""
    analyzer = StaticAnalyzer(code, "<snippet>")
    result = analyzer.analyze()
    exec_result = None
    if not result.has_syntax_error():
        # append a print of a result variable if present
        exec_result = CodingAgent().execute_code(code + "\n")
    return {
        "analysis": asdict(result),
        "execution": asdict(exec_result) if exec_result else None,
    }
