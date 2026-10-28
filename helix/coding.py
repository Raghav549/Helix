"""HELIX advanced coding & software-engineering agent.

Independently implemented. Provides repository-level coding capabilities:
  * Repository inspection (file listing, reading, metadata)
  * Static analysis via Python AST (functions, classes, imports, complexity, TODOs)
  * Sandboxed code execution (reuses agent.PythonTool with limits)
  * Patch generation (unified diff) for proposed fixes
  * Unit-test generation from function signatures
  * LLM-assisted explanation, bug finding, and fixing

Designed for CPU-first operation; the LLM assists but real compiler/interpreter
feedback and static analysis drive correctness. Execution uses a fail-closed Linux namespace/seccomp sandbox with resource limits.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import tempfile
import hashlib
from pathlib import Path
import re
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
        self.root = os.path.realpath(root)

    def _safe_path(self, rel_path: str) -> str:
        if not rel_path or os.path.isabs(rel_path):
            raise ValueError("path must be relative to the repository")
        parts = Path(rel_path).parts
        sensitive = {".git", ".env", ".ssh", ".aws", ".kube", ".gnupg", ".netrc", ".npmrc", ".pypirc",
                     ".git-credentials", "credentials.json", "service-account.json", "id_rsa", "id_ed25519"}
        if any(part.lower() in sensitive or part.lower().startswith(".env.") for part in parts):
            raise ValueError("sensitive path is not readable")
        full = os.path.realpath(os.path.join(self.root, rel_path))
        if os.path.commonpath((self.root, full)) != self.root:
            raise ValueError("path escapes repository")
        return full

    def list_files(self, max_files: int = 200) -> List[str]:
        files: List[str] = []
        for root, dirs, names in os.walk(self.root):
            # skip common non-source dirs
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules",
                                                    "build", "dist", ".idea", ".ssh", ".aws", ".kube", ".gnupg"}
                       and not os.path.islink(os.path.join(root, d))]
            for n in sorted(names):
                ext = os.path.splitext(n)[1].lower()
                if (os.path.islink(os.path.join(root, n)) or n.lower() in
                        {".env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials.json", "service-account.json", "id_rsa", "id_ed25519"}
                        or n.lower().startswith(".env.")):
                    continue
                if ext in self.CODE_EXTS or n.lower() in {
                    "makefile", "dockerfile", "readme", "license", "changelog",
                }:
                    files.append(os.path.relpath(os.path.join(root, n), self.root))
                if len(files) >= max_files:
                    return files
        return files

    def read_file(self, rel_path: str) -> Optional[str]:
        full = self._safe_path(rel_path)
        if not os.path.isfile(full):
            return None
        if os.path.getsize(full) > 1024 * 1024:
            raise ValueError("file exceeds 1 MiB read limit")
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def file_info(self, rel_path: str) -> Optional[FileInfo]:
        full = self._safe_path(rel_path)
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
        """Apply a unified diff in memory, rejecting stale/malformed hunks."""
        if diff == "# No changes\n":
            return original
        lines = diff.splitlines(keepends=True)
        if len(lines) < 2 or not lines[0].startswith("--- ") or not lines[1].startswith("+++ "):
            raise ValueError("invalid unified diff headers")
        orig = original.splitlines(keepends=True)
        result, oi, i = [], 0, 2
        header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        seen = False
        while i < len(lines):
            match = header.match(lines[i])
            if not match:
                raise ValueError("invalid hunk header")
            seen = True
            old_start, old_count = int(match[1]), int(match[2] or 1)
            new_count = int(match[4] or 1)
            start = old_start - 1 if old_count else old_start
            if start < oi or start > len(orig):
                raise ValueError("overlapping or out-of-range hunk")
            result.extend(orig[oi:start]); oi = start
            i += 1; consumed = produced = 0
            while i < len(lines) and not lines[i].startswith("@@ "):
                line = lines[i]
                if not line or line[0] not in " +-":
                    raise ValueError("invalid hunk line")
                marker, content = line[0], line[1:]
                if marker in " -":
                    if oi >= len(orig) or orig[oi] != content:
                        raise ValueError("patch context does not match original")
                    oi += 1; consumed += 1
                if marker in " +":
                    result.append(content); produced += 1
                i += 1
            if consumed != old_count or produced != new_count:
                raise ValueError("hunk line counts do not match header")
        if not seen:
            raise ValueError("no patch hunks")
        result.extend(orig[oi:])
        return "".join(result)


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

class TestGenerator:
    def generate_tests(self, filename: str, source: str) -> str:
        """Generate honest import/callable smoke checks, not fake correctness tests."""
        tree = ast.parse(source, filename=filename)
        functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        lines = ["import importlib.util", "import pathlib", "import unittest", "",
                 f"_path = pathlib.Path(__file__).parent / {filename!r}",
                 "_spec = importlib.util.spec_from_file_location('_helix_target', _path)",
                 "module = importlib.util.module_from_spec(_spec)",
                 "_spec.loader.exec_module(module)", "",
                 "class TestImports(unittest.TestCase):",
                 "    def test_module_imports(self):", "        self.assertIsNotNone(module)"]
        for name in functions:
            lines += [f"    def test_{name}_is_callable(self):",
                      f"        self.assertTrue(callable(module.{name}))"]
        lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
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

    def execute_code(self, code: str, *, approved: bool = False) -> ExecutionResult:
        if approved is not True:
            return ExecutionResult(False, "", "explicit execution approval required", 0.0)
        result = self.executor.execute(code=code)
        return ExecutionResult(result.ok, result.output, result.error,
                               result.duration_s)

    def run_python_file(self, rel_path: str, *, approved: bool = False) -> ExecutionResult:
        source = self.read_file(rel_path)
        if source is None:
            return ExecutionResult(False, "", f"file not found: {rel_path}", 0.0)
        return self.execute_code(source, approved=approved)

    def generate_patch(self, rel_path: str, modified: str) -> Optional[Patch]:
        original = self.read_file(rel_path)
        if original is None:
            return None
        return PatchGenerator().make_patch(rel_path, original, modified)

    def apply_patch(self, rel_path: str, diff: str) -> str:
        original = self.read_file(rel_path) or ""
        return PatchGenerator().apply_patch(original, diff)

    def apply_patch_to_repo(self, rel_path: str, diff: str, expected_sha256: str,
                            *, approved: bool = False) -> str:
        """Write a verified patch only with explicit approval and a fresh hash."""
        if approved is not True:
            raise PermissionError("repository modification requires explicit approval")
        path = self.inspector._safe_path(rel_path)
        original = self.read_file(rel_path)
        if original is None:
            raise FileNotFoundError(rel_path)
        actual = hashlib.sha256(original.encode("utf-8")).hexdigest()
        if actual != expected_sha256:
            raise ValueError("file changed since patch was proposed")
        modified = PatchGenerator().apply_patch(original, diff)
        fd, temp = tempfile.mkstemp(prefix=".helix-patch-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(modified)
            os.chmod(temp, os.stat(path).st_mode & 0o777)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return hashlib.sha256(modified.encode("utf-8")).hexdigest()

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


def demo_code_fix(code: str, *, approved: bool = False) -> dict:
    """Demonstrate static analysis + execution feedback on a snippet."""
    analyzer = StaticAnalyzer(code, "<snippet>")
    result = analyzer.analyze()
    exec_result = None
    if not result.has_syntax_error() and approved is True:
        exec_result = CodingAgent().execute_code(code + "\n", approved=True)
    return {
        "analysis": asdict(result),
        "execution": asdict(exec_result) if exec_result else None,
    }
