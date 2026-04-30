"""AST safety checks for generated dynamic FlowForge code."""
from __future__ import annotations

import ast

# ---------------------------------------------------------------------------
# AST safety validation — reject dangerous patterns before exec_module()
# ---------------------------------------------------------------------------

# Attribute calls that are never allowed in generated code.
_BLOCKED_ATTR_CALLS: set[str] = {
    "os.system",
    "os.popen",
    "os.exec",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawn",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "shutil.rmtree",
}

# Module-level imports that are never allowed.
_BLOCKED_IMPORTS: set[str] = {
    "subprocess",
    "shutil",
    "ctypes",
    "socket",
}

# Built-in function names that are never allowed as top-level calls.
_BLOCKED_BUILTINS: set[str] = {
    "__import__",
    "exec",
    "eval",
    "compile",
}


def _validate_generated_ast(code: str) -> str | None:
    """Parse *code* and reject dangerous patterns.

    Returns ``None`` when the code is safe, or a human-readable error
    string describing the first violation found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        # --- blocked imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    return (
                        f"import of '{alias.name}' is not allowed in "
                        f"generated code (line {node.lineno})"
                    )
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    return (
                        f"import from '{node.module}' is not allowed in "
                        f"generated code (line {node.lineno})"
                    )

        # --- blocked builtins (bare calls like eval(...)) ---
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                return (
                    f"call to '{func.id}()' is not allowed in "
                    f"generated code (line {node.lineno})"
                )

            # --- blocked attribute calls (os.system(...), subprocess.run(...)) ---
            if isinstance(func, ast.Attribute):
                parts = _resolve_attr_chain(func)
                if parts:
                    full = ".".join(parts)
                    for blocked in _BLOCKED_ATTR_CALLS:
                        if full == blocked or full.endswith(f".{blocked}"):
                            return (
                                f"call to '{full}()' is not allowed in "
                                f"generated code (line {node.lineno})"
                            )

    return None


def _resolve_attr_chain(node: ast.Attribute) -> list[str] | None:
    """Walk an ``ast.Attribute`` chain and return dotted names, e.g. ``['os', 'system']``."""
    parts: list[str] = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return parts
    return None
