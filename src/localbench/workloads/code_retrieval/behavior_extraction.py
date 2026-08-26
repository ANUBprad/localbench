"""Deterministic behavior extraction from Python source code.

Stage A of the two-stage query generation pipeline.  Extracts
``StructuredBehaviorFacts`` from source code using AST analysis
only — no LLM, no docstring, no randomness.

The extracted facts feed into Stage B (docstring-blind prompt) so
the LLM generates queries grounded in code behavior rather than
paraphrasing docstrings.

Scope:
- AST-based structural analysis
- Side-effect detection (attribute writes, global mutations)
- Return pattern classification
- Error handling extraction
- Control flow summary
- Parameter count inference

Out of scope:
- Natural language generation
- Docstring processing
- Query generation

Provenance constraint:
All extracted fields must be free of implementation identifiers
(class names, function names, method names, parameter names,
variable names, file paths).  Descriptions are generic behavioral
summaries derived from AST structure only.
"""

from __future__ import annotations

import ast

from localbench.workloads.code_retrieval.schemas import StructuredBehaviorFacts

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _get_function_name(source_code: str) -> str | None:
    """Extract the function/method name from source code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def _classify_return(tree: ast.Module) -> str:
    """Classify what the function returns using only structural signals."""
    has_bare_return = False
    has_return_value = False
    has_return_none = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            if node.value is None:
                has_bare_return = True
            elif isinstance(node.value, ast.Constant) and node.value.value is None:
                has_return_none = True
            elif isinstance(node.value, ast.Name) and node.value.id == "None":
                has_return_none = True
            else:
                has_return_value = True

    if not has_return_value and not has_bare_return and not has_return_none:
        return "implicit None return"

    parts: list[str] = []
    if has_bare_return:
        parts.append("bare return")
    if has_return_none:
        parts.append("returns None")
    if has_return_value:
        parts.append("returns a computed value")

    return "; ".join(parts) if parts else "has return statements"


def _detect_side_effects(tree: ast.Module) -> list[str]:
    """Detect side effects using generic descriptions (no attribute names).

    Reports the category of mutation rather than the specific target
    (e.g. ``"modifies instance state"`` instead of ``"modifies self.counter"``).
    """
    effects: list[str] = []
    has_instance_write = False
    has_collection_mutation = False
    has_print = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            target = (
                node.target
                if isinstance(node, ast.AugAssign)
                else (node.targets[0] if node.targets else None)
            )
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name):
                    obj = target.value.id
                    if obj in ("self", "cls"):
                        has_instance_write = True
                    else:
                        has_instance_write = True

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                has_print = True
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("append", "extend", "insert", "clear"):
                    has_collection_mutation = True

    if has_instance_write:
        effects.append("modifies instance state")
    if has_collection_mutation:
        effects.append("mutates a collection")
    if has_print:
        effects.append("prints to stdout")

    return effects


def _detect_raises(tree: ast.Module) -> list[str]:
    """Detect exception types explicitly raised."""
    raised: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc:
            if isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name):
                    name = node.exc.func.id
                    if name not in seen:
                        seen.add(name)
                        raised.append(name)
                elif isinstance(node.exc.func, ast.Attribute):
                    name = node.exc.func.attr
                    if name not in seen:
                        seen.add(name)
                        raised.append(name)
            elif isinstance(node.exc, ast.Name):
                name = node.exc.id
                if name not in seen:
                    seen.add(name)
                    raised.append(name)

    return raised


def _detect_error_handling(tree: ast.Module) -> str:
    """Detect try/except patterns."""
    patterns: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            handlers: list[str] = []
            for handler in node.handlers:
                if handler.type is None:
                    handlers.append("bare except")
                elif isinstance(handler.type, ast.Name):
                    handlers.append(handler.type.id)
                elif isinstance(handler.type, ast.Tuple):
                    names = [
                        e.id for e in handler.type.elts if isinstance(e, ast.Name)
                    ]
                    handlers.append(", ".join(names))
            if handlers:
                patterns.append(f"catches {', '.join(handlers)}")

    return "; ".join(patterns) if patterns else ""


def _detect_control_flow(tree: ast.Module) -> str:
    """Detect loops, conditionals, and early returns."""
    parts: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            parts.append("while loop")
        elif isinstance(node, ast.For):
            parts.append("for loop")
        elif isinstance(node, ast.AsyncFor):
            parts.append("async for loop")
        elif isinstance(node, ast.ListComp):
            if "list comprehension" not in parts:
                parts.append("list comprehension")
        elif isinstance(node, ast.GeneratorExp):
            if "generator expression" not in parts:
                parts.append("generator expression")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            for i, stmt in enumerate(body):
                if isinstance(stmt, ast.Return) and i < len(body) - 1:
                    if "early return" not in parts:
                        parts.append("early return on condition")
                    break

    return ", ".join(parts) if parts else "straight-line"


def _count_parameters(tree: ast.Module) -> str:
    """Summarize parameter count without naming any identifiers."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = [
                a for a in args.args
                if not (a == args.args[0] and a.arg in ("self", "cls"))
            ]
            vararg_count = 1 if args.vararg else 0
            kwonly_count = len(args.kwonlyargs)
            kwarg_count = 1 if args.kwarg else 0

            total = len(positional) + vararg_count + kwonly_count + kwarg_count
            has_defaults = len(args.defaults) > 0

            if total == 0:
                return "takes no arguments"
            if total == 1 and not has_defaults:
                return "takes a single parameter"
            desc = f"takes {total} parameters"
            if has_defaults:
                desc += ", at least one with a default value"
            return desc

    return "takes unknown parameters"


def _count_operation_categories(source_code: str) -> list[str]:
    """Count and classify operations without using call names.

    Returns generic categories like ``"invokes method calls"``
    instead of the actual method names.
    """
    categories: list[str] = []
    method_call_count = 0
    func_call_count = 0

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return categories

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                method_call_count += 1
            elif isinstance(node.func, ast.Name):
                func_call_count += 1

    if method_call_count > 0:
        categories.append(f"performs {method_call_count} method call(s)")
    if func_call_count > 0:
        categories.append(f"invokes {func_call_count} function call(s)")

    return categories


def _derive_primary_purpose(
    tree: ast.Module,
    raises: list[str],
    side_effects: list[str],
    operation_categories: list[str],
) -> str:
    """Derive a generic verb-phrase describing the primary purpose.

    Uses only AST-derived structural signals (control flow patterns,
    error handling, side effects, operation counts).  Does NOT use
    the function name, which would leak implementation identifiers.
    """
    has_while = any(isinstance(n, ast.While) for n in ast.walk(tree))
    has_for = any(isinstance(n, ast.For) for n in ast.walk(tree))
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(tree))

    # Build purpose from structural signals
    parts: list[str] = []

    if has_while and has_try:
        parts.append("retries an operation with error handling")
    elif has_while:
        parts.append("iterates with a while loop")
    elif has_for:
        parts.append("iterates over items")

    if has_raise and not has_try:
        parts.append("validates input and raises on error")
    elif has_raise and has_try:
        parts.append("handles errors and may raise")

    if side_effects:
        parts.append(side_effects[0])

    if not parts:
        if operation_categories:
            parts.append("performs processing operations")
        else:
            parts.append("performs an operation")

    return "; ".join(parts[:2])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_behavior_facts(source_code: str) -> StructuredBehaviorFacts:
    """Extract structured behavioral facts from Python source code.

    This is a deterministic, AST-based extraction.  No LLM, no
    docstring, no randomness.  All output fields are free of
    implementation identifiers.

    Parameters
    ----------
    source_code:
        Valid Python function/method source code.

    Returns
    -------
    A ``StructuredBehaviorFacts`` instance with behavioral metadata.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return StructuredBehaviorFacts(
            primary_purpose="parses as invalid code",
            input_summary="unknown",
            output_summary="unknown",
        )

    raises = _detect_raises(tree)
    side_effects = _detect_side_effects(tree)
    error_handling = _detect_error_handling(tree)
    control_flow = _detect_control_flow(tree)
    input_summary = _count_parameters(tree)
    output_summary = _classify_return(tree)
    operation_categories = _count_operation_categories(source_code)
    primary_purpose = _derive_primary_purpose(
        tree, raises, side_effects, operation_categories
    )

    return StructuredBehaviorFacts(
        primary_purpose=primary_purpose,
        input_summary=input_summary,
        output_summary=output_summary,
        side_effects=side_effects,
        key_operations=operation_categories,
        error_handling=error_handling,
        control_flow=control_flow,
        raises=raises,
    )
