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
import textwrap

from localbench.workloads.code_retrieval.schemas import StructuredBehaviorFacts

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_source(source_code: str) -> ast.Module | None:
    """Parse Python source after normalizing leading indentation.

    Method/class-nested code units are stored with leading indentation
    (e.g. ``    def method(...)``).  A raw ``ast.parse`` would raise
    ``IndentationError`` even though the code is valid, silently degrading
    behavior extraction to the "invalid code" fallback.  ``textwrap.dedent``
    removes the common indentation so indented-but-valid source parses,
    while genuinely invalid syntax still raises ``SyntaxError`` and returns
    ``None``.
    """
    try:
        return ast.parse(textwrap.dedent(source_code))
    except SyntaxError:
        return None


def _get_function_name(source_code: str) -> str | None:
    """Extract the function/method name from source code."""
    tree = _parse_source(source_code)
    if tree is None:
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


def _looks_like_get(node: ast.Call) -> bool:
    """Detect a dictionary-style ``.get(...)`` lookup call.

    Checks the method attribute name internally to decide the category but
    never surfaces the identifier in extracted facts.
    """
    fn = node.func
    return isinstance(fn, ast.Attribute) and fn.attr == "get"


def _count_operation_categories(source_code: str) -> list[str]:
    """Count and classify operations without using call names.

    Returns generic categories like ``"invokes method calls"``
    instead of the actual method names, plus identifier-free semantic
    operation categories derived from structural call patterns.
    """
    categories: list[str] = []
    method_call_count = 0
    func_call_count = 0

    tree = _parse_source(source_code)
    if tree is None:
        return categories

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                method_call_count += 1
            elif isinstance(node.func, ast.Name):
                func_call_count += 1

    # Semantic operation categories based on structural call patterns.
    if any(isinstance(n, ast.Call) and _looks_like_get(n) for n in ast.walk(tree)):
        categories.append("looks up a value by key")
    has_membership_check = any(
        isinstance(n, ast.Compare)
        and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops)
        for n in ast.walk(tree)
    )
    if has_membership_check:
        categories.append("tests membership in a collection")

    if method_call_count > 0:
        categories.append(f"performs {method_call_count} method call(s)")
    if func_call_count > 0:
        categories.append(f"invokes {func_call_count} function call(s)")

    return categories


# ---------------------------------------------------------------------------
# Identifier-free semantic behavior-signal detectors
# ---------------------------------------------------------------------------

_STRING_MANIPULATION_METHODS = frozenset({
    "split", "strip", "rstrip", "lstrip", "replace", "lower", "upper",
    "join", "format", "translate", "partition", "splitlines",
})


def _detect_string_operation(tree: ast.Module) -> str | None:
    """Detect string slicing/parsing/transformation (no element identifiers)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.attr in _STRING_MANIPULATION_METHODS:
                return "extracts or transforms part of a string"
    return None


def _detect_container_construction(tree: ast.Module) -> str | None:
    """Detect construction of containers/mappings without element identifiers."""
    for node in ast.walk(tree):
        if isinstance(node, ast.DictComp):
            return "builds a computed mapping"
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(k is not None for k in node.keys):
            return "constructs a mapping of related values"
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            return "builds a filtered list"
    for node in ast.walk(tree):
        if isinstance(node, ast.SetComp):
            return "builds a set of items"
    for node in ast.walk(tree):
        if isinstance(node, ast.GeneratorExp):
            return "iterates with a generator expression"
    return None


def _detect_membership(tree: ast.Module) -> str | None:
    """Detect containment/membership checks (``in`` / ``not in``)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.In, ast.NotIn)):
                    return "checks whether a value is contained in a collection"
    return None


def _detect_comparison(tree: ast.Module) -> str | None:
    """Detect value comparison/validation predicates."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot,
                                   ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    return (
                        "checks whether a value satisfies a comparison condition"
                    )
    return None


def _detect_delegation(tree: ast.Module) -> str | None:
    """Detect forwarding to another routine (returning a call result)."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Return)
            and node.value
            and isinstance(node.value, ast.Call)
        ):
            return "delegates to another routine and returns its result"
    return None


def _detect_attribute_read(tree: ast.Module) -> str | None:
    """Detect reading a stored value from an associated object.

    Reports only attribute *reads* (not method calls or assignment targets),
    and never surfaces the attribute name itself.
    """
    method_call_attr_ids = {
        id(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in method_call_attr_ids):
            return "reads a value from an associated object"
    return None


def _derive_primary_purpose(
    tree: ast.Module,
    raises: list[str],
    side_effects: list[str],
    operation_categories: list[str],
) -> str:
    """Derive an identifier-free verb-phrase describing the primary purpose.

    Prefers specific structural behavior signals (string operations,
    container construction, membership, comparisons, delegation, attribute
    reads) over coarse generics, then appends validation/error and
    side-effect behavior.  Does NOT use the function name, which would leak
    implementation identifiers.
    """
    has_while = any(isinstance(n, ast.While) for n in ast.walk(tree))
    has_for = any(isinstance(n, ast.For) for n in ast.walk(tree))
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(tree))
    has_return_call = any(
        isinstance(n, ast.Return) and n.value and isinstance(n.value, ast.Call)
        for n in ast.walk(tree)
    )

    parts: list[str] = []

    # Most specific structural signal first.
    lead = (
        _detect_string_operation(tree)
        or _detect_container_construction(tree)
        or _detect_membership(tree)
        or _detect_comparison(tree)
        or _detect_delegation(tree)
        or _detect_attribute_read(tree)
    )
    if lead:
        parts.append(lead)
    elif has_while:
        if has_try:
            parts.append("retries an operation with error handling")
        else:
            parts.append("repeatedly iterates with a while loop")
    elif has_for:
        parts.append("iterates over items")
    elif has_return_call:
        parts.append("delegates to another routine and returns its result")

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
# Identifier-free domain concept & observable effect detectors
# ---------------------------------------------------------------------------

_DOMAIN_IMPORT_TABLE: tuple[tuple[tuple[str, ...], str], ...] = (
    # (module-prefix parts, generic domain concept)
    # Only imports that are domain-discriminative are retained; ubiquitous
    # stdlib modules (os, io, logging, typing, datetime, re, ...) are omitted
    # so domain facts stay specific rather than echoing on every unit.
    (("rich", "console"), "terminal/console rendering"),
    (("rich", "table"), "tabulated console rendering"),
    (("rich", "text"), "text formatting"),
    (("rich", "progress"), "progress display"),
    (("rich", "panel"), "panel rendering"),
    (("rich", "spinner"), "terminal spinner display"),
    (("rich", "align"), "text alignment/rendering"),
    (("rich", "pretty"), "pretty-printing"),
    (("rich", "logging"), "logging to a rich console"),
    (("subprocess",), "subprocess/process execution"),
    (("pexpect",), "interactive subprocess automation"),
    (("ctypes",), "native OS/terminal API calls"),
    (("urllib", "request"), "HTTP transfer"),
    (("requests",), "HTTP transfer"),
    (("json",), "JSON serialization/parsing"),
    (("csv",), "CSV serialization/parsing"),
    (("argparse",), "command-line configuration parsing"),
    (("configparser",), "configuration parsing"),
    (("iniconfig",), "configuration parsing"),
    (("tomllib",), "TOML configuration parsing"),
    (("calendar",), "calendar handling"),
    (("pluggy",), "plugin hooks"),
    (("_pytest", "hookspec"), "plugin hooks"),
    (("_pytest", "fixtures"), "fixtures"),
    (("_pytest", "collect"), "test collection"),
    (("_pytest", "config"), "test configuration"),
    (("_pytest", "cacheprovider"), "test result caching"),
    (("_pytest", "warning_types"), "warning handling"),
    (("_pytest", "pytester"), "test harness / temporary test directories"),
)

# Generic domain concept inferred from called attribute/method names without
# surfacing those names.  The attribute is matched internally to choose the
# phrase; it is never written into the facts.
_DOMAIN_CALL_TABLE: tuple[tuple[tuple[str, ...], str], ...] = (
    # (attribute names matched on a call/attribute, domain concept)
    (("dup2",), "file-descriptor redirection"),
    (("align",), "text alignment/measurement"),
    (("add_column", "add_row"), "tabulated row/column layout"),
    (("add_task",), "progress tracking"),
    (("wrap_file",), "progress-tracked file reading"),
    (("_setconsolecursorposition", "setconsolecursorposition"),
     "terminal cursor positioning"),
    (("maketxtfile",), "temporary file creation"),
    (("maketmpdir",), "temporary directory creation"),
    (("mkdir", "makedirs"), "directory creation"),
)


def _import_parts(imports: list[str]) -> list[str]:
    """Normalize an import/module string into dot-separated lower parts."""
    parts: list[str] = []
    for entry in imports or []:
        name = entry.strip().lstrip("@")
        for token in name.split("."):
            token = "".join(ch for ch in token if ch.isalnum() or ch == "_")
            if token and token not in parts:
                parts.append(token.lower())
    return parts


def _call_attribute_names(tree: ast.Module) -> set[str]:
    """Collect called attribute names (used only for concept selection)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr.lower())
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id.lower())
    return names


def _detect_domain_concepts(
    tree: ast.Module,
    imports: list[str] | None,
) -> list[str]:
    """Derive generic identifier-free domain concepts from source signals.

    Grounded in the module imports the unit is part of and the API
    families its body calls.  Every phrase is a fixed generic concept and
    never an implementation identifier.
    """
    concepts: list[str] = []
    seen: set[str] = set()

    import_parts = _import_parts(imports)
    call_names = _call_attribute_names(tree)

    for prefixes, concept in _DOMAIN_IMPORT_TABLE:
        if all(p in import_parts for p in prefixes) and concept not in seen:
            seen.add(concept)
            concepts.append(concept)

    for attrs, concept in _DOMAIN_CALL_TABLE:
        if any(a in call_names for a in attrs) and concept not in seen:
            seen.add(concept)
            concepts.append(concept)

    return concepts


def _detect_observable_effects(
    tree: ast.Module,
    imports: list[str] | None,
    side_effects: list[str],
) -> list[str]:
    """Derive identifier-free observable effects supported by the source.

    Emits only effects with structural evidence: prints, file-descriptor
    redirection, subprocess execution, terminal cursor movement, tabulated
    console output, temporary directory creation, and HTTP file transfer.
    """
    effects: list[str] = []
    seen: set[str] = set()
    import_parts = _import_parts(imports)

    call_attrs: set[str] = set()
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_attrs.add(node.func.attr.lower())
            elif isinstance(node.func, ast.Name):
                call_names.add(node.func.id.lower())

    def add(effect: str) -> None:
        if effect not in seen:
            seen.add(effect)
            effects.append(effect)

    has_print = any("prints to stdout" in s for s in side_effects)
    if has_print:
        add("prints to the console")

    if "dup2" in call_attrs:
        add("redirects or duplicates a file descriptor")
        add("captures standard output/error")
    if "set_console_cursor_position" in call_attrs or (
        "ctypes" in import_parts and any("cursor" in a for a in call_attrs)
    ):
        add("moves the terminal cursor")
    if any(c in call_names for c in ("popen",)) or any(
        "subprocess" in p or "pexpect" in p for p in import_parts
    ):
        add("launches a subprocess")
    if any("tempfile" in p for p in import_parts):
        add("creates or uses temporary files/directories")
    if any("urllib.request" == ".".join(i.split(".")[:2]) for i in (imports or [])):
        add("transfers data over HTTP to a local file")
    if any("add_column" in a or "add_row" in a for a in call_attrs):
        add("renders a table of items to the console")
    if any("align" in a for a in call_attrs):
        add("formats text for display and alignment")

    return effects


def _domain_lead(domain_concepts: list[str]) -> str | None:
    """Return the most specific domain concept as a purpose lead, if any."""
    if not domain_concepts:
        return None
    return domain_concepts[0] + " behaviour"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_behavior_facts(
    source_code: str,
    imports: list[str] | None = None,
) -> StructuredBehaviorFacts:
    """Extract structured behavioral facts from Python source code.

    This is a deterministic, AST-based extraction.  No LLM, no
    docstring, no randomness.  All output fields are free of
    implementation identifiers.

    Parameters
    ----------
    source_code:
        Valid Python function/method source code.
    imports:
        Optional module-level import list for the code unit's context,
        used to ground identifier-free domain concepts and observable
        effects (e.g. ``["rich.console", "subprocess"]``).

    Returns
    -------
    A ``StructuredBehaviorFacts`` instance with behavioral metadata.
    """
    tree = _parse_source(source_code)
    if tree is None:
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
    domain_concepts = _detect_domain_concepts(tree, imports)
    observable_effects = _detect_observable_effects(
        tree, imports, side_effects
    )

    if (primary_purpose == "performs an operation"
            or primary_purpose == "performs processing operations"):
        lead = _domain_lead(domain_concepts)
        if lead:
            primary_purpose = lead

    return StructuredBehaviorFacts(
        primary_purpose=primary_purpose,
        input_summary=input_summary,
        output_summary=output_summary,
        side_effects=side_effects,
        key_operations=operation_categories,
        error_handling=error_handling,
        control_flow=control_flow,
        raises=raises,
        domain_concepts=domain_concepts,
        observable_effects=observable_effects,
    )
