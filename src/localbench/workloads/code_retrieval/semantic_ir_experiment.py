"""EXPERIMENT ONLY (pilot): language-agnostic Semantic IR over source code.

This module exists to test one hypothesis — that repeated human-review
failures are caused partly by *semantic information loss in Stage A*, not
merely by prompt wording.  It is deliberately isolated:

- No production code imports it.
- It never writes to dataset artifacts.
- It never calls a model (no Ollama, no LLM of any kind).
- It does not touch selection, review, or regeneration.

The current pipeline's ``StructuredBehaviorFacts`` (see
``behavior_extraction.py``) is a Python-``ast`` bag of structural facts such
as "performs 3 method call(s)" and "checks whether a value satisfies a
comparison condition".  Those are *syntax facts*: they discard the
information a developer would actually search for.  This module tries to
preserve behavior instead — what arrives, what is done to it, what leaves,
and what an outside observer can see.

Provenance constraint (same as the production facts, and enforced by
``assert_identifier_free``): no function, method, class, parameter,
variable, file-path or repository identifier is ever emitted.  Identifiers
are inspected internally only, to classify a known API into a fixed
identifier-free semantic category::

    internally detect:  text.splitlines
    emitted:            "splits textual content into lines"

Dependency: the optional ``semantic-ir`` extra (Tree-sitter).  Only the
Python, Java, Go, Rust and JavaScript grammars are used.

ponytail: the data flow behind ``relations`` is a single forward pass with
lexical identifier containment, not SSA or interprocedural analysis.  A
transformation reached through an alias that is not an input parameter is
reported unlinked, and a conditional reassignment can mis-attribute a
value.  The ceiling is acceptable for a pilot; the upgrade path is a real
def-use graph over the same cue table.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from functools import cache
from typing import Any

# ---------------------------------------------------------------------------
# Public contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticIR:
    """Behavior-level description of one code unit.

    Every field is a fixed, identifier-free phrase drawn from the tables
    below.  Lists are tuples so two runs over identical source produce
    structurally identical output.
    """

    language: str
    unit_kind: str
    parse_ok: bool
    input_roles: tuple[str, ...]
    output_behavior: str
    domain_signals: tuple[str, ...]
    transformations: tuple[str, ...]
    relations: tuple[str, ...]
    conditions: tuple[str, ...]
    state_changes: tuple[str, ...]
    side_effects: tuple[str, ...]
    error_behavior: str
    observable_effects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view, for printing and side-by-side comparison."""
        return {
            "language": self.language,
            "unit_kind": self.unit_kind,
            "parse_ok": self.parse_ok,
            "input_roles": list(self.input_roles),
            "output_behavior": self.output_behavior,
            "domain_signals": list(self.domain_signals),
            "transformations": list(self.transformations),
            "relations": list(self.relations),
            "conditions": list(self.conditions),
            "state_changes": list(self.state_changes),
            "side_effects": list(self.side_effects),
            "error_behavior": self.error_behavior,
            "observable_effects": list(self.observable_effects),
        }


@dataclass(frozen=True)
class Informativeness:
    """Deterministic count of the distinct semantic anchors an IR carries.

    EXPERIMENTAL DIAGNOSTIC ONLY.  There is deliberately no pass/fail
    threshold: a production cutoff is not justified until anchor counts have
    been inspected against real development CodeUnits, which is what the
    pilot probe exists for.  Each field contributes one anchor per distinct
    item, so a unit cannot inflate the count by repeating one operation.
    """

    anchor_count: int
    breakdown: dict[str, int]


# ---------------------------------------------------------------------------
# Language table (Section 7: the five grammars are NOT the same syntax)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LangSpec:
    ts_name: str
    unit_nodes: frozenset[str]
    param_nodes: frozenset[str]
    call_nodes: frozenset[str]
    return_nodes: frozenset[str]
    cond_nodes: frozenset[str]
    loop_nodes: frozenset[str]
    assign_nodes: frozenset[str]
    try_nodes: frozenset[str]
    catch_nodes: frozenset[str]
    throw_nodes: frozenset[str]
    await_nodes: frozenset[str]
    closure_nodes: frozenset[str]
    #: unit node types that are unambiguously methods
    method_units: frozenset[str]
    #: parent node types that make a plain unit node a method
    method_parents: frozenset[str]
    #: True when a unit's value is returned by a bare tail expression
    tail_value: bool
    #: grammar field naming the declared result type, when the language has one
    result_field: str | None
    #: node types that name a member (``.x``, ``a.b()``); their last child is an
    #: API name, not an author-chosen variable name
    member_access_nodes: frozenset[str]


_SPECS: dict[str, _LangSpec] = {
    "python": _LangSpec(
        ts_name="python",
        unit_nodes=frozenset({"function_definition"}),
        param_nodes=frozenset({"parameters"}),
        call_nodes=frozenset({"call"}),
        return_nodes=frozenset({"return_statement"}),
        cond_nodes=frozenset(
            {"if_statement", "if_expression", "conditional_expression"}
        ),
        loop_nodes=frozenset({
            "for_statement", "while_statement", "list_comprehension",
            "dictionary_comprehension", "set_comprehension",
            "generator_expression",
        }),
        assign_nodes=frozenset({"assignment", "augmented_assignment"}),
        try_nodes=frozenset({"try_statement"}),
        catch_nodes=frozenset({"except_clause"}),
        throw_nodes=frozenset({"raise_statement"}),
        await_nodes=frozenset({"await"}),
        closure_nodes=frozenset({"lambda"}),
        method_units=frozenset(),
        method_parents=frozenset({"class_definition"}),
        tail_value=False,
        result_field="return_type",
        member_access_nodes=frozenset(
            {"attribute", "call", "function_call", "keyword_argument", "decorator"}
        ),
    ),
    "java": _LangSpec(
        ts_name="java",
        unit_nodes=frozenset({"method_declaration", "constructor_declaration"}),
        param_nodes=frozenset({"formal_parameters"}),
        call_nodes=frozenset(
            {"method_invocation", "object_creation_expression"}
        ),
        return_nodes=frozenset({"return_statement"}),
        cond_nodes=frozenset(
            {"if_statement", "ternary_expression", "switch_expression"}
        ),
        loop_nodes=frozenset(
            {"for_statement", "enhanced_for_statement", "while_statement"}
        ),
        assign_nodes=frozenset({
            "local_variable_declaration", "assignment_expression",
        }),
        try_nodes=frozenset({"try_statement", "try_with_resources_statement"}),
        catch_nodes=frozenset({"catch_clause"}),
        throw_nodes=frozenset({"throw_statement"}),
        await_nodes=frozenset(),
        closure_nodes=frozenset({"lambda_expression"}),
        method_units=frozenset(
            {"method_declaration", "constructor_declaration"}
        ),
        method_parents=frozenset(),
        tail_value=False,
        result_field="type",
        member_access_nodes=frozenset(
            {"method_invocation", "field_access", "method_reference",
             "object_creation_expression"}
        ),
    ),
    "go": _LangSpec(
        ts_name="go",
        unit_nodes=frozenset({"function_declaration", "method_declaration"}),
        param_nodes=frozenset({"parameter_list"}),
        call_nodes=frozenset({"call_expression", "type_conversion_expression"}),
        return_nodes=frozenset({"return_statement"}),
        cond_nodes=frozenset({"if_statement", "expression_switch_statement"}),
        loop_nodes=frozenset({"for_statement", "go_statement"}),
        assign_nodes=frozenset({
            "short_var_declaration", "var_declaration", "assignment_statement",
            "inc_statement", "dec_statement",
        }),
        try_nodes=frozenset(),
        catch_nodes=frozenset(),
        # Go signals failure by returning an error value or panicking.
        throw_nodes=frozenset(),
        await_nodes=frozenset(),
        closure_nodes=frozenset({"func_literal"}),
        method_units=frozenset({"method_declaration"}),
        method_parents=frozenset(),
        tail_value=False,
        result_field="result",
        member_access_nodes=frozenset(
            {"selector_expression", "call_expression", "keyed_element"}
        ),
    ),
    "rust": _LangSpec(
        ts_name="rust",
        unit_nodes=frozenset({"function_item"}),
        param_nodes=frozenset({"parameters"}),
        call_nodes=frozenset({"call_expression", "macro_invocation"}),
        return_nodes=frozenset({"return_expression"}),
        cond_nodes=frozenset({"if_expression", "match_expression"}),
        loop_nodes=frozenset(
            {"for_expression", "while_expression", "loop_expression"}
        ),
        assign_nodes=frozenset({"let_declaration", "assignment_expression"}),
        try_nodes=frozenset(),
        catch_nodes=frozenset(),
        # Rust signals failure via Result / the ? operator / panic!.
        throw_nodes=frozenset(),
        await_nodes=frozenset({"await_expression"}),
        closure_nodes=frozenset({"closure_expression"}),
        method_units=frozenset(),
        method_parents=frozenset({"impl_item", "trait_item"}),
        tail_value=True,
        result_field="return_type",
        member_access_nodes=frozenset(
            {"field_expression", "scoped_identifier", "macro_invocation",
             "call_expression"}
        ),
    ),
    "javascript": _LangSpec(
        ts_name="javascript",
        unit_nodes=frozenset({
            "function_declaration", "generator_function_declaration",
            "method_definition",
        }),
        param_nodes=frozenset({"formal_parameters"}),
        call_nodes=frozenset({"call_expression", "new_expression"}),
        return_nodes=frozenset({"return_statement"}),
        cond_nodes=frozenset({
            "if_statement", "ternary_expression", "switch_statement",
        }),
        loop_nodes=frozenset({
            "for_statement", "for_in_statement", "while_statement",
            "do_statement",
        }),
        assign_nodes=frozenset({
            "variable_declarator", "variable_declaration",
            "lexical_declaration", "assignment_expression",
            "augmented_assignment_expression",
        }),
        try_nodes=frozenset({"try_statement"}),
        catch_nodes=frozenset({"catch_clause"}),
        throw_nodes=frozenset({"throw_statement"}),
        await_nodes=frozenset({"await_expression"}),
        closure_nodes=frozenset({"arrow_function", "function_expression"}),
        method_units=frozenset({"method_definition"}),
        method_parents=frozenset(),
        tail_value=False,
        result_field=None,
        member_access_nodes=frozenset(
            {"member_expression", "call_expression", "pair", "new_expression"}
        ),
    ),
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(_SPECS)

# Canonical domain vocabulary (Section 8).  A domain is reachable only
# through a structural signal in _CUES; nothing is asserted by name alone.
_D_TEXT = "string/text processing"
_D_COLL = "collection processing"
_D_JSON = "JSON parsing"
_D_FILE = "file I/O"
_D_HTTP = "HTTP/networking"
_D_PROC = "subprocess execution"
_D_LOG = "logging"
_D_CONSOLE = "terminal/console"
_D_CONF = "configuration parsing"
_D_DB = "database interaction"
_D_SER = "serialization"
_D_TIME = "time and scheduling"
_D_MATH = "numeric and statistical computation"
_D_CRYPTO = "hashing and cryptography"
_D_CACHE = "caching"
_D_STATE = "object and value construction"

# ---------------------------------------------------------------------------
# API cue table
# ---------------------------------------------------------------------------
#
# Each entry is (cue key, behavior phrase, domain).  A key is a dot-joined
# tail of a callee's dotted/:: path, normalized to lowercase alphanumerics
# ("os.ReadFile" -> "os.readfile", "std::fs::read_to_string" ->
# "std.fs.readtostring").  The LONGEST matching key wins, so a specific
# entry always beats a generic one ("log.printf" beats "printf").
#
# ponytail: the table is hand-curated and deliberately under-calls rather
# than over-calls — an unlisted API yields no transformation at all, which
# costs recall but never invents behavior.  The upgrade path is deriving
# entries from package documentation; the matcher itself does not change.

_CUES: tuple[tuple[str, str, str], ...] = (
    # --- string / text ---------------------------------------------------
    ("splitlines", "splits textual content into lines", _D_TEXT),
    ("readlines", "splits textual content into lines", _D_TEXT),
    ("lines", "splits textual content into lines", _D_TEXT),
    ("splitn", "splits textual content on a delimiter", _D_TEXT),
    ("rsplit", "splits textual content on a delimiter", _D_TEXT),
    ("split", "splits textual content on a delimiter", _D_TEXT),
    ("join", "joins a collection of items into one delimited text", _D_TEXT),
    ("lstrip", "trims leading whitespace from text", _D_TEXT),
    ("rstrip", "trims trailing whitespace from text", _D_TEXT),
    ("strip", "trims surrounding whitespace from text", _D_TEXT),
    ("trimspace", "trims surrounding whitespace from text", _D_TEXT),
    ("trimstart", "trims leading whitespace from text", _D_TEXT),
    ("trimend", "trims trailing whitespace from text", _D_TEXT),
    ("trim", "trims surrounding whitespace from text", _D_TEXT),
    ("toupper", "converts text to upper case", _D_TEXT),
    ("tolower", "converts text to lower case", _D_TEXT),
    ("uppercase", "converts text to upper case", _D_TEXT),
    ("lowercase", "converts text to lower case", _D_TEXT),
    ("upper", "converts text to upper case", _D_TEXT),
    ("lower", "converts text to lower case", _D_TEXT),
    ("capitalize", "re-cases the first character of text", _D_TEXT),
    ("replace", "substitutes occurrences within text", _D_TEXT),
    ("startswith", "tests whether text begins with a given prefix", _D_TEXT),
    ("endswith", "tests whether text ends with a given suffix", _D_TEXT),
    ("padstart", "pads text out to a fixed width", _D_TEXT),
    ("padend", "pads text out to a fixed width", _D_TEXT),
    ("repeat", "repeats text a fixed number of times", _D_TEXT),
    ("concat", "concatenates text fragments", _D_TEXT),
    ("substring", "extracts a contiguous portion of text", _D_TEXT),
    ("substr", "extracts a contiguous portion of text", _D_TEXT),
    ("escape", "escapes special characters for safe embedding", _D_TEXT),
    ("unescape", "decodes escaped characters back to raw form", _D_TEXT),
    ("sprintf", "formats values into a text template", _D_TEXT),
    ("format", "formats values into a text template", _D_TEXT),
    ("fstring", "interpolates values into a text template", _D_TEXT),
    # --- collections -----------------------------------------------------
    ("map", "transforms each item of a collection", _D_COLL),
    ("flatmap", "transforms and flattens each item of a collection", _D_COLL),
    ("filter", "keeps only the items of a collection that match a predicate",
     _D_COLL),
    ("reduce", "folds a collection into one accumulated value", _D_COLL),
    ("fold", "folds a collection into one accumulated value", _D_COLL),
    ("foreach", "applies an action to every item of a collection", _D_COLL),
    ("sorted", "orders the items of a collection", _D_COLL),
    ("sortby", "orders the items of a collection by a derived key", _D_COLL),
    ("groupby", "groups the items of a collection by a derived key", _D_COLL),
    ("sort", "orders the items of a collection", _D_COLL),
    ("append", "adds an item to a collection", _D_COLL),
    ("extend", "adds several items to a collection", _D_COLL),
    ("push", "adds an item to a collection", _D_COLL),
    ("insert", "places an item into a collection at a position", _D_COLL),
    ("prepend", "places an item at the front of a collection", _D_COLL),
    ("pop", "removes and yields an item from a collection", _D_COLL),
    ("delete", "removes an item from a collection", _D_COLL),
    ("remove", "removes an item from a collection", _D_COLL),
    ("count", "counts items or occurrences", _D_COLL),
    ("len", "counts the items of a collection", _D_COLL),
    ("length", "counts the items of a collection", _D_COLL),
    ("size", "counts the items of a collection", _D_COLL),
    ("isempty", "tests whether a collection or input is empty", _D_COLL),
    ("enumerate", "iterates items together with their position", _D_COLL),
    ("zip", "pairs items drawn from two collections", _D_COLL),
    ("keys", "enumerates the keys held in a stored mapping", _D_COLL),
    ("values", "enumerates the values held in a stored mapping", _D_COLL),
    ("items", "enumerates the entries held in a stored mapping", _D_COLL),
    ("range", "produces a sequence of positions", _D_COLL),
    ("clone", "copies a collection or value", _D_COLL),
    ("copy", "copies a value into a new independent form", _D_COLL),
    ("reserve", "pre-allocates capacity for a growing collection", _D_COLL),
    ("iter", "iterates over a collection of items", _D_COLL),
    ("next", "advances an iterator to its next item", _D_COLL),
    ("reverses", "reverses the order of a collection", _D_COLL),
    ("reverse", "reverses the order of a collection", _D_COLL),
    # --- numeric ---------------------------------------------------------
    ("sum", "aggregates numeric items into a single total", _D_MATH),
    ("aggregate", "aggregates items into one reduced value", _D_MATH),
    ("mean", "averages numeric items", _D_MATH),
    ("median", "computes the median of numeric items", _D_MATH),
    ("min", "selects the smallest of a set of values", _D_MATH),
    ("max", "selects the largest of a set of values", _D_MATH),
    # --- mapping lookup / store -----------------------------------------
    ("containskey", "checks whether a stored mapping holds a key", _D_COLL),
    ("hasownproperty", "checks whether a stored mapping holds a key", _D_COLL),
    ("contain", "checks whether a collection holds a value", _D_COLL),
    ("getordefault", "looks up a key and falls back to a default value",
     _D_COLL),
    ("getkey", "looks up a value by key in a stored mapping", _D_COLL),
    ("setdefault", "stores a default value under a key when absent", _D_COLL),
    ("get", "looks up a value by key in a stored mapping", _D_COLL),
    ("put", "stores a value under a key in a mapping", _D_COLL),
    ("update", "stores or refreshes a value under a key in a mapping", _D_COLL),
    ("lookup", "looks up a value by key", _D_COLL),
    ("has", "checks for the presence of a key or entry", _D_COLL),
    ("find", "searches a collection for a matching entry", _D_COLL),
    ("indexof", "locates the position of an entry", _D_COLL),
    ("index", "locates the position of an entry", _D_COLL),
    # --- JSON / structured data -----------------------------------------
    ("json.loads", "parses JSON data", _D_JSON),
    ("json.load", "parses JSON data", _D_JSON),
    ("json.loadsall", "parses a stream of JSON values", _D_JSON),
    ("json.parse", "parses JSON data", _D_JSON),
    ("jsonparse", "parses JSON data", _D_JSON),
    ("jsondecode", "parses JSON data", _D_JSON),
    ("json.unmarshal", "parses JSON data", _D_JSON),
    ("json.fromstr", "parses JSON data", _D_JSON),
    ("json.fromreader", "parses JSON data from a stream", _D_JSON),
    ("newdecoder", "parses JSON data from a stream", _D_JSON),
    ("json", "parses JSON data", _D_JSON),
    ("json.dumps", "serializes data to JSON text", _D_JSON),
    ("json.stringify", "serializes data to JSON text", _D_JSON),
    ("json.marshal", "serializes data to JSON text", _D_JSON),
    ("tojsonstring", "serializes data to JSON text", _D_JSON),
    ("tojson", "serializes data to JSON text", _D_JSON),
    # --- serialization / encoding ---------------------------------------
    ("marshal", "serializes a value into a portable representation", _D_SER),
    ("unmarshal", "deserializes a value from a portable representation", _D_SER),
    ("serialize", "serializes a value into a portable representation", _D_SER),
    ("deserialize", "deserializes a value from a portable representation",
     _D_SER),
    ("fromstr", "deserializes a value from its portable representation", _D_SER),
    ("tostring", "renders a value as text", _D_SER),
    ("b64encode", "encodes binary data as base64 text", _D_SER),
    ("b64decode", "decodes base64 text back to binary data", _D_SER),
    ("encode64", "encodes binary data as base64 text", _D_SER),
    ("decode64", "decodes base64 text back to binary data", _D_SER),
    ("encodebase64", "encodes binary data as base64 text", _D_SER),
    ("decodebase64", "decodes base64 text back to binary data", _D_SER),
    ("encode", "encodes a value into bytes or another representation", _D_SER),
    ("decode", "decodes bytes or an encoded representation into values", _D_SER),
    ("compress", "compresses data into a smaller representation", _D_SER),
    ("decompress", "restores data from a compressed representation", _D_SER),
    ("gzip", "compresses data into a smaller representation", _D_SER),
    ("deflate", "compresses data into a smaller representation", _D_SER),
    # --- hashing / secrets ----------------------------------------------
    ("sha256", "derives a fixed-size digest from a value", _D_CRYPTO),
    ("sha1", "derives a fixed-size digest from a value", _D_CRYPTO),
    ("md5", "derives a fixed-size digest from a value", _D_CRYPTO),
    ("digest", "derives a fixed-size digest from a value", _D_CRYPTO),
    ("hash", "derives a fixed-size digest from a value", _D_CRYPTO),
    ("urandom", "generates random bytes", _D_CRYPTO),
    ("randombytes", "generates random bytes", _D_CRYPTO),
    ("secrets", "generates random values suitable for secrets", _D_CRYPTO),
    # --- file I/O --------------------------------------------------------
    ("readtostring", "reads a file as text", _D_FILE),
    ("readfile", "reads a file's bytes into memory", _D_FILE),
    ("readbytes", "reads a file's bytes into memory", _D_FILE),
    ("readtext", "reads a file as text", _D_FILE),
    ("writebytes", "writes bytes to a file", _D_FILE),
    ("writetext", "writes text to a file", _D_FILE),
    ("writefile", "writes data to a file", _D_FILE),
    ("readdirsync", "lists the entries of a directory", _D_FILE),
    ("readdir", "lists the entries of a directory", _D_FILE),
    ("readdirectory", "lists the entries of a directory", _D_FILE),
    ("listdir", "lists the entries of a directory", _D_FILE),
    ("scandir", "lists the entries of a directory", _D_FILE),
    ("walkdir", "recursively walks a directory tree", _D_FILE),
    ("removedir", "removes a directory from the filesystem", _D_FILE),
    ("removedirectory", "removes a directory from the filesystem", _D_FILE),
    ("removefile", "removes a file from the filesystem", _D_FILE),
    ("os.remove", "removes a filesystem entry", _D_FILE),
    ("fs.remove", "removes a filesystem entry", _D_FILE),
    ("files.remove", "removes a filesystem entry", _D_FILE),
    ("path.remove", "removes a filesystem entry", _D_FILE),
    ("os.rename", "renames a filesystem entry", _D_FILE),
    ("fs.rename", "renames a filesystem entry", _D_FILE),
    ("makedirs", "creates a directory and any missing parents", _D_FILE),
    ("createdirs", "creates a directory and any missing parents", _D_FILE),
    ("createdir", "creates a directory", _D_FILE),
    ("mkdirs", "creates a directory and any missing parents", _D_FILE),
    ("mkdir", "creates a directory", _D_FILE),
    ("mkdtemp", "creates a temporary directory", _D_FILE),
    ("namedtempfile", "creates a temporary file", _D_FILE),
    ("tempfile", "uses a temporary file or directory", _D_FILE),
    ("existsync", "tests whether a filesystem path exists", _D_FILE),
    ("exists", "tests whether a filesystem path exists", _D_FILE),
    ("metadata", "inspects filesystem metadata for a path", _D_FILE),
    ("readall", "reads a stream to its end", _D_FILE),
    ("writeall", "writes all provided bytes to a stream", _D_FILE),
    ("flush", "flushes buffered output to its destination", _D_FILE),
    ("writeln", "writes a formatted line of text to a stream", _D_FILE),
    ("fs.write", "writes data to a file", _D_FILE),
    ("close", "closes an open file or resource", _D_FILE),
    ("open", "opens a file for reading or writing", _D_FILE),
    ("read", "reads data from a file or stream", _D_FILE),
    ("write", "writes data to a file or stream", _D_FILE),
    # --- HTTP / networking ----------------------------------------------
    ("requests.get", "issues an HTTP request to a remote service", _D_HTTP),
    ("requests.post", "sends data to a remote service over HTTP", _D_HTTP),
    ("requests.put", "sends data to a remote service over HTTP", _D_HTTP),
    ("httpx.get", "issues an HTTP request to a remote service", _D_HTTP),
    ("httpx.post", "sends data to a remote service over HTTP", _D_HTTP),
    ("http.get", "issues an HTTP request to a remote service", _D_HTTP),
    ("http.post", "sends data to a remote service over HTTP", _D_HTTP),
    ("http.postform", "sends a form body to a remote service", _D_HTTP),
    ("client.get", "issues an HTTP request to a remote service", _D_HTTP),
    ("client.post", "sends data to a remote service over HTTP", _D_HTTP),
    ("session.get", "issues an HTTP request to a remote service", _D_HTTP),
    ("session.post", "sends data to a remote service over HTTP", _D_HTTP),
    ("reqwest.get", "issues an HTTP request to a remote service", _D_HTTP),
    ("sendrequest", "transmits a request to a remote service", _D_HTTP),
    ("newrequest", "builds a request for a remote service", _D_HTTP),
    ("urlopen", "issues an HTTP request to a remote endpoint", _D_HTTP),
    ("fetch", "issues an HTTP request to a remote endpoint", _D_HTTP),
    ("send", "transmits data to a remote service", _D_HTTP),
    # --- subprocess / process -------------------------------------------
    ("subprocess.run", "executes an external process", _D_PROC),
    ("subprocess.call", "executes an external process", _D_PROC),
    ("subprocess.popen", "starts an external process and captures it", _D_PROC),
    ("exec.command", "starts an external process", _D_PROC),
    ("command.new", "starts an external process", _D_PROC),
    ("processbuilder", "starts an external process", _D_PROC),
    ("getoutput", "runs an external process and captures its output", _D_PROC),
    ("checkoutput", "runs an external process and checks its status", _D_PROC),
    ("checkcall", "runs an external process and checks its status", _D_PROC),
    ("popen", "starts an external process and captures it", _D_PROC),
    ("spawn", "starts a new process", _D_PROC),
    ("exec", "executes an external process", _D_PROC),
    ("system", "runs a command through the operating system shell", _D_PROC),
    ("kill", "terminates a running process", _D_PROC),
    # --- logging ---------------------------------------------------------
    ("log.fatal", "records a fatal diagnostic message", _D_LOG),
    ("log.panic", "records a fatal diagnostic message and aborts", _D_LOG),
    ("log.printf", "records a formatted diagnostic message", _D_LOG),
    ("log.println", "records a diagnostic message", _D_LOG),
    ("log.print", "records a diagnostic message", _D_LOG),
    ("log.warn", "records a warning diagnostic message", _D_LOG),
    ("logger.info", "records an informational diagnostic message", _D_LOG),
    ("logger.debug", "records a low-level diagnostic message", _D_LOG),
    ("logger.warning", "records a warning diagnostic message", _D_LOG),
    ("logger.error", "records an error diagnostic message", _D_LOG),
    ("logger.exception", "records a failure diagnostic message", _D_LOG),
    ("logging.info", "records an informational diagnostic message", _D_LOG),
    ("logging.debug", "records a low-level diagnostic message", _D_LOG),
    ("logging.warning", "records a warning diagnostic message", _D_LOG),
    ("logging.error", "records an error diagnostic message", _D_LOG),
    ("logging.exception", "records a failure diagnostic message", _D_LOG),
    ("info", "records an informational diagnostic message", _D_LOG),
    ("debug", "records a low-level diagnostic message", _D_LOG),
    ("warning", "records a warning diagnostic message", _D_LOG),
    ("warn", "records a warning diagnostic message", _D_LOG),
    ("critical", "records a critical diagnostic message", _D_LOG),
    ("fatal", "records a fatal diagnostic message", _D_LOG),
    ("log", "records a diagnostic message", _D_LOG),
    # --- terminal / console ---------------------------------------------
    ("system.out", "writes to the standard output stream", _D_CONSOLE),
    ("system.err", "writes a diagnostic message to the error stream",
     _D_CONSOLE),
    ("sys.stdout", "writes to the standard output stream", _D_CONSOLE),
    ("sys.stderr", "writes a diagnostic message to the error stream",
     _D_CONSOLE),
    ("stdout.write", "writes to the standard output stream", _D_CONSOLE),
    ("stderr.write", "writes a diagnostic message to the error stream",
     _D_CONSOLE),
    ("console.log", "writes a line to the console", _D_CONSOLE),
    ("console.info", "writes a line to the console", _D_CONSOLE),
    ("console.debug", "writes a line to the console", _D_CONSOLE),
    ("console.warn", "writes a warning line to the console", _D_CONSOLE),
    ("console.error", "writes an error line to the console", _D_CONSOLE),
    ("console.table", "renders a table of items to the console", _D_CONSOLE),
    ("fmt.print", "writes to the standard output stream", _D_CONSOLE),
    ("fmt.println", "writes a line to the standard output stream", _D_CONSOLE),
    ("fmt.printf", "writes formatted text to the output stream", _D_CONSOLE),
    ("setconsolecursorposition", "moves the terminal cursor to a position",
     _D_CONSOLE),
    ("setcursor", "moves the terminal cursor to a position", _D_CONSOLE),
    ("movecursor", "moves the terminal cursor", _D_CONSOLE),
    ("showcursor", "shows the terminal cursor", _D_CONSOLE),
    ("hidecursor", "hides the terminal cursor", _D_CONSOLE),
    ("gotoxy", "moves the terminal cursor to a position", _D_CONSOLE),
    ("print", "writes a line to the console", _D_CONSOLE),
    ("println", "writes a line to the console", _D_CONSOLE),
    ("eprintln", "writes a line to the error stream", _D_CONSOLE),
    # --- configuration ---------------------------------------------------
    ("getenv", "reads a setting from the process environment", _D_CONF),
    ("getproperty", "reads a configuration property", _D_CONF),
    ("getproperties", "reads the full set of configuration properties", _D_CONF),
    ("yaml.load", "parses structured configuration data", _D_CONF),
    ("yaml.safe_load", "parses structured configuration data", _D_CONF),
    ("loadyaml", "parses structured configuration data", _D_CONF),
    ("parseyaml", "parses structured configuration data", _D_CONF),
    ("toml.load", "parses TOML configuration data", _D_CONF),
    ("toml.loads", "parses TOML configuration data", _D_CONF),
    ("loadtoml", "parses TOML configuration data", _D_CONF),
    ("parsetoml", "parses TOML configuration data", _D_CONF),
    ("tomllib", "parses TOML configuration data", _D_CONF),
    ("iniconfig", "parses INI-style configuration data", _D_CONF),
    ("configparser", "parses INI-style configuration data", _D_CONF),
    ("argparse", "parses command-line configuration", _D_CONF),
    ("config", "reads configuration values", _D_CONF),
    # --- database --------------------------------------------------------
    ("executemany", "runs a batch of statements against a database", _D_DB),
    ("fetchone", "reads one row from a database result", _D_DB),
    ("fetchall", "reads all rows from a database result", _D_DB),
    ("fetchmany", "reads a batch of rows from a database result", _D_DB),
    ("commit", "commits a database transaction", _D_DB),
    ("rollback", "discards a database transaction", _D_DB),
    ("cursor", "opens a handle for issuing database statements", _D_DB),
    ("connect", "opens a connection to a database service", _D_DB),
    ("execute", "runs a statement against a database", _D_DB),
    ("query", "runs a query against a database", _D_DB),
    # --- time / scheduling -----------------------------------------------
    ("sleep", "pauses execution for a fixed interval", _D_TIME),
    ("utcnow", "reads the current time", _D_TIME),
    ("now", "reads the current time", _D_TIME),
    ("today", "reads the current date", _D_TIME),
    ("timer", "schedules work against a deadline or interval", _D_TIME),
    ("timeout", "bounds how long an operation may take", _D_TIME),
    # --- object construction ---------------------------------------------
    ("init", "initializes an owned object", _D_STATE),
    ("new", "constructs a new value", _D_STATE),
    ("builder", "builds a value incrementally through chained setters",
     _D_STATE),
    # --- ubiquitous stdlib module names, used to ground domains from the
    #     unit's import list the way the production extractor does ---------
    ("os", "interacts with the operating system", _D_FILE),
    ("pathlib", "manipulates filesystem paths", _D_FILE),
    ("shutil", "manipulates files and directories", _D_FILE),
    ("tempfilepackage", "uses a temporary file or directory", _D_FILE),
    ("io", "reads and writes byte or text streams", _D_FILE),
    ("csv", "parses delimiter-separated data", _D_JSON),
    ("re", "matches text against a pattern", _D_TEXT),
    ("requests", "exchanges data with remote HTTP services", _D_HTTP),
    ("httpx", "exchanges data with remote HTTP services", _D_HTTP),
    ("urllib", "exchanges data with remote HTTP services", _D_HTTP),
    ("socket", "exchanges raw data with a network peer", _D_HTTP),
    ("asyncio", "runs asynchronous concurrent work", _D_STATE),
    ("threading", "runs work concurrently on threads", _D_STATE),
    ("multiprocessing", "runs work concurrently in separate processes",
     _D_STATE),
    ("sqlite3", "queries an embedded SQL database", _D_DB),
    ("pickle", "serializes Python objects to bytes", _D_SER),
    ("base64", "encodes and decodes base64 text", _D_SER),
    ("hashlib", "derives fixed-size digests from values", _D_CRYPTO),
    ("numpy", "computes over numeric arrays", _D_MATH),
    ("pandas", "computes over tabular data frames", _D_MATH),
    ("statistics", "computes summary statistics", _D_MATH),
    ("decimal", "computes exact decimal arithmetic", _D_MATH),
    ("fractions", "computes exact rational arithmetic", _D_MATH),
    ("itertools", "builds and consumes iterators", _D_COLL),
    ("functools", "composes and caches callables", _D_CACHE),
    ("lru_cache", "reuses a previously computed result", _D_CACHE),
    ("cache", "reuses a previously computed result", _D_CACHE),
    ("random", "generates random values", _D_MATH),
    ("subprocess", "executes an external process", _D_PROC),
    ("logging", "records diagnostic messages", _D_LOG),
    ("warnings", "emits a warning about suspicious code", _D_LOG),
    ("dataclasses", "declares a data shape", _D_STATE),
    ("enum", "declares a closed set of named variants", _D_STATE),
    ("collections", "provides container types", _D_COLL),
    ("typing", "annotates types only", ""),
)

_CUE_MAP: dict[str, tuple[str, str]] = {
    key: (phrase, dom) for key, phrase, dom in _CUES
}

# Cue phrases that mean "stores a value for a later lookup".
_STORE_CUES = frozenset({
    "put", "update", "setdefault", "insert", "add", "cache", "lru_cache",
    "store", "addcolumn", "addrow", "addtask",
})
# Cue phrases that mean "reads a value out of a stored structure".
_LOOKUP_CUES = frozenset({
    "get", "containskey", "getordefault", "lookup", "getkey", "find", "has",
    "index", "indexof", "hasownproperty", "contain",
})
_MAPPING_TYPES = frozenset({
    "map", "dict", "hashmap", "treemap", "mapping", "json", "record", "tbl",
    "htable", "object", "struct",
})

# ---------------------------------------------------------------------------
# Tree-sitter plumbing
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_OP_CHARS = frozenset("+-*/%<>=!&|^~?:.")
_IDENT_TYPES = frozenset({
    "identifier",
    "property_identifier",
    "field_identifier",
    "package_identifier",
    "type_identifier",
    "primitive_type",
    "shorthand_property_identifier",
})
_OPAQUE = "opaque input"
_NONE_HANDLING = "no explicit failure handling; failures are not intercepted"
_FAILS = re.compile(r"\b(err|error|exception|failure|panic)\b", re.IGNORECASE)


def _norm(text: str) -> str:
    """Lowercase and drop every non-alphanumeric character from a segment."""
    return _NON_ALNUM.sub("", text.lower())


@cache
def _parser(language: str):
    """Return a cached Tree-sitter parser for a supported pilot language."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "the semantic IR pilot needs the optional 'semantic-ir' extra: "
            "pip install -e '.[semantic-ir]'"
        ) from exc
    return get_parser(_SPECS[language].ts_name)


def _add(bucket: list[str], value: str) -> None:
    """Append to an ordered set: first-seen order, duplicates dropped."""
    if value and value not in bucket:
        bucket.append(value)


#: assignment targets that write a field rather than a local
_ATTRIBUTE_TARGETS = frozenset({
    "attribute", "field_access", "field_expression", "member_expression",
    "selector_expression", "scoped_identifier",
})
#: wrappers whose children hold a declared name rather than a value
_DECLARATION_WRAPPERS = frozenset({
    "parameters", "formal_parameters", "parameter", "typed_parameter",
    "default_parameter", "typed_default_parameter", "optional_parameter",
    "variadic_parameter", "typed_variadic_parameter", "parameter_list",
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "method", "func_declaration", "function_item",
    "arrow_function", "lambda", "lambda_parameters", "catch_formal_parameter",
})
#: destructuring patterns, which bind a name per element
_PATTERN_NODES = frozenset({
    "pattern_list", "tuple_pattern", "list_pattern", "expression_list",
    "tuple", "list", "array_pattern", "slice_pattern",
})
#: the receiver a language gives its own instances.  These are the language's
#: token, not the author's: ``self``/``this`` are keywords or convention that
#: the grammar and every linter agree on, so keying on them is structural in
#: the same sense a node type is.  An author-named receiver is not an instance
#: of anything the unit owns, and used to read as one.
_SELF_RECEIVERS: dict[str, frozenset[str]] = {
    "python": frozenset({"self", "cls"}),
    "java": frozenset({"this", "super"}),
    "javascript": frozenset({"this"}),
    "rust": frozenset({"self"}),
    "go": frozenset(),
}


def _writes_own_field(left, spec: _LangSpec, unit_kind: str) -> bool:
    """True when an assignment target is a field on the unit's own receiver.

    Structural, not spelling: the receiver is taken from the grammar's
    ``object``/``expression`` field and matched against the language's
    documented instance name.  The old test was a regex over the rendered
    text, so ``me.foo = 1`` counted as writing an owned field while
    ``instance.foo = 1`` did not, and the answer moved whenever the author
    renamed a local.

    The unit must also be a method.  A free function taking a parameter named
    ``cls`` owns nothing -- it is handed the object and mutates it -- and
    calling that an owned field is the word ``cls`` deciding the semantics.
    """
    if unit_kind != "method":
        return False
    for receiver_field in ("object", "expression", "value"):
        receiver = left.child_by_field_name(receiver_field)
        if receiver is None:
            continue
        return _text(receiver) in _SELF_RECEIVERS.get(spec.ts_name, frozenset())
    return False


def _text(node) -> str:
    return node.text.decode("utf-8", "replace")


def _assign_sides(node) -> tuple[Any, Any]:
    """An assignment's target and value, across the five grammars' spellings."""
    for left_field, right_field in (
        ("left", "right"),
        ("pattern", "value"),
        ("name", "value"),
    ):
        left = node.child_by_field_name(left_field)
        right = node.child_by_field_name(right_field)
        if left is not None:
            return left, right
    for child in node.children:
        if child.is_named and child.type in {
            "variable_declarator", "var_spec", "const_spec", "expression_list"
        }:
            left = child.child_by_field_name("name")
            right = child.child_by_field_name("value")
            if left is not None:
                return left, right
            names = [c for c in child.children if c.is_named]
            if names:
                return names[0], names[-1] if len(names) > 1 else None
    named = [c for c in node.children if c.is_named]
    if len(named) >= 2:
        return named[0], named[-1]
    return (named[0] if named else None), None


def _identifiers(node) -> set[str]:
    """Every identifier-shaped token in a subtree (internal use only)."""
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _IDENT_TYPES:
            found.add(_text(current))
        stack.extend(current.children)
    return found


def _value_identifiers(node, spec: _LangSpec) -> set[str]:
    """Identifiers naming a value an expression consumes.

    The ``url`` in ``self.url`` is the object's API, not one of the unit's
    own bindings, so it must not be mistaken for a parameter that merely
    shares the name.  That coincidence linked ``self.url`` to a parameter
    ``url`` and credited the parameter with a chain it never had; renaming
    the parameter broke the illusion and the IR moved.
    """
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _IDENT_TYPES \
                and not _is_member_name(current, spec):
            found.add(_text(current))
        stack.extend(current.children)
    return found


_OP_CHARS = frozenset("+-*/%<>=!&|^~")
#: operators that make a value a *derived quantity* rather than a pass-through
_ARITH_OPS = frozenset({"+", "-", "*", "/", "%", "++", "--", "+=", "-=", "*=",
                        "/=", "%="})


def _operator_tokens(node) -> set[str]:
    """Collect operator token texts found anywhere under ``node``."""
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child in current.children:
            raw = _text(child)
            if not child.is_named:
                if raw and len(raw) <= 3 and set(raw) <= _OP_CHARS:
                    found.add(raw)
            elif child.type.endswith("_operator"):
                found.update(
                    part
                    for part in re.split(r"[^-+*/%<>=!&|^~]+", raw)
                    if part
                )
            else:
                stack.append(child)
    return found


#: node types that dispatch on a value's type/kind by grammar, not by text
_MATCH_SWITCH_NODES = frozenset({
    "match_expression", "switch_expression", "switch_statement",
    "expression_switch_statement",
})
_TYPE_TEST_TOKENS = frozenset({"instanceof", "typeof"})
_TYPE_TEST_CALLS = frozenset({"isinstance", "type"})


def _is_type_branch(node, cond, spec: _LangSpec) -> bool:
    """True when the branch dispatches on the type or kind of a value.

    Structural, not textual.  A Python local named ``match`` is not a match
    expression, and a regex over the branch text that could not tell the two
    apart made the IR move whenever the local was renamed: ``if match is
    None`` looked like a type dispatch.
    """
    if node.type in _MATCH_SWITCH_NODES or cond.type in _MATCH_SWITCH_NODES:
        return True
    stack = [cond]
    while stack:
        current = stack.pop()
        if not current.is_named:
            if _text(current) in _TYPE_TEST_TOKENS:
                return True
            continue
        if current.type in spec.call_nodes:
            for field in ("function", "name"):
                target = current.child_by_field_name(field)
                if target is not None and target.type in _IDENT_TYPES \
                        and _text(target) in _TYPE_TEST_CALLS:
                    return True
        stack.extend(current.children)
    return False


def _cue_for(segments: list[str]) -> tuple[str, str] | None:
    """Longest-match a normalized callee path against the cue table."""
    best: str | None = None
    for i in range(len(segments)):
        key = ".".join(segments[i:])
        if key in _CUE_MAP and (best is None or len(key) > len(best)):
            best = key
    return _CUE_MAP[best] if best else None


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


@dataclass
class _Facts:
    transformations: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    state_changes: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    observable_effects: list[str] = field(default_factory=list)
    error_signals: list[str] = field(default_factory=list)
    params: list[tuple[str, str]] = field(default_factory=list)
    param_chains: dict[str, list[str]] = field(default_factory=dict)
    returns: list[Any] = field(default_factory=list)
    has_value_return: bool = False
    has_early_return: bool = False
    has_loop: bool = False
    has_mapping_decl: bool = False
    has_lookup: bool = False
    has_store: bool = False
    has_retained_store: bool = False
    has_await: bool = False
    has_construct: bool = False
    has_closure: bool = False
    #: transformation that directly produced each returned value
    return_cues: list[str] = field(default_factory=list)
    #: locals assigned from an arithmetic expression
    derived: set[str] = field(default_factory=set)
    #: the unit's declared result type names a failure value
    declared_failure: bool = False


class _Scanner:
    """One forward pass over a single unit's subtree."""

    def __init__(self, unit, spec: _LangSpec) -> None:
        self.unit = unit
        self.spec = spec
        self.facts = _Facts()
        self.unit_kind = _classify_unit_kind(unit, spec)
        name_node = unit.child_by_field_name("name")
        self.local_names = _scope_bindings(
            unit, spec, _text(name_node) if name_node is not None else ""
        )
        # local variable name -> transformations already applied to its value
        self.env: dict[str, list[str]] = {}
        # local variable name -> parameters its value ultimately came from
        self.env_params: dict[str, set[str]] = {}
        self.loop_depth = 0

    # -- entry points -----------------------------------------------------

    def run(self) -> _Facts:
        self._collect_params()
        self._walk(self.unit)
        self._collect_tail_value()
        self._resolve_cache()
        return self.facts

    def _collect_params(self) -> None:
        for child in self.unit.children:
            if child.type not in self.spec.param_nodes:
                continue
            for entry in child.children:
                if not entry.is_named or entry.type in {"variadic_parameter"}:
                    continue
                name = self._declared_name(entry)
                if not name:
                    continue
                type_node = entry.child_by_field_name("type")
                self.facts.params.append(
                    (name, _text(type_node) if type_node is not None else "")
                )

    def _declared_name(self, entry) -> str:
        if entry.type in {"identifier", "identifier_pattern"}:
            return _text(entry)
        for key in ("name", "pattern", "left"):
            target = entry.child_by_field_name(key)
            if target is None:
                continue
            names = _identifiers(target)
            return sorted(names)[0] if names else ""
        return ""

    def _collect_tail_value(self) -> None:
        """Rust (and friends) return their value as a bare tail expression."""
        if not self.spec.tail_value or self.facts.returns:
            return
        block = next(
            (c for c in self.unit.children if c.type in {"block", "body"}),
            None,
        )
        if block is None:
            return
        named = [c for c in block.children if c.is_named]
        if named and named[-1].type in {
            "expression_statement",
            "call_expression",
            "macro_invocation",
            "binary_expression",
            "identifier",
            "try_expression",
        }:
            self.facts.returns.append(named[-1])
            self.facts.has_value_return = True

    def _resolve_cache(self) -> None:
        """Cache = a key lookup plus a store into state the unit did not build."""
        if not (self.facts.has_lookup and self.facts.has_store):
            return
        if self.facts.has_mapping_decl or self.facts.has_retained_store:
            _add(self.facts.domains, "caching")
            _add(
                self.facts.state_changes,
                "retains a computed value under a key for reuse by later calls",
            )

    # -- traversal --------------------------------------------------------

    def _walk(self, node) -> None:
        node_type = node.type
        if node_type in self.spec.loop_nodes:
            self.facts.has_loop = True
            self.loop_depth += 1
            for child in node.children:
                self._walk(child)
            self.loop_depth -= 1
            return
        if node_type in self.spec.call_nodes:
            self._on_call(node)
        elif node_type in self.spec.return_nodes:
            self._on_return(node)
        elif node_type in self.spec.cond_nodes:
            self._on_condition(node)
        elif node_type in self.spec.throw_nodes:
            _add(self.facts.error_signals, "raises")
        elif node_type in self.spec.catch_nodes | self.spec.try_nodes:
            _add(self.facts.error_signals, "catches")
        elif node_type in self.spec.assign_nodes:
            self._on_assign(node)
        elif node_type in self.spec.await_nodes:
            self.facts.has_await = True
        elif node_type in self.spec.closure_nodes:
            self.facts.has_closure = True

        if node_type in _IDENT_TYPES and _norm(_text(node)) in _MAPPING_TYPES:
            self.facts.has_mapping_decl = True
        if node_type in {"dictionary", "object", "composite_literal"}:
            self.facts.has_mapping_decl = True
        if node_type in {
            "object_creation_expression",
            "struct_expression",
            "new_expression",
            "array_creation_expression",
        }:
            self.facts.has_construct = True
        if node_type == "try_expression":
            _add(self.facts.error_signals, "propagates")

        for child in node.children:
            self._walk(child)

    # -- calls ------------------------------------------------------------

    def _callee_segments(self, node) -> list[str]:
        # Java (and Kotlin-shaped) grammars put the receiver in a separate
        # field from the method name, so the dotted path has to be rebuilt.
        receiver = node.child_by_field_name("object")
        target = None
        for key in ("function", "name", "macro", "type", "constructor"):
            target = node.child_by_field_name(key)
            if target is not None:
                break
        if receiver is not None and target is not None:
            raw = f"{_text(receiver)}.{_text(target)}"
        elif target is None:
            return []
        elif target.type in _IDENT_TYPES and not _is_member_name(target, self.spec) \
                and _text(target) in self.local_names:
            # A bare callee the unit itself binds (a nested def or a
            # parameter) is the author's function, not a library API.  Cueing
            # on its name made ``filter(items)`` -- a local -- read as the
            # builtin ``filter``, so the IR moved with the local's name.
            return []
        else:
            raw = _text(target)
        parts = [p for p in re.split(r"[.:]+", raw) if p]
        return [s for s in (_norm(p) for p in parts) if s][-3:]

    def _on_call(self, node) -> None:
        """Classify the whole call, including anything its callbacks do."""
        hits: list[tuple[str, str, str]] = []  # (cue key, phrase, domain)
        for descendant in self._calls_under(node):
            segments = self._callee_segments(descendant)
            found = _cue_for(segments) if segments else None
            if found:
                hits.append((segments[-1], found[0], found[1]))
        if not hits:
            return

        chain: list[str] = []
        for key, phrase, domain in hits:
            _add(chain, phrase)
            self._record_effect(phrase, domain)
            if domain:
                _add(self.facts.domains, domain)
            if key in _STORE_CUES:
                self.facts.has_store = True
                if len(self._callee_segments(node)) > 1:
                    # The mapping is a receiver, not something built here.
                    self.facts.has_retained_store = True
                _add(self.facts.state_changes,
                     "stores a computed value under a key in a mapping")
            if key in _LOOKUP_CUES:
                self.facts.has_lookup = True

        # Link the transformation chain to the parameters it consumed, and
        # prepend whatever transformations already produced those operands.
        used = _value_identifiers(node, self.spec) - self._closure_params(node)
        upstream: list[str] = []
        for ident in sorted(used):
            upstream.extend(self.env.get(ident, ()))
        # A parameter reaches this call either by appearing here or by way of
        # a local that was itself computed from it.
        reached = {n for n, _ in self.facts.params if n in used}
        for ident in sorted(used):
            reached |= self.env_params.get(ident, set())
        for name in sorted(reached):
            linked = self.facts.param_chains.setdefault(name, [])
            for prior in upstream:
                _add(linked, prior)
            for step in chain:
                _add(linked, step)

    def _calls_under(self, node) -> list[Any]:
        """The node itself plus every call nested inside it, outermost first."""
        found: list[Any] = []
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current.type in self.spec.call_nodes:
                found.append(current)
            queue.extend(current.children)
        return found

    def _closure_params(self, node) -> set[str]:
        """Names bound by a callback inside the call, not the caller's inputs."""
        names: set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current is not node and current.type in self.spec.closure_nodes:
                for child in current.children:
                    if child.type in self.spec.param_nodes:
                        names |= _identifiers(child)
            stack.extend(current.children)
        return names

    def _record_effect(self, phrase: str, domain: str) -> None:
        _add(self.facts.transformations, phrase)
        if domain in {
            _D_FILE, _D_HTTP, _D_PROC, _D_LOG, _D_CONSOLE, _D_DB, _D_TIME
        }:
            _add(self.facts.side_effects, phrase)
        effect = {
            _D_CONSOLE: "emits output visible on the console or terminal",
            _D_LOG: "leaves a diagnostic record in a log",
            _D_HTTP: "exchanges data with a remote service over the network",
            _D_PROC: "starts or controls an external process",
            _D_DB: "reads from or writes to a database",
            _D_TIME: "advances or observes the passage of time",
        }.get(domain)
        if effect:
            _add(self.facts.observable_effects, effect)
        elif domain == _D_FILE:
            writes = phrase.startswith(
                ("writes", "creates", "removes", "renames")
            )
            _add(
                self.facts.observable_effects,
                "produces or modifies a file on disk"
                if writes
                else "depends on files on disk",
            )

    # -- returns ----------------------------------------------------------

    def _on_return(self, node) -> None:
        value = node.child_by_field_name("value")
        if value is None:
            # Go and friends list the returned expressions positionally.
            value = next(
                (c for c in node.children if c.is_named and c.type != "comment"),
                None,
            )
        if value is not None and _text(value).strip(" ;\t\n"):
            self.facts.has_value_return = True
            self.facts.returns.append(value)
            own = self._own_cues(value)
            self.facts.return_cues.append(own[0] if own else "")
        else:
            self.facts.returns.append(None)
            self.facts.return_cues.append("")
        parent = node.parent
        if (
            parent is not None
            and parent.type in self.spec.cond_nodes
            and self.loop_depth == 0
        ):
            self.facts.has_early_return = True

    # -- conditions -------------------------------------------------------

    def _on_condition(self, node) -> None:
        found: list[str] = []
        cond = node.child_by_field_name("condition") or node
        ops = _operator_tokens(cond)
        text = _text(cond)
        if re.search(r"\b(nil|null|None|undefined)\b", text) and (
            ops & {"!=", "==", ">", "<", "===", "!=="}
        ):
            _add(found, "branches on whether an operation succeeded or failed")
        if re.search(
            r"containsKey|contains_key|hasOwnProperty|existsSync|getOrDefault",
            text,
        ):
            _add(found, "branches on whether a stored entry is present")
        if ops & {"==", "!=", "===", "!==", "<", ">", "<=", ">="}:
            _add(found, "branches on a value comparison")
        if re.search(r"\bcontains\(|\bin\b\s*[\[(]|includes\(", text):
            _add(found, "branches on membership in a collection")
        if _is_type_branch(node, cond, self.spec):
            _add(found, "branches on a type or kind of value")
        if re.search(
            # isEmpty is a real API; the snake_case spelling is a Python local
            # name, and keying on that would make the IR depend on it.
            r"isEmpty|\.empty\b|len\([^)]*\)\s*==\s*0|!\w*\.?length",
            text,
        ):
            _add(found, "branches on whether an input is empty")
        if not found:
            _add(found, "branches on a predicate")
        for item in found:
            _add(self.facts.conditions, item)

    # -- assignments ------------------------------------------------------

    def _on_assign(self, node) -> None:
        left, right = _assign_sides(node)
        if left is None:
            return
        # For ``obj.field = x`` the thing being rebound is ``obj``; counting
        # the field name put it in no binding class at all, so a write to a
        # field of a caught exception fell through to "computes a local
        # value" and the phrase depended on what the field was called.
        target_names = _bound_names(left)
        left_names = _identifiers(left)
        param_names = {p for p, _ in self.facts.params}
        produced = self._transformation_chain(right) if right is not None else []
        fed_by = self._feeds_params(right) if right is not None else set()
        # State the unit did not build itself cannot be a local cache.
        retained = bool(left_names) and not (left_names & param_names) and not (
            left_names & set(self.env)
        )
        if right is not None and _ARITH_OPS & _operator_tokens(right):
            self.facts.derived |= left_names
        for name in left_names:
            self.env[name] = list(produced)
            self.env_params[name] = fed_by

        if left.type in _ATTRIBUTE_TARGETS and _writes_own_field(
            left, self.spec, self.unit_kind
        ):
            _add(self.facts.state_changes, "writes a field of an owned object")
        elif left.type in {
            "index_expression", "subscript", "subscript_expression", "array_access"
        }:
            self.facts.has_store = True
            if retained:
                self.facts.has_retained_store = True
            _add(
                self.facts.state_changes,
                "stores a computed value under a key in a retained mapping",
            )
        elif target_names & param_names:
            _add(
                self.facts.state_changes,
                "reassigns an incoming value before using it",
            )
        elif self.loop_depth > 0 and produced:
            _add(
                self.facts.state_changes,
                "grows a collection as items are processed",
            )
        elif self.loop_depth > 0:
            _add(
                self.facts.state_changes,
                "accumulates or refines a value across iterations",
            )
        else:
            _add(
                self.facts.state_changes, "computes a local value from its inputs"
            )

    def _own_cues(self, node) -> list[str]:
        """Cue phrases applied by calls inside ``node``, outermost first."""
        own: list[str] = []
        for call in self._calls_under(node):
            segments = self._callee_segments(call)
            hit = _cue_for(segments) if segments else None
            if hit:
                _add(own, hit[0])
        return own

    def _transformation_chain(self, node) -> list[str]:
        """What happened to the operands, then what this expression did.

        Order matters: the chain is meant to read as
        ``input -> ... -> output``, so what produced the operands has to come
        before what the operands were then put through.
        """
        chain: list[str] = []
        for ident in sorted(_value_identifiers(node, self.spec)):
            for prior in self.env.get(ident, ()):
                _add(chain, prior)
        for phrase in self._own_cues(node):
            _add(chain, phrase)
        return chain

    def _feeds_params(self, node) -> set[str]:
        """Parameters (directly or through a local) that produced ``node``."""
        used = _value_identifiers(node, self.spec)
        params = {name for name, _ in self.facts.params} & used
        for ident in used:
            params |= self.env_params.get(ident, set())
        return params


# ---------------------------------------------------------------------------
# Output classification
# ---------------------------------------------------------------------------

_COLLECTION_NODES = frozenset({
    "list_comprehension", "array_expression", "list_literal",
    "array_creation_expression", "comprehension_expression", "tuple",
})
_COLLECTION_TEXT = re.compile(
    r"\[\s*[^\[\]]*for\s|for\s+\w+\s+in\s|\.(map|filter)\("
)
_LOOKUP_TEXT = re.compile(
    r"\b(get|getOrDefault|lookup|readValue|getString|getInt)\s*\(|"
    r"^\s*[\w.]+\s*\["
)
_ARITH_TEXT = re.compile(r"[-+*/%]|\+\+|--")
_FAILURE_RETURN = (
    "returns a result together with an explicit failure value",
    "a result paired with a failure value",
)
_NO_VALUE = (
    "performs its work as a side effect and returns nothing",
    "no returned value",
)
#: what the produced value actually IS, keyed off the transformation that made
#: it.  Section 9 lives or dies on this: "returns a value" is useless, "returns
#: a collection of transformed items" is searchable.
_OUTPUT_NOUNS: tuple[tuple[str, str], ...] = (
    ("joins a collection", "one delimited text"),
    ("transforms each item", "a collection of transformed items"),
    ("transforms and flattens", "a flattened collection of transformed items"),
    ("keeps only the items", "a filtered collection of items"),
    ("orders the items", "an ordered collection of items"),
    ("groups the items", "a grouping of items by a derived key"),
    ("folds a collection", "one accumulated value"),
    ("aggregates numeric items", "a single numeric total"),
    ("splits textual content into lines", "a list of text lines"),
    ("splits textual content on a delimiter", "a list of delimited text parts"),
    ("trims surrounding whitespace from text", "whitespace-trimmed text"),
    ("parses JSON data", "a structured record"),
    ("serializes data to JSON text", "JSON text"),
    ("looks up a value by key", "a looked-up value"),
    ("reads a file's bytes into memory", "the file's contents"),
    ("reads a file as text", "the file's text"),
    ("writes data to a file or stream", "written output"),
)


def _output_noun(cue: str) -> str:
    for prefix, noun in _OUTPUT_NOUNS:
        if cue.startswith(prefix):
            return noun
    return "a value derived from its inputs"


def _classify_output(facts: _Facts, cache: bool, spec: _LangSpec) -> tuple[str, str]:
    """Return (output_behavior sentence, short output noun for relations)."""
    if cache:
        return (
            "returns a previously stored value when a validity condition "
            "holds, otherwise computes a new value and stores it for reuse",
            "a cached or freshly computed value",
        )
    values = [v for v in facts.returns if v is not None]
    if not values:
        return _NO_VALUE

    # A declared failure type is stronger evidence than anything the returned
    # expression happens to look like, so it is checked first.
    returned = " | ".join(_text(v).strip() for v in values)
    if (
        facts.declared_failure
        or _FAILS.search(returned)
        or any(
            len(_text(v).split(",")) >= 2 for v in values
        )
    ):
        return _FAILURE_RETURN

    for node, cue in zip(values, facts.return_cues, strict=False):
        text = _text(node).strip()
        if cue:
            return (
                f"returns a value produced by {cue}",
                _output_noun(cue),
            )
        if node.type in _COLLECTION_NODES or _COLLECTION_TEXT.search(text):
            return (
                "builds and returns a collection of items derived from its "
                "inputs",
                "a collection of derived items",
            )
        if _LOOKUP_TEXT.search(text) or node.type in {
            "subscript", "index_expression", "subscript_expression",
            "array_access",
        }:
            return (
                "returns a value looked up from stored data by key",
                "a looked-up value",
            )
        if re.fullmatch(r"(true|false)", text):
            return ("returns a boolean outcome", "a boolean outcome")
        if _ARITH_TEXT.search(text) and (
            {p for p, _ in facts.params} & _value_identifiers(node, spec)
        ):
            return (
                "computes and returns a derived quantity from its inputs",
                "a derived quantity",
            )
        if text in facts.derived:
            return (
                "computes and returns a derived quantity from its inputs",
                "a derived quantity",
            )
        if text in {p for p, _ in facts.params}:
            return ("returns one of its inputs unchanged", "an input value")
        if re.fullmatch(r"[\"'].*[\"']|\d+", text):
            return ("returns a fixed constant value", "a constant value")
    if facts.has_early_return:
        return (
            "returns early on a condition and otherwise returns a later "
            "computed value",
            "an early or computed value",
        )
    return (
        "returns a value derived from its inputs",
        "a value derived from its inputs",
    )


# ---------------------------------------------------------------------------
# Input roles
# ---------------------------------------------------------------------------

#: A parameter's role from its *declared* type.  Structure is checked before
#: type names because "&[i32]" is a collection, not a number, and the
#: normalized form of every one of these collapses to the same token.
_MAPPING_SHAPE = re.compile(r"\bmap\b|dict|hashmap|treemap|record|\[\s*string\s*:\s*")
_COLLECTION_SHAPE = re.compile(r"\[|&|\bvec\b|slice|\blist\b|iterable|buffer")
_TYPE_ROLES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"string", "str", "char", "varchar", "bytestring",
                "stringliteral"}), "textual input"),
    (frozenset({"int", "int32", "int64", "i32", "i64", "u32", "u64", "long",
                "short", "byte", "integer", "float", "float32", "float64",
                "double", "number", "usize", "size"}), "numeric input"),
    (frozenset({"list", "slice", "array", "vec", "sequence", "iterable",
                "buffer", "set"}), "collection input"),
    (frozenset({"map", "dict", "hashmap", "mapping", "object", "record",
                "table"}), "mapping input"),
    (frozenset({"bool", "boolean"}), "flag input"),
    (frozenset({"io", "result", "option", "context", "generator", "coroutine",
                "channel", "stream", "error"}), "effectful input"),
)


def _declared_role(declared: str) -> str | None:
    if not declared:
        return None
    if _MAPPING_SHAPE.search(declared):
        return "mapping input"
    if _COLLECTION_SHAPE.search(declared):
        return "collection input"
    normalized = _norm(declared)
    for tokens, label in _TYPE_ROLES:
        if normalized and any(t in normalized for t in tokens):
            return label
    return None


def _input_roles(facts: _Facts) -> list[str]:
    roles: list[str] = []
    for name, declared in facts.params:
        role = _declared_role(declared)
        if role is None:
            # No usable type: infer the role from what the unit does to it.
            applied = " ".join(facts.param_chains.get(name, ()))
            if re.search(r"text|line|case|prefix|suffix|whitespace", applied):
                role = "textual input"
            elif re.search(r"collection|items|entries|iterat", applied):
                role = "collection input"
            elif re.search(r"mapping|key|stored|lookup", applied):
                role = "mapping input"
            elif re.search(r"file|stream", applied):
                role = "file or stream input"
            elif re.search(r"http|remote|service|endpoint", applied):
                role = "remote resource input"
            role = role or _OPAQUE
        _add(roles, role)
    return roles


# ---------------------------------------------------------------------------
# Relations (Section 9: the reason a flat operation list is not enough)
# ---------------------------------------------------------------------------


def _build_relations(
    facts: _Facts, roles: list[str], output_noun: str
) -> list[str]:
    """input -> transformation -> output chains, one per linked input."""
    relations: list[str] = []
    for (name, _declared), role in zip(facts.params, roles, strict=False):
        chain = list(dict.fromkeys(facts.param_chains.get(name, ())))
        if not chain:
            continue
        _add(relations, f"{role} -> {' -> '.join(chain)} -> returns {output_noun}")
    if relations:
        return relations
    # Say so rather than implying a link that the structure does not show.
    if facts.transformations:
        detail = f"performs {facts.transformations[0]}"
    elif facts.params:
        detail = "passes its inputs through unchanged"
    else:
        return relations
    _add(
        relations,
        f"no input-to-output link is visible; the unit {detail} and returns "
        f"{output_noun}",
    )
    return relations


# ---------------------------------------------------------------------------
# Error behavior
# ---------------------------------------------------------------------------

_FAILURE_TYPES = re.compile(r"\b(result|option|error|either)\b", re.IGNORECASE)


def _declared_result(unit, spec: _LangSpec) -> str:
    """The unit's declared result type, read from the grammar's own field.

    Reading it structurally (rather than grepping the body) is what keeps
    "this unit reports failure" separate from "this unit mentions the word
    error somewhere".
    """
    if spec.result_field is None:
        return ""
    node = unit.child_by_field_name(spec.result_field)
    return _text(node) if node is not None else ""


def _error_behavior(facts: _Facts, unit_text: str) -> str:
    signals = set(facts.error_signals)
    parts: list[str] = []
    if "propagates" in signals:
        parts.append("propagates a failure from a nested operation to its caller")
    if re.search(r"\.unwrap\(|\.expect\(", unit_text):
        parts.append("assumes an operation succeeded and aborts on failure")
    if "raises" in signals:
        parts.append("signals a failure to its caller by raising")
    if "catches" in signals:
        parts.append("catches failures and falls back to a defined result")
    if facts.declared_failure:
        parts.append("reports failure in a dedicated return value")
    if re.search(r"panic!|\bos\.Exit\b", unit_text):
        parts.append("aborts execution on an unrecoverable failure")
    return "; ".join(dict.fromkeys(parts)) or _NONE_HANDLING


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_semantic_ir(
    source_code: str,
    language: str,
    unit_kind: str | None = None,
    imports: list[str] | None = None,
) -> SemanticIR:
    """Build the Semantic IR for one code unit.

    Parameters
    ----------
    source_code:
        Source of a single function/method.  Leading indentation (how
        method-nested units are stored) is normalized first, so an indented
        but valid unit is not mistaken for malformed source.
    language:
        One of :data:`SUPPORTED_LANGUAGES`.
    unit_kind:
        Optional ``"function"``/``"method"`` override.  A unit extracted
        without its enclosing class cannot be told from a free function by
        the tree alone, so the dataset's own ``symbol_type`` is the
        authoritative answer when the caller already has it.
    imports:
        Optional module-path list from the unit's file, used to ground
        domain signals the way the production extractor does.

    Never raises on malformed source: a unit that does not parse cleanly
    comes back with ``parse_ok=False`` plus whatever evidence survived.  An
    unsupported ``language`` is a caller bug and raises.
    """
    if language not in _SPECS:
        raise ValueError(
            f"unsupported language {language!r}; the pilot covers "
            f"{', '.join(SUPPORTED_LANGUAGES)}"
        )
    spec = _SPECS[language]
    text = textwrap.dedent(source_code)
    if not text.strip():
        return SemanticIR(
            language=language,
            unit_kind=unit_kind or "function",
            parse_ok=False,
            input_roles=(),
            output_behavior="no source text was supplied to describe",
            domain_signals=(),
            transformations=(),
            relations=(),
            conditions=(),
            state_changes=(),
            side_effects=(),
            error_behavior="not determinable without a parsed unit",
            observable_effects=(),
        )
    tree = _parser(language).parse(text.encode("utf-8"))
    root = tree.root_node

    unit = _find_unit(root, spec)
    if unit is None:
        return SemanticIR(
            language=language,
            unit_kind=unit_kind or "function",
            parse_ok=not root.has_error,
            input_roles=(),
            output_behavior="source exposes no callable unit to describe",
            domain_signals=(),
            transformations=(),
            relations=(),
            conditions=(),
            state_changes=(),
            side_effects=(),
            error_behavior="not determinable without a parsed unit",
            observable_effects=(),
        )

    scanner = _Scanner(unit, spec)
    facts = scanner.run()
    facts.declared_failure = bool(_FAILURE_TYPES.search(_declared_result(unit, spec)))
    for module in imports or []:
        parts = [s for s in (_norm(p) for p in re.split(r"[.:/]+", module)) if s]
        hit = _cue_for(parts[-3:]) if parts else None
        if hit and hit[1]:
            _add(facts.domains, hit[1])

    if facts.has_construct:
        _add(facts.transformations, "constructs a new value of a given shape")
    if facts.has_await:
        _add(facts.transformations,
             "waits on an asynchronous result before continuing")
    if facts.has_loop:
        _add(facts.transformations,
             "repeats its work over a collection of items")
    if facts.has_closure:
        _add(facts.transformations,
             "delegates per-item work to an inline callback")

    cache = "caching" in facts.domains
    output, output_noun = _classify_output(facts, cache, spec)
    roles = _input_roles(facts)
    return SemanticIR(
        language=language,
        unit_kind=unit_kind or _classify_unit_kind(unit, spec),
        parse_ok=not root.has_error,
        input_roles=tuple(roles),
        output_behavior=output,
        domain_signals=tuple(facts.domains),
        transformations=tuple(facts.transformations),
        relations=tuple(_build_relations(facts, roles, output_noun)),
        conditions=tuple(facts.conditions),
        state_changes=tuple(facts.state_changes),
        side_effects=tuple(facts.side_effects),
        error_behavior=_error_behavior(facts, _text(unit)),
        observable_effects=tuple(facts.observable_effects),
    )


def _find_unit(root, spec: _LangSpec):
    """First callable unit in the tree, or ``None``."""
    if root.type in spec.unit_nodes:
        return root
    queue = list(root.children)
    while queue:
        node = queue.pop(0)
        if node.type in spec.unit_nodes:
            return node
        queue.extend(node.children)
    return None


def _classify_unit_kind(unit, spec: _LangSpec) -> str:
    if unit.type in spec.method_units:
        return "method"
    parent = unit.parent
    if parent is not None and parent.type in spec.method_parents:
        return "method"
    return "function"


def informativeness(ir: SemanticIR) -> Informativeness:
    """Count the distinct semantic anchors an IR carries.

    EXPERIMENTAL DIAGNOSTIC ONLY — no threshold is applied and nothing here
    feeds the production candidate pool.  Each field contributes one anchor
    per distinct item, so a unit cannot inflate its score by repeating a
    single operation.
    """
    produced = not ir.output_behavior.startswith("performs its work")
    breakdown = {
        "input_roles": len(set(ir.input_roles)),
        "output_behavior": int(produced),
        "domain_signals": len(set(ir.domain_signals)),
        "transformations": len(set(ir.transformations)),
        "relations": len(set(ir.relations)),
        "conditions": len(set(ir.conditions)),
        "state_changes": len(set(ir.state_changes)),
        "observable_effects": len(set(ir.observable_effects)),
        "error_behavior": int(not ir.error_behavior.startswith("no explicit")),
    }
    return Informativeness(
        anchor_count=sum(breakdown.values()), breakdown=breakdown
    )


# ---------------------------------------------------------------------------
# Provenance guard (Section 6)
# ---------------------------------------------------------------------------

#: names that must survive renaming, because they are the language's own
#: vocabulary rather than something the author chose
#: subtree types that hold a declared type; nothing inside is an author name
_TYPE_CONTEXTS = frozenset({
    "type", "union_type", "generic_type", "type_parameter", "sized_type",
    "type_annotation", "type_identifier", "scoped_type_identifier",
    "primitive_type", "integral_type", "floating_point_type", "boolean_type",
    "void_type", "array_type", "object_type", "function_type",
})
#: the receiver segment of every receiver-qualified cue key.  A name the
#: detector itself keys off (json.dumps, logging.error) is a library's name,
#: not the author's, so the renamer must leave it alone -- exactly as the
#: production extractor treats module imports.  Derived from the table so a new
#: qualified cue cannot silently reintroduce the dependency.
_CUE_RECEIVERS = frozenset(
    key.split(".")[0] for key in _CUE_MAP if "." in key
) | {"json"}
#: Names the renamer must not touch even when the unit binds them.  This used
#: to be one flat list of every keyword in all five languages, which quietly
#: protected real author words: a Python unit with a parameter called ``path``,
#: ``error`` or ``result`` was only half renamed, so the guard compared two
#: different programs and the leftovers were invisible.  Language keywords no
#: longer need listing -- the grammar does not parse them as identifiers, so
#: they can never reach here.  What remains is the two things that are
#: genuinely not the author's to rename: the instance receivers the
#: classifiers key off, and the library names in the cue table.
_RESERVED = _CUE_RECEIVERS | frozenset(
    name for receivers in _SELF_RECEIVERS.values() for name in receivers
)
_RENAME_TOKEN = "qqqqzzz"
#: node types whose text is a type/field/property name: renaming those changes
#: the program's meaning, so they are left alone
_NOT_RENAMEABLE = frozenset({
    "field_identifier", "property_identifier", "class_name", "type_identifier",
    "primitive_type", "integral_type", "floating_point_type", "boolean_type",
    "void_type", "scoped_type_identifier", "generic_type", "type_annotation",
})
#: when this many words accumulate into one identifier it is a phrase, not a
#: name, and the IR must never contain it verbatim
_PHRASE_FLOOR = 3


def source_identifiers(source_code: str, min_length: int = 4) -> set[str]:
    """Identifier-shaped tokens of at least ``min_length`` characters."""
    return {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source_code)
        if len(token) >= min_length
    }


#: grammar fields that hold "the name after the dot"
_MEMBER_NAME_FIELDS = frozenset({
    "attribute", "field", "property", "method", "function", "macro",
    "constructor", "name", "type",
})
#: ...but only on a node that is a *member access*.  ``call`` is in
#: ``member_access_nodes`` so that ``x.foo()`` keeps ``foo``, and it shares
#: the ``function`` field with a bare ``f()``, whose name is a local binding
#: and must be renamed with every other reference to it.
_CALLEE_FIELDS = frozenset({"function"})


def _scope_bindings(unit, spec: _LangSpec, own_name: str) -> set[str]:
    """Names bound by this scope: its parameters, locals and nested defs.

    Not recursive: a nested definition's parameters and body belong to the
    nested scope, so only its *name* is collected here.  That distinction is
    what makes shadowing work.  A nested ``def filter(items)`` reusing the
    outer name is a different variable, and giving both the same stub merged
    them, so the scanner could no longer tell which one a phrase came from
    and the IR moved whenever the outer name changed.

    Alpha-renaming is only a *test* if the renamed program is the same
    program, so a local must be renamed at every reference -- including the
    one where it is called.  Renaming by name alone let ``name_factory`` be
    renamed in the signature and left alone at the call site, and the guard
    then compared two different programs.  So: rename what the unit binds,
    never what it merely references.  ``fetch`` and ``len`` are referenced,
    so they stay; ``factory`` is bound, so it goes everywhere.
    """
    names: set[str] = set()

    def walk(node, is_root: bool) -> None:
        if node.type in spec.unit_nodes and not is_root:
            # a nested definition's name binds here, in the enclosing scope;
            # its parameters and body belong to the scope it opens
            own = node.child_by_field_name("name")
            if own is not None and _text(own) != own_name:
                names.add(_text(own))
            return
        if node.type in spec.assign_nodes:
            left, right = _assign_sides(node)
            names.update(_bound_names(left))
            if right is not None:
                walk(right, False)
            return
        if node.type in spec.loop_nodes:
            for loop_field in ("left", "name", "identifier", "pattern"):
                target = node.child_by_field_name(loop_field)
                if target is not None:
                    names.update(_bound_names(target))
            for child in node.children:
                walk(child, False)
            return
        if node.type in spec.param_nodes:
            names.update(_declared_names(node, spec, own_name))
            for child in node.children:
                walk(child, False)
            return
        for child in node.children:
            walk(child, False)

    walk(unit, True)
    return names


def _declared_names(node, spec: _LangSpec, own_name: str) -> set[str]:
    """The names a parameter list or definition *declares*.

    A parameter is not a bare identifier: ``x: int = 5`` is a
    ``default_parameter`` wrapping the name, its type and its default.  Only
    the name binds -- the default's identifiers are values the unit
    references, and collecting them renamed free globals.
    """
    names: set[str] = set()

    def add(child) -> None:
        if child is None or child.type in _TYPE_CONTEXTS \
                or child.type in _NOT_RENAMEABLE:
            return
        if child.type in _IDENT_TYPES:
            name = _text(child)
            if name != own_name:
                names.add(name)
            return
        for declared_field in ("name", "pattern", "declarator", "left"):
            declared = child.child_by_field_name(declared_field)
            if declared is not None:
                add(declared)
                return
        if child.type in _DECLARATION_WRAPPERS or child.type in _PATTERN_NODES:
            for grandchild in child.children:
                add(grandchild)

    if node.type in spec.unit_nodes:
        own = node.child_by_field_name("name")
        if own is not None and _text(own) != own_name:
            names.add(_text(own))
        return names
    for child in node.children:
        add(child)
    return names


def _bound_names(target) -> set[str]:
    """The bindings an assignment target rebinds, ignoring any field name.

    ``self.count = 1`` rebinds nothing; ``items = ...`` rebinds ``items``;
    ``a, b = pair`` rebinds both.  Resolving to the base is what makes the
    phrase depend on the object being written to rather than on what its
    field happens to be called.
    """
    if target is None:
        return set()
    if target.type in _ATTRIBUTE_TARGETS:
        for receiver_field in ("object", "expression", "value"):
            receiver = target.child_by_field_name(receiver_field)
            if receiver is not None:
                return _bound_names(receiver)
        return set()
    return _identifiers(target)


def _field_of(node, parent) -> str | None:
    """The grammar field ``node`` fills in ``parent``, or None.

    py-tree-sitter hands back a fresh wrapper per access, so the node is
    located by position rather than by identity.
    """
    for index, child in enumerate(parent.named_children):
        if (child.start_byte, child.end_byte) == (node.start_byte, node.end_byte):
            return parent.field_name_for_named_child(index)
    return None


def _is_member_name(node, spec: _LangSpec) -> bool:
    """True for ``x.foo``'s ``foo``: the name side of a member access."""
    parent = node.parent
    if parent is None or parent.type not in spec.member_access_nodes:
        return False
    if parent.type in spec.call_nodes:
        # a call's ``function`` child is either the member (``x.foo``) or the
        # whole callee expression; only the former is a member name.
        return _field_of(node, parent) not in _CALLEE_FIELDS
    return _field_of(node, parent) in _MEMBER_NAME_FIELDS


def _renameable(node) -> bool:
    return node.type not in _NOT_RENAMEABLE


def rename_identifiers(
    source_code: str, language: str, token: str = _RENAME_TOKEN
) -> str:
    """Rewrite every author-chosen name in the unit to a unique stub.

    Tree-sitter edits are collected in document order and applied to the
    original text, so this round-trips exactly and needs no re-parse.  Three
    things are deliberately left alone:

    * names the language owns -- keywords, declared types, fields,
      properties -- and the name of anything being called (``splitlines``,
      ``get``, ``fetch``, ``len``), because renaming those changes what the
      program *does*, not what it is called.
    * the receivers the cue table keys off (``json``, ``logging``): those are
      library names, the same way module imports are in the production
      extractor.
    * the unit's own name, which is part of the dataset's identity.

    Each distinct name gets a *different* stub, so the aliasing structure the
    scanner depends on (which local came from which parameter) survives.
    """
    if language not in _SPECS:
        raise ValueError(f"unsupported language {language!r}")
    spec = _SPECS[language]
    text = textwrap.dedent(source_code)
    tree = _parser(language).parse(text.encode("utf-8"))
    unit = _find_unit(tree.root_node, spec)
    if unit is None:
        return text
    own = unit.child_by_field_name("name")
    own_name = _text(own) if own is not None else ""
    edits: list[tuple[int, int, str]] = []
    stubs: dict[str, str] = {}

    def stub(name: str) -> str:
        if name not in stubs:
            stubs[name] = f"{token}{len(stubs)}"
        return stubs[name]

    def emit(node, replacement: str) -> None:
        start = node.start_byte
        edits.append((start, node.end_byte - start, replacement))

    # Rename the unit's own bindings -- and only those -- uniformly: a name is
    # rewritten at every occurrence, binding or use.  Uniform renaming is a
    # true alpha-rename (binding resolution depends on the name, and every
    # occurrence of a name moves together), so a nested ``filter(items)`` that
    # shares an outer name stays shared and the scanner, which also keys on
    # the bare name, keeps seeing the same program.
    #
    # Free references are left alone: ``fetch`` and ``len`` are not bound here,
    # so renaming them would rename a *library* and change what the program
    # does.  The set of bound names is what separates the two.
    bound = _scope_bindings(unit, spec, own_name)

    def visit(node) -> None:
        if node.type in _TYPE_CONTEXTS:
            return
        if node.type == "identifier" and _renameable(node) \
                and not _is_member_name(node, spec):
            name = _text(node)
            if name in bound and name not in _RESERVED:
                emit(node, stub(name))
            return
        for child in node.children:
            visit(child)

    visit(unit)
    # Edits are byte offsets into the encoded source, so the splice has to
    # happen there too.  Slicing the decoded str with them silently shifted
    # every identifier after the first non-ASCII character -- a '✓' in a list
    # literal is 3 bytes and 1 char -- and produced source that did not parse.
    # That is the same class of bug: a mangled program scores as an IR
    # difference and reads like a classifier leak.
    data = text.encode("utf-8")
    for start, length, replacement in sorted(edits, reverse=True):
        data = data[:start] + replacement.encode("utf-8") + data[start + length:]
    return data.decode("utf-8")


def assert_identifier_free(
    ir: SemanticIR, source_code: str, language: str
) -> None:
    """Raise unless the IR is byte-identical under a full rename.

    Comparing the IR built from renamed source against the original is the
    only check that actually proves provenance.  A wordlist check cannot:
    half the phrase vocabulary is ordinary English ("string", "total",
    "items"), so it either misses real leaks or cries wolf.  If every
    author-chosen variable and parameter changes and the output does not,
    none of them reached the output.

    This is a statement about *names*, not about the program: member and API
    names are left in place, so a phrase that leaked ``splitlines`` would
    pass.  Those are covered by the fixture tests and by :func:`_phrase_leak`,
    which catches a phrase *copied* out of the source wholesale.

    The failure names the language and prints each differing field's original
    and renamed value, which is what a caller needs to find the heuristic that
    read the text.  Measured at 1000/1000 on the probe sample; rerun
    scripts/run_semantic_ir_probe.py to reproduce.
    """
    if ir.parse_ok and not ir.relations and not ir.transformations:
        # Nothing was extracted, so there is nothing that could have leaked.
        return
    renamed = build_semantic_ir(
        rename_identifiers(source_code, language), language, ir.unit_kind
    )
    before, after = ir.to_dict(), renamed.to_dict()
    differing = sorted(key for key in before if before[key] != after[key])
    if differing:
        detail = "\n".join(
            f"  {key}:\n    original: {before[key]!r}\n    renamed : {after[key]!r}"
            for key in differing
        )
        raise AssertionError(
            "semantic IR changed when every source identifier was renamed "
            f"({language}); fields that leaked:\n{detail}"
        )


def _all_phrases(ir: SemanticIR) -> str:
    """Every emitted phrase on one line, for provenance checks and diffing."""
    return " | ".join(
        [ir.output_behavior, ir.error_behavior, *ir.input_roles,
         *ir.domain_signals, *ir.transformations, *ir.relations,
         *ir.conditions, *ir.state_changes, *ir.side_effects,
         *ir.observable_effects]
    )


def _phrase_leak(ir: SemanticIR, source_code: str) -> str | None:
    """An identifier-shaped run of words, if the IR repeats one verbatim."""
    lines = textwrap.dedent(source_code).splitlines()
    runs: set[str] = set()
    for line in lines:
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)
        for size in range(_PHRASE_FLOOR, len(words) + 1):
            for start in range(len(words) - size + 1):
                runs.add(" ".join(words[start:start + size]))
    emitted = _all_phrases(ir)
    for run in sorted(runs, key=len, reverse=True):
        if run in emitted:
            return run
    return None
