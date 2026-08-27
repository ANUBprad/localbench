"""Tests for Stage A: deterministic behavior extraction from source code.

All tests are deterministic, network-free, and use local fixtures.
No LLM is involved.
"""

from __future__ import annotations

import ast as _ast

from localbench.workloads.code_retrieval.behavior_extraction import (
    extract_behavior_facts,
)
from localbench.workloads.code_retrieval.schemas import StructuredBehaviorFacts


# ===========================================================================
# extract_behavior_facts
# ===========================================================================


class TestExtractBehaviorFacts:
    def test_returns_structured_behavior_facts(self) -> None:
        src = (
            "def retry_with_backoff(tid, max_attempts=3):\n"
            '    """Retry a failed transaction with backoff."""\n'
            "    attempts = 0\n"
            "    while attempts < max_attempts:\n"
            "        try:\n"
            "            return process(tid)\n"
            "        except Exception:\n"
            "            attempts += 1\n"
            '    raise RuntimeError("Failed")\n'
        )
        facts = extract_behavior_facts(src)
        assert isinstance(facts, StructuredBehaviorFacts)
        assert facts.primary_purpose
        assert facts.input_summary
        assert facts.output_summary

    def test_extracts_parameters(self) -> None:
        src = (
            "def fetch(url, timeout=30, retries=3):\n"
            "    pass\n"
        )
        facts = extract_behavior_facts(src)
        assert "3 parameters" in facts.input_summary

    def test_extracts_side_effects(self) -> None:
        src = (
            "def update_state(self):\n"
            "    self.counter += 1\n"
            "    self.log.append('updated')\n"
        )
        facts = extract_behavior_facts(src)
        assert any("instance" in se for se in facts.side_effects)

    def test_extracts_raises(self) -> None:
        src = (
            "def validate(x):\n"
            "    if x < 0:\n"
            '        raise ValueError("negative")\n'
        )
        facts = extract_behavior_facts(src)
        assert "ValueError" in facts.raises

    def test_extracts_error_handling(self) -> None:
        src = (
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            '        return 0\n'
        )
        facts = extract_behavior_facts(src)
        assert "ZeroDivisionError" in facts.error_handling

    def test_extracts_control_flow_loop(self) -> None:
        src = (
            "def count(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += 1\n"
            "    return total\n"
        )
        facts = extract_behavior_facts(src)
        assert "for loop" in facts.control_flow

    def test_extracts_key_operations(self) -> None:
        src = (
            "def process(data):\n"
            "    result = transform(data)\n"
            "    return result\n"
        )
        facts = extract_behavior_facts(src)
        assert any("function call" in op for op in facts.key_operations)

    def test_invalid_syntax_returns_safe_defaults(self) -> None:
        facts = extract_behavior_facts("not valid python!!!")
        assert facts.primary_purpose
        assert facts.input_summary == "unknown"

    def test_no_side_effects(self) -> None:
        src = (
            "def pure_func(x):\n"
            "    return x * 2\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.side_effects == []

    def test_no_raises(self) -> None:
        src = (
            "def safe_func(x):\n"
            "    return x + 1\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.raises == []


class TestStructuredBehaviorFactsSchema:
    def test_fields_are_strings(self) -> None:
        facts = StructuredBehaviorFacts(
            primary_purpose="does something",
            input_summary="takes x",
            output_summary="returns y",
        )
        assert isinstance(facts.primary_purpose, str)
        assert isinstance(facts.input_summary, str)
        assert isinstance(facts.output_summary, str)

    def test_optional_fields_default_empty(self) -> None:
        facts = StructuredBehaviorFacts(
            primary_purpose="does something",
            input_summary="takes x",
            output_summary="returns y",
        )
        assert facts.side_effects == []
        assert facts.key_operations == []
        assert facts.error_handling == ""
        assert facts.control_flow == ""
        assert facts.raises == []
        assert facts.domain_concepts == []
        assert facts.observable_effects == []

    def test_frozen_dataclass(self) -> None:
        facts = StructuredBehaviorFacts(
            primary_purpose="does something",
            input_summary="takes x",
            output_summary="returns y",
        )
        import pytest

        with pytest.raises(AttributeError):
            facts.primary_purpose = "changed"  # type: ignore[misc]


# ===========================================================================
# Provenance: no implementation identifiers in extracted facts
# ===========================================================================


import re

# Pattern matching Python identifiers (snake_case, camelCase, PascalCase)
_IDENTIFIER_PATTERN = re.compile(
    r"\b(self|cls|[A-Z][a-zA-Z0-9]+|"
    r"[a-z][a-z_]+_[a-z][a-z_]+|"
    r"_[a-z][a-z_]+)\b"
)


def _contains_identifier(text: str) -> bool:
    """Check if text contains a Python identifier (name-like string)."""
    return bool(_IDENTIFIER_PATTERN.search(text))


class TestNoIdentifiersInExtractedFacts:
    """Ensure extracted facts are free of implementation identifiers."""

    def test_primary_purpose_no_function_name(self) -> None:
        src = (
            "def process_retry(self, tid, max_attempts=3):\n"
            '    """Retry a failed transaction."""\n'
            "    attempts = 0\n"
            "    while attempts < max_attempts:\n"
            "        try:\n"
            "            return self._do_process(tid)\n"
            "        except Exception:\n"
            "            attempts += 1\n"
            '    raise RuntimeError("Failed")\n'
        )
        facts = extract_behavior_facts(src)
        assert "process_retry" not in facts.primary_purpose

    def test_input_summary_no_param_names(self) -> None:
        src = (
            "def fetch(url, timeout=30, retries=3):\n"
            "    pass\n"
        )
        facts = extract_behavior_facts(src)
        assert "url" not in facts.input_summary
        assert "timeout" not in facts.input_summary
        assert "retries" not in facts.input_summary

    def test_side_effects_no_attribute_names(self) -> None:
        src = (
            "def update_state(self):\n"
            "    self.counter += 1\n"
            "    self.log.append('updated')\n"
        )
        facts = extract_behavior_facts(src)
        for se in facts.side_effects:
            assert "self.counter" not in se
            assert "self.log" not in se

    def test_key_operations_no_call_names(self) -> None:
        src = (
            "def process(data):\n"
            "    result = transform(data)\n"
            "    return result\n"
        )
        facts = extract_behavior_facts(src)
        for op in facts.key_operations:
            assert "transform" not in op

    def test_raises_still_contains_exception_names(self) -> None:
        """Exception class names are domain identifiers, not implementation identifiers."""
        src = (
            "def validate(x):\n"
            "    if x < 0:\n"
            '        raise ValueError("negative")\n'
        )
        facts = extract_behavior_facts(src)
        assert "ValueError" in facts.raises

    def test_error_handling_still_contains_exception_names(self) -> None:
        """Exception class names in error handling are domain identifiers."""
        src = (
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            '        return 0\n'
        )
        facts = extract_behavior_facts(src)
        assert "ZeroDivisionError" in facts.error_handling


# ===========================================================================
# v3.0 Regression: indented/class-nested source must not be misparsed
# ===========================================================================
# Regression for the Stage-A indentation defect that collapsed 32/45 of the
# selected v3.0 candidates into "parses as invalid code" queries: method and
# class-nested code units are stored with leading indentation and a raw
# ast.parse() raised IndentationError. These tests pin that the source is
# normalized before parsing.


class TestIndentedSourceParsesAsValid:
    def test_leading_indented_property(self) -> None:
        src = (
            "    @property\n"
            "    def example(self):\n"
            "        return self.value\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.primary_purpose != "parses as invalid code"
        assert facts.input_summary != "unknown"
        assert facts.output_summary != "unknown"

    def test_indented_method_parses(self) -> None:
        src = (
            "    def process(self, data):\n"
            "        result = transform(data)\n"
            "        return result\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.primary_purpose != "parses as invalid code"
        assert facts.input_summary != "unknown"

    def test_class_nested_source_parses(self) -> None:
        src = (
            "class Foo:\n"
            "    def bar(self, x):\n"
            "        if x in self.cache:\n"
            "            return self.cache[x]\n"
            "        return x\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.primary_purpose != "parses as invalid code"
        assert facts.input_summary != "unknown"


class TestInvalidSyntaxStillSafe:
    def test_genuine_syntax_error_returns_safe_defaults(self) -> None:
        facts = extract_behavior_facts("not valid python!!!")
        assert facts.primary_purpose == "parses as invalid code"
        assert facts.input_summary == "unknown"
        assert facts.output_summary == "unknown"

    def test_indented_genuine_syntax_error_still_invalid(self) -> None:
        facts = extract_behavior_facts("    if this is not @ valid\n")
        assert facts.primary_purpose == "parses as invalid code"
        assert facts.input_summary == "unknown"


# ===========================================================================
# v3.0 Regression: enriched identifier-free purpose extraction
# ===========================================================================


class TestEnrichedPurposeExtraction:
    def test_purpose_distinguishes_behaviors(self) -> None:
        string_src = (
            "def parse_line(text):\n"
            "    return text.split(',')\n"
        )
        container_src = (
            "def config():\n"
            '    return {"retries": 3, "timeout": 30}\n'
        )
        member_src = (
            "def is_member(x, items):\n"
            "    return x in items\n"
        )
        p_string = extract_behavior_facts(string_src).primary_purpose
        p_container = extract_behavior_facts(container_src).primary_purpose
        p_member = extract_behavior_facts(member_src).primary_purpose
        assert len({p_string, p_container, p_member}) == 3
        assert p_string != "performs an operation"
        assert p_container != "performs an operation"
        assert p_member != "performs an operation"

    def test_container_construction_informative(self) -> None:
        src = (
            'def config():\n'
            '    return {"retries": 3, "timeout": 30}\n'
        )
        assert "mapping" in extract_behavior_facts(src).primary_purpose

    def test_dict_comprehension_construction_informative(self) -> None:
        src = (
            "def index(items):\n"
            "    return {item: item for item in items}\n"
        )
        assert "mapping" in extract_behavior_facts(src).primary_purpose

    def test_comparison_informative(self) -> None:
        src = (
            "def threshold(x):\n"
            "    return x >= 10\n"
        )
        assert "comparison" in extract_behavior_facts(src).primary_purpose

    def test_validation_raises_informative(self) -> None:
        src = (
            "def must_be_positive(x):\n"
            "    if x < 0:\n"
            '        raise ValueError("negative")\n'
        )
        assert "comparison" in extract_behavior_facts(src).primary_purpose
        assert "raises on error" in extract_behavior_facts(src).primary_purpose

    def test_string_operation_informative(self) -> None:
        src = (
            "def parse_line(text):\n"
            "    return text.split(',')\n"
        )
        assert "string" in extract_behavior_facts(src).primary_purpose

    def test_membership_check_informative(self) -> None:
        src = (
            "def is_member(x, items):\n"
            "    return x in items\n"
        )
        purpose = extract_behavior_facts(src).primary_purpose
        assert "contained in a collection" in purpose


class TestNoIdentifiersLeakedByEnrichedPurpose:
    def _all_fact_text(self, facts: StructuredBehaviorFacts) -> str:
        return " ".join(
            [
                facts.primary_purpose,
                facts.input_summary,
                facts.output_summary,
                facts.error_handling,
                facts.control_flow,
                " ".join(facts.side_effects),
                " ".join(facts.key_operations),
            ]
        )

    def test_attribute_read_does_not_leak_attribute(self) -> None:
        src = (
            "class Store:\n"
            "    def current(self):\n"
            "        return self.internal_counter\n"
        )
        facts = extract_behavior_facts(src)
        text = self._all_fact_text(facts)
        assert "internal_counter" not in text
        assert not _contains_identifier(text)

    def test_function_call_does_not_leak_call_identifier(self) -> None:
        src = (
            "def wrapper(self, raw_payload):\n"
            "    return self.handle_payload(raw_payload)\n"
        )
        facts = extract_behavior_facts(src)
        text = self._all_fact_text(facts)
        assert "handle_payload" not in text
        assert "raw_payload" not in text
        assert not _contains_identifier(text)

    def test_parameter_names_never_in_facts(self) -> None:
        src = (
            "def connect(hostname, port_number, timeout_seconds=30):\n"
            "    return hostname, port_number, timeout_seconds\n"
        )
        facts = extract_behavior_facts(src)
        text = self._all_fact_text(facts)
        assert "hostname" not in text
        assert "port_number" not in text
        assert "timeout_seconds" not in text
        assert not _contains_identifier(text)

    def test_class_and_function_names_never_in_facts(self) -> None:
        src = (
            "class ConnectionManager:\n"
            "    def establish_session(self):\n"
            "        return self.open_stream()\n"
        )
        facts = extract_behavior_facts(src)
        text = self._all_fact_text(facts)
        assert "ConnectionManager" not in text
        assert "establish_session" not in text
        assert "open_stream" not in text
        assert not _contains_identifier(text)

    def test_dict_comprehension_variables_not_leaked(self) -> None:
        src = (
            "def index(items):\n"
            "    return {item: item for item in items}\n"
        )
        facts = extract_behavior_facts(src)
        text = self._all_fact_text(facts)
        assert "item" not in text
        assert not _contains_identifier(text)


# ===========================================================================
# v3.0 Regression: existing consumers remain compatible and docstring-blind
# ===========================================================================


class TestConsumerCompatibility:
    def test_schema_unchanged(self) -> None:
        from dataclasses import fields

        field_names = {f.name for f in fields(StructuredBehaviorFacts)}
        expected = {
            "primary_purpose",
            "input_summary",
            "output_summary",
            "side_effects",
            "key_operations",
            "error_handling",
            "control_flow",
            "raises",
        }
        assert expected.issubset(field_names)

    def test_prompt_builder_consumes_new_facts_and_stays_docstring_blind(
        self,
    ) -> None:
        from localbench.workloads.code_retrieval.query_prompt import (
            build_query_generation_prompt_v2,
        )

        src = (
            "class Store:\n"
            "    def current(self):\n"
            "        return self.internal_counter\n"
        )
        facts = extract_behavior_facts(src)
        prompt = build_query_generation_prompt_v2(
            facts=facts,
            source_code=src,
            symbol_type="method",
            class_name="Store",
        )
        assert "parses as invalid code" not in prompt
        assert "internal_counter" not in prompt
        assert src not in prompt
        assert "reads a value from an associated object" in prompt

    def test_provenance_and_meta_query_checks_unchanged(self) -> None:
        from localbench.workloads.code_retrieval.query_generator import (
            check_meta_query,
            check_query_provenance,
        )

        assert check_meta_query(
            "Can you generate a retrieval query for this code?"
        )
        assert not check_meta_query("Where does this function get its config?")
        assert check_query_provenance(
            "quality the exact docstring sentence appears verbatim here",
            "the exact docstring sentence appears verbatim here",
        ).passed is False
        assert check_query_provenance(
            "a totally independent query about configuration", "unrelated docstring"
        ).passed


class TestInvestigationDerivedRegression:
    """Cases modelled on the 45 selected v3.0 candidates."""

    def test_indented_deleting_utility_not_invalid(self) -> None:
        src = (
            "    def maybe_delete_a_numbered_dir(self, path):\n"
            "        try:\n"
            "            self.remove_dir(path)\n"
            "        except OSError:\n"
            "            return\n"
            "        finally:\n"
            "            self.cleanup()\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.primary_purpose != "parses as invalid code"
        assert facts.input_summary != "unknown"

    def test_indented_delegating_wrapper_not_invalid(self) -> None:
        src = (
            "    def genitems(self, colitems):\n"
            "        return self._pytester.genitems(colitems)\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.primary_purpose != "parses as invalid code"
        assert "delegates" in facts.primary_purpose
        assert "pytester" not in facts.primary_purpose


# ===========================================================================
# v3.2 regression: identifier-free domain concepts & observable effects
# ===========================================================================


class TestDomainConceptsExtraction:
    """Domain concepts derived from imports and call patterns are generic and
    deterministic and never surface implementation identifiers."""

    def test_empty_without_imports(self) -> None:
        src = "def f():\n    pass\n"
        facts = extract_behavior_facts(src)
        assert facts.domain_concepts == []

    def test_rich_console_import(self) -> None:
        src = "def render():\n    console.print('hi')\n"
        facts = extract_behavior_facts(src, imports=["rich.console"])
        assert "terminal/console rendering" in facts.domain_concepts

    def test_subprocess_import(self) -> None:
        src = "def run(cmd):\n    return subprocess.run(cmd)\n"
        facts = extract_behavior_facts(src, imports=["subprocess"])
        assert "subprocess/process execution" in facts.domain_concepts

    def test_urllib_http_import(self) -> None:
        src = "def grab(url):\n    return urlopen(url)\n"
        facts = extract_behavior_facts(src, imports=["urllib.request"])
        assert "HTTP transfer" in facts.domain_concepts

    def test_ctypes_native_import(self) -> None:
        src = "def f():\n    return ctypes.cdll.LoadLibrary('x')\n"
        facts = extract_behavior_facts(src, imports=["ctypes"])
        assert "native OS/terminal API calls" in facts.domain_concepts

    def test_dup2_call_concept(self) -> None:
        src = (
            "def capture():\n"
            "    os.dup2(fd, 1)\n"
        )
        facts = extract_behavior_facts(src, imports=["os", "sys"])
        assert "file-descriptor redirection" in facts.domain_concepts

    def test_cursor_placement_via_call(self) -> None:
        src = (
            "def move(handle, pos):\n"
            "    ctypes.windll.kernel32._SetConsoleCursorPosition(handle, pos)\n"
        )
        facts = extract_behavior_facts(src, imports=["ctypes"])
        assert "terminal cursor positioning" in facts.domain_concepts

    def test_determinism(self) -> None:
        src = "def render():\n    console.print('hi')\n"
        a = extract_behavior_facts(src, imports=["rich.console"])
        b = extract_behavior_facts(src, imports=["rich.console"])
        assert a.domain_concepts == b.domain_concepts
        assert a.observable_effects == b.observable_effects

    def test_docstring_independent(self) -> None:
        plain = (
            "def render():\n"
            "    console.print('hi')\n"
        )
        with_doc = (
            "def render():\n"
            '    """Render the company dashboard to the terminal."""\n'
            "    console.print('hi')\n"
        )
        a = extract_behavior_facts(plain, imports=["rich.console"])
        b = extract_behavior_facts(with_doc, imports=["rich.console"])
        assert a.domain_concepts == b.domain_concepts
        assert a.observable_effects == b.observable_effects

    def test_generic_purpose_gets_domain_lead(self) -> None:
        src = "def f():\n    do_thing()\n"
        facts = extract_behavior_facts(src, imports=["subprocess"])
        assert "subprocess" in facts.primary_purpose or (
            "process" in facts.primary_purpose
        )


class TestObservableEffectsExtraction:
    """Observable effects are emitted only with structural evidence."""

    def test_print_effect(self) -> None:
        src = "def show():\n    print('hi')\n"
        facts = extract_behavior_facts(src)
        assert "prints to the console" in facts.observable_effects

    def test_dup2_capture_effect(self) -> None:
        src = (
            "def capture(self):\n"
            "    os.dup2(self.fd, 1)\n"
        )
        facts = extract_behavior_facts(src, imports=["os"])
        assert "captures standard output/error" in facts.observable_effects

    def test_subprocess_launch_effect(self) -> None:
        src = "def go(cmd):\n    subprocess.Popen(cmd)\n"
        facts = extract_behavior_facts(src, imports=["subprocess"])
        assert "launches a subprocess" in facts.observable_effects

    def test_cursor_movement_effect(self) -> None:
        src = (
            "def move(handle, pos):\n"
            "    ctypes.windll.kernel32._SetConsoleCursorPosition(handle, pos)\n"
        )
        facts = extract_behavior_facts(src, imports=["ctypes"])
        assert "moves the terminal cursor" in facts.observable_effects

    def test_http_download_effect(self) -> None:
        src = (
            "def copy(url):\n"
            "    return urllib.request.urlretrieve(url)\n"
        )
        facts = extract_behavior_facts(src, imports=["urllib.request"])
        assert "transfers data over HTTP to a local file" in facts.observable_effects

    def test_table_rendering_effect(self) -> None:
        src = (
            "def show():\n"
            "    t = table.Table()\n"
            "    t.add_column('title')\n"
        )
        facts = extract_behavior_facts(src, imports=["rich.table"])
        assert "renders a table of items to the console" in facts.observable_effects

    def test_no_fabricated_effects(self) -> None:
        src = (
            "def total(items):\n"
            "    return sum(items)\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.observable_effects == []
        assert facts.domain_concepts == []


class TestDomainEffectsIdentifierFree:
    """The new fields must never surface implementation identifiers.

    Mirrors the production leak check: identifier-like tokens in the fact text
    must not overlap with identifiers actually present in the source code.
    """

    def _source_identifiers(self, src: str) -> set[str]:
        """Collect implementation identifiers present in the source text."""
        tree = _ast.parse(src)
        ids: set[str] = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
                ids.add(node.name)
            elif isinstance(node, _ast.Name):
                ids.add(node.id)
            elif isinstance(node, _ast.arg):
                ids.add(node.arg)
            elif isinstance(node, _ast.Attribute):
                ids.add(node.attr)
        return ids

    def _assert_new_fields_identifier_free(self, src, imports=None) -> None:
        facts = extract_behavior_facts(src, imports=imports)
        text = " ".join(facts.domain_concepts + facts.observable_effects)
        src_ids = self._source_identifiers(src)
        fact_words = set(_IDENTIFIER_PATTERN.findall(text))
        overlap = fact_words & src_ids
        assert overlap == set(), f"identifier leak in new fields: {overlap}"

    def test_rich_table_unit_identifier_free(self) -> None:
        src = (
            "def print_table(movie_rows):\n"
            "    t = table.Table()\n"
            "    for row in movie_rows:\n"
            "        t.add_column(row[0])\n"
        )
        self._assert_new_fields_identifier_free(
            src, imports=["rich.table", "rich.console"]
        )

    def test_fd_capture_unit_identifier_free(self) -> None:
        src = (
            "def start(self, fd):\n"
            "    self.old = os.dup2(fd, 1)\n"
        )
        self._assert_new_fields_identifier_free(src, imports=["os", "sys"])

    def test_cursor_unit_identifier_free(self) -> None:
        src = (
            "def SetConsoleCursorPosition(handle, pos):\n"
            "    ctypes.windll.kernel32._SetConsoleCursorPosition(handle, pos)\n"
        )
        self._assert_new_fields_identifier_free(src, imports=["ctypes"])

    def test_copy_url_unit_identifier_free(self) -> None:
        src = (
            "def copy_url(src_url, dst_path):\n"
            "    return urllib.request.urlretrieve(src_url, dst_path)\n"
        )
        self._assert_new_fields_identifier_free(src, imports=["urllib.request"])

    def test_domain_class_name_not_emitted(self) -> None:
        src = (
            "def render():\n"
            "    console.print('hi')\n"
        )
        facts = extract_behavior_facts(
            src, imports=["rich.console", "rich.windows"]
        )
        for phrase in facts.domain_concepts:
            assert "Console" not in phrase
            assert "Windows" not in phrase

    def test_effects_identifier_free_when_domain_present(self) -> None:
        src = (
            "def run_steps(steps):\n"
            "    for step in steps:\n"
            "        progress.advance(step)\n"
        )
        self._assert_new_fields_identifier_free(src, imports=["rich.progress"])


class TestCanaryDerivedDomainFacts:
    """Assertions grounded in the real 48-unit canary extraction results."""

    def test_fd_capture_start(self) -> None:
        src = (
            "def start(self):\n"
            "    os.dup2(self.fd, 1)\n"
        )
        facts = extract_behavior_facts(src, imports=["os", "sys"])
        assert "file-descriptor redirection" in facts.domain_concepts
        assert "captures standard output/error" in facts.observable_effects

    def test_set_console_cursor_position(self) -> None:
        src = (
            "def SetConsoleCursorPosition(handle, pos):\n"
            "    ctypes.windll.kernel32._SetConsoleCursorPosition(handle, pos)\n"
        )
        facts = extract_behavior_facts(src, imports=["ctypes"])
        assert "terminal cursor positioning" in facts.domain_concepts
        assert "moves the terminal cursor" in facts.observable_effects

    def test_print_table(self) -> None:
        src = (
            "def print_table(rows):\n"
            "    t = table.Table()\n"
            "    for row in rows:\n"
            "        t.add_column(row[0])\n"
            "    console.print(t)\n"
        )
        facts = extract_behavior_facts(src, imports=["rich.table", "rich.console"])
        assert "tabulated console rendering" in facts.domain_concepts
        assert "tabulated row/column layout" in facts.domain_concepts

    def test_copy_url(self) -> None:
        src = (
            "def copy_url(src, dst):\n"
            "    return urllib.request.urlretrieve(src, dst)\n"
        )
        facts = extract_behavior_facts(src, imports=["urllib.request", "rich.progress"])
        assert "HTTP transfer" in facts.domain_concepts
        assert "transfers data over HTTP to a local file" in facts.observable_effects

    def test_text_alignment(self) -> None:
        src = (
            "def align_center(text, width):\n"
            "    return text.align('center', width)\n"
        )
        facts = extract_behavior_facts(src, imports=["rich.text"])
        assert "text formatting" in facts.domain_concepts
        assert "formats text for display and alignment" in facts.observable_effects

