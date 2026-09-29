"""Tests for the semantic-IR pilot (EXPERIMENTAL, not production).

The fixtures use deliberately distinctive identifier names so that
``assert_identifier_free`` is a real check rather than a formality: if the
extractor ever interpolated a name into a phrase, these names would be
caught.

Assertions are semantic (category membership, relation shape), never exact
prose, except for the short contract phrases the fixtures pin deliberately.
"""

from __future__ import annotations

import textwrap

import pytest

from localbench.workloads.code_retrieval import semantic_ir_experiment
from localbench.workloads.code_retrieval.behavior_extraction import (
    extract_behavior_facts,
)
from localbench.workloads.code_retrieval.semantic_ir_experiment import (
    SUPPORTED_LANGUAGES,
    SemanticIR,
    assert_identifier_free,
    build_semantic_ir,
    informativeness,
    rename_identifiers,
)

# ---------------------------------------------------------------------------
# Fixtures: one per pilot language, plus deliberately trivial/hostile cases
# ---------------------------------------------------------------------------

PYTHON_TEXT = '''\
def render_report(zephyr_payload):
    """Docstring is deliberately present; it must never reach the IR."""
    warmed = zephyr_payload.splitlines()
    kept = [row.strip() for row in warmed if row.strip()]
    return "\\n".join(kept)
'''

PYTHON_CACHE = '''\
def zephyr_lookup(quartz_key, quartz_source):
    if quartz_source.get(quartz_key) is not None:
        return quartz_source[quartz_key]
    hydrated = quartz_source.get(quartz_key)
    if hydrated is not None:
        return hydrated
    computed = zephyr_transform(quartz_key)
    quartz_source[quartz_key] = computed
    return computed
'''

JAVA_CACHE = '''\
public class Vault {
    private final Map<String, String> reservoir = new HashMap<>();

    public String hydrate(String vellumKey) {
        if (reservoir.containsKey(vellumKey)) {
            return reservoir.get(vellumKey);
        }
        String emberValue = vellumFetch(vellumKey);
        reservoir.put(vellumKey, emberValue);
        return emberValue;
    }
}
'''

GO_MANIFEST = '''\
package ledger

func ingest(beaconPath string) ([]Beacon, error) {
    beaconBytes, beaconErr := os.ReadFile(beaconPath)
    if beaconErr != nil {
        return nil, beaconErr
    }
    var gathered []Beacon
    for _, beaconLine := range strings.Split(string(beaconBytes), "\\n") {
        if strings.TrimSpace(beaconLine) == "" {
            continue
        }
        gathered = append(gathered, Beacon{Name: beaconLine})
    }
    return gathered, nil
}
'''

RUST_TOTAL = '''\
fn tally(cobalt_items: &[i32], cobalt_done: i32) -> Result<i32, String> {
    let cobalt_sum: i32 = cobalt_items.iter().sum();
    if cobalt_sum < 0 {
        return Err("negative total".to_string());
    }
    Ok(cobalt_sum - cobalt_done)
}
'''

JAVASCRIPT_PULL = '''\
async function hydrate(nimbusUrl) {
    const nimbusReply = await fetch(nimbusUrl);
    if (!nimbusReply.ok) {
        throw new Error("request failed");
    }
    const nimbusBody = await nimbusReply.json();
    return nimbusBody.items.map(nimbusRow => nimbusRow.name.trim());
}
'''

TRIVIAL_PYTHON = '''\
def ordinality(nimbus_counter):
    nimbus_counter = nimbus_counter + 1
    return nimbus_counter
'''

#: Author-chosen names that collide with the vocabulary the classifiers used
#: to read out of source text.  Each one produced a real leak: a local named
#: ``match`` read as a type dispatch, a locally defined ``filter`` read as
#: the builtin, and a parameter named ``url`` was matched against the ``url``
#: in ``self.url``.  Renaming any of them used to change the IR.
ADVERSARIAL: dict[str, str] = {
    "python": '''\
def gather(zephyr_items):
    match = zephyr_items
    if match is None:
        return []
    length = len(match)
    if length == 0:
        return []
    error = None
    kept = [row for row in match if row]
    if not kept:
        error = "empty"
    return kept
''',
    "java": '''\
public class Ledger {
    public List<String> gather(List<String> items) {
        int length = items.size();
        if (length == 0) {
            return new ArrayList<>();
        }
        return items;
    }
}
''',
    "go": '''\
package ledger

func gather(items []string) ([]string, error) {
\tlength := len(items)
\tif length == 0 {
\t\treturn nil, nil
\t}
\treturn items, nil
}
''',
    "rust": '''\
fn gather(items: &[String]) -> usize {
    let length = items.len();
    if length == 0 {
        return 0;
    }
    items.len()
}
''',
    "javascript": '''\
function gather(items) {
    const length = items.length;
    if (length === 0) {
        return [];
    }
    const error = null;
    return items.filter((item) => item.length > 0);
}
''',
}


FIXTURES: dict[str, str] = {
    "python": PYTHON_TEXT,
    "java": JAVA_CACHE,
    "go": GO_MANIFEST,
    "rust": RUST_TOTAL,
    "javascript": JAVASCRIPT_PULL,
}


def _ir(source: str, language: str, **kwargs) -> SemanticIR:
    return build_semantic_ir(source, language, **kwargs)


_all_phrases = semantic_ir_experiment._all_phrases


# ---------------------------------------------------------------------------
# 1. all five languages parse, and the function boundary is found
# ---------------------------------------------------------------------------


class TestCrossLanguageParsing:
    def test_pilot_covers_exactly_five_languages(self) -> None:
        assert set(SUPPORTED_LANGUAGES) == {
            "python", "java", "go", "rust", "javascript",
        }

    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_fixture_parses_cleanly_and_finds_the_unit(self, language) -> None:
        ir = _ir(FIXTURES[language], language)
        assert ir.parse_ok is True, ir
        assert ir.language == language
        assert ir.unit_kind in {"function", "method"}
        assert ir.input_roles, "no input role recovered"
        assert ir.transformations or ir.state_changes

    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_docstrings_never_reach_the_ir(self, language) -> None:
        ir = _ir(FIXTURES[language], language)
        assert "docstring" not in _all_phrases(ir).lower()
        assert "deliberately present" not in _all_phrases(ir).lower()

    def test_unit_kind_override_is_respected(self) -> None:
        # A method extracted without its class is indistinguishable from a
        # free function in the tree, so the dataset's symbol_type wins.
        extracted = textwrap.indent(PYTHON_TEXT, "    ")
        assert _ir(extracted, "python").unit_kind == "function"
        assert _ir(extracted, "python", unit_kind="method").unit_kind == "method"

    def test_unsupported_language_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported language"):
            _ir("def f(): pass", "cobol")


# ---------------------------------------------------------------------------
# 2. identifier sanitization (Section 6)
# ---------------------------------------------------------------------------

#: one API name per language that the rename must leave alone
_MEMBER_NAMES = {
    "python": ("splitlines", "join", "strip"),
    "java": ("containsKey", "put", "get"),
    "go": ("ReadFile", "Split", "TrimSpace"),
    "rust": ("sum", "to_string"),
    "javascript": ("fetch", "json", "trim"),
}


class TestIdentifierSanitization:
    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_no_source_identifier_leaks(self, language) -> None:
        source = FIXTURES[language]
        ir = _ir(source, language)
        assert_identifier_free(ir, source, language)

    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_rename_stubs_variables_but_keeps_api_names(self, language) -> None:
        # The guard is only meaningful if the rename actually renames the
        # right things: variables become stubs, and the API names that supply
        # the evidence survive untouched.
        renamed = rename_identifiers(FIXTURES[language], language)
        assert "qqqqzzz" in renamed
        for member in _MEMBER_NAMES[language]:
            assert member in renamed, f"rename clobbered the API name {member!r}"

    def test_internal_api_detection_becomes_an_identifier_free_phrase(self) -> None:
        # Internally: splitlines.  Emitted: a behavior phrase, never the name.
        source = "def nimbus_split(nimbus_text):\n    return nimbus_text.splitlines()\n"
        ir = _ir(source, "python")
        assert "splits textual content into lines" in ir.transformations
        assert "splitlines" not in _all_phrases(ir)

    def test_json_api_becomes_a_parse_semantic(self) -> None:
        source = (
            "def nimbus_load(nimbus_blob):\n"
            "    return json.loads(nimbus_blob)\n"
        )
        ir = _ir(source, "python")
        assert "JSON parsing" in ir.domain_signals
        assert "parses JSON data" in ir.transformations
        # "parses JSON data" legitimately contains the English word; what must
        # not appear is the API's own name.
        assert "loads" not in _all_phrases(ir)
        assert "json.loads" not in _all_phrases(ir)

    def test_directory_names_do_not_leak(self) -> None:
        ir = _ir(PYTHON_TEXT, "python")
        for token in ("zephyr", "render_report", "payload", "warmed"):
            assert token not in _all_phrases(ir)

    def test_phrase_leak_catches_a_copied_source_phrase(self) -> None:
        # The rename guard cannot see this one: a phrase copied out of the
        # source verbatim survives renaming intact.
        source = (
            "def nimbus():\n"
            "    return 'nimbus strips surrounding whitespace from text'\n"
        )
        leaked = SemanticIR(
            language="python",
            unit_kind="function",
            parse_ok=True,
            input_roles=(),
            output_behavior="nimbus strips surrounding whitespace from text",
            domain_signals=(),
            transformations=(),
            relations=(),
            conditions=(),
            state_changes=(),
            side_effects=(),
            error_behavior="no explicit error handling is visible",
            observable_effects=(),
        )
        assert semantic_ir_experiment._phrase_leak(leaked, source) is not None
        # The real extractor never produces that phrase for this source.
        assert semantic_ir_experiment._phrase_leak(
            _ir(source, "python"), source
        ) is None


class TestAdversarialAuthorNames:
    """Names an author may pick that collide with classifier vocabulary."""

    @pytest.mark.parametrize("language", sorted(ADVERSARIAL))
    def test_ir_is_identifier_free(self, language) -> None:
        source = ADVERSARIAL[language]
        ir = _ir(source, language)
        assert ir.parse_ok is True, ir
        assert_identifier_free(ir, source, language)

    @pytest.mark.parametrize("language", sorted(ADVERSARIAL))
    def test_renaming_them_changes_the_source(self, language) -> None:
        # Otherwise the first test passes for the wrong reason: if nothing is
        # renamed, nothing can leak.
        renamed = rename_identifiers(ADVERSARIAL[language], language)
        assert "qqqqzzz" in renamed, renamed

    def test_local_named_match_is_not_a_type_branch(self) -> None:
        # `if match is None` is a null check.  A regex for the *words* of a
        # match expression could not tell it from one.
        ir = _ir(ADVERSARIAL["python"], "python")
        assert "branches on a type or kind of value" not in ir.conditions

    def test_locally_defined_filter_is_not_the_builtin(self) -> None:
        source = '''\
def screen(zephyr_items):
    def filter(rows):
        return [row for row in rows if row]

    return filter(zephyr_items)
'''
        ir = _ir(source, "python")
        assert "keeps only the items of a collection that match a predicate" \
            not in ir.transformations
        assert_identifier_free(ir, source, "python")

    def test_member_named_url_does_not_credit_a_url_parameter(self) -> None:
        source = '''\
class Session:
    def hydrate(self, url):
        return load(self.url)
'''
        ir = _ir(source, "python", unit_kind="method")
        assert "collection input" not in ir.input_roles
        assert_identifier_free(ir, source, "python")

    def test_guard_reports_the_language_and_the_differing_field(self) -> None:
        source = "def nimbus(zephyr_blob):\n    return zephyr_blob.strip()\n"
        ir = _ir(source, "python")
        tampered = SemanticIR(**{**ir.to_dict(),
                                  "output_behavior": "a leaked value"})
        with pytest.raises(AssertionError) as caught:
            assert_identifier_free(tampered, source, "python")
        message = str(caught.value)
        assert "(python)" in message
        assert "output_behavior" in message
        assert "a leaked value" in message


# ---------------------------------------------------------------------------
# 3. semantic anchors, not renamed syntax facts (Section 5)
# ---------------------------------------------------------------------------


class TestSemanticAnchors:
    def test_syntax_facts_do_not_masquerade_as_semantics(self) -> None:
        ir = _ir(TRIVIAL_PYTHON, "python")
        for banned in (
            "performs 1 method call",
            "invokes 1 function call",
            "contains 2 if",
            "uses a for loop",
            "method call",
            "function call",
        ):
            assert banned not in _all_phrases(ir)
        # The only real signal is that a derived number comes back out.
        assert "derived quantity" in ir.output_behavior

    def test_text_transformation_is_described_as_behavior(self) -> None:
        ir = _ir(PYTHON_TEXT, "python")
        assert "splits textual content into lines" in ir.transformations
        assert "trims surrounding whitespace from text" in ir.transformations
        assert "joins a collection of items into one delimited text" in (
            ir.transformations
        )
        assert "string/text processing" in ir.domain_signals

    def test_java_cache_lookup_is_detected(self) -> None:
        ir = _ir(JAVA_CACHE, "java")
        assert "caching" in ir.domain_signals
        assert "checks whether a stored mapping holds a key" in (
            ir.transformations
        )
        assert "stores a value under a key in a mapping" in ir.transformations
        assert "previously stored value" in ir.output_behavior
        assert any(
            "branches on whether a stored entry is present" in c
            for c in ir.conditions
        )

    def test_go_json_and_iteration_signals(self) -> None:
        ir = _ir(GO_MANIFEST, "go")
        assert "file I/O" in ir.domain_signals
        assert "reads a file's bytes into memory" in ir.transformations
        assert "splits textual content on a delimiter" in ir.transformations
        assert "trims surrounding whitespace from text" in ir.transformations
        assert "adds an item to a collection" in ir.transformations
        assert "repeats its work over a collection of items" in ir.transformations

    def test_rust_result_and_error_signals(self) -> None:
        ir = _ir(RUST_TOTAL, "rust")
        assert "aggregates numeric items into a single total" in ir.transformations
        assert "collection input" in ir.input_roles
        assert "numeric input" in ir.input_roles
        assert "reports failure in a dedicated return value" in ir.error_behavior
        assert "branches on a value comparison" in ir.conditions

    def test_javascript_http_then_transform_chain(self) -> None:
        ir = _ir(JAVASCRIPT_PULL, "javascript")
        assert "HTTP/networking" in ir.domain_signals
        assert "JSON parsing" in ir.domain_signals
        assert "issues an HTTP request to a remote endpoint" in ir.transformations
        assert "parses JSON data" in ir.transformations
        assert "transforms each item of a collection" in ir.transformations
        assert "signals a failure to its caller by raising" in ir.error_behavior


# ---------------------------------------------------------------------------
# 4. relational semantics (Section 9) — the point of the pilot
# ---------------------------------------------------------------------------


class TestRelations:
    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_every_fixture_yields_at_least_one_linked_relation(
        self, language
    ) -> None:
        ir = _ir(FIXTURES[language], language)
        assert ir.relations, "no input->transformation->output link found"
        for relation in ir.relations:
            assert relation.startswith(("textual", "numeric", "collection",
                                        "mapping", "flag", "opaque", "remote",
                                        "effectful", "file"))
            assert " -> returns " in relation

    def test_chain_preserves_input_transformation_output_order(self) -> None:
        ir = _ir(JAVASCRIPT_PULL, "javascript")
        linked = next(r for r in ir.relations if "issues an HTTP request" in r)
        steps = linked.split(" -> ")
        assert steps[0].endswith("input")
        assert steps.index("parses JSON data") < steps.index(
            "transforms each item of a collection"
        )
        assert steps[-1].startswith("returns ")

    def test_text_to_lines_relation_for_python(self) -> None:
        ir = _ir(PYTHON_TEXT, "python")
        assert any(
            "textual input -> " in r
            and "splits textual content into lines" in r
            and r.endswith("returns one delimited text")
            for r in ir.relations
        )

    def test_rust_relation_links_the_collection_input_to_the_total(self) -> None:
        ir = _ir(RUST_TOTAL, "rust")
        assert any(
            r.startswith("collection input -> ")
            and "aggregates numeric items into a single total" in r
            for r in ir.relations
        )

    def test_unlinked_unit_says_so_instead_of_inventing_a_link(self) -> None:
        ir = _ir(
            "def nimbus_passthrough(nimbus_value):\n    return nimbus_value\n",
            "python",
        )
        assert any("no input-to-output link is visible" in r for r in ir.relations)


# ---------------------------------------------------------------------------
# 5. observable effects and error behavior (Sections 7.9, 7.10)
# ---------------------------------------------------------------------------


class TestEffectsAndErrors:
    def test_observable_effects_are_observer_facing(self) -> None:
        js = _ir(JAVASCRIPT_PULL, "javascript")
        assert "exchanges data with a remote service over the network" in (
            js.observable_effects
        )
        go = _ir(GO_MANIFEST, "go")
        assert "depends on files on disk" in go.observable_effects

    def test_file_write_is_reported_as_a_disk_effect(self) -> None:
        source = (
            "def nimbus_dump(nimbus_rows, nimbus_target):\n"
            "    with open(nimbus_target, 'w') as nimbus_stream:\n"
            "        nimbus_stream.write(nimbus_rows)\n"
        )
        ir = _ir(source, "python")
        assert "file I/O" in ir.domain_signals
        assert "produces or modifies a file on disk" in ir.observable_effects
        assert "writes data to a file or stream" in ir.side_effects

    def test_missing_parent_directory_creation_is_visible(self) -> None:
        source = (
            "def nimbus_dump(nimbus_target):\n"
            "    os.makedirs(nimbus_target, exist_ok=True)\n"
            "    open(nimbus_target, 'w').write('x')\n"
        )
        ir = _ir(source, "python")
        assert "creates a directory and any missing parents" in ir.transformations

    def test_console_output_is_an_observable_effect(self) -> None:
        source = "def nimbus_say(nimbus_text):\n    print(nimbus_text)\n"
        ir = _ir(source, "python")
        assert "terminal/console" in ir.domain_signals
        assert "emits output visible on the console or terminal" in (
            ir.observable_effects
        )

    def test_logging_is_a_separate_domain_from_console(self) -> None:
        source = (
            "def nimbus_note(nimbus_text):\n"
            "    logging.warning(nimbus_text)\n"
        )
        ir = _ir(source, "python")
        assert "logging" in ir.domain_signals
        assert "leaves a diagnostic record in a log" in ir.observable_effects

    def test_try_except_is_reported_as_recovery(self) -> None:
        source = (
            "def nimbus_safe(nimbus_blob):\n"
            "    try:\n"
            "        return json.loads(nimbus_blob)\n"
            "    except ValueError:\n"
            "        return {}\n"
        )
        ir = _ir(source, "python")
        assert "catches failures and falls back to a defined result" in (
            ir.error_behavior
        )

    def test_rust_question_mark_propagates_failure(self) -> None:
        source = (
            "fn nimbus_slurp(nimbus_path: &str) -> Result<String, String> {\n"
            "    let nimbus_text = std::fs::read_to_string(nimbus_path)?;\n"
            "    Ok(nimbus_text)\n"
            "}\n"
        )
        ir = _ir(source, "rust")
        assert "propagates a failure from a nested operation to its caller" in (
            ir.error_behavior
        )

    def test_absent_error_handling_is_stated_not_invented(self) -> None:
        ir = _ir(TRIVIAL_PYTHON, "python")
        assert ir.error_behavior.startswith("no explicit failure handling")


# ---------------------------------------------------------------------------
# 6. informativeness (Section 10)
# ---------------------------------------------------------------------------


class TestInformativeness:
    def test_anchor_count_is_positive_and_breakdown_sums(self) -> None:
        for language, source in FIXTURES.items():
            score = informativeness(_ir(source, language))
            assert score.anchor_count > 0, language
            assert sum(score.breakdown.values()) == score.anchor_count
            assert "relations" in score.breakdown
            assert "observable_effects" in score.breakdown

    def test_semantically_richer_unit_outscores_a_trivial_one(self) -> None:
        rich = informativeness(_ir(JAVASCRIPT_PULL, "javascript"))
        thin = informativeness(_ir(TRIVIAL_PYTHON, "python"))
        assert rich.anchor_count > thin.anchor_count

    def test_no_production_threshold_is_encoded(self) -> None:
        # The pilot must not decide what is "good enough"; it only counts.
        score = informativeness(_ir(PYTHON_TEXT, "python"))
        assert not hasattr(score, "passed")
        assert not hasattr(score, "is_sufficient")


# ---------------------------------------------------------------------------
# 7. robustness: malformed input, determinism, isolation
# ---------------------------------------------------------------------------


class TestRobustness:
    @pytest.mark.parametrize("broken", [
        "def nimbus(:\n  return ???\n",
        "class {{{",
        "}}}\n)))\n",
        "",
    ])
    def test_malformed_source_fails_safely(self, broken) -> None:
        ir = _ir(broken, "python")
        assert isinstance(ir, SemanticIR)
        assert ir.parse_ok is False
        assert ir.transformations == ()
        informativeness(ir)  # must not raise

    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_malformed_non_python_source_fails_safely(self, language) -> None:
        ir = _ir("((((( === )))", language)
        assert ir.parse_ok is False

    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_identical_input_produces_identical_ir(self, language) -> None:
        first = _ir(FIXTURES[language], language).to_dict()
        second = _ir(FIXTURES[language], language).to_dict()
        assert first == second
        assert informativeness(_ir(FIXTURES[language], language)).anchor_count == (
            informativeness(_ir(FIXTURES[language], language)).anchor_count
        )

    def test_indented_unit_source_is_not_treated_as_malformed(self) -> None:
        ir = _ir(textwrap.indent(PYTHON_TEXT, "        "), "python")
        assert ir.parse_ok is True
        assert "splits textual content into lines" in ir.transformations


# ---------------------------------------------------------------------------
# 8. production isolation (Section 16)
# ---------------------------------------------------------------------------


class TestProductionIsolation:
    def test_production_facts_still_work_for_the_same_source(self) -> None:
        # Stage A is untouched: the same source still yields the old facts.
        facts = extract_behavior_facts(PYTHON_TEXT)
        assert facts.primary_purpose
        assert isinstance(facts.key_operations, list)

    def test_ir_keeps_information_the_old_facts_drop(self) -> None:
        # The whole hypothesis: the generic facts lose the behavior.
        facts = extract_behavior_facts(PYTHON_TEXT)
        ir = _ir(PYTHON_TEXT, "python")
        assert "splits textual content into lines" in ir.transformations
        assert not any("split" in op for op in facts.key_operations)
        assert "joins a collection of items into one delimited text" in (
            ir.transformations
        )
        assert not any("join" in op for op in facts.key_operations)
        assert ir.relations, "IR has relations the fact bag cannot express"
